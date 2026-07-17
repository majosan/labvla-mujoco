项目路径：~/projects/labvla-mujoco/
  ▎ 目标：在 WSL/Linux 环境下，为 LabVLA 4-bit 量化推理实验 搭建可用的 Conda 环境
  ▎ 当前硬件：NVIDIA GeForce RTX 4060 Ti
  ▎ 当前验证结果：PyTorch + CUDA 12.4 已可用

  ---
  一、为什么从 venv 改成 conda
  
  原来的 venv 方案主要卡在两类问题：

  1. 现有 .venv 不完整，缺少标准激活脚本
  2. pip 安装 torch + cu126 时下载过慢且不稳定
  3. GPU 版 PyTorch 在 pip 路径下更容易遇到 CUDA wheel、依赖和网络问题

  改用 conda 的优点：

  - 安装 GPU 版 PyTorch 更稳定
  - CUDA 运行时依赖更容易配齐
  - 更适合后续 bitsandbytes、flash_attn、transformers 这类依赖组合
  - 对 LabVLA 这种量化推理实验更省排障时间

  ---
  二、保留原 .venv，不要先删除
 
  先保留项目根目录原有的：

  ~/projects/labvla-mujoco/.venv

  原因：

  - 它不会影响 conda 环境使用
  - 便于回退和对照
  - 在 conda 方案完全跑通前，不建议删除旧环境

  ---
  三、安装 Miniforge
 
  如果系统里还没有 conda，先安装 Miniforge。

  1）下载安装脚本

  如果终端能访问 GitHub：
  
  curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o ~/Miniforge3.sh

  如果终端下载慢，也可以直接用浏览器下载到主目录。

  2）执行安装
  
  bash ~/Miniforge3.sh

  安装时建议：

  - 安装路径使用默认值：~/miniforge3
  - shell 初始化选 yes

  3）让当前 shell 生效

  source ~/miniforge3/bin/activate
  conda --version

  看到版本号说明安装成功。

  ---
  四、创建新的 Conda 环境
 
  不要直接复用旧的 labvla 环境，建议新建一个明确带 CUDA 12.4 的环境：

  conda create -n labvla-cu124 python=3.10 -y
  conda activate labvla-cu124
  python --version

  预期输出应接近：

  Python 3.10.x

  ---
  五、安装 GPU 版 PyTorch
 
  ▎ 注意：pytorch-cuda=12.6 在当前 conda 源里不可用，所以改用 12.4。

  执行：

  conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia --strict-channel-priority -y

  为什么用 --strict-channel-priority

  这样能减少 solver 从其他 channel 里拿到 CPU 版 torch 的概率。
  
  ---
  六、验证 PyTorch 和 CUDA 是否正常
 
  先确认系统 GPU 可见：

  nvidia-smi

  然后确认 conda 环境中的 torch 已启用 CUDA：

  python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('torch cuda:', torch.version.cuda); 
  print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

  你当前已经验证通过，预期类似：
  
  2.5.1
  CUDA: True
  torch cuda: 12.4
  NVIDIA GeForce RTX 4060 Ti

  如果这里出现：

  - CUDA: False
  - torch.version.cuda == None

  说明装到了 CPU 版 PyTorch，需要重新建环境或重装。

  ---
  七、安装量化推理基础依赖
 
  先安装量化推理最关键的几个包：

  pip install bitsandbytes
  pip install huggingface-hub

  验证 bitsandbytes：

  python -c "import bitsandbytes as bnb; print(bnb.__version__)"

  如果这条命令能正常输出版本号，说明 4-bit 量化的核心依赖基本可用。

  ---
  八、安装 LabVLA 基础依赖
 
  进入项目目录：

  cd ~/projects/labvla-mujoco/LabVLA
  pip install -r requirements.txt

  已知问题：flash_attn 可能失败

  requirements.txt 中包含：
  
  flash_attn==2.8.3

  它在默认 pip install -r requirements.txt 过程中，容易因为构建隔离环境缺少依赖而失败，例如：

  - ModuleNotFoundError: No module named 'torch'
  - ModuleNotFoundError: No module named 'psutil'

  ---
  九、单独安装 flash_attn 的推荐方法
 
  先补构建依赖：

  pip install psutil packaging ninja wheel setuptools

  然后单独安装：

  pip install flash_attn==2.8.3 --no-build-isolation

  为什么要 --no-build-isolation

  这样 flash_attn 编译时会直接使用当前 conda 环境里的：
  
  - torch
  - CUDA 配置
  - 已安装的 Python 包

  否则它会在隔离构建环境里找不到必要依赖。

  如果还是失败
  
  先检查编译链：
  
  g++ --version
  nvcc --version

  如果 nvcc 缺失，或者 CUDA toolkit 没装完整，flash_attn 可能继续失败。
  
  ---
  十、再次安装 requirements
  
  如果 flash_attn 单独装成功，再回到 LabVLA 目录执行：

  pip install -r requirements.txt

  这时 pip 通常会跳过已安装的 flash_attn。

  ---
  十一、安装模型下载工具并拉取模型
  
  确保仍在 labvla-cu124 环境中：

  conda activate labvla-cu124
  cd ~/projects/labvla-mujoco

  创建模型目录：

  mkdir -p LabVLA-5B-Base

  下载模型：

  huggingface-cli download zjunlp/LabVLA-5B-Base \
    --local-dir LabVLA-5B-Base \
    --resume-download

  如果网络慢，建议始终保留 --resume-download。

  ---
  十二、验证 GPU 量化推理基础条件
 
  建议先验证下面三件事：

  1）PyTorch GPU 可用

  python -c "import torch; print(torch.cuda.is_available())"

  2）bitsandbytes 可导入

  python -c "import bitsandbytes as bnb; print(bnb.__version__)"

  3）显卡显存可见

  python -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')"

  ---
  十三、运行 LabVLA 量化推理脚本
  
  项目根目录下执行：

  conda activate labvla-cu124
  cd ~/projects/labvla-mujoco
  python scripts/infer_quantized.py

  成功标志：

  - 模型能正常加载
  - 推理过程不报 CUDA / bitsandbytes / transformers 错误
  - 显存占用在可接受范围内
  - 输出结果看起来合理

  ---
  十四、启动量化推理服务
 
  启动服务端：

  conda activate labvla-cu124
  cd ~/projects/labvla-mujoco
  python scripts/serve_labvla_4bit.py \
    --pretrained_path ./LabVLA-5B-Base \
    --vlm_path Qwen/Qwen3-VL-4B-Instruct \
    --port 8000 \
    --device cuda

  另开一个终端，运行客户端：

  conda activate labvla-cu124
  cd ~/projects/labvla-mujoco
  python scripts/test_client.py --port 8000

  ---
  十五、推荐的实际安装顺序
 
  这是最稳的一套顺序：

  source ~/miniforge3/bin/activate
  conda create -n labvla-cu124 python=3.10 -y
  conda activate labvla-cu124

  conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia --strict-channel-priority -y

  python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('torch cuda:', torch.version.cuda); 
  print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

  pip install bitsandbytes
  pip install huggingface-hub

  pip install psutil packaging ninja wheel setuptools
  pip install flash_attn==2.8.3 --no-build-isolation

  cd ~/projects/labvla-mujoco/LabVLA
  pip install -r requirements.txt
  cd ~/projects/labvla-mujoco

  ---
  十六、常见问题排查
 
  1）pytorch-cuda=12.6 找不到

  现象：

  PackagesNotFoundInChannelsError: pytorch-cuda=12.6

  处理：
  改用：

  pytorch-cuda=12.4

  ---
  2）torch.cuda.is_available() == False
 
  先检查：

  nvidia-smi

  如果 nvidia-smi 正常，再检查是否装成了 CPU 版 torch：

  python -c "import torch; print(torch.version.cuda)"

  如果输出 None，说明不是 GPU 版 torch。

  ---
  3）flash_attn 构建时报缺 torch
 
  现象：

  ModuleNotFoundError: No module named 'torch'

  处理：

  pip install flash_attn==2.8.3 --no-build-isolation

  ---
  4）flash_attn 构建时报缺 psutil
 
  现象：

  ModuleNotFoundError: No module named 'psutil'

  处理：

  pip install psutil

  再重试：

  pip install flash_attn==2.8.3 --no-build-isolation

  ---
  5）命令行出现 > 提示符卡住
 
  说明你输入的引号或命令没有结束。
  按：

  Ctrl+C

  然后重新粘贴完整命令，不要换行。

  ---
  十七、最终结论
 
  从你当前已经验证过的结果看，用 conda 替代原来的 venv 是更稳定的路线，尤其是为了完成：

  - GPU 版 PyTorch
  - CUDA 可用
  - bitsandbytes 4-bit 量化
  - LabVLA 推理实验

  当前最关键的里程碑已经完成：

  - labvla-cu124 环境创建成功
  - torch 2.5.1 + CUDA 12.4 可用
  - GPU 已被 PyTorch 正确识别

  接下来主要就是继续打通：

  1. bitsandbytes
  2. flash_attn
  3. LabVLA/requirements.txt
  4. scripts/infer_quantized.py

  如果你要，我下一步可以把这份内容直接整理成一个适合保存到项目里的 labvla-conda-setup-guide.md 成品版本。