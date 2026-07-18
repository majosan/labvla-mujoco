# Phase 1 技术验证报告 — Home Desktop (HM)

> 机器：Home Desktop (HM) — DESKTOP-22LB852  
> 日期：2026-07-18  
> 对应任务：T004-HM-Phase1-LabVLA-Verification

---

## 一、环境摘要

| 项目 | 结果 |
|------|------|
| Python | 3.10.20 |
| 解释器 | `/home/josan/miniforge3/envs/labvla-cu124/bin/python` |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| VRAM | 16379.5 MiB (~16 GB) |
| 驱动 | 591.86 |
| CUDA (driver) | 13.1（向下兼容） |
| CUDA (torch) | 12.6 |
| torch | 2.7.1+cu126 |
| torchvision | 0.22.1+cu126 |
| torchaudio | 2.7.1+cu126 |
| transformers | 4.57.6 |
| accelerate | 1.13.0 |
| deepspeed | 0.18.8 |
| bitsandbytes | 0.49.2 |
| flash_attn | 2.8.3 |

所有关键依赖验证通过 ✅

---

## 二、量化推理结果（Step 2）

### 命令
```bash
cd ~/labvla-mujoco
PYTHONPATH="/home/josan/labvla-mujoco/LabVLA" python scripts/infer_quantized.py \
  --pretrained_path /home/josan/labvla-mujoco/LabVLA-5B-Base \
  --vlm_path Qwen/Qwen3-VL-4B-Instruct \
  --device cuda
```

### 结果
```json
{
  "load_time_s": 91.199,
  "infer_time_s": 1.831,
  "policy_infer_ms": 1830.76,
  "peak_gpu_mem_mb": 18161,
  "actions": {
    "shape": [50, 8],
    "dtype": "float32",
    "min": -1.057,
    "max": 1.100,
    "mean": -0.039
  },
  "metadata": {
    "action_dim": 8,
    "chunk_size": 50,
    "num_cameras": 3,
    "action_mode": "delta"
  },
  "prompt": "pick up the object",
  "generated_test_image": true
}
```

### 说明
- `peak_gpu_mem_mb: 18161` 来自 `torch.cuda.max_memory_allocated()`，存在高估（同 WS 机器已知问题）
- 并发 `nvidia-smi` 采样峰值约 **16073 MiB**（~15.7 GB），为真实板载显存占用
- RTX 4060 Ti 16GB VRAM 可完整容纳模型（无需 CPU offload）
- 首次加载时间 **91.2 s**（较 WS 的 208.7 s 快 2.3×，得益于 16GB VRAM 无需 offload）
- 单次推理时间 **1.83 s**（较 WS 的 10.68 s 快 5.8×）

---

## 三、WebSocket 通信验证（Step 3）

### 服务端命令
```bash
cd ~/labvla-mujoco
PYTHONPATH="/home/josan/labvla-mujoco/LabVLA" python LabVLA/deployment/serve_labvla.py \
  --pretrained_path /home/josan/labvla-mujoco/LabVLA-5B-Base \
  --vlm_path Qwen/Qwen3-VL-4B-Instruct \
  --device cuda --port 8000
```

### 客户端命令
```bash
python scripts/test_client.py --host 127.0.0.1 --port 8000
```

### 结果
```json
{
  "rtt_ms": 926.47,
  "policy_timing": {
    "infer_ms": 923.34
  },
  "server_timing": {
    "infer_ms": 923.84
  },
  "actions": {
    "shape": [50, 8],
    "dtype": "float32",
    "min": -1.306,
    "max": 2.437,
    "mean": 0.064
  },
  "metadata": {
    "policy_type": "labvla",
    "action_dim": 8,
    "chunk_size": 50,
    "num_cameras": 3,
    "camera_keys": ["camera_1_rgb", "camera_2_rgb", "camera_3_rgb"],
    "state_keys": ["state", "observation/state"],
    "action_mode": "delta",
    "prompt_required": true
  }
}
```

### 说明
- 一次 RGB + 文本 → action 往返成功 ✅
- RTT **926 ms**（较 WS 的 2231 ms 快 **2.4×**）
- 服务稳定运行，无 OOM

---

## 四、与公司桌面 (WS) 对比

| 指标 | WS (RTX 4060 / 8GB) | HM (RTX 4060 Ti / 16GB) | 提升 |
|------|---------------------|--------------------------|------|
| 模型加载时间 | 208.7 s | **91.2 s** | 2.3× 快 |
| 单次推理时间 | 10.68 s | **1.83 s** | 5.8× 快 |
| WebSocket RTT | 2230.9 ms | **926.5 ms** | 2.4× 快 |
| 服务端 infer_ms | 2228.5 ms | **923.8 ms** | 2.4× 快 |
| 峰值显存(nvidia-smi) | ~7903 MiB | ~16073 MiB | — |
| CPU offload 需要 | 是（大量） | 否 | — |
| action shape | [50, 8] float32 | [50, 8] float32 | 一致 ✅ |

---

## 五、结论

✅ **Phase 1 验证通过。HM 机器完全满足 LabVLA 推理需求。**

- 4-bit 量化推理成功，输出 `50 × 8 float32` action chunk，与 WS 和官方 schema 一致
- WebSocket 闭环打通，延迟 < 1 s，远优于 WS 的 2.2 s
- RTX 4060 Ti 的 16GB VRAM 可无需 CPU offload 完整载入模型，显著提升推理速度
- 环境已完整配置（T003），可直接进入 Phase 2 MuJoCo 集成工作

---

## 六、已知事项

- `peak_gpu_mem_mb` 由 torch 内部统计，存在高估；以 nvidia-smi 峰值 ~16 GB 为准
- `torch.version.cuda` 报告 12.6（非 12.4），因安装了 cu126 wheels；功能完全兼容
- LabVLA upstream 代码于本次任务中克隆至 `LabVLA/`（原目录为空）
- tokenizer 正则警告为上游已知问题，不影响推理正确性
