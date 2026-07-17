# PHASE1 Report

## 1. Scope
- Goal: complete the Phase 1 baseline for LabVLA in this repository without modifying `LabVLA/` official source.
- Steps:
  1. Environment verification
  2. Download `zjunlp/LabVLA-5B-Base` and run 4-bit local inference
  3. Launch WebSocket inference service and validate one RGB + text round trip
  4. Record findings and limitations

## 2. Environment Baseline
- Python: `/home/josan/miniforge3/envs/labvla-cu124/bin/python`
- GPU: NVIDIA GeForce RTX 4060
- Driver / VRAM: `576.80`, `8188 MiB`
- Torch stack:
  - `torch 2.7.1+cu126`
  - `torchvision 0.22.1+cu126`
  - `torchaudio 2.7.1+cu126`
- Key runtime deps:
  - `transformers 4.57.6`
  - `accelerate 1.13.0`
  - `deepspeed 0.18.8`
  - `bitsandbytes 0.49.2`
  - `flash_attn 2.8.3`
- CUDA availability: `True`
- CUDA runtime reported by torch: `12.6`

## 3. Local 4-bit Inference
### Command
```bash
PYTHONPATH="/home/josan/projects/labvla-mujoco/LabVLA" /home/josan/miniforge3/envs/labvla-cu124/bin/python scripts/infer_quantized.py --pretrained_path /home/josan/projects/labvla-mujoco/LabVLA-5B-Base --vlm_path Qwen/Qwen3-VL-4B-Instruct --device cuda
```

### Expected outputs
- model load time
- one-shot inference time
- peak GPU memory
- action output shape / dtype / range

### Actual results
```json
{
  "load_time_s": 208.70435506200068,
  "infer_time_s": 10.683479493999504,
  "policy_infer_ms": 10677.10251299981,
  "peak_gpu_mem_mb": 7903,
  "actions": {
    "shape": [50, 8],
    "dtype": "float32",
    "min": -0.06039096415042877,
    "max": 0.02522941306233406,
    "mean": -0.0012085360949859023
  },
  "metadata": {
    "action_dim": 8,
    "state_dim": 8,
    "action_horizon": 50,
    "model_type": "LabVLA"
  },
  "prompt": "pick up the object",
  "image_path": null,
  "generated_test_image": true
}
```

结论：本地 4-bit 推理链路已跑通，输出为 `50 x 8` 的 `float32` action chunk，和官方 Space 中的 schema 设定一致。

显存说明：最初脚本里记录的 `18.16 GB` 来自 `torch.cuda.max_memory_allocated()`，它在当前这条 4-bit 推理链路上明显高估了真实板载显存占用，和 `RTX 4060 8GB` 的硬件上限冲突。后续按更可靠口径复测时，使用并发 `nvidia-smi` 采样得到的峰值约为 `7903 MB`，这才是应作为 Phase 1 基线引用的值。

## 4. WebSocket Service Validation
### Server command
```bash
PYTHONPATH="/home/josan/projects/labvla-mujoco/LabVLA" /home/josan/miniforge3/envs/labvla-cu124/bin/python LabVLA/deployment/serve_labvla.py --pretrained_path /home/josan/projects/labvla-mujoco/LabVLA-5B-Base --vlm_path Qwen/Qwen3-VL-4B-Instruct --device cuda --port 8000
```

### Client command
```bash
/home/josan/miniforge3/envs/labvla-cu124/bin/python scripts/test_client.py --host 127.0.0.1 --port 8000
```

### Expected outputs
- RTT
- action output shape / dtype / range
- server-side `policy_timing.infer_ms`

### Actual results
```json
{
  "rtt_ms": 2230.944206996355,
  "metadata": {
    "policy_type": "labvla",
    "action_dim": 8,
    "chunk_size": 50,
    "model_chunk_size": 50,
    "output_chunk_size": 50,
    "state_dim": 8,
    "max_state_dim": 32,
    "max_action_dim": 32,
    "num_inference_steps": 10,
    "num_cameras": 3,
    "camera_keys": ["camera_1_rgb", "camera_2_rgb", "camera_3_rgb"],
    "camera_key_aliases": {},
    "state_keys": ["state", "observation/state"],
    "action_mode": "delta",
    "prompt_required": true,
    "default_prompt_set": false
  },
  "policy_timing": {
    "infer_ms": 2228.132153992192
  },
  "server_timing": {
    "infer_ms": 2228.482755002915
  },
  "actions": {
    "shape": [50, 8],
    "dtype": "float32",
    "min": -1.2278354167938232,
    "max": 1.0727245807647705,
    "mean": -0.03920865058898926
  },
  "prompt": "pick up the object",
  "image_path": null,
  "generated_test_image": true
}
```

结论：WebSocket 推理服务已成功启动并完成一次 RGB + 文本 -> action 往返。协议首帧先返回 metadata，后续 observation 需要使用与服务端兼容的 msgpack ndarray 编码。

## 5. Artifacts Added
- `scripts/infer_quantized.py` — local 4-bit inference benchmark and output summarizer
- `scripts/test_client.py` — msgpack WebSocket round-trip validator for RGB + text observations
- `scripts/labvla_schema.json` — localized schema baseline copied from `spaces/zjunlp/lab-vla`, now managed in-repo for future schema customization

## 6. Known Constraints / Notes
- `LabVLA-5B-Base/` was initially empty and requires a local model download.
- No repository-owned RGB sample image was found during initial Phase 1 exploration, so the first validation path uses a generated RGB test image unless a real image is supplied later.
- `LabVLA/requirements.txt` contains `pyyaml-include==1.4.0`, which had to be replaced with `1.4.1` during environment setup.
- `websocket-client` was not pulled in by the base dependency set and had to be installed explicitly for `scripts/test_client.py`.
- Direct `huggingface.co` access was unreachable from this machine. Downloading through `HF_ENDPOINT=https://hf-mirror.com` succeeded for metadata and small files, and the large `model.safetensors` transfer required resumable retries.
- The downloaded `zjunlp/LabVLA-5B-Base` checkpoint did not include `labvla_schema.json`. A compatible schema sidecar was found in the Hugging Face Space `zjunlp/lab-vla`, then localized into `scripts/labvla_schema.json` and auto-copied into the model directory when needed.

## 7. Next Actions
- Phase 1 baseline is complete.
- Replace the generated RGB test image with a real task image for a more representative benchmark.
- Decide whether to keep using the copied `labvla_schema.json` from `spaces/zjunlp/lab-vla` as the canonical local sidecar, or vendor it into a project-managed assets path.
- Start the Mujoco-side integration work using the validated `50 x 8` action chunk contract.
