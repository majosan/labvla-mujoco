# Claude Code Handoff

## Project background
This repository is a local integration workspace for running `LabVLA` under Mujoco-oriented experimentation. The immediate goal was to complete Phase 1 from `labvla-phase1-mission.md`: verify the environment, run one local 4-bit inference with `zjunlp/LabVLA-5B-Base`, validate the OpenPI-style msgpack WebSocket service, and capture baseline metrics for later Mujoco-side integration.

## Current repository state
Key paths:

- `LabVLA/` — upstream model and deployment code. Do not modify upstream source unless explicitly required.
- `scripts/infer_quantized.py` — local 4-bit benchmark script implemented in this project.
- `scripts/test_client.py` — msgpack WebSocket validation client implemented in this project.
- `scripts/labvla_schema.json` — localized schema template copied from `spaces/zjunlp/lab-vla`; this is now the project-managed baseline for action/state schema customization.
- `LabVLA-5B-Base/` — local model directory for `zjunlp/LabVLA-5B-Base`.
- `PHASE1-REPORT.md` — measured Phase 1 results.
- `项目环境与跨电脑操作手册.md` — durable operator manual for re-entry, rebuild, and cross-machine setup.
- `labvla-phase1-mission.md` — the mission brief and acceptance criteria.

## What was validated
Environment:

- Python interpreter: `/home/josan/miniforge3/envs/labvla-cu124/bin/python`
- GPU: `NVIDIA GeForce RTX 4060`
- Torch stack: `torch 2.7.1+cu126`, `torchvision 0.22.1+cu126`, `torchaudio 2.7.1+cu126`
- Key runtime deps: `transformers 4.57.6`, `accelerate 1.13.0`, `deepspeed 0.18.8`, `bitsandbytes 0.49.2`, `flash_attn 2.8.3`, `websocket-client`

Phase 1 baseline:

- Local 4-bit inference succeeded.
- First model load time: about `208.7 s`
- One-shot local inference time: about `10.68 s`
- Peak GPU memory by external `nvidia-smi` sampling: about `7903 MB`
- Output action chunk: shape `50 x 8`, dtype `float32`

WebSocket baseline:

- Local service on port `8000` succeeded.
- One RGB + text -> action round trip succeeded.
- RTT: about `2230.94 ms`
- Server/policy infer time: about `2228 ms`
- Returned action chunk: shape `50 x 8`, dtype `float32`

## Important implementation facts
1. Direct access to `huggingface.co` was unreliable from this machine. The working path used:
   - `HF_ENDPOINT=https://hf-mirror.com`
2. `zjunlp/LabVLA-5B-Base` did not ship `labvla_schema.json` in the model directory.
3. A compatible schema was found in the Hugging Face Space `zjunlp/lab-vla` and localized into `scripts/labvla_schema.json`.
4. `scripts/infer_quantized.py` and `scripts/test_client.py` now auto-copy `scripts/labvla_schema.json` into `LabVLA-5B-Base/labvla_schema.json` if the model directory is missing the sidecar.
5. The WebSocket protocol is msgpack-based and sends a metadata frame first. The client must read metadata first, then send the observation payload.
6. The observation payload must be encoded with the same ndarray msgpack convention as the upstream server expects; plain JSON-like lists are not enough for a faithful client.

## Known pitfalls
- `LabVLA/requirements.txt` contains `pyyaml-include==1.4.0`; this version was not installable here and was replaced with `1.4.1` during setup.
- The local environment name is still `labvla-cu124`, but the working torch runtime is actually `cu126`. Do not assume the environment name reflects the current CUDA wheel version.
- If `torch.cuda.is_available()` becomes false on another machine, first verify the interpreter path and GPU visibility before changing packages.

## Recommended next focus
Use the validated `50 x 8` action chunk contract as the starting point for Mujoco-side integration. If state/action formats need to change, treat `scripts/labvla_schema.json` as the editable baseline and keep upstream `LabVLA/` unchanged until the new contract is well understood.
