#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
LABVLA_ROOT = REPO_ROOT / "LabVLA"
if str(LABVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(LABVLA_ROOT))

from deployment.serve_labvla import LabVLALabUtopiaPolicy


LOCAL_SCHEMA_PATH = REPO_ROOT / "scripts" / "labvla_schema.json"


def ensure_schema(pretrained_path: str) -> Path:
    target = Path(pretrained_path) / "labvla_schema.json"
    if target.exists():
        return target
    if not LOCAL_SCHEMA_PATH.exists():
        raise FileNotFoundError(f"missing local schema template: {LOCAL_SCHEMA_PATH}")
    shutil.copy2(LOCAL_SCHEMA_PATH, target)
    return target


def build_test_image(size: int) -> np.ndarray:
    x = np.linspace(0, 255, size, dtype=np.uint8)
    y = np.linspace(0, 255, size, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    rgb = np.stack([xv, yv, np.full_like(xv, 127)], axis=-1)
    return rgb


def load_image(path: str | None, size: int) -> np.ndarray:
    if path is None:
        return build_test_image(size)
    image = Image.open(path).convert("RGB").resize((size, size))
    return np.asarray(image, dtype=np.uint8)


def summarize_actions(actions: np.ndarray) -> dict:
    return {
        "shape": list(actions.shape),
        "dtype": str(actions.dtype),
        "min": float(np.min(actions)),
        "max": float(np.max(actions)),
        "mean": float(np.mean(actions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", default=str(REPO_ROOT / "LabVLA-5B-Base"))
    parser.add_argument("--vlm_path", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--state_dim", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    args = parser.parse_args()

    image = load_image(args.image_path, args.image_size)
    state = np.zeros(args.state_dim, dtype=np.float32)
    obs = {
        "camera_1_rgb": image,
        "camera_2_rgb": image.copy(),
        "camera_3_rgb": image.copy(),
        "state": state,
        "prompt": args.prompt,
    }

    schema_path = ensure_schema(args.pretrained_path)

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    load_start = time.perf_counter()
    policy = LabVLALabUtopiaPolicy(
        pretrained_path=args.pretrained_path,
        vlm_path=args.vlm_path,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_s = time.perf_counter() - load_start

    infer_start = time.perf_counter()
    result = policy.infer(obs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_s = time.perf_counter() - infer_start

    actions = np.asarray(result["actions"])
    peak_mem_mb = None
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    payload = {
        "load_time_s": load_s,
        "infer_time_s": infer_s,
        "policy_infer_ms": float(result["policy_timing"]["infer_ms"]),
        "peak_gpu_mem_mb": peak_mem_mb,
        "actions": summarize_actions(actions),
        "metadata": policy.metadata,
        "schema_path": str(schema_path),
        "prompt": args.prompt,
        "image_path": args.image_path,
        "generated_test_image": args.image_path is None,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
