#!/usr/bin/env python3
import argparse
import json
import shutil
import time
from pathlib import Path

import msgpack
import numpy as np
import websocket
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SCHEMA_PATH = REPO_ROOT / "scripts" / "labvla_schema.json"
MODEL_DIR = REPO_ROOT / "LabVLA-5B-Base"


def ensure_schema() -> Path:
    target = MODEL_DIR / "labvla_schema.json"
    if target.exists():
        return target
    if not LOCAL_SCHEMA_PATH.exists():
        raise FileNotFoundError(f"missing local schema template: {LOCAL_SCHEMA_PATH}")
    shutil.copy2(LOCAL_SCHEMA_PATH, target)
    return target


def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if isinstance(obj, dict) and (obj.get("__ndarray__") or obj.get(b"__ndarray__")):
        data = obj.get("data", obj.get(b"data"))
        dtype = np.dtype(obj.get("dtype", obj.get(b"dtype")))
        shape = tuple(obj.get("shape", obj.get(b"shape")))
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    if isinstance(obj, dict) and (obj.get("__npgeneric__") or obj.get(b"__npgeneric__")):
        dtype = np.dtype(obj.get("dtype", obj.get(b"dtype")))
        data = obj.get("data", obj.get(b"data"))
        return dtype.type(data)
    if isinstance(obj, list):
        return [_unpack_array(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _unpack_array(v) for k, v in obj.items()}
    return obj


def build_test_image(size: int) -> np.ndarray:
    x = np.linspace(0, 255, size, dtype=np.uint8)
    y = np.linspace(0, 255, size, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    rgb = np.stack([xv, np.full_like(xv, 127), yv], axis=-1)
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--state_dim", type=int, default=8)
    args = parser.parse_args()

    image = load_image(args.image_path, args.image_size)
    schema_path = ensure_schema()
    obs = {
        "camera_1_rgb": image,
        "camera_2_rgb": image.copy(),
        "camera_3_rgb": image.copy(),
        "state": np.zeros(args.state_dim, dtype=np.float32),
        "prompt": args.prompt,
    }

    ws = websocket.create_connection(f"ws://{args.host}:{args.port}")
    metadata_reply = ws.recv()
    metadata = msgpack.unpackb(metadata_reply, raw=False)

    start = time.perf_counter()
    ws.send_binary(msgpack.packb(obs, default=_pack_array, use_bin_type=True))
    reply = ws.recv()
    rtt_ms = (time.perf_counter() - start) * 1000.0
    ws.close()

    result = _unpack_array(msgpack.unpackb(reply, raw=False))
    if "actions" not in result:
        print(json.dumps({"rtt_ms": rtt_ms, "metadata": metadata, "response": result}, indent=2, ensure_ascii=False))
        return
    actions = np.asarray(result["actions"])
    payload = {
        "rtt_ms": rtt_ms,
        "metadata": metadata,
        "policy_timing": result.get("policy_timing", {}),
        "server_timing": result.get("server_timing", {}),
        "actions": summarize_actions(actions),
        "prompt": args.prompt,
        "image_path": args.image_path,
        "generated_test_image": args.image_path is None,
        "schema_path": str(schema_path),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
