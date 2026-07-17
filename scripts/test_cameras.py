#!/usr/bin/env python3
"""Step 2 verification: render all 3 cameras and save PNG samples."""
from pathlib import Path
import numpy as np
import mujoco

SCRIPT_DIR = Path(__file__).parent
SCENE = SCRIPT_DIR / "mujoco_scene.xml"
CAMERA_NAMES = ["camera_1_rgb", "camera_2_rgb", "camera_3_rgb"]
IMG_SIZE = 224


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=IMG_SIZE, width=IMG_SIZE)

    for cam in CAMERA_NAMES:
        renderer.update_scene(data, camera=cam)
        img = renderer.render()

        assert img.shape == (IMG_SIZE, IMG_SIZE, 3), f"{cam}: unexpected shape {img.shape}"
        assert img.dtype == np.uint8, f"{cam}: dtype {img.dtype}"

        out_path = SCRIPT_DIR / f"{cam}_test.png"
        _save_png(img, out_path)

        print(f"{cam}: shape={img.shape} dtype={img.dtype} "
              f"min={img.min()} max={img.max()} mean={img.mean():.1f}  -> {out_path.name}")

    print("\n[PASS] Camera verification complete.")


def _save_png(img: np.ndarray, path: Path) -> None:
    """Minimal PNG save without PIL dependency."""
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
        return
    except ImportError:
        pass
    # Fallback: use mujoco's built-in if PIL not available
    import struct, zlib

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        c += struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return c

    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    compressed = zlib.compress(raw, 9)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", compressed))
        f.write(chunk(b"IEND", b""))


if __name__ == "__main__":
    main()
