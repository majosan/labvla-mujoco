#!/usr/bin/env python3
"""Step 1 verification: load mujoco_scene.xml and print scene statistics."""
from pathlib import Path
import mujoco

SCENE = Path(__file__).parent / "mujoco_scene.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    print(f"Scene loaded: {SCENE}")
    print(f"  Bodies   : {model.nbody}")
    print(f"  Joints   : {model.njnt}")
    print(f"  Actuators: {model.nu}")
    print(f"  Geoms    : {model.ngeom}")
    print(f"  Cameras  : {model.ncam}")

    cam_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
    print(f"\nCamera names:")
    for n in cam_names:
        print(f"  {n}")

    jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    print(f"\nJoint names (DOF={model.nv}):")
    for n in jnt_names:
        print(f"  {n}")

    # Reset to home keyframe and print qpos
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    print(f"\nHome qpos (first 9): {data.qpos[:9]}")
    print(f"Home ctrl          : {data.ctrl}")

    assert "camera_1_rgb" in cam_names, "camera_1_rgb not found"
    assert "camera_2_rgb" in cam_names, "camera_2_rgb not found"
    assert "camera_3_rgb" in cam_names, "camera_3_rgb not found"
    assert model.njnt >= 9, "expected at least 9 joints (7 arm + 2 finger)"
    print("\n[PASS] Scene verification complete.")


if __name__ == "__main__":
    main()
