#!/usr/bin/env python3
"""
MuJoCo passive viewer — shows the Franka Panda scene with live physics.

Usage
-----
# 纯物理仿真（arm 保持 home 姿态，可在 viewer 里拖动关节）：
    python scripts/view_sim.py

# 连接 LabVLA 推理服务，arm 受模型控制实时运动（需先启动服务）：
    python scripts/view_sim.py --labvla --prompt "pick up the beaker" --num_steps 20
"""

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

SCENE = Path(__file__).parent / "mujoco_scene.xml"
CAM_NAMES = ["camera_1_rgb", "camera_2_rgb", "camera_3_rgb"]


# ── msgpack helpers (only imported when --labvla is used) ──────────────────

def _make_helpers():
    import msgpack
    import websocket

    def pack_array(obj):
        if isinstance(obj, np.ndarray):
            return {b"__ndarray__": True, b"data": obj.tobytes(),
                    b"dtype": obj.dtype.str, b"shape": obj.shape}
        return obj

    def unpack_array(obj):
        if isinstance(obj, dict):
            if obj.get(b"__ndarray__") or obj.get("__ndarray__"):
                d = obj.get(b"data", obj.get("data"))
                dt = np.dtype(obj.get(b"dtype", obj.get("dtype")))
                sh = tuple(obj.get(b"shape", obj.get("shape")))
                return np.frombuffer(d, dtype=dt).reshape(sh)
            return {k: unpack_array(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [unpack_array(v) for v in obj]
        return obj

    def query(host, port, obs):
        ws = websocket.create_connection(f"ws://{host}:{port}")
        meta = msgpack.unpackb(ws.recv(), raw=False)
        t0 = time.perf_counter()
        ws.send_binary(msgpack.packb(obs, default=pack_array, use_bin_type=True))
        result = unpack_array(msgpack.unpackb(ws.recv(), raw=False))
        rtt = (time.perf_counter() - t0) * 1000
        ws.close()
        return np.asarray(result["actions"], dtype=np.float32), rtt, meta

    return query


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labvla", action="store_true",
                        help="Connect to LabVLA service and control the arm")
    parser.add_argument("--host",      default="127.0.0.1")
    parser.add_argument("--port",      type=int, default=8000)
    parser.add_argument("--prompt",    default="pick up the beaker")
    parser.add_argument("--num_steps", type=int, default=20,
                        help="Number of LabVLA inference steps (--labvla only)")
    parser.add_argument("--substeps",  type=int, default=50,
                        help="Physics substeps between viewer syncs")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)   # home 姿态

    # 关节限位（用于 delta 动作裁剪）
    arm_lo = np.array([model.jnt_range[model.joint(f"joint{i+1}").id, 0] for i in range(7)])
    arm_hi = np.array([model.jnt_range[model.joint(f"joint{i+1}").id, 1] for i in range(7)])

    if args.labvla:
        renderer = mujoco.Renderer(model, height=224, width=224)
        query_labvla = _make_helpers()

    print("=" * 55)
    print(" MuJoCo viewer 启动")
    print(f" 场景: {SCENE.name}")
    if args.labvla:
        print(f" 模式: LabVLA 控制  ({args.num_steps} 步)")
        print(f" 服务: ws://{args.host}:{args.port}")
        print(f" 提示: '{args.prompt}'")
    else:
        print(" 模式: 纯物理仿真 (arm 保持 home 姿态)")
        print(" 提示: 在 viewer 里拖动关节可看弹回效果")
    print("=" * 55)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置默认视角
        viewer.cam.lookat[:] = [0.1, 0.4, 0.65]
        viewer.cam.distance  = 2.2
        viewer.cam.elevation = -22
        viewer.cam.azimuth   = 160

        if not args.labvla:
            # ── 纯物理模式：持续步进，viewer 保持响应 ──────────────────────
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()

        else:
            # ── LabVLA 控制模式 ────────────────────────────────────────────
            for step in range(args.num_steps):
                if not viewer.is_running():
                    break

                # 渲染三路相机
                images = {}
                for cam in CAM_NAMES:
                    renderer.update_scene(data, camera=cam)
                    images[cam] = renderer.render().astype(np.uint8)

                obs = {
                    "camera_1_rgb": images["camera_1_rgb"],
                    "camera_2_rgb": images["camera_2_rgb"],
                    "camera_3_rgb": images["camera_3_rgb"],
                    "state":  np.array(data.qpos[:8], dtype=np.float32),
                    "prompt": args.prompt,
                }

                print(f"[step {step+1}/{args.num_steps}] 等待推理...", end="", flush=True)
                try:
                    actions, rtt, _ = query_labvla(args.host, args.port, obs)
                except Exception as e:
                    print(f"\n  WebSocket 错误: {e}")
                    break

                # 应用动作
                delta = actions[0, :7].astype(float)
                data.ctrl[:7] = np.clip(data.qpos[:7] + delta, arm_lo, arm_hi)
                data.ctrl[7]  = 255.0 if float(actions[0, 7]) > 0.5 else 0.0

                print(f"  RTT={rtt:.0f}ms | delta=[{delta.min():.3f}, {delta.max():.3f}]")

                # 步进物理 + 持续刷新 viewer（让 viewer 在推理等待期间也保持响应）
                for _ in range(args.substeps):
                    if not viewer.is_running():
                        break
                    mujoco.mj_step(model, data)
                    viewer.sync()

            # 推理结束后保持 viewer 开启
            print("\n推理完成，viewer 保持开启，按 Esc 或关闭窗口退出。")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()


if __name__ == "__main__":
    main()
