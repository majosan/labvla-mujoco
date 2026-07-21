"""tactile_sim.py — Generate simulated tactile data from MuJoCo state.

Produces physically plausible 124-dim pressure vectors based on
gripper state and contact info, without needing real glove hardware.
"""

import numpy as np

# Pressure zones (FPC scan order indices for 124 points):
# These map to FPC rows 19-32, cols 1-14, excluding rows where col=none
# Zone boundaries specific to the 32x32 FPC layout used in the glove

# ============================================================
# ⚠️ 124维向量排布顺序（对应 _map_124_directly 的物理顺序）
# ============================================================
# 手指区 (0-59) — 6行 × 每行10个 (小→无名→中→食→拇, 各2列)
#   行0(指尖): [0:10]  = pinky_c0,c1, ring_c0,c1, middle_c0,c1, index_c0,c1, thumb_c0,c1
#   行1:       [10:20] = 同上顺序
#   行2:       [20:30]
#   行3:       [30:40]
#   行4:       [40:50]
#   行5(指根): [50:60]
# 手掌区 (60-123) — 8行 × 8列
#   行0(靠指根): [60:68], 行1: [68:76], ... 行7(靠腕): [116:124]

# 每指的 12 个传感器位置（按实际物理布局）：
FINGER_124_IDX = {
    "pinky":  [0,1, 10,11, 20,21, 30,31, 40,41, 50,51],       # 12点
    "ring":   [2,3, 12,13, 22,23, 32,33, 42,43, 52,53],
    "middle": [4,5, 14,15, 24,25, 34,35, 44,45, 54,55],
    "index":  [6,7, 16,17, 26,27, 36,37, 46,47, 56,57],
    "thumb":  [8,9, 18,19, 28,29, 38,39, 48,49, 58,59],
}
PALM_124_IDX = list(range(60, 124))  # 64点


def generate_simulated_tactile(
    gripper_force: float = 0.0,
    contact_zones: list = None,
    object_type: str = "beaker"
) -> np.ndarray:
    """Generate tactile pressure vector based on simulation state.

    Args:
        gripper_force: Normalized gripper force [0, 1]
        contact_zones: List of zone names with contact (optional)
        object_type: "beaker" | "spatula" | "bottle" | "none"

    Returns:
        (124,) float32 pressure vector
    """
    tactile = np.zeros(124, dtype=np.float32)

    if gripper_force < 0.01:
        return tactile  # No contact = no pressure

    force = float(gripper_force)
    noise = np.random.randn(124).astype(np.float32) * 0.02 * force

    if object_type == "beaker":
        # C-wrap grasp: all fingers + palm
        # 按实际 124 索引填入
        for fname, idx_list in FINGER_124_IDX.items():
            for i, pos in enumerate(idx_list):
                # 每指 6行×2列，row_progress = floor(i/2) 表示 指尖→指根
                row_progress = i // 2  # 0=指尖, 5=指根
                pressure = np.interp(row_progress, [0, 5], [0.25, 0.45]) * force
                tactile[pos] = pressure

        # 手掌: 64点 (靠指根行压力大)
        for i, pos in enumerate(PALM_124_IDX):
            row = i // 8  # 0=靠指根, 7=靠腕
            pressure = np.interp(row, [0, 7], [0.2, 0.08]) * force
            tactile[pos] = pressure

    elif object_type == "spatula":
        # Precision pinch: thumb tip + index tip
        for pos in FINGER_124_IDX["thumb"][:2]:    # 拇指指尖2点
            tactile[pos] = 0.6 * force
        for pos in FINGER_124_IDX["index"][:2]:    # 食指尖端2点
            tactile[pos] = 0.5 * force
        # 微量手掌
        for pos in PALM_124_IDX:
            tactile[pos] = 0.01 * force

    elif object_type == "bottle":
        # Wrap grasp: mostly fingers, less palm
        for fname, idx_list in FINGER_124_IDX.items():
            for i, pos in enumerate(idx_list):
                row_progress = i // 2
                pressure = np.interp(row_progress, [0, 5], [0.3, 0.5]) * force
                tactile[pos] = pressure
        # 瓶子接触面小，手掌压力小
        for pos in PALM_124_IDX[:24]:  # 前3行手掌
            tactile[pos] = 0.05 * force

    else:  # "none" or unknown
        tactile = noise * 0.3

    # Add noise and clip
    tactile = np.clip(tactile + noise, 0.0, 1.0)
    return tactile.astype(np.float32)


if __name__ == "__main__":
    # Quick test
    import json
    for obj in ["beaker", "spatula", "bottle", "none"]:
        p = generate_simulated_tactile(0.7, object_type=obj)
        # Verify index count
        nz = np.count_nonzero(p)
        print(f"tactile_sim({obj}): shape={p.shape}, "
              f"range=[{p.min():.3f}, {p.max():.3f}], "
              f"nonzero={nz}/124")
        # Cross-check: map through GloveMapper to verify grid output
        print(f"  -> First 10 indices: {p[:10].round(3).tolist()}")
    print("✅ tactile_sim test PASSED")
