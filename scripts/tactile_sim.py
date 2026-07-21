"""tactile_sim.py — Generate simulated tactile data from MuJoCo state.

Produces physically plausible 124-dim pressure vectors based on gripper
state and contact info, without needing real glove hardware.

===============================================================================
124-dim vector layout (matches the actual FPC scan order used by the glove):
===============================================================================
Finger region — indices 0..59  (6 fabric rows × 10 sensors per row)
  Row 0 (finger tip):    [0:10]  = pinky_c0,c1, ring_c0,c1, middle_c0,c1,
                                    index_c0,c1, thumb_c0,c1
  Row 1..5:              [10:20], [20:30], [30:40], [40:50], [50:60]
                          — same finger order per row
  Row 5 = finger root.

  Per-finger indices (12 sensors each, one row-tip → row-root):
      pinky  : [0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51]
      ring   : [2, 3, 12, 13, 22, 23, 32, 33, 42, 43, 52, 53]
      middle : [4, 5, 14, 15, 24, 25, 34, 35, 44, 45, 54, 55]
      index  : [6, 7, 16, 17, 26, 27, 36, 37, 46, 47, 56, 57]
      thumb  : [8, 9, 18, 19, 28, 29, 38, 39, 48, 49, 58, 59]

Palm region — indices 60..123  (8 rows × 8 cols)
  Row 0 (near finger roots): [60:68]
  Row 7 (near wrist):        [116:124]
"""

import numpy as np


FINGER_124_IDX = {
    "pinky":  [0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51],
    "ring":   [2, 3, 12, 13, 22, 23, 32, 33, 42, 43, 52, 53],
    "middle": [4, 5, 14, 15, 24, 25, 34, 35, 44, 45, 54, 55],
    "index":  [6, 7, 16, 17, 26, 27, 36, 37, 46, 47, 56, 57],
    "thumb":  [8, 9, 18, 19, 28, 29, 38, 39, 48, 49, 58, 59],
}
PALM_124_IDX = list(range(60, 124))  # 64 palm sensors, 8×8 grid


def generate_simulated_tactile(
    gripper_force: float = 0.0,
    contact_zones: list | None = None,
    object_type: str = "beaker",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a (124,) float32 pressure vector shaped by object_type.

    Args:
        gripper_force: normalized gripper closing force [0, 1]
        contact_zones: optional zone names (kept for API compat; unused here)
        object_type:   "beaker" | "spatula" | "bottle" | "none"
        rng:           optional numpy Generator for deterministic output
    """
    if rng is None:
        rng = np.random.default_rng()

    tactile = np.zeros(124, dtype=np.float32)

    if gripper_force < 0.01:
        return tactile

    force = float(np.clip(gripper_force, 0.0, 1.0))
    noise = rng.standard_normal(124).astype(np.float32) * 0.02 * force

    if object_type == "beaker":
        # C-wrap grasp: all fingers + broad palm contact.
        # Pressure grows from finger tip (row 0) toward the root (row 5).
        for idx_list in FINGER_124_IDX.values():
            for i, pos in enumerate(idx_list):
                row_progress = i // 2   # 0=tip, 5=root
                tactile[pos] = np.interp(row_progress, [0, 5], [0.25, 0.45]) * force
        # Palm: pressure stronger near the finger roots, weaker toward the wrist.
        for i, pos in enumerate(PALM_124_IDX):
            row = i // 8               # 0=near-fingers, 7=near-wrist
            tactile[pos] = np.interp(row, [0, 7], [0.20, 0.08]) * force

    elif object_type == "spatula":
        # Precision pinch: thumb tip + index tip only (2 sensors each).
        for pos in FINGER_124_IDX["thumb"][:2]:
            tactile[pos] = 0.60 * force
        for pos in FINGER_124_IDX["index"][:2]:
            tactile[pos] = 0.50 * force
        for pos in PALM_124_IDX:
            tactile[pos] = 0.01 * force   # trace palm contact

    elif object_type == "bottle":
        # Wrap grasp: strong finger contact, minimal palm (top rows only).
        for idx_list in FINGER_124_IDX.values():
            for i, pos in enumerate(idx_list):
                row_progress = i // 2
                tactile[pos] = np.interp(row_progress, [0, 5], [0.30, 0.50]) * force
        for pos in PALM_124_IDX[:24]:      # top 3 rows only
            tactile[pos] = 0.05 * force

    else:  # "none" or unknown
        tactile = noise * 0.3

    tactile = np.clip(tactile + noise, 0.0, 1.0)
    return tactile.astype(np.float32)


if __name__ == "__main__":
    for obj in ["beaker", "spatula", "bottle", "none"]:
        p = generate_simulated_tactile(0.7, object_type=obj)
        print(f"tactile_sim({obj:>7}): shape={p.shape} "
              f"range=[{p.min():.3f}, {p.max():.3f}] "
              f"nonzero={np.count_nonzero(p):3d}/124 "
              f"mean={p.mean():.3f}")

    # Zero-force invariant
    z = generate_simulated_tactile(0.0, object_type="beaker")
    assert z.sum() == 0.0, "gripper_force=0 must produce all-zero tactile"

    # Index-layout invariant: no per-finger index appears in another finger
    all_idx = [i for lst in FINGER_124_IDX.values() for i in lst]
    assert len(set(all_idx)) == 60, "finger index layout has duplicates"
    assert max(all_idx) < 60 and min(all_idx) == 0

    print("[PASS] tactile_sim test PASSED")
