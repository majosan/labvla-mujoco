"""tactile_sim.py — Generate simulated tactile data from MuJoCo state.

Produces physically plausible 124-dim pressure vectors based on gripper
state and contact info, without needing real glove hardware.

Zones (indices into the 124-dim vector):
    thumb  : 0..12
    index  : 12..24
    middle : 24..36
    ring   : 36..48
    pinky  : 48..60
    palm   : 60..124
"""

import numpy as np


FINGER_ZONES = {
    "thumb":  slice(0, 12),
    "index":  slice(12, 24),
    "middle": slice(24, 36),
    "ring":   slice(36, 48),
    "pinky":  slice(48, 60),
}
PALM_SLICE = slice(60, 124)  # 64 sensors


def generate_simulated_tactile(
    gripper_force: float = 0.0,
    contact_zones: list | None = None,
    object_type: str = "beaker",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a (124,) float32 pressure vector shaped by object_type.

    Args:
        gripper_force: normalized gripper closing force [0, 1]
        contact_zones: optional list of zone names with contact (unused for
            the built-in patterns but kept for API compat with future overrides)
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
        # C-wrap grasp: all fingers + broad palm contact
        for zone in FINGER_ZONES.values():
            n = zone.stop - zone.start
            tactile[zone] = np.linspace(0.25, 0.45, n, dtype=np.float32) * force
        pn = PALM_SLICE.stop - PALM_SLICE.start
        tactile[PALM_SLICE] = rng.uniform(0.08, 0.20, pn).astype(np.float32) * force

    elif object_type == "spatula":
        # Precision pinch: thumb tip + index tip only
        # NOTE: assign into the vector via composed indices; tactile[slice][:6]=x
        # would write to a copy and lose the update.
        thumb = FINGER_ZONES["thumb"]
        idxf  = FINGER_ZONES["index"]
        tactile[thumb.start : thumb.start + 6] = 0.60 * force  # thumb tip
        tactile[idxf.start  : idxf.start  + 6] = 0.50 * force  # index tip
        tactile[PALM_SLICE] = 0.01 * force  # trace palm contact

    elif object_type == "bottle":
        # Wrap grasp: mostly fingers, minimal palm
        for zone in FINGER_ZONES.values():
            n = zone.stop - zone.start
            tactile[zone] = np.linspace(0.30, 0.50, n, dtype=np.float32) * force
        tactile[PALM_SLICE] = 0.05 * force

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

    # Zero force = zero tactile
    z = generate_simulated_tactile(0.0, object_type="beaker")
    assert z.sum() == 0.0, "gripper_force=0 must produce all-zero tactile"
    print("[PASS] tactile_sim test PASSED")
