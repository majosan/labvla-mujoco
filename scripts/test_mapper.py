"""Quick test: verify glove_grid_mapper standalone."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from glove_grid_mapper import GloveMapper

mapper = GloveMapper()

# Simulate 124 random pressure values
fake_pressure = np.random.rand(124).astype(np.float32) * 255

# Process
grid = mapper.process_frame(fake_pressure)

# Check output
print(f"Input : 124 floats")
print(f"Output: shape={grid.shape} dtype={grid.dtype}")
print(f"Value range: {grid.min():.2f} - {grid.max():.2f}")
print(f"Non-zero cells: {np.count_nonzero(grid)} / 144")
assert grid.shape == (12, 12), f"Expected (12,12), got {grid.shape}"
assert grid.dtype == np.float32, f"Expected float32, got {grid.dtype}"
print("[PASS] Mapper test PASSED")
