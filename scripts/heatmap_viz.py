"""heatmap_viz.py — 实时 12×12 触觉热力图显示"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque


class TactileHeatmap:
    """实时显示 12×12 触觉压力热力图（非阻塞）"""

    def __init__(self, title="Tactile Pressure (12x12)", history_len=50):
        plt.ion()  # 交互模式，非阻塞
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig.canvas.manager.set_window_title(title)

        # 初始网格（全零）
        self.grid = np.zeros((12, 12), dtype=np.float32)
        self.im = self.ax.imshow(
            self.grid, cmap="hot", vmin=0, vmax=1.0,
            interpolation="nearest", aspect="equal"
        )
        self.colorbar = self.fig.colorbar(self.im, ax=self.ax, label="Pressure")

        # 标签
        self.ax.set_xticks(range(12))
        self.ax.set_yticks(range(12))
        labels = ["", "index", "", "middle", "", "ring", "", "pinky", "", "", "thumb", ""]
        self.ax.set_xticklabels(labels, fontsize=7)
        self.ax.set_yticklabels(range(12), fontsize=7)
        self.ax.set_xlabel("Finger / Palm Region")
        self.ax.set_ylabel("Tip -> Base (row)")
        self.ax.set_title(title)

        # 日志
        self.max_history = history_len
        self.energy_log = deque(maxlen=history_len)
        self.frame_count = 0

    def update(self, grid_12x12: np.ndarray):
        """更新热力图。在主循环每帧调用。"""
        assert grid_12x12.shape == (12, 12), f"Expected (12,12), got {grid_12x12.shape}"

        self.grid = grid_12x12
        self.im.set_data(grid_12x12)
        self.im.set_clim(vmin=0, vmax=max(1.0, grid_12x12.max()))

        # 窗口标题显示总能量
        energy = grid_12x12.sum()
        self.frame_count += 1
        self.energy_log.append(energy)
        avg = np.mean(self.energy_log) if self.energy_log else 0
        self.fig.suptitle(
            f"Frame {self.frame_count} | Energy: {energy:.1f} | Avg: {avg:.1f}"
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)  # 极小延迟，让 GUI 事件循环有机会处理

    def close(self):
        plt.ioff()
        plt.close(self.fig)


if __name__ == "__main__":
    # 快速测试 — 模拟 50 帧随机数据
    import time

    viz = TactileHeatmap()
    try:
        for i in range(50):
            # 模拟一个移动的亮点
            grid = np.zeros((12, 12), dtype=np.float32)
            rx, ry = 3 + int(i / 5) % 6, 2 + int(i / 3) % 8
            grid[rx, ry] = 0.8
            grid[rx + 1, ry - 1: ry + 2] = 0.3  # 扩散
            viz.update(grid)
            time.sleep(0.1)
    finally:
        viz.close()
    print("[PASS] heatmap_viz test PASSED")
