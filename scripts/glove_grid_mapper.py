"""
glove_grid_mapper.py — 触觉手套 124 压力点 → 12×12 网格（供 CNN 编码）

输出：12×12 压力灰度图（单通道），空位 = 0
后续：CNN 编码器将 12×12 → 64 维特征 → 融合进 LabVLA

物理布局（织物坐标）：
  手指区：织物行 1-6（1=指尖, 6=指根），每指2列 × 5指 = 60点
  手掌区：织物行 7-14（7=靠腕, 14=靠指根），8列 = 64点
  合计：124 点

12×12 网格布局（对应 tactile-encoding-fusion.md 方案）：
  ┌────────────────────────────────────────────────────────┐
  │ 行0-5 (6行) 手指区                                      │
  │   [0:2]空 │ [2:4]食指 │ [4:6]中指 │ [6:8]无名 │ [8:10]小指 │ [10:12]空 │
  │   每指 2列 × 6行 = 12 格，60 指传感器按坐标填入          │
  │ 行3-9 (7行) 拇指区                                      │
  │   [0:2]拇指(行3-9) → 14 格放 12 个拇指传感器              │
  │ 行6-11 (6行) 手掌区                                      │
  │   [2:10]掌心(8列) → 48 格放 64 个掌心传感器（部分合并）   │
  │ 四角留空 = 0                                            │
  └────────────────────────────────────────────────────────┘

用法：
  from glove_grid_mapper import GloveMapper
  mapper = GloveMapper()
  grid_12x12 = mapper.process_frame(frame_124_or_1024)
  # grid_12x12.shape → (12, 12), dtype float32
"""

import numpy as np


# ============================================================
# FPC → 织物坐标映射
# ============================================================

# FPC行号 → 织物行号 (FPC 19-32 共14行有效)
FPC_ROW_TO_FABRIC = {
    19: 7, 20: 8, 21: 9, 22: 10, 23: 11,
    24: 14,   # 反接
    25: 13,   # 反接
    26: 12,   # 反接
    27: 6, 28: 5, 29: 4, 30: 3, 31: 2, 32: 1,
}

# FPC列号 → (手指名, 手指内列号, 是否共享手掌)
# 列1-14, 有些列在不同行上同时连接手指和手掌
FPC_COL_INFO = {
    1:  ("pinky",  0, False),
    2:  ("pinky",  1, False),
    3:  ("ring",   1, True),   # 行1-6 → 无名指列2; 行7-14 → 掌心列8
    4:  (None,    None, "palm"),  # 纯手掌列7
    5:  (None,    None, "palm"),  # 纯手掌列6
    6:  ("ring",   0, True),   # 行1-6 → 无名指列1; 行7-14 → 掌心列5
    7:  ("middle", 0, True),   # 行1-6 → 中指列1;   行7-14 → 掌心列1
    8:  (None,    None, "palm"),  # 纯手掌列3
    9:  (None,    None, "palm"),  # 纯手掌列2
    10: ("middle", 1, True),   # 行1-6 → 中指列2;   行7-14 → 掌心列4
    11: ("index",  0, False),
    12: ("index",  1, False),
    13: ("thumb",  0, False),
    14: ("thumb",  1, False),
}

# 共享列上：FPC列 → 手掌列号
FPC_TO_PALM_COL = {3: 8, 4: 7, 5: 6, 6: 5, 7: 1, 8: 3, 9: 2, 10: 4}


def _extract_valid(raw_1024_or_124):
    """
    将 1024 原始帧或 124 提取帧统一为 (196,) 数组
    196 = 14行 × 14列，按 FPC 行19→32、列1→14 顺序
    """
    if len(raw_1024_or_124) == 1024:
        vals = []
        for fpc_r in range(19, 33):
            for fpc_c in range(1, 15):
                idx = (fpc_r - 1) * 32 + (fpc_c - 1)
                vals.append(raw_1024_or_124[idx])
        return np.array(vals, dtype=np.float32)
    elif len(raw_1024_or_124) == 196:
        return np.array(raw_1024_or_124, dtype=np.float32)
    elif len(raw_1024_or_124) == 124:
        # 已经是124个有效值，但需要知道排列顺序
        # 按 FPC 19→32 × 1→14 遍历，忽略未接线的位置
        # 如果已经是提取好的124个，假设是按 FPC 扫描顺序排列
        # 实际只有14×14共196个扫描位置中的124个接线的
        # 但既然只有124个值，我们回到物理布局分别填充
        return raw_1024_or_124
    else:
        raise ValueError(f"输入长度应为 1024、196 或 124，收到 {len(raw_1024_or_124)}")


class GloveMapper:
    """124 压力点 → 12×12 网格映射"""

    def __init__(self):
        pass

    def process_frame(self, raw_data, timestamp=0):
        """
        输入: 原始帧数据 (1024 bytes 或 124 values)
        输出: (12, 12) 压力网格，float32
        """
        grid = np.zeros((12, 12), dtype=np.float32)

        # --- 步骤1: 解析 196 个扫描位置 ---
        if len(raw_data) == 1024:
            # 从1024帧提取196个扫描位置
            scanned = {}
            for fpc_r in range(19, 33):
                for fpc_c in range(1, 15):
                    idx = (fpc_r - 1) * 32 + (fpc_c - 1)
                    scanned[(fpc_r, fpc_c)] = float(raw_data[idx])
        elif len(raw_data) == 124:
            # 124个值，需要按物理位置重新分布
            # 用 fabric 坐标重建
            return self._map_124_directly(np.array(raw_data, dtype=np.float32))
        else:
            raise ValueError(f"输入长度: {len(raw_data)}")

        # --- 步骤2: 按 FPC 行扫描，填入织物位置 ---
        for fpc_r in range(19, 33):
            fabric_r = FPC_ROW_TO_FABRIC[fpc_r]
            for fpc_c in range(1, 15):
                val = scanned[(fpc_r, fpc_c)]
                info = FPC_COL_INFO[fpc_c]

                if fabric_r <= 6:  # 手指区
                    f_name, f_col, _ = info
                    if f_name is not None:
                        f_row = 6 - fabric_r  # 织物1=指尖→grid行0
                        g_r, g_c = self._finger_to_grid(f_name, f_row, f_col)
                        if 0 <= g_r < 12 and 0 <= g_c < 12:
                            grid[g_r, g_c] = self._blend(grid[g_r, g_c], val)

                else:  # 手掌区 (fabric_r 7-14)
                    p_col = FPC_TO_PALM_COL.get(fpc_c)
                    if p_col is not None:
                        p_row = fabric_r - 7  # 0-based 手掌行
                        g_r, g_c = self._palm_to_grid(p_row, p_col-1)
                        if 0 <= g_r < 12 and 0 <= g_c < 12:
                            grid[g_r, g_c] = self._blend(grid[g_r, g_c], val)

        return grid

    def _map_124_directly(self, vals):
        """
        当输入是124个值时，按织物坐标直接填入12×12网格
        顺序假设：按织物行1→14、织物列按FPC物理顺序
        """
        grid = np.zeros((12, 12), dtype=np.float32)

        # 手指60点: 织物行1-6
        idx = 0
        finger_order = ["pinky", "ring", "middle", "index", "thumb"]
        finger_cols_12x12 = {"pinky": 8, "ring": 6, "middle": 4, "index": 2, "thumb": 0}

        for fabric_r in range(1, 7):      # 手指行1→6
            f_row = 6 - fabric_r           # 12×12 grid行: 0→5
            for f_name in finger_order:    # 小→无名→中→食→拇
                for f_c in range(2):       # 每指2列
                    if idx < 124:
                        g_c = finger_cols_12x12[f_name] + f_c
                        # 拇指从 grid 行3开始
                        if f_name == "thumb":
                            g_r = f_row + 3  # thumb在grid行3-8
                        else:
                            g_r = f_row
                        if 0 <= g_r < 12 and 0 <= g_c < 12:
                            grid[g_r, g_c] = max(grid[g_r, g_c], float(vals[idx]))
                        idx += 1

        # 手掌64点: 织物行7-14
        for fabric_r in range(7, 15):
            p_row = fabric_r - 7           # 0-based
            for p_c in range(8):
                if idx < 124:
                    g_r, g_c = self._palm_to_grid(p_row, p_c)
                    if 0 <= g_r < 12 and 0 <= g_c < 12:
                        grid[g_r, g_c] = max(grid[g_r, g_c], float(vals[idx]))
                    idx += 1

        return grid

    def _finger_to_grid(self, finger_name, f_row, f_col):
        """
        手指织物坐标 → 12×12 网格坐标
        finger_name: index/middle/ring/pinky/thumb
        f_row: 0=指尖 → 5=指根
        f_col: 0或1
        """
        col_map = {"index": 2, "middle": 4, "ring": 6, "pinky": 8, "thumb": 0}
        base_col = col_map.get(finger_name, 0)

        if finger_name == "thumb":
            # 拇指在网格中下移3行（从行3开始）
            return f_row + 3, base_col + f_col
        else:
            return f_row, base_col + f_col

    def _palm_to_grid(self, p_row, p_col):
        """
        手掌织物坐标 → 12×12 网格坐标
        p_row: 0-7（织物行7-14）
        p_col: 0-7（织物列1-8）
        """
        if p_row < 4:
            # 掌心上区: grid行6-9, 列2-9
            return p_row + 6, p_col + 2
        else:
            # 掌心下区(最靠腕): grid行10-11, 列4-7
            return p_row + 6, p_col + 2

    def _blend(self, existing, new_val):
        """多个传感器映射到同一网格时的合并策略"""
        if existing == 0:
            return new_val
        # 已有值则取最大值（压力由最强传感器代表）
        return max(existing, new_val)


# ============================================================
# 3 通道编码器（方案B：压力 + 变化率 + 局部方差）
# 对应 tactile-encoding-fusion.md 方案B
# ============================================================

class TactileEncoder3Ch:
    """
    滑动窗口 → 3 通道 12×12 输出
    每帧调用 append() 更新，输出 3 通道网格

    用法：
    encoder = TactileEncoder3Ch(window=5)
    for each_frame:
        grid_12x12 = mapper.process_frame(raw_data)
        ch3 = encoder.append(grid_12x12)  # (12, 12, 3)
        # ch3[..., 0] = 当前压力
        # ch3[..., 1] = 压力变化率 (ΔP/Δt)
        # ch3[..., 2] = 局部方差 (最近5帧)
    """
    def __init__(self, window=5):
        self.window = window
        self.history = []

    def append(self, grid_12x12):
        """
        输入: (12, 12) 当前帧压力图
        输出: (12, 12, 3) 三通道编码
        """
        self.history.append(grid_12x12.copy())
        if len(self.history) > self.window:
            self.history.pop(0)

        ch1 = grid_12x12                                     # Ch1: 当前压力

        if len(self.history) >= 2:
            ch2 = grid_12x12 - self.history[-2]               # Ch2: ΔP/Δt
        else:
            ch2 = np.zeros_like(grid_12x12)

        if len(self.history) >= self.window:
            ch3 = np.var(np.stack(self.history), axis=0)      # Ch3: 局部方差
        else:
            ch3 = np.zeros_like(grid_12x12)

        return np.stack([ch1, ch2, ch3], axis=-1)  # (12, 12, 3)


# ============================================================
# CNN 编码器（PyTorch）— 放入 LabVLA 模型前向中
# ============================================================

"""
建议在 LabVLA 的 observation_encoder.py 中加入：

class TactileCNN(nn.Module):
    '''12×12 压力图 → 64 维触觉特征'''
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 12→6
            nn.ReLU(), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 6→3
            nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 3→2
            nn.ReLU(), nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(1),  # → (64, 1, 1)
            nn.Flatten(),             # → 64
        )
    def forward(self, x):
        return self.cnn(x)  # (B, 64)

# 在融合层：
# tactile_feat = tactile_cnn(tactile_3ch)       # (B, 64)
# vision_feat = vision_encoder(images)           # (B, 1024)  
# combined = torch.cat([vision_feat, tactile_feat, proprio_feat], dim=-1)
"""


# ============================================================
# 快速验证
# ============================================================

if __name__ == "__main__":
    mapper = GloveMapper()

    # 模拟 1024 原始帧（全部0，仅测试映射不报错）
    mock_1024 = [0.0] * 1024
    grid = mapper.process_frame(mock_1024)
    print(f"模拟1024帧 → 网格形状: {grid.shape}")  # (12, 12)
    print(f"非零点数: {np.count_nonzero(grid)}")    # 应为0（模拟全0帧）

    # 模拟 124 有效值（全1）
    mock_124 = [1.0] * 124
    grid2 = mapper.process_frame(mock_124)
    print(f"\n模拟124帧全1 → 非零点数: {np.count_nonzero(grid2)}")
    print(f"总能量: {grid2.sum():.1f} (期望 ~124)")

    # 三通道编码器测试
    encoder = TactileEncoder3Ch(window=5)
    for i in range(5):
        ch3 = encoder.append(grid2)
    print(f"\n3通道输出形状: {ch3.shape}")  # (12, 12, 3)
    print(f"Ch1(压力) 非零: {np.count_nonzero(ch3[...,0])} / 144")
    print(f"Ch2(变化率) 非零: {np.count_nonzero(ch3[...,1])} / 144")
    print(f"Ch3(方差) 非零: {np.count_nonzero(ch3[...,2])}")
