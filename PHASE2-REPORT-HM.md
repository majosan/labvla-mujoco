# Phase 2 技术验证报告 — Home Desktop (HM)

> 机器：Home Desktop (HM) — DESKTOP-22LB852  
> 日期：2026-07-18  
> 对应任务：T004-HM-Phase1-LabVLA-Verification（Phase 2 扩展）

---

## 1. 场景配置

### Franka 模型来源
- 模型：`/home/josan/ai-chem-lab/mujoco_menagerie/franka_emika_panda/panda.xml`
- 通过 `<include file="..."/>` 引入 `scripts/mujoco_scene.xml`
- Assets 软链接：`scripts/assets/ → .../franka_emika_panda/assets/`
- 注：WS 机器的旧路径（`venv/ai-chem-lab/...`）在 HM 上已修正为 `ai-chem-lab/...`

### 场景摘要
| 元素 | 数量 |
|------|------|
| Bodies | 12 |
| Joints (DOF) | 9 |
| Actuators | 8 |
| Cameras | 3 |

### 桌面物体
| 物体 | 类型 | 位置 (x, y, z) | 尺寸 | 颜色 |
|------|------|----------------|------|------|
| 烧杯 | Cylinder | (0.20, 0.35, 0.055) | r=0.035, h=0.11 | 玻璃蓝 80% |
| 试管 | Cylinder | (-0.12, 0.30, 0.065) | r=0.013, h=0.13 | 黄色 90% |

（均为静态视觉 geom，contype=0，不参与物理碰撞）

### 初始关节位置（Home Keyframe）
```
joint1..7: [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
finger_joint1, 2: [0.04, 0.04]
gripper ctrl: 255（完全张开）
```

### 摄像头位置与朝向
| 摄像头 | 位置 (x, y, z) | xyaxes | FOVy | 描述 |
|--------|----------------|--------|------|------|
| camera_1_rgb | (0, -1.5, 1.2) | `1 0 0  0 0.426 0.905` | 60° | 正前上方，俯视工作区 |
| camera_2_rgb | (-1.2, 0.2, 1.0) | `0 -1 0  0.447 0 0.894` | 60° | 左侧，45° |
| camera_3_rgb | (1.2, 0.2, 1.0) | `0 1 0  -0.447 0 0.894` | 60° | 右侧，45° |

---

## 2. 摄像头测试结果

渲染分辨率：224×224，dtype uint8，从 home keyframe 姿态渲染。

| 摄像头 | Shape | dtype | min | max | mean |
|--------|-------|-------|-----|-----|------|
| camera_1_rgb | (224,224,3) | uint8 | 15 | 255 | 104.0 |
| camera_2_rgb | (224,224,3) | uint8 | 15 | 255 | 104.2 |
| camera_3_rgb | (224,224,3) | uint8 | 15 | 255 | 100.5 |

- 三路摄像头均正常渲染，无全黑帧
- 样本图像保存：`scripts/camera_1_rgb_test.png`, `camera_2_rgb_test.png`, `camera_3_rgb_test.png`
- 验证通过 ✅

---

## 3. 闭环测试结果

**Prompt：** `"pick up the beaker"` | **Steps：** 5 | **物理子步：** 20

### 时序

| 步骤 | RTT (ms) | Total step (ms) |
|------|----------|-----------------|
| 1 | 740.1 | 1252.0 |
| 2 | 258.2 | 416.6 |
| 3 | 174.6 | 331.0 |
| 4 | 173.0 | 340.5 |
| 5 | 171.5 | 328.2 |

- Step 1 RTT 740 ms：CUDA 首次推理预热（与 WS 的 10820 ms 对比，快 **14.6×**）
- Steps 2–5 平均 RTT：**194 ms**（与 WS 的 2692 ms 对比，快 **13.9×**）
- 服务加载时间：**90 s**（与 WS 的 230 s 对比，快 **2.6×**）

### Action 统计

| 步骤 | delta_arm min | delta_arm max | gripper_raw | gripper_ctrl |
|------|---------------|---------------|-------------|--------------|
| 1 | -1.7689 | 0.9728 | -0.8215 | 0 (夹紧) |
| 2 | -2.1290 | 1.7254 | -0.9791 | 0 (夹紧) |
| 3 | -2.6515 | 1.5913 | -1.0211 | 0 (夹紧) |
| 4 | -2.9899 | 1.9884 | 0.7327 | 255 (张开) |
| 5 | -3.0744 | 2.0616 | 1.0585 | 255 (张开) |

### 最终关节位置
```
qpos[:8] = [0.847, -1.086, 0.579, -2.821, -0.519, 2.811, -2.036, 0.025]
```

### 机械臂行为描述
- 每步都产生有效的 delta action，关节限位夹紧正常工作
- delta 幅度（最大 ±3 rad/step）与 WS 结果一致
- 夹爪在步骤 4-5 切换为张开状态，行为与 WS 相似
- 全程无 OOM，无 WebSocket 连接中断

---

## 4. 与公司桌面 (WS) 对比

| 指标 | WS (RTX 4060 / 8GB) | HM (RTX 4060 Ti / 16GB) | 提升 |
|------|---------------------|--------------------------|------|
| 服务加载时间 | 230 s | **90 s** | 2.6× 快 |
| Step 1 RTT（首次推理） | 10820 ms | **740 ms** | 14.6× 快 |
| Steps 2-5 平均 RTT | ~2692 ms | **~194 ms** | 13.9× 快 |
| 总步数 | 5 | 5 | 一致 |
| Action shape | [50,8] float32 | [50,8] float32 | 一致 ✅ |
| OOM | 无 | 无 | 一致 ✅ |

> RTT 大幅下降的原因：HM 的 16GB VRAM 可完整容纳 4-bit 量化模型，无需 CPU offload，GPU 利用率更高，推理延迟从 ~2.2s 降至 ~170ms。

---

## 5. 遇到的问题

### 5.1 路径适配（从 WS 迁移）
- `scripts/mujoco_scene.xml` 的 `<include>` 路径引用了 WS 的 `venv/ai-chem-lab/...`
- `scripts/assets` 软链接目标同样是 WS 路径
- `scripts/run_phase2.sh` 的 `PROJECT_DIR` 为 `~/projects/labvla-mujoco`
- **解决：** 三处路径均修正为 HM 的实际路径（`~/labvla-mujoco`，`~/ai-chem-lab/...`）

### 5.2 LabVLA/ 目录为空
- 迁移时未带上游代码
- **解决：** 从 `https://github.com/zjunlp/LabVLA` 克隆

---

## 6. 结论

✅ **MuJoCo 闭环验证通过，可进入 Phase 3**

完整 pipeline 端到端运行正常：
- MuJoCo 渲染 3×224×224 RGB 帧（Franka Panda 场景）
- 帧 + 关节状态编码为 msgpack，通过 WebSocket 发送给 LabVLA
- LabVLA 返回 (50, 8) float32 delta action chunk
- Action 施加到 MuJoCo（含关节限位夹紧），物理步进正常

HM 机器的推理性能（~194 ms RTT）大幅优于 WS（~2692 ms RTT），满足后续 Phase 3 实时控制要求。

---

## 7. 产物清单

| 文件 | 用途 |
|------|------|
| `scripts/mujoco_scene.xml` | MuJoCo 场景（已修正路径） |
| `scripts/assets` | 软链接 → franka_emika_panda/assets（已修正） |
| `scripts/run_phase2.sh` | 一键自动化脚本（已修正路径） |
| `scripts/camera_*_test.png` | 摄像头渲染样本图像 |
| `phase2_console_hm.log` | 完整终端输出 |
| `PHASE2-REPORT-HM.md` | 本报告 |
