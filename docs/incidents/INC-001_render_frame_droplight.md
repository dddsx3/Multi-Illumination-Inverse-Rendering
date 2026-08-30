# INC-001 · 确认集渲染器偶发整帧丢光（depsgraph 未同步）

- **发现**：2026-08-31 · R4′-C canary（conf_sphere_r05，seed 20260901）
- **触发轮次**：R4′-C 数据生成前的 canary 纪律（任务书 §6/纪律 #15）——
  canary 正是为此设的，未进入批量。
- **严重度**：数据完整性高（静默坏帧会污染 Fisher 分数与 solver 误差），
  物理协议无影响（正常帧与 discovery 逐像素统计一致）。

## 现象（两次独立 canary 渲染，同 seed 20260901）

| run | 坏帧（1-based） | 形态 |
|---|---|---|
| run1 | light_015 | 前景全黑（masked mean 0.0003 vs 正常 0.48）+ 顶部横贯亮带（bg mean 0.10 vs 正常 0.0017） |
| run2 | light_003 / 004 / 006 | 整帧全黑（fg≈bg≈0.0001），无亮带 |

- 退化帧**位置在重跑间随机移动**（非方向依赖：同方向在另一 run 正常，且
  正常帧两次均值一致到 5 位小数）；
- discovery 数据（seed 20260830，4 scene ×32 = 128 帧）逐帧检查**无坏帧**。

## 根因（判定）

单 Blender 进程内 33 次 render 之间创建/删除灯（每灯独立 render call，
这是 synthetic_v3 事故后的冻结设计，不能改回 frame animation），
新建 SUN 灯**偶发未同步进 Cycles depsgraph** → 渲染用了没有该灯的场景状态
→ 整帧丢光；亮带变体是同一同步故障的残留伪影。定位实验：
- 方向假设排除（light0 离相机轴 9.4° 正常、light14 14.2° 坏，重跑后互换）；
- 确定性排除（同 seed 重跑坏帧集合不同）；
- 帧级统计确认正常帧物理上界（albedo_max×1.0625×I_eff≈0.58 < 1，
  任何 ≥1.0 像素必为渲染伪影）。

## 修复（render_multilight.py，物理语义零改动）

1. 每灯 render 前强制 `bpy.context.view_layer.update()`（depsgraph 同步）；
2. 渲染后帧级完整性校验：解析期望 `mean_p a_p·ReLU(Y_pᵀc_k)`（L2 SH 单灯
   模型，无自阴影）与渲染 masked mean 之比 ratio，加背景均值检查；
3. 坏帧 → 重新同步 + 重渲（≤3 次），仍坏 raise 干净失败（场景级重跑）。

**阈值标定**（discovery 128 健康帧）：ratio∈[0.86, 1.69]（中位 1.29，
系统性偏移来自 L2 截断 vs 真实 diffuse），bg≤0.0074；坏帧实测 ratio≈0.001、
bg≈0.10。冻结阈值取宽安全带：**ratio∈[0.15, 3.5]，bg_mean≤0.05**
（下界放宽 0.86→0.15 给复合 mesh 自阴影留余量；坏帧比值距阈值 ≥15×，
分离度充分）。

## 效力与下游

- 生效：本次 commit 起，R4′-C 批量生成必须用修复后渲染器；
- discovery 数据已逐帧核验干净（ratio/bg 全在健康带内），**不作废**；
- canary run1/run2 目录保留作证据（`_canary_run1_degenerate/` 与
  run2 由本修复后的重渲覆盖前保留）；
- 每场景渲染日志含 `[INC-001]` 重试行时，验收报告必须提及重试次数。

签发：R4′-C · ZCode agent · 2026-08-31
