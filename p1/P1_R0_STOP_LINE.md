# P1_R0_STOP_LINE · 立即停线决议（2026-08-30，外部专家审查触发）

> 触发：外部专家对 `EXPERT_REVIEW_PACKAGE_20d2ba8.zip` 的代码级审查。
> 效力：立即。以下禁令在 R1/R2 完成前有效。

## 1. 两个已确认的 Stop-the-line 问题

### 问题 1 · SH Lambertian 卷积系数错误（实质性问题，非 typo）

- 当前实现（`p1/source/physics/sh.py`）：
  `K_L = [√π, √(π/3), √(π/5)] = [1.7725, 1.0233, 0.8862]`
- 正确值（正交归一实 SH，Ramamoorthi & Hanrahan 2001 Eq.7-9）：
  `Â = [π, 2π/3, π/4] = [3.1416, 2.0944, 0.7854]`
- 逐带比值 [0.564, 0.489, 1.128] —— **不是全局尺度差，是真实的逐带畸变**。
- 专家独立 Monte Carlo：正确系数下 MAE vs `max(0,n·d)` ≈ **0.031**；
  我们当前系数 ≈ **0.148**。
- **被禁止的归因**："L=2 有 75% relative MAE、需升级 L=4"。
  该数字来自我们 Test5 的错误对照（把 reproducing kernel `Σ Y(d)Y(n)`
  当作 irradiance 对照），作废。**L=2 本身就是正确且充分的选择**
  （Ramamoorthi 原文：worst-case pixel error ~9%，照明分布平均 <3%）。

### 问题 2 · P-domain 生成器用近场点光却标全局方向光 SH

- `render_multilight.py`：`light.set_type("POINT"); light.set_location(d_w)`
  其中 `d_w` 是单位半球点（**灯距原点仅 1.0**），而 mesh 尺度归一到 1.6
  → 强近场效应（逐像素入射方向不同 + 距离衰减）。
- 但 metadata `sh_coeffs_irradiance.npy` 用场景中心方向构造全局方向光 SH
  → **物理模型不匹配**。
- 因此 **22.25 dB oracle floor 不能归因为"SH-2 截断误差"**（当前文档如此
  归因，作废）。真实 floor 是多因素混合：错误卷积系数 + 近场失配 +
  距离衰减 + 逐像素方向变化 + 阴影/路径追踪，最后才是 L=2 截断。

## 2. 立即生效的行动

1. **暂停**用当前 `render_multilight.py` 生成任何 P-domain 数据
   （P1-13 本就未启动，零浪费）。
2. **不删除**已生成的 point-light calibration 数据；重命名语义标记：
   `p1/calibration_set/data` → 保留，标记
   `domain: nearfield_stress`（未来 robustness 实验资产）。
3. **禁止**在一切文档/汇报中继续引用 "75% SH error"、"22.25 dB 归因
   SH 截断"、"N≥8 饱和"（旧 N 曲线是近场失配数据上的数字，
   待 SUN 重渲后重测）。
4. 冻结当前 commit；后续修正走新 commit。

## 3. 修正路径（R1→R4，见任务序列）

- R1：`K_L → Â=[π,2π/3,π/4]`；P-domain 生成器 `POINT → SUN`（真远场
  方向光）；重写解析测试（n=d / n⊥d / n=−d 三必测点 + Monte Carlo）。
- R2：只重渲 5×32 SUN calibration → 重算 oracle。**通过条件**：数值
  oracle residual 与正确 L=2 analytic floor 对得上，无大块 unexplained
  error（不设武断 dB 阈值）。
- R3：实现真正 gauge-aware 联合 Fisher（Schur 补 + gauge projection），
  禁止再用 `F=(AY)ᵀ(AY)` 充当 H-COND 证据（当前 F 与光照子集无关，
  只测了法线分布对 SH 基的可观测性）。
- R4：同 N 不同子集扫描（GA-ISI vs solver error 相关性）= 定核 Gate。

## 4. 一句话状态

> 论文方向可以定了（illumination-set information governs joint
> decomposition identifiability）；核心贡献还差一轮 5×32 calibration
> 级的"定核 Gate"。不要按原协议烧 27h GPU。

签发：P1-R0 · ZCode agent · 2026-08-30 · commit 20d2ba8 之后
