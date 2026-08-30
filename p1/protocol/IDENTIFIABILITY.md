# IDENTIFIABILITY · 光照集合可辨识性理论（章节骨架 v0.1）

> 论文 §5 的底稿。按专家建议的 10 节顺序搭建；公式以本仓库实现的
> 记号为准（`p1/source/physics/sh.py`、`gauge_fisher.py`）。
> 状态：1-7 节已可写；8-10 部分依赖 R4 数值结果。

## 1. Image formation

固定针孔相机，像素 p 的线性域观测：
`I_k(p) = a_p · Ŝ_k(p) + ε_k(p)`，
`Ŝ_k(p) = ReLU(E_k(n_p))`，`E_k(n) = Σ_lm c_km Y_lm(n)`（Route A，
c 已含 Lambertian 卷积 Â_l 与相机系方向）。噪声 ε 来自采样/量化。

## 2. Unknown scene/shared parameters

θ_s = (a_p)_{p∈Ω}（canonical albedo，Ω = 前景掩码），
几何 (n_p) 在 identifiability 分析中视为已知/已恢复量（分阶段论证）。

## 3. Per-light nuisance parameters

θ_ℓ = (c_1, …, c_N)，每图 9 维；**非定标** ⇒ 每图自带 9 维未知量。

## 4. Full Jacobian

`J = ∂I/∂(θ_s, θ_ℓ)`：
`∂I_k(p)/∂a_p = Ŝ_k(p)`；`∂I_k(p)/∂c_km = a_p Y_m(n_p) 1[Ŝ_k(p)>0]`。

## 5. Fisher block matrix

`F = [[F_ss, F_sℓ], [F_ℓs, F_ℓℓ]]`，
`F_ss = diag_p Σ_k Ŝ_k(p)²`，
`F_ℓℓ = blockdiag_k(Σ_p a_p² 1[Ŝ_k>0] Y Yᵀ)`，
`F_sℓ(p,k块) = a_p 1[Ŝ_k(p)>0] Y(n_p)ᵀ`。

## 6. Schur complement（消去 per-light nuisance）

`F_eff = F_ss − F_sℓ F_ℓℓ† F_ℓs`
逐像素形式（实现于 `gauge_fisher.ga_isi_scores`）：
`F_eff(p) = a_p² [ Σ_k Ŝ_k(p)² − Σ_k 1[Ŝ_k>0] Y(n_p)ᵀ F_k⁻¹ Y(n_p) ]`。

**解读**：第一项是"把光照当已知"时的 naive 信息（与子集相关）；
第二项是"这些信息有多少能被重新优化光照解释掉"——两灯几乎同向时，
第二项吃掉第一项的大部分 → F_eff 小。**这正是"相同 N、不同质量"的
数学来源**。

## 7. Gauge nullspace

解析已知规范方向：
- 全局尺度：δa = ε·a，δc = −ε·c（乘积不变）；
- （对 L=1 分量的相机系讨论：normal/depth 与光照的坐标耦合。）
实现：a 归一化到单位 RMS 固定尺度规范；完全的 nullspace 投影
`F_eff ← Π_g F_eff Π_gᵀ` 列为正式版升级（数值上 λ⁺ 系列已隐式
排除零特征值）。

## 8. Projected effective Fisher（候选分数）

- `λ⁺_min(F_eff)`：最坏像素有效信息（鲁棒性视角）；
- `logdet⁺(F_eff)`：全局信息量（D-最优）；
- `tr(F_eff†)`（A-optimal）：平均恢复方差；
- 简单基线：angular diversity（均值成对夹角）、N。
R4 在 calibration 数据上比较哪一个最能预测实际恢复误差。

## 9. Propositions / toy examples（写作清单）

- P1（平凡的）：N=1 时 F_eff(p) = a²(Ŝ² − YᵀF⁻¹Y) = 0（一图一像素
  完全被一盏未知光解释）⇒ 单图无联合可辨识性。
- P2：重复光不增信息：子集含重复光时 F_eff 不变（与 Δ_dup=0 一致）。
- P3：两光方向夹角 γ → 的 2D 闭式解（均匀 albedo + 单像素 SH 截断），
  展示 F_eff 随 γ 增长（"互补光"的形式化）。
- P4：固定 N、变化子集 → F_eff 的谱范围上下界（与 diversity 指标挂钩）。

## 10. 与实验的接口

- 数值验证：`defining_gate_r4.py`（E2：固定 N 内 score↔error 相关）；
- 神经验证：C0 Gate（EXPERIMENT_CONTRACT）——probe 误差随 GA-ISI 改善；
- 失败模式：若固定 N 内相关性弱，检查 (a) gauge 投影是否充分、
  (b) solver 收敛偏差、(c) ReLU 阴影区信息真空（valid_frac 报告）。
