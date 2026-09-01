# R5-P1-A · 结构性边界异常子集诊断 (Diagnostic, 2026-09-01)

> **触发条件**：P1-A smoke 完成后，仅 0.08% (5/6000) 子集 |I_O − I_A| > 1e-3。
> **问题**：是否说明 albedo 在 trace 归一化后仍影响 GSIQ？
> **结论**：**不**。是 `spec_cutoff=1e-8` 在边界 eigenvalue 上的离散判定造成的伪差。

---

## 1. 异常子集全表

| scene | N | subset | I_O | I_A | ΔI | d_extra_null_O | d_extra_null_A | d_pos_O | d_pos_A | rank Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| cylinder_r06_d06 | 3 | {0, 1, 2} | -6.0624 | -6.0312 | **-0.0312** | 2 | 3 | 397 | 396 | 308 |
| cylinder_r06_d06 | 5 | {0, 17, 25, 26, 28} | -6.1075 | -6.0765 | **-0.0310** | 1 | 2 | 398 | 397 | 42 |
| prism8 | 3 | {0, 18, 21} | -6.0818 | -6.1128 | **+0.0311** | 3 | 2 | 397 | 398 | 130 |
| ellipsoid_z06 | 3 | {1, 2, 18} | -6.6358 | -6.6063 | **-0.0294** | 0 | 1 | 397 | 397 | 0 |
| ellipsoid_z06 | 3 | {1, 2, 25} | -6.9616 | -6.9915 | **+0.0299** | 2 | 1 | 397 | 397 | 0 |

5/6000 = 0.083%。其余 5995 个子集 |ΔI| 中位 ~1e-5。

## 2. 根因分析：spec_cutoff 边界 eigenvalue 离散判定

### 2.1 直接证据（cylinder {0,1,2}）

trace 归一化后最小 8 个 eigenvalue 对照：

| idx | a_gt (normalised λ̃) | a=1 (normalised λ̃) | abs_diff | ratio | 在 spec_cutoff=1e-8 下 |
|---|---|---|---|---|---|
| 0 | -1.24e-14 | 8.68e-17 | — | — | (negative → excluded) |
| 1 | 1.12e-12 | 1.15e-12 | 3.63e-14 | 0.969 | excluded（< 1e-8）|
| 2 | 2.50e-09 | 2.49e-09 | 4.94e-12 | 1.002 | excluded（< 1e-8）|
| **3** | **1.00e-08** | **9.97e-09** | **5.18e-11** | 1.005 | **DIFFERS** |
| 4 | 5.83e-05 | 5.83e-05 | 4.33e-08 | 1.001 | included |
| 5 | 9.09e-04 | 9.09e-04 | 7.07e-09 | 1.000 | included |
| 6 | 9.15e-04 | 9.15e-04 | 7.18e-09 | 1.000 | included |
| 7 | 9.26e-04 | 9.26e-04 | 1.23e-07 | 1.000 | included |

a_gt 的 eigenvalue 3 是 `1.0018e-8` —— **刚好大于** `spec_cutoff=1e-8` → 计入 mean log。
a=1 的 eigenvalue 3 是 `9.97e-9` —— **刚好小于** `spec_cutoff=1e-8` → 不计入 mean log。

**两者差距 5.18e-11**（来自 a_gt_per_pixel 与 a=1 的微小差别），但通过 `log` 与 `1/d_pos` 平均后放大为 I_GS 层面 **3.12e-02** 的差异。

### 2.2 验证：换 cutoff 即可消除差异

| spec_cutoff | cylinder {0,1,2} |ΔI| | cylinder {0,17,25,26,28} |ΔI| | prism {0,18,21} |ΔI| | ellipsoid {1,2,18} |ΔI| | ellipsoid {1,2,25} |ΔI| |
|---|---|---|---|---|---|
| 1e-8（当前 smoke 默认）| **3.12e-02** | **3.10e-02** | **3.11e-02** | **2.94e-02** | **2.99e-02** |
| 1e-10 | 1.69e-05 | 1.90e-05 | 8.80e-06 | 1.07e-05 | 1.50e-06 |
| 1e-12 | 6.33e-05 | 2.75e-05 | 8.80e-06 | 5.97e-05 | 1.49e-04 |

- cutoff = 1e-10：5 个异常子集 |ΔI| 全部 ≤ 2e-5（≈ 噪声级）；
- cutoff = 1e-12：5 个异常子集 |ΔI| 全部 ≤ 2e-4（仍然很低，因 eigenvalue 真值差异被抹平）；
- cutoff = 1e-8：**5/5 异常子集都跨过临界 eigenvalue**。

⇒ 异常完全归因于 `spec_cutoff=1e-8` 在数值上的硬离散判定。

### 2.3 物理机制

a_gt 在 `fix_gauge=True` 下被归一到 RMS=1；`conf_cylinder_r06_d06` 的 a_gt ∈ [0.74, 1.02] —— 几乎接近 a=1。

但 `F_eff = F_ss − Σ_k B_k F_k† B_kᵀ` 中 `B_k ∝ a_p · s_kp · h_kp`，`F_k ∝ a_p²`，导致 F_eff **per-pixel albedo-modulated**：

- 大多数像素上 a_p ≈ 1.00 ⇒ F_eff 与 a=1 几乎相同；
- 但 s_kp² 与 albedo 的耦合在**边界像素**（h_kp 模糊或 s_kp → 0）放大：
  - a_gt[p] > 1 → F_ss_diag 增大 → d_pos 可能 +1
  - a_gt[p] < 1 → F_ss_diag 减小 → d_pos 可能 -1
- 当 eigenvalue 3 距 cutoff < 1e-10 时，per-pixel albedo 的微小调制足以让它跨过 cutoff。

**结论**：per-pixel albedo **不**改变 GSIQ 的物理量（rank、bulk spectrum、bulk trace），但**改变哪些 eigenvalue 进入 / 离开"正谱"集合**。这是一个**报告口径的离散问题**，不是物理学问题。

## 3. 影响评估

### 3.1 对 P1-A PASS-A 裁决的影响

**无影响**。原因：

1. 异常数 0.083%（5/6000）属于统计噪声；
2. 即便异常 subset 在 O vs A 间 rank 漂移 308 名（cylinder {0,1,2}），
   其他 5995 个 subset 的 ρ=1.0 仍给出 median ρ=1.0；
3. P1-A Gate 阈值 median ρ ≥ 0.95 是 robust 到这种边界噪声的。

### 3.2 对 P1-B proxy 选择的影响

**无影响**。P1-B 用 `ga_isi_v2_scores(1, Ŷ, Ĉ)`，与 a=1 = (1, Y_GT, C_GT) 共用 spec_cutoff，ρ 计算时两者都受同样边界判定误差，**误差对称** ⇒ ρ 仍接近 1.0。

### 3.3 对论文 claim 的影响

**无影响**。当前 wording 已经明示：

- "trace-normalized spectral quality"（IDENTIFIABILITY_v3 §6.1）
- "spectra balance / bulk conditioning quality"（§6.2）
- 不声称 absolute information amount（§6.2）

boundary eigenvalue 切分问题本质上是 **bulk spectrum 在 cutoff 附近的颗粒度**——`spec_cutoff=1e-8` 是任务书 §5 冻结值（与 R4″ v2 同），不应在 P1-A 阶段调它。

### 3.4 报告口径建议（additive，不替换）

在 R5-P1-A 全量（P0→P1→P2 完成后）正文 / appendix 加一段：

> *GSIQ operates on positive eigenvalues above a frozen threshold (spec_cutoff = 1e-8,
> IDENTIFIABILITY_v3 §6.1). For a small fraction of structurally-deficient subsets
> (~0.1% in smoke), per-pixel albedo modulation can shift the smallest positive
> eigenvalue across this threshold, leading to a |ΔI| ≤ 3×10⁻² outlier. The bulk
> spectrum and ranking are preserved to ≤ 1e-5; we report structural-null status
> (d_expected, d_pos, d_extra_null) alongside M1 to surface these cases
> (IDENTIFIABILITY_v3 §7).*

> 这是 additive documentation，不是新 claim，不改 wording。

## 4. 为什么这不改变 M1 primary 选择

1. **M2 / M5 在 boundary 上同样有颗粒度问题**：A-optimal 与 λ_min⁺ 在边界 eigenvalue 上的精度问题更严重（M2 在 cylinder {0,1,2} 上 a_opt_pos_norm 从 478 → 251913，相差 528×）；
2. **spec_cutoff=1e-8 已是 R4″ 冻结值**，CLAIM_REGISTRY v0.4 与 IDENTIFIABILITY_v3 §6.1 均引用同一冻结值；
4. **structural-null gate 正是为这种颗粒度问题而设计的** —— deficient 状态与 d_pos 变化让 reviewer 看见这些 case。

**禁止**根据本诊断调 spec_cutoff（任务书 §25 KILL 条件隐含：不允许靠改 metric / threshold 救 correlation）。

## 5. P0 不需要回头修改

- IDENTIFIABILITY_v3 §7.3 已要求 deficient subset **必须**连同 structural-null 一起报告 —— 本诊断的边界异常子集全部 d_extra_null ∈ {1, 2, 3}（deficient），符合 gating 要求；
- CLAIM_REGISTRY v0.4 §"字面禁词清单" 不需要增加任何新条目（不是新禁词，是新 caveat）；
- `gauge_fisher_v2.py::structural_null_gate` 行为正确（5 个异常子集全部标记 deficient）；
- `r5_p1_albedo_ablation.py` 不需要修改（smoke 上输出正确，gate verdict 正确）。

## 6. 给后续 P1-A full (Linux H100) 的建议

1. **保持 spec_cutoff=1e-8**（per 任务书 §5 冻结）；
2. **在 ranking 输出里增加 `n_at_cutoff` 字段**（每个 cell 报有多少 subset 落在 spec_cutoff 边界）—— additive、不影响裁决；
3. **gate memo 增加 boundary-outlier 表**（自动从 CSV 中筛 |ΔI| > 1e-3 的 subset 列出）—— 让 reviewer 一眼看见颗粒度问题的全貌；
4. **不要**为了消除这 0.08% 异常而：
   - 调 spec_cutoff（任务书 §25 KILL 条件）
   - 调 cutoff=1e-8（per-taskbook 冻结）
   - 引入 albedo 重加权（任务书 §4 拍板禁止）

---

*作者: ZCode agent · 2026-09-01 · 基于 R5-P1-A smoke 数据 + 直 eigenvalue 对照*
*本诊断不修改任何代码 / 文档；仅作为 P1-A 完整 closure 的 supporting evidence*