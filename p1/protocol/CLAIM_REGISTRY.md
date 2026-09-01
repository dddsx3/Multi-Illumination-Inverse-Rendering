# CLAIM_REGISTRY · R5-B′ 论文四句话契约

> **本文件是论文的宪法**：R5-B′ 阶段任何实验、图表、章节都必须服务于以下四句 claim（C1–C4）。
> 修改本文件 = 修改论文核心 = 需要 R5 级证据 + 显式版本号。
> **版本**：v0.5（R5-B′ P1-C 实测后；R4″ PIVOT B′ → 本机实测 Q1 PASS-A + Q2 Case 2
> 触发 → 论文 claim 按任务书 §R5-P1-C Case 2 wording 降级为"practical optimization
> recoverability predictor"；前版 v0.4 wording 已兼容但本版显式记录）。
> **落盘**：2026-09-02 · R5-P1-C 实测点（本机, P0 修复后）。
> **上游继承**：v0.4（R5-B′ P0）；v0.5 **不重审** v0.4 wording，只增补"实测裁决
> 状态"字段。

---

## Research Question（R5-B′ 主线）

> At fixed illumination cardinality, can a gauge-aware, nuisance-marginalized
> information-quality metric (a) predict reconstruction error, and ultimately
> (b) help select subsets that reconstruct better than random subsets?

（在**相同光照基数**下，一个 gauge-aware、nuisance-marginalized 的信息质量指标能否（a）预测重构误差、并最终（b）帮助选择比重随机子集更好的子集？）

> 该问题分解为三个层次（H-PRED / H-SEL / H-EXT），对应 R5-P2 / P3 / P4。

---

## System Claim

> We study this in feed-forward inverse rendering with **known geometry**,
> reconstructing canonical reflectance and explicit per-image illumination
> from a fixed-N set of linear-domain multi-illumination observations.

> 已知 geometry 是当前理论硬性收紧的一部分（与 R4″ 的"feed-forward 2.5D geometry
> 联合估计"区分；R4″ joint geometry–albedo–lighting 在 R5-B′ 内不再 claim）。

---

## 四句话主 claim（C1–C4 · 冻结）

### C1 · Mathematical construction

> We construct a **gauge-aware, per-light-nuisance-marginalized Gauss–Newton / Fisher information**
> for the unknown parameter vector θ = (a, c_1, …, c_N), with **geometry / normals treated as known**.

> 不得写 "joint geometry–albedo–lighting Fisher"。未知 geometry 扩展留给后续 TPAMI/IJCV 版本
> （任务书 §26）。

### C2 · Metric interpretation

> The primary continuous indicator is the **Gauge-Schur Information Quality (GSIQ)**, also
> referred to as the **Nuisance-Marginalized Spectral Information**:
>
> `I_GS = (1/d⁺) Σ_{i=1}^{d⁺} log(λ̃_i)`，其中 `λ̃ = λ/trace(F_eff)`。
>
> It measures the **spectral balance / bulk conditioning quality on the identifiable subspace**,
> not the absolute information magnitude.

> **不**声称 I_GS 直接衡量 absolute information amount。
> **必须**与 structural-null gate（d_expected, d_pos, d_extra_null, structural_status）同时报告
> （见 C1+C2 配套 + 任务书 §2 T0.2）。

### C3 · Fixed-budget effect（仅在 R5 held-out selection 通过后才允许升级）

> At fixed illumination cardinality, higher Gauge-Schur information quality is associated
> with lower reconstruction error.

> 升级路径：在 R5-P2 / P3 held-out selection **通过**后才允许写
> *"predicts reconstruction quality"* 或 *"enables subset selection"*。
> 升级前**不得**使用 "predict" / "select"。

### C4 · High-N result

> Across increasing N, **subset-sensitivity saturates** (σ_subset / Ē → 几个百分点) while
> **residual subset effect remains well above solver-repeat noise** (R_signal ≫ 1).
> We describe this as **selection-leverage compression**, not noise-limited saturation.

> 高 N 现象正式 wording：selection-leverage compression / relative subset-sensitivity saturation。
> 禁止使用：noise-floor saturation / hits the noise floor / render noise floor。
> 报告要求：同时画 CV_subset(N) = σ_subset/Ē 与 R_signal(N) = σ_subset/σ_repeat
> （任务书 §19）。

---

## v0.5 实测裁决状态 (R5-P1-C 本机实测, 2026-09-02)

> **来源**: r4pp/07_local_vs_global_init.csv (本机, P0 修复后, 240 unique runs,
> 6 scene × 2 N × 10 subset × 2 init_mode, R4″ 任务书 §R5-P1-C Case 1/2/3 判定)

| 模式 | n | β (logE vs I) | pearson r | 解读 |
|---|---:|---:|---:|---|
| global | 120 | **-0.558** | -0.559 | ✓ 信息多 → global solver 误差小 |
| **oracle_local** | 120 | **+0.029** | +0.053 | ✗ 信息多少与 local 误差无关 |

**判定: 任务书 §R5-P1-C Case 2 触发**
- β_g < 0 AND β_o ≥ 0 → 信息效应只在 standard (global-initialized) reconstruction
  下成立, 在 local-perturbed 初始化下不成立
- **C3 升级路径关闭**: 不能 claim "intrinsic identifiability → error"
- **C3 当前 wording (兼容)**: "At fixed illumination cardinality, higher GSIQ is
  *associated with* lower reconstruction error" (保守 wording, Case 1 / Case 2 都成立)
- **论文最终 wording** (任务书 §17 Case 2 降级, 待 v0.6 写入):
  > "At fixed illumination cardinality, Gauge-Schur information quality
  >  predicts the difficulty of standard (global-initialised) reconstruction
  >  and, consequently, the relative quality of selected subsets under
  >  such reconstruction pipelines."

**P1-A smoke 同时验证 (本机, 2026-09-02 1 scene × 2 N)**:
- ρ(O vs A) = 0.99997 @ P=500, 12/12 cells PASS-A (从 in-domain 6 scene 推论)
- 验证 P2 (held-out) 数据收集中, 见 `r5/r5_p2_heldout.csv`

---

## 明确不是核心的东西（系统组件，不许写进贡献列表）

- Set Transformer / FiLM / attention（PRE-0 与 P1 任务书双重禁止）
- variable-N 本身（IDArb 已占）
- 9D SH 本身（Basri & Jacobs 系已占）
- active light acquisition 本身（ReLeaPS 已占）；本文只 claim **subset selection / subset curation**
- 2.5D depth 本身
- "光照质量比数量重要"的直觉（ICCV05 经典 + GeoUniPS 已占）

---

## Primary metric 冻结理由（C2 配套 + 任务书 §2 T0.3）

GSIQ / M1 作为 primary 的**五条冻结理由**（禁止根据下一批 error 再重新选择）：

1. 数值稳定（`r4pp/03_metric_stability.csv` 5/5 PASS）
2. 有标准 spectral-volume interpretation（D-optimal design 文献直接对应）
3. 与 full Schur 数学结构直接对应
4. 无额外超参数（仅 cutoff=1e-8 + spec_cutoff=1e-8，二者已在 v2 冻结）
5. 已在 R4″ 预注册冻结（CLAIM_REGISTRY v0.3 + `archive/R4prime_frozen/`）

M2 / M5 / λ_max / d_pos 留作 sensitivity / diagnostic，不得再 claim primary。

---

## Structural-null gate（C1+C2 配套 · 任务书 §2 T0.2 · 必须与 M1 同时报告）

每个 (scene, subset) 的 metric 输出**必须**包含以下全部字段：

```
P, n_dead, d_expected (= P − n_dead − 1), d_pos, d_extra_null (= d_expected − d_pos),
structural_status ∈ {full, deficient, flip, unknown}, I_GS, d_pos_observed, ...
```

判定：

- `d_extra_null = 0`：structurally full；可单独报告 I_GS
- `d_extra_null > 0`：structurally deficient；正文**不得**单独引用 I_GS，必须连同
  (d_expected, d_pos, d_extra_null) 一并报告
- `d_extra_null < 0`：spectrum-flip，触发 review（当前实测未见）
- `unknown`：仅 operator 路径出现（大 P），此时 d_pos 不可得；正文不得引用 I_GS

实现：`p1/source/information_audit/gauge_fisher_v2.py::structural_null_gate`，
已集成到 `ga_isi_v2_scores` 的 dense 与 operator 两条路径。

---

## 字面禁词清单（任务书 §2 T0.4 · 全文生效）

R5-B′ 正文、figure caption、appendix、CSV 注释、commit message **不得**使用：

- `joint geometry recoverability` / `joint recoverability`
- `joint geometry–albedo–lighting Fisher`
- `noise-floor saturation at N=8` / `noise floor saturation` / `hits the noise floor`
- `N curve is projection of conditioning` / `N-curve as projection`
- `M1 is the only stable metric` / `M1 uniquely stable` / `M1 是唯一稳定度量`
- `render noise floor` / `渲染噪声地板`（renderer 当前 deterministic；σ_render=0）
- `noise-limited saturation`

替换：

- `joint recoverability` → （删；geometry 已知）
- `noise-floor saturation` → `selection-leverage compression` / `subset-sensitivity saturation`
- `N curve is projection` → （删；非冻结结论）
- `M1 uniquely stable` → `M1 chosen for 5 frozen reasons`
- `render noise floor` → `solver-repeat noise` / `repeatability floor`

## 用语映射表（v2 → v3 / v0.4）

| v2 / v0.3 | v3 / v0.4 | 出处 |
|---|---|---|
| log pdet / M1 logdet | GSIQ / Gauge-Schur Information Quality | C2 |
| primary metric uniquely stable | primary metric chosen for 5 frozen reasons | T0.3 |
| joint recoverability | （删；geometry 已知） | T0.4 |
| noise-floor saturation at N=8 | selection-leverage compression | C4 |
| M1 is the only stable metric | M1 chosen for 5 frozen reasons | T0.3 |
| render noise floor | solver-repeat noise / repeatability floor | §20 |
| N curve is projection of conditioning | （删；非冻结结论） | T0.4 |
| (only M1 reported) | M1 + structural-null gate 全列同时报告 | T0.2 |

---

## 执行 Gate（R5-B′ P0 → P5）

按任务书 §27 顺序：

- **P0**（当前阶段）：文档 / claim / structural-null 修复 → 已完成本文件 + IDENTIFIABILITY_v3.md + `structural_null_gate` 集成。
- **P1**：Oracle→Proxy availability audit（5 种 score × ranking fidelity / top-decile overlap / selection regret）。
  - PASS-PRACTICAL：`median ρ ≥ 0.70` 且 ≥75% scene×N cell proxy>random。
  - FAIL-PRACTICAL：oracle 强但 proxy `ρ < 0.5` 且 selection 无改善 → 停止宣传 "selection method"，退为 identifiability diagnostic paper。
- **P2**（条件 P1 PASS 后）：Held-out fixed-budget selection benchmark（≥12 held-out scene，N={3,5,8}，B0–B6 baselines）。
- **P3**（条件 P1 PASS 后）：Task G local-vs-global + external estimator / real benchmark（至少一项）。
- **P4**：冻结 figures（Figure 1–6）+ hierarchical statistics。
- **P5**：CVPR / ICCV 主会稿。

KILL 主线触发条件（任务书 §25）：K1（held-out 负 slope 不复现）/ K2（M1 top 与 random 无实际 error 差异）/ K3（angular heuristic 等价或更好）/ K4（proxy ranking 完全崩溃）/ K5（local-init + external 都失败）—— 任意一条触发即停。

---

## 6 行 Gate 数字（继承 v0.3 · 不重审 · 见 `r4pp/08_go_no_go_dashboard.md`）

- **Instrument**：PASS（M1 5/5 stability）
- **Signal**：PASS（全 24 cell R_signal > 2，median 27.2）
- **Direction**：PASS（β median −0.348, 81% 负号）
- **Interaction**：FAIL（family A N=3/5 ρ 反向；geometry × information interaction 无稳定机制证据）
- **Saturation**：PASS（N=8 R_signal=22.6, σ_subset/Ē=3.2%；高 N 子集敏感性仍远高于 solver-repeat noise，但 practical leverage 快速压缩 → "selection-leverage compression"）
- **Externality**：PENDING（Task G 因本机 WinError 1455 未执行；HANDOFF §4.1 已记）

> 当前状态（v0.4）：R4″ 6 行 Gate = 4 PASS + 1 FAIL + 1 PENDING；
> 裁决 PIVOT (B′) → R5-B′ 四句话 claim + structural-null gate 报告口径冻结。
>
> 与 R4prime_frozen 旧数据的根本区别（详见 `archive/R4prime_frozen/R4prime_failure_audit.md`
> 与 `r4pp/09_R4pp_decision.md`）：
> - primary metric 已从退化的 `λ_min⁺` 换为 M1 / GSIQ（v2 stability 5/5）；
> - 收敛判据已从 P75×P75 内生筛选换为 absolute-step200 无内生准则；
> - Solver 加 seed/theta0/trace/proj_grad_norm 可复现；
> - 新增 R5-B′：structural-null gate（d_expected/d_pos/d_extra_null）、GSIQ 命名、
>   C1–C4 四句话 claim、selection-leverage compression wording。
> **所有引用旧 R4′ 数字的论文段落必须在更新版删除或重做。**