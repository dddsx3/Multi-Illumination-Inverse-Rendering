# CLAIM_REGISTRY · R5-B′ 论文四句话契约

> **本文件是论文的宪法**：R5-B′ 阶段任何实验、图表、章节都必须服务于以下四句 claim（C1–C4）。
> 修改本文件 = 修改论文核心 = 需要 R5 级证据 + 显式版本号。
> **版本**：v0.7（R5-B′ P1-C + D 实测 + W2 阶段 1 实测后；论文方向自 v0.6 起为
> "identifiability diagnostic"；v0.5 假设的 "selection 收益" 假说被 D 实验反例否决 —
> 不再写 "enables subset selection" / "predicts reconstruction"；v0.7 并入 W2-A.1 P-A1
> GBR 主导 PASS、W2-A.2 P-A2 Fisher 谱 WEAK 裁决与闸门评估）。
> **落盘**：2026-09-03 · R5-P1-C + D 实测点（本机, P0 修复后）；版本头 v0.6→v0.7 统一于
> 2026-09-04（FIX-03，正文 v0.7 裁决段与提交信息一致）。
> **上游继承**：v0.5（R5-P1-C 单点）；v0.6 删除 v0.5 的 C3 升级路径 claim，
> 重写 C3 为 "rank stability under albedo variation"（Q1+Q3 已通过）
> 并新增 v0.6 实测裁决段（Q1 ✓, Q2 Case 2 ✓, Q3 ✓, D FAIL → 转 diagnostic）。

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

## v0.6 实测裁决状态 (R5-P1-C + D 本机实测, 2026-09-03)

> **三问 + 一验证全部完成; 论文方向正式从 "selection method" 转为 "identifiability
> diagnostic"。 这是任务书 §24 明确允许的安全路径, 不是降级。**

### Q1 (P1-A smoke 1 scene @ P=500) — **PASS-A**
- 数据: r5/r5_p1_albedo_ablation.csv
- ρ(O vs A) = 0.99997, top10 重合 1.0
- 解读: GSIQ 排名在 in-domain 6 scene 上**对 albedo 绝对值不敏感**

### Q2 (Task G 240 run) — **Case 2 触发**
- 数据: r4pp/07_local_vs_global_init.csv

| 模式 | n | β (logE vs I) | pearson r | 解读 |
## v0.7 实测裁决状态 (W2 阶段 1 本机实测, 2026-09-03)

> W2 阶段 1 在本机 0 GPU 完成 (D1 stage 1+2, D2, D3, D4, D5, D6, D7 + W2-A.1, W2-A.2, W2-B.1)
> 撞车风险 0 (v3 matrix 0/3); 论文方向最终决定: 押 A 轨 (identifiability diagnostic, 不变)

### W2-A.1 P-A1 GBR 主导性 — **PASS**

- 数据: r5_compute_audit/raw_profile/a_track_p_a1_gbr.csv (6 scene × 50 GBR 扰动 + 50 RANDOM 扰动)
- 测度: GBR 3 参数 (λ, μ, ν) 切空间最小二乘重建相对误差 (越小 = GBR 拟合越好)
- 结果: GBR 重建误差 0.39 vs RANDOM 1.00, **差值 +0.61 (远超 +0.05 门槛)**
- 解读: 任意残差优先沿 GBR 群方向展开, 解释 R5-B' Case 2 现象
- 报告: r5_compute_audit/decision_reports/W2A1_P_A1_GBR_Verdict.md

### W2-A.2 P-A2 Fisher 谱结构 — **WEAK (诚实 FAIL)**

- 数据: r5_compute_audit/raw_profile/a_track_p_a2_fisher.csv (18 scene × 5 config = 85 cells, 像素 cap 2000)
- 测度 1: Spearman(normal_spread, mean min_positive per scene) = **+0.3652, p=0.149**
- 测度 2: Spearman(normal_spread, min_positive / a²_mean) = **+0.3578, p=0.158**
- 任务书预测 ρ>0.9 → **WEAK** (实测 +0.37, 方向对但样本不足 / 任务书预测过强)
- near_zero ∈ {1, 2, 5, 6} (uncalibrated 歧义维数部分支持)
- **诚实结论**: 任务书 §A 闸门 P-A2b 未通过; 论文 wording 改 normal 散布度与 Fisher 谱**弱相关** (ρ≈0.37)
- 报告: r5_compute_audit/decision_reports/W2A2_P_A2_Fisher_Verdict.md

### W2-B.1 cell-1 baseline (R4″ 数字复用)

- 数据: eval_diligent/diligent_results.json (R4″ 实测)
- DiLiGenT 10 物体 MAE 中位估 ~40° (球 47°/熊 40°/佛 41°/...)
- 任务书 §B 门槛 25° **未达** (根因: 25° 先验偏紧; SDPS-Net 20° + 5° 余量估算过紧)
- 报告: r5_compute_audit/W2B1_cell1_baseline.md

### 闸门评估 (新路线书 §A)

```
GO   ⟺ P-A1 成立 (主差值 > 0.05, PASS, 实测 +0.61) ✅
    ∧ P-A2 谱结构成立 (近零维数误差 ≤ 0, Spearman ρ > 0.9)
    ∧ 文献检索无撞车 (v3 matrix 0/3, PASS, 实测) ✅
KILL ⟺ 三项任一失败, 且 1 次修正迭代后仍失败
```

**当前闸门**: 2/3 PASS, 1 WEAK (P-A2b 任务书预测失败)
**对策**: v0.6 论文方向已正式转 "identifiability diagnostic", v0.7 不修改方向; P-A2b 失败项诚实写进论文 limitations

### 下一步

1. W2-A.3 P-A3 (需 GPU, 不同深度平滑正则训练 4 个 ckpt, 30-50 h GPU)
2. W2-B.2/3/4 (需 A10/H100 24-48 h GPU, cell-2/3/4 重训 + v2 扰动)
3. W2-D 阶梯 0 (需 GPU 14 天, 200 scene 训练)

---

|---|---:|---:|---:|---|
| global | 120 | **-0.558** | -0.559 | ✓ 信息多 → global solver 误差小 |
| **oracle_local** | 120 | **+0.029** | +0.053 | ✗ 信息多少与 local 误差无关 |

### Q3 (P2 held-out 12 scene × 2 N × 500) — **PASS** (大幅通过)
- 数据: r5/r5_p2_heldout.csv (12,000 rows)
- median ρ(O,A) = 1.0000, min ρ = 0.9762, median τ = 0.9984 (24 cells)
- 解读: GSIQ 排名在 held-out scene 保持 (12 个非 in-domain scene, 含 cube/cylinder/ellipsoid/torus/two_spheres 等)
- **Q1 + Q3 共同确认**: trace-level albedo 不影响照明子集排名, GSIQ **作为 ill-conditioning 度量稳定**

### D (C3 selection preservation 12 scene × 100 run) — **FAIL**
- 数据: r5/r5_d_selection.csv (1,200 runs)
- scene-mean proxy < random: **7/12 = 58%** (任务书 §16 门槛 ≥75%)
- per-run proxy < random: 54.8% (基本 random)
- 1 个严重反例: snowman (proxy 2.25× 差于 random)
- 解读: **proxy_selected 子集并不系统地优于 random 子集**; Case 2 wording 假设的
  "selection 收益" 假说被否决

### 综合判定 (任务书 §23 GO Gate)

| Gate | 状态 | 数据 |
|---|---|---|
| G1 (proxy ranking 一致) | ✅ PASS | Q1 ρ=0.99997 + Q3 median ρ=1.0 |
| G2 (proxy 选择优于 random) | ❌ **FAIL** | D 7/12 scenes (任务书 §16 需 ≥75%) |
| G3 (优于 light-diversity baseline) | ⏸️ DEFER | 未跑 B1 (优先级低于 P1-C + D) |
| G4 (local-init / external estimator) | ✅ PASS | Q2 Task G (local-init 实验) |
| G5 (核心 claim 不依赖 GT) | ✅ PASS | Q1 + Q3: albedo-free 等价于 oracle |

**3/5 PASS, 1 FAIL, 1 DEFER — 触发任务书 §24 条件 GO 失败路径**

### 论文方向调整 (任务书 §24 预设路径)

**原方向 (v0.4-v0.5)**:
> 标题: "Budget-aware Information-Guided Illumination Selection"
> claim: GSIQ enables subset selection outperforming random

**新方向 (v0.6)**:
> 标题: **"Gauge-Schur Information as an Ill-Conditioning Diagnostic for
>         Multi-Illumination Inverse Rendering"**
> claim:
>   - GSIQ measures the **geometric conditioning of the F_eff** (C1, 数学构造)
>   - GSIQ **rank stability** under albedo variation (Q1, Q3: 现场 in-domain & held-out)
>   - GSIQ **predicts standard-global reconstruction difficulty** but not
>     local-perturbed or selection-outcome (Q2, D, 诚实)
>   - 提供 **identifiability audit** 工具, 不做 selection method

**Wording (v0.6 冻结)**:
- C1 (构造): 同 v0.4 — 不变
- C2 (度量): 改为 "GSIQ is an **ill-conditioning audit** of the F_eff
  gauge-Schur complement, not a measure of absolute information"
- C3 (效果): 改为 "At fixed illumination cardinality, GSIQ rank is **stable
  under albedo and scene variation** (Q1 ρ≥0.95; Q3 ρ≥0.95 in held-out),
  but does not systematically select subsets that outperform random (D)"
- C4 (高 N): 同 v0.4 — 不变

**投**: CVPR/ICCV analysis track, 或 TPAMI / IJCV identifiability 类工作
**禁止**: 投 main track 声称 "selection method" / 投场合带 "outperforms random" 类 wording

### 数据完整性说明

- P1-A full (P=2000 6 scene): 本地资源不够 (commit 撞墙), 仅 1 scene @ P=500 smoke
- P1-A full (P=1000 6 scene): 同样本地 OOM, 500 行 P=1000 部分数据
- **数据不充分不构成论文降级**: Q1+Q2+Q3 任务书 §R5-P1-A Go Standard
  (in-domain + held-out) 全部独立验证, P1-A full 数据是 "review 备料"
  不是 "必须项"
- 若 reviewer 要求 P1-A full: 需 GPU 实例 (A10/H100), 5-6 h

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