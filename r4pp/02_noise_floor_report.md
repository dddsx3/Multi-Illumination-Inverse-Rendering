# 02 · Solver-Repeat Noise Floor 报告（Task C 全量 480 runs）

> **R5-B′ 术语说明（2026-09-01）**：本文原题 "Noise-Floor Report"。当前 setup
> 下 `n_renders=1`，renderer 为 deterministic ⇒ σ_render = 0；本报告测得
> 的实际是 *solver repeat noise / solver repeatability floor*（同一 solver
> 不同 seed 引起的误差方差），不是 render noise。后续正文统一改为
> *solver-repeat noise* / *repeatability floor*。
>
> **实验**：6 scene × N{2,3,5,8} × 4 subset × 5 solver seed = 480 runs（restarts=1）
> **目的**：判定 N=8 是 saturation 还是噪声；标定绝对收敛判据；为 Gate 2 提供 R_signal。
> **脚本**：`p1/source/information_audit/r4pp_noise_floor.py`
> **数据**：`r4pp/02_noise_floor.csv`（480 行）+ `r4pp/02_noise_floor_summary.csv`（24 cell）

---

## 1. 方差分解（24 cell 汇总）

| 量 | median | 说明 |
|---|---|---|
| **R_signal** = σ_subset / σ_repeat | **27.2**（min 9.0, max 113.9）| 全部 24 cell > 2 |
| σ_solver / err_median | **0.3%**（0.1%~3.5%）| solver 高度可复现 |
| σ_subset / err_median | 9.0%（N 相关，见下）| 信息质量效应 |

## 2. 按 N 的效应衰减（saturation 的核心证据）

| N | R_signal 中位 | σ_subset/err | σ_solver/err |
|---|---|---|---|
| 2 | 31.3 | **77.9%** | 3.5% |
| 3 | 43.0 | **19.6%** | 0.3% |
| 5 | 30.4 | **4.8%** | 0.2% |
| 8 | 22.6 | **3.2%** | 0.1% |

## 3. 对任务书 §8 的裁决

**Gate 2（low-N measurable signal）初判 = PASS**：

> 任务书 §8："如果在 N≤5，R_signal > 2 在大多数高 geometry-observability
> scenes 上成立 ⇒ **强 GO**"。

- 全部 24 cell（含低 G 的 cube_axis / prism8）R_signal > 9，远超阈值 2；
- N=8 的 R_signal = 22.6 仍 > 2，但 σ_subset/err 从 N=2 的 77.9% 单调降到 3.2%。

**N=8 结论**：**不是噪声**（R_signal 22.6），而是**效应真实衰减**
（saturation regime）。这与旧 R4′ 观测到的"N=8 动态范围坍塌"（IQR/med 0.058）
**一致**——旧数据的问题不是信号不存在，而是 30 subsets 上 Spearman 功效不足
+ 收敛判据内生筛选掩盖了真实信号。

## 4. 对 P0-2/P0-3 的最终回应

| 旧失效项 | 本次证据 |
|---|---|
| P0-2 "N=8 可能 saturation" | **证实**：σ_subset/err 3.2% 且 R_signal 22.6，是真实衰减不是噪声 |
| P0-3 "收敛判据内生" | **修复**：新判据（finite ∧ abs_step200<1e-6 ∧ loss<3e-4）已冻结；pgn 仅诊断 |

## 5. 关键 caveat

- 本次只用了 **4 subsets/cell**，σ_subset 的自由度小（ddof=1，n=4）；
- **需要 Task F（controlled geometry，500-2000 candidate subsets）的
  information 分层抽样**来确认 σ_subset 的组成（是否与 information 相关）；
- render realization 全部 = 0（本次未做 §C5 的 render repeat——渲染器确定性
  已由 INC-001 帧级校验保证，且 R_signal 远高于阈值，render 分量不影响结论）。

---

*02 · Noise-Floor 报告 · 2026-08-31 · 依据 R4″ 任务书 §7-§8*
