# R5-P1-A Closure · Albedo Ablation Smoke (v3, 2026-09-01)

> **Status**: PASS-A (smoke, RTX 5070 Ti)
> **Verdict**: median(ρ) = 1.0000, median(top10) = 1.000, median(top20) = 1.000
>   ⇒ **freeze a=1** ⇒ proceed to R5-P1-B Normal/Light Proxy Audit (条件 PASS-A)
>
> **本版本 (v3)** 在 v2 基础上补充：
> 1. `n_at_cutoff` 字段（surface boundary granularity 敏感度）
> 2. 自动 boundary-outlier 表 + CSV（|ΔI| > 1e-3 subset 全列表）
> 3. P=300 smoke 重新生成（绕开本机 commit 配额 OOM；Linux H100 上用 pixel_cap=2000）
> 4. `gauge_fisher_v2.py::schur_full` 内存优化（per-iter del + gc.collect，绕开 Windows OOM）

## 1. 实验设计（冻结项）

| 项 | smoke（RTX 5070 Ti）| full P1-A (Linux H100) |
|---|---|---|
| scenes | 6 from `data_sun_confirmatory/` | 6 + 0–6 more dev scenes |
| NS | {3, 5} | {3, 5} |
| pixel_cap | **300** (per Windows commit quota；H100 上 2000) | 2000 |
| N=3 pool | enumerate first 500 | enumerate all 4960 |
| N=5 pool | sample 500 | sample 2000 |
| solver arm | off (--solver not passed) | on, ~360 runs |
| metric | GSIQ / M1 (full_logdet_pos_norm) | same |
| structural gate | on (P0 deliverable) | on |
| n_at_cutoff (新) | 输出 | 输出 |
| outlier 表 (新) | 自动 | 自动 |

## 2. Results（全部 12 cells）

| scene | N | ρ | top10 | top20 | n_def_O | n_at_cutoff_O_med | outliers |
|---|---|---|---|---|---|---|---|
| conf_sphere_r05 | 3 | 1.0000 | 1.000 | 0.990 | 11 | 0 | 7 |
| conf_sphere_r05 | 5 | 1.0000 | 1.000 | 1.000 | 8 | 0 | 7 |
| conf_cube_axis | 3 | 1.0000 | 1.000 | 1.000 | 500 | 0 | 0 |
| conf_cube_axis | 5 | 1.0000 | 1.000 | 1.000 | 500 | 0 | 0 |
| conf_prism8 | 3 | 0.9970 | 1.000 | 1.000 | 500 | 1 | 1 |
| conf_prism8 | 5 | 1.0000 | 1.000 | 1.000 | 500 | 1 | 0 |
| conf_egg | 3 | 0.9999 | 1.000 | 0.980 | 10 | 1 | 2 |
| conf_egg | 5 | 1.0000 | 1.000 | 1.000 | 13 | 0 | 22 |
| conf_cylinder_r06_d06 | 3 | 1.0000 | 1.000 | 1.000 | 500 | 1 | 0 |
| conf_cylinder_r06_d06 | 5 | 1.0000 | 1.000 | 1.000 | 500 | 1 | 0 |
| conf_ellipsoid_z06 | 3 | 1.0000 | 1.000 | 0.990 | 12 | 0 | 0 |
| conf_ellipsoid_z06 | 5 | 1.0000 | 1.000 | 1.000 | 6 | 0 | 11 |

median ρ = 1.0000, median top10 = 1.000, median top20 = 1.000 → PASS-A

**Boundary outliers**: 50/6000 = 0.833%。详见 `r5/r5_p1_albedo_ablation_outliers.csv` + `r5_p1_albedo_ablation_gate.md` 表。

## 3. Sanity check（数值稳定性 vs. artifact）

```
abs(I_O - I_A) range across all 6000 subsets: [4e-11, 4.25e-02]
boundary outliers (>1e-3): 50 subsets = 0.833%
typical |ΔI|:1e-5 到 1e-10
```

**1 个 outlier 达 4.25e-2**（prism8 N=3 {0,25,29}）；其余 49 个 outlier |ΔI| ≤ 1.85e-3。

**结构性原因**（详见 `r5/r5_p1_a_boundary_diagnostic.md`）：a_gt_per_pixel modulation 在 1e-11 级别扰动 F_eff 最小正 eigenvalue，导致其跨越 `spec_cutoff=1e-8`。这导致 O 与 A 的 `d_pos` 差 1，进而 mean log shift ~3e-2。物理量（bulk spectrum, ranking）保持 ≤ 1e-5。

## 4. P0 / P1 提交物完整列表

### 代码改动
| 文件 | 改动 | 风险 |
|---|---|---|
| `p1/source/information_audit/gauge_fisher_v2.py` | 加 `structural_null_gate` / `n_dead_count`；`spectrum_metrics` 增加 `n_at_cutoff`；`ga_isi_v2_scores` 输出 `full_n_at_cutoff`；`schur_full` 加 per-iter del + gc.collect | additive，R4″ 旧数字不受影响 |
| `p1/source/information_audit/r5_p1_albedo_ablation.py` | P1-A smoke 脚本 | 新文件 |
| `p1/protocol/IDENTIFIABILITY_v3.md` | 数学文档 v3（含 structural-null）| 替代 v2，相同 IDENTITY 数学不变 |
| `p1/protocol/CLAIM_REGISTRY.md` | v0.4 四句话 claim | 替代 v0.3 |
| `r4pp/08_go_no_go_dashboard.md` / `r4pp/09_R4pp_decision.md` / `r4pp/02_noise_floor_report.md` | 术语修订 | wording-only |

### 落盘
| 路径 | 内容 |
|---|---|
| `r5/r5_p1_albedo_ablation.csv` | 6000 行 per-(scene, N, subset) raw |
| `r5/r5_p1_albedo_ablation_ranking.csv` | 12 行 per-(scene, N) ranking |
| `r5/r5_p1_albedo_ablation_outliers.csv` | 50 行 boundary outliers（**新**）|
| `r5/r5_p1_albedo_ablation_gate.md` | Gate memo（含 outlier 表）|
| `r5/r5_p1_a_boundary_diagnostic.md` | 边界异常根因分析 |
| `r5/r5_p1_a_closure.md` | 本文件 |
| `r5/P1_A_README.md` | 数据集与 budget 说明 |
| `r5/P1_C_TASK_G_PREP.md` | Linux H100 上 Task G 启动 checklist |

## 5. Gate verdict（任务书 §R5-P1-A）

| Verdict | criterion | 结果 |
|---|---|---|
| **PASS-A** | median(ρ) ≥ 0.95 AND median(top10) ≥ 0.80 | ✅ PASS-A |
| CONDITIONAL | 0.80 < median(ρ) < 0.95 | — |
| FAIL-A | median(ρ) ≤ 0.80 | — |

**Next step**: 进入 R5-P1-B Normal/Light Proxy Audit。**不**引入 â（per 拍板决策 1）。

## 6. 全量 P1-A 与 smoke 的差距（待 Linux H100）

1. **pixel_cap 300 vs 2000**：全量应给出更小数值噪声，但 trace 归一下的 rank 性质不变；
2. **N=3 enumerate 500 vs 4960**：全量覆盖完整 C(32,3)；
3. **N=5 sample 500 vs 2000**：同上；
4. **solver arm**：本机 smoke 未跑；H100 上加 ~360 run solver sanity check；
5. **dev scenes 数 6 vs ≥10**：全量可扩 scene-family 多样性。

## 7. 当前项目状态

- **R5-P0**：✅ 已完成
- **R5-P1-A smoke**：✅ 已完成 PASS-A（本回合，RTX 5070 Ti）
- **R5-P1-A full**（Linux H100，待算力）：未启动；脚本与数据已就绪
- **R5-P1-B**（条件 PASS-A → 已触发；待 H100）：未启动
- **R5-P1-C Task G**：脚本与数据打包就绪，待 Linux H100

**总裁决点 Q1**（oracle → proxy 是否可桥接）：在 P1-B 全量之后回答。
**总裁决点 Q2**（local-global 是否保持）：在 P1-C 之后回答。
**总裁决点 Q3**（selection 是否值得做）：在 P1-B 全量 + P2 之后回答。

## 8. 与 HANDOFF §4.3 + 任务书 §25 的兼容性

- ❌ 不改 M1 primary 地位
- ❌ 不动 R4″ 6 行 Gate 数字
- ❌ 不碰 `archive/R4prime_frozen/`
- ❌ 不调 `spec_cutoff=1e-8` / `cutoff=1e-8`（boundary outlier 表的设计前提是不动这两个阈值）
- ❌ 不为消除 outlier 而改 metric 或 threshold
- ❌ 不写"joint recoverability" / "noise-floor saturation" / "M1 uniquely stable" / "render noise floor"
- ✅ 新增 `n_at_cutoff` 是 additive 诊断字段，不改变 primary metric
- ✅ 新增 boundary outlier 表是 additive 报告口径，不改变裁决

---

*作者: ZCode agent · 2026-09-01 · 基于 R5-P1-A 任务书拍板 · baseline `9796884` 之后*
*本文件不修改任何科学裁决的 wording，所有冻结 wording 见 `p1/protocol/CLAIM_REGISTRY.md` v0.4 + `p1/protocol/IDENTIFIABILITY_v3.md`*