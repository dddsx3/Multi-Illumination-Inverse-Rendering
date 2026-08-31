# 01_master_trial_table · 说明

**产出**：`01_master_trial_table.parquet`（988 行 × 91 列，436 KB）+ 同名 `.csv` + `_summary.json`
**脚本**：`p1/source/information_audit/r4pp_master_table.py`
**输入**：全部来自 `archive/R4prime_frozen/data/`（只读归档，sha256 已校验）

## 契约断言（全部 PASS）

| 断言 | 结果 |
|---|---|
| 行数守恒（join 未丢行） | 988 == 988 |
| `old_p75_success_flag` 均值 ≈ 0.5625 | **0.5840**（复现 endogenous filtering） |
| scores 全部匹配 | 缺失 0 |
| eigenspectrum 全部匹配 | 缺失 0 |
| `objective_rel_change` 全 NaN | 旧 trial 无 loss trace，刻意留缺口 |
| `geometry_family` 全部已分类 | unknown = 0 |

## 覆盖

- 11 scene（旧 R4′ 终止于 11/18）× N∈{3,5,8} = 33 cell
- family：smooth 450 / cluster 360 / composite 178
- `old_success_asrecorded` 均值 = **0.0000**（Discovery-P75 固定阈值在确认集上恒为 0）

## 列族

`identity`（scene_id / geometry_family / dataset_tag）·
`budget`（N / n_lights / illumination_ids）·
`randomness`（solver_seed_base / solver_restarts / solver_seed=NaN / pixel_seed）·
`optimization`（solver_status / iteration_count / final_objective / grad_norm /
proj_grad_norm=NaN / objective_rel_change=NaN / tail_range_abs）·
`error`（reconstruction_error / ho_psnr）·
`old filtering`（old_converged_flag / old_success_asrecorded / old_p75_success_flag
+ 两个阈值）·
`Fisher scores`（11 列）· `Fisher 全谱`（q0..q100 共 14 + n_above_* 共 9 + bulk 候选
6 + rank/active/dead 诊断 8）· `geometry`（13 列 scene 级）

## 三个 NaN 列的原因（不得填充）

| 列 | 原因 |
|---|---|
| `solver_seed` | 旧 trial 取 restarts=3 的 best，未记录 winning restart |
| `proj_grad_norm` | C-1 改动之后的新 trial 才有 |
| `objective_rel_change` | 旧 `joint_solve` 未落盘 loss trace |

## 本表独立复现的失效证据

| 项 | 从本表算出的数字 |
|---|---|
| P0-1 | primary median 2.89e-07 vs `eig_norm_q1` median 4.35e-05 ⇒ 低 **105×**；14.9% 贴 cutoff |
| P0-2 | scene 内 IQR/median = 0.469 / 0.141 / 0.058（N=3/5/8） |
| P0-3 | `old_p75_success` 均值 0.5840；`old_success_asrecorded` 0.0000 |
| **P0-3 新增** | ρ(grad_norm, err) median **−0.471**（91% cell 为负）；E[err|筛入]/E[err|筛出] median **1.068**（76% cell >1）⇒ 旧判据筛掉更准的重建 |
| P3-7 | scene 分层表（sh_gram_rank 4→9，normal_cov_eff_rank 1.00→2.27） |

## Task C 的 6 scene 选择依据（本表 P3-7 分层）

| scene | family | sh_gram_rank | normal_eff_rank | G 档 |
|---|---|---|---|---|
| `conf_cube_axis` | cluster | 4 | 1.00 | low |
| `conf_cylinder_r06_d06` | cluster | 6 | 1.17 | low-med |
| `conf_cylinder_r03_d12` | cluster | 6 | 1.20 | low-med |
| `conf_cone_r04_d12` | cluster | 9 | 1.24 | med |
| `conf_cube_plus_cone` | composite | 9 | 1.17 | med |
| `conf_egg` | smooth | 9 | 2.16 | high |
| `conf_icosphere_sub3` | smooth | 9 | 2.27 | high |

注：`conf_prism8`（sh_gram_rank 5）在旧 R4′ 终止前未跑到，Task C 若要纳入需新渲染
（渲染数据已存在于 `data_sun_confirmatory/`，只是没有 solver trial）。
