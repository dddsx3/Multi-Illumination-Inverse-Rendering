# Q-D · C3 Selection Preservation (本机, 2026-09-03)

> **G2 失败: 7/12 scene-mean proxy<random, 58% (任务书 §16 门槛 ≥75% → FAIL)**
> **per-run proxy<random = 54.8% (基本 random)**
> **1 个严重反例: snowman (proxy 2.25× worse than random)**

## 数据
- 来源: r5/r5_d_selection.csv (1200 rows)
- 12 held-out scene × 1 N=3 × 100 solver run per cell
- 50 proxy_top (按 I_O top-10%) + 50 random
- 全部使用 same global-initialized solver (restarts=1, 400 iters)

## 关键数字 (per-scene)
| scene | proxy mean | random mean | Δ | proxy<random% | p-value |
|---|---:|---:|---:|---:|---:|
| conf_cone_r04_d12 | 2.17e-4 | 2.36e-4 | -1.9e-5 | 72% | 0.0000 |
| conf_cube_plus_cone | 3.37e-4 | 3.11e-4 | +2.5e-5 | 54% | 0.7601 |
| conf_cyl_plus_sphere | 4.41e-4 | 4.43e-4 | -2.0e-6 | 48% | 0.2954 |
| conf_cylinder_r03_d12 | 1.21e-4 | 1.26e-4 | -4.5e-6 | 62% | 0.0118 |
| conf_ellipsoid_x13z07 | 2.68e-4 | 2.91e-4 | -2.3e-5 | 72% | 0.0000 |
| conf_hemisphere_sq | 2.05e-4 | 2.07e-4 | -2.7e-6 | 58% | 0.1086 |
| conf_icosphere_sub3 | 1.78e-4 | 1.86e-4 | -7.7e-6 | 68% | 0.0010 |
| **conf_snowman** | **1.25e-3** | **5.56e-4** | **+6.96e-4** | **8%** | 1.0000 |
| conf_sphere_on_cube | 4.98e-4 | 4.53e-4 | +4.6e-5 | 32% | 0.9295 |
| conf_torus_R05_r02 | 2.71e-4 | 2.62e-4 | +9.1e-6 | 44% | 0.8738 |
| conf_torus_R06_r035 | 2.53e-4 | 2.49e-4 | +4.1e-6 | 54% | 0.6257 |
| conf_two_spheres_row | 1.28e-3 | 1.40e-3 | -1.3e-4 | 86% | 0.0000 |

## 汇总
- scene-mean proxy < random: **7/12 = 58%** (任务书 §16 ≥75% → **FAIL**)
- per-run proxy < random: 54.8% (基本 random)
- Wilcoxon paired test (per scene mean): 不显著

## 解读 (为什么 G2 失败但论文不"降级")
- Q2 (Task G) 已经说: β_oracle_local ≈ 0 → GSIQ 几乎不预测 oracle_local 误差
- D 数据 **加深**了这个结论: GSIQ 不仅"不预测" local 误差, 它**甚至无法选出比 random 更好的子集**
- 这是 **任务书 §16 主动允许的结局**: "若 7/12 不满足, 退为 identifiability diagnostic / 转 Case 2 wording"
- **论文若坚持"selection method"是错的**; 若改为:
  - "**GSIQ measures information about reconstruction difficulty (when global-init optimization is the practical reconstruction pipeline)**"
  - "**but does not select strictly better subsets than random in the global-init setting**"
  - 这是 **合法的 Case 2 wording 升级版** (比 CLAIM_REGISTRY v0.5 更诚实)
- 任务书 §24: Case 2 + D FAIL = 转 identifiability diagnostic / 不宣传 selection method
  → 论文标题从 "selection method" 改为 "GSIQ as identifiability audit"

## 论文方向调整
- ❌ 移除 "selection method" / "predicts reconstruction" / "enables subset selection" 类 wording
- ✅ 保留 "GSIQ measures **illumination-geometry conditioning** on F_eff" / "useful as **identifiability diagnostic**" / "rank stability across albedo variations (Q1+Q3)"
- 投: CVPR/ICCV workshop (identifiability) 或 analysis track, 不是 main track
- 或: 转 TPAMI/IJCV 综述类, 强调 GSIQ 作为工具的价值而非选择方法
