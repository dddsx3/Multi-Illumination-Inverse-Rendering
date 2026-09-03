# R5-B′ · 本机全任务汇总裁决报告 (v2, 2026-09-03)

> **结论 (大白话)**: **任务书 5 个 GO Gate 中 3 过 1 失败 1 未做**。
> - **失败的是 G2 (selection preservation)**: proxy 选的子集跟 random 差不多 (54.8%)。
> - **任务书 §24 预设路径是"转 identifiability diagnostic paper"** — 不再宣传 selection method,
>   改为宣传"GSIQ 作为 ill-conditioning 审计工具 + rank 稳定性"。
> - **全部本机完成, 0 云算力, 0 元成本**。论文方向已调整, CLAIM_REGISTRY v0.6 冻结。

## 1. 四问最终答案 (大白话)

| Q | 问的是什么 | 答 | 数据 | 任务书门槛 |
|---|---|---|---|---|
| **Q1** | GSIQ 排名是否依赖 albedo 绝对值 | **不依赖** (PASS-A) | 1 scene × P=500: ρ(O,A)=0.99997 | ≥ 0.95 ✅ |
| **Q2** | GSIQ 能否预测 reconstruction error | **只能预测 standard-global, 不预测 local (Case 2)** | Task G 240 run: β_g=-0.56, β_o=+0.03 | Case 2 触发 ✅ |
| **Q3** | GSIQ 排名在 held-out scene 上是否保持 | **完全保持** | 12 scene × 2 N: median ρ=1.0 | ≥ 0.95 ✅ |
| **D** | proxy 选择优于 random | **不优于** (54.8% ≈ random) | 12 scene × 100 run: scene-mean 7/12=58% | ≥ 75% ❌ **FAIL** |

**4/4 任务书核心问题全部得到诚实数据 (3 PASS + 1 FAIL)**

## 2. 任务书 §23 GO Gate 全部 5 项

| Gate | 状态 | 数据 |
|---|---|---|
| G1 (proxy ranking 一致) | ✅ **PASS** | Q1 ρ=0.99997 + Q3 median ρ=1.0 |
| G2 (proxy 选择优于 random) | ❌ **FAIL** | D 7/12 scenes (任务书 §16 需 ≥75%) |
| G3 (优于 light-diversity baseline) | ⏸️ DEFER | 未跑 (B1 路线) |
| G4 (local-init / external estimator) | ✅ **PASS** | Q2 Task G (local-init 实验) |
| G5 (核心 claim 不依赖 GT) | ✅ **PASS** | Q1 + Q3: albedo-free ≡ oracle |

**3/5 PASS, 1 FAIL, 1 DEFER — 触发任务书 §24 条件 GO 失败路径 (预设)**

## 3. 论文方向调整 (任务书 §24 安全路径)

**原方向 (v0.4-v0.5)** — 已被否决:
- ❌ 标题: "Budget-aware Information-Guided Illumination Selection"
- ❌ claim: GSIQ enables subset selection outperforming random

**新方向 (v0.6)** — 任务书 §24 明确允许的"identifiability diagnostic"路径:
- ✅ 标题: **"Gauge-Schur Information as an Ill-Conditioning Diagnostic for
  Multi-Illumination Inverse Rendering"**
- ✅ claim:
  - GSIQ measures the **ill-conditioning of the F_eff** (C1, 数学构造)
  - GSIQ **rank stability** under albedo + scene variation (Q1, Q3 数据)
  - GSIQ **predicts standard-global reconstruction difficulty** (Q2)
  - 提供 **identifiability audit** 工具, 不做 selection method

**投**: CVPR/ICCV analysis track, 或 TPAMI/IJCV identifiability 类工作
**禁止**: 投 main track 声称 "selection method" / "outperforms random"

## 4. 阶段 A (P1-A full) - 本地资源不足 (最终完成率如实注记)

**最终现场 (2026-09-03 复查)**: `r5/r5_p1a_full.csv` 含 **37,128 行干净数据 = 任务书 §R5-P1-A 目标 (6 scene × 83,520 行) 的 44.5%**。
- 已完成 3/6 scene: conf_sphere_r05 (N3+N5)、conf_cube_axis (N3+N5)、conf_prism8 (仅 N3; P=2000 OOM 中断)
- 整批缺失 3 scene: conf_egg / conf_cylinder_r06_d06 / conf_ellipsoid_z06 (OOM 反复 kill + 24GB commit 配额不足)
- 复查时清理 1 行截断脏数据 (原行 6962, scene='8' subset_id 字段错位, I_A 列丢失)
- 根因: 物理 15.2GB + 24GB commit 限额, 单进程实际 ~2-3GB; P=2000 启动 10 min 内 OOM
- **不阻断论文主结论** (Q1+Q3 已独立验证, 任务书 §R5-P1-A 通过)
- 若 reviewer 要求补齐 P1-A full: 需 GPU 实例 (A10/H100), 补 3 scene 约 5-6 h

## 5. 产物索引 (本次会话)

| 路径 | 内容 |
|---|---|
| r5/r5_p1_albedo_ablation.csv | P1-A smoke 1 scene @ P=500 (6,000 rows) |
| r5/r5_p1a_full.csv | P1-A full 3/6 scene × N{3,5} = 37,128 rows (44.5%, 脏行已清; 缺 conf_egg/conf_cylinder/conf_ellipsoid) |
| **r5/r5_d_selection.csv** | **D C3 selection 12 scene × 100 run = 1,200 rows (FAIL 数据)** |
| r4pp/07_local_vs_global_init.csv | Task G 240 run (Case 2 数据) |
| r5/r5_p2_heldout.csv | P2 held-out 12 scene × 2 N × 500 (12,000 rows) |
| r5/r5_q2_taskG_verdict.md | Q2 大白话裁决 (Case 2) |
| r5/r5_q3_p2_verdict.md | Q3 大白话裁决 (PASS) |
| **r5/r5_q_d_verdict.md** | **D 大白话裁决 (FAIL)** |
| r5/R5_FINAL_REPORT.md | 本报告 v2 |
| r5_p1a_full.py | P1-A full 启动器 (受 commit 限制) |
| r5_p2_heldout.py | P2 启动器 |
| r5_d_selection.py | D 启动器 |
| r5_local_smoke.py | 6 项能力一键自检 |
| **p1/protocol/CLAIM_REGISTRY.md** | **v0.6 (论文方向正式转为 identifiability diagnostic)** |
| r5_compute_audit/LOCAL_MACHINE_DIAGNOSIS.md | v2 (P0 修复实测) |
| r5_compute_audit/CAMPAIGN_REPORT.md | Compute-Aware Campaign (B+C 路线) |

## 6. 一句话总结

> **R5-B′ 论文主线 4/4 任务书核心问题 (Q1-Q3 + D) 全部得到诚实数据, 3 PASS 1 FAIL。**
> 失败的是 selection method 假说 (D FAIL), 触发任务书 §24 预设路径 —
> **论文方向从 "selection method" 转为 "identifiability diagnostic"**。
> CLAIM_REGISTRY v0.6 已是投稿冻结 wording。0 云算力, 0 元成本, 全部本机完成。
> 论文可以开始写, 投 CVPR/ICCV analysis track 或 TPAMI/IJCV。

---

*作者: ZCode agent · 2026-09-03 · 本机 RTX 5070 Ti Laptop · 0 元云算力*
*基线 commit: 9796884 (R4″ sprint 收官)*
*CLAIM_REGISTRY v0.6 已落地: p1/protocol/CLAIM_REGISTRY.md*
*最终 commit: 32af0e7 → 待 v0.6 推送*
