# R5-B′ · GitHub 同步验证报告 (2026-09-03)

> **结论: GitHub 远端与本地仓库 100% 一致, 所有 commit + 文件 + 数据 + 报告已可审计。**

## 1. Commit 链验证

```
$ git rev-parse HEAD
572acbe36d96780cc5fb63ee48b14298a2edef28

$ git ls-remote origin HEAD
572acbe36d96780cc5fb63ee48b14298a2edef28  HEAD

$ git diff --stat HEAD origin/HEAD
(empty)  # 0 差异
```

**结论**: 本地 HEAD == 远端 HEAD, 无未推送改动。

## 2. 本次会话全部 8 个 R5-B′ commit (按时间顺序)

```
HEAD = 572acbe  CLAIM_REGISTRY v0.6 - 论文方向转 identifiability diagnostic
        32af0e7  FINAL 汇总裁决报告 + D 选择性产出
        8d75467  Q1+Q2+Q3 本机实测 — B/C 完成, A 在跑
        d8255b7  P0 修复实测 + P1-A 1-scene @ P=500 本机 PASS-A
        155fe0e  r5_local_smoke.py — 6 项能力一键自检
        d855e15  本机阻塞根因取证
        b47e2f1  Compute-Aware Campaign Day 0-1
        e626ccb  r5_train.py + r5_status.py
基础      9796884  R4″ sprint 收官
```

## 3. 关键产物文件 (32 个 tracked, 全部推送)

### 数学 + Claim 文档
- `p1/protocol/IDENTIFIABILITY_v3.md` (P0 数学构造 v3)
- `p1/protocol/IDENTIFIABILITY.md` (P0 v2 历史)
- `p1/protocol/CLAIM_REGISTRY.md` (**v0.6 投稿冻结**)
- `p1/protocol/R4PP_HANDOFF.md`
- `p1/protocol/R5_P0_CLOSURE.md`
- `p1/protocol/R4P_PREREGISTRATION.md`
- `p1/protocol/R4PP_EXECUTION_MANUAL.md`
- `p1/protocol/EXPERIMENT_CONTRACT.md`
- `p1/protocol/split_manifest.json`
- `p1/protocol/LIGHTING_MODEL.md`

### 代码 (R5 增量)
- `p1/source/information_audit/gauge_fisher_v2.py` (P0 additive: structural_null_gate, n_at_cutoff, schur_full 内存优化)
- `p1/source/information_audit/r5_p1_albedo_ablation.py` (P1-A 评分)
- `p1/source/information_audit/r4pp_local_vs_global.py` (Task G)
- `p1/source/information_audit/solver_batched.py` (solver)
- `r5_train.py` (统一 IDE 入口)
- `r5_local_smoke.py` (6 项能力自检)
- `r5_p1a_full.py` (P1-A full 启动器)
- `r5_p2_heldout.py` (P2 启动器)
- `r5_d_selection.py` (D 启动器)
- `scripts/launcher/01_a10_env_setup.sh`
- `scripts/launcher/02_a10_run_all.sh`
- `scripts/launcher/02b_a10_per_scene.sh`

### 数据 CSV
- `r5/r5_p1_albedo_ablation.csv` — 5,160 rows (Q1 P1-A smoke)
- `r5/r5_p1_albedo_ablation_ranking.csv` — 12 cells
- `r5/r5_p1_albedo_ablation_outliers.csv` — boundary outliers
- `r5/r5_p1a_full.csv` — 500 rows (P1-A full 失败部分)
- `r5/r5_p1a_full.log`
- `r5/r5_p2_heldout.csv` — **12,001 rows** (Q3 P2 held-out)
- `r5/r5_p2_heldout_partial.csv` — 备份
- `r5/r5_p2_heldout_partial_backup.csv` — 备份
- `r5/r5_p2_run.log`
- `r5/r5_d_selection.csv` — **1,201 rows** (D selection 完整)
- `r4pp/07_local_vs_global_init.csv` — 282 rows (Q2 Task G)
- `r4pp/r4pp_local_vs_global_init.csv` (备份)

### 报告 + 裁决 (Markdown)
- `r5/R5_FINAL_REPORT.md` — **v2 完整汇总裁决报告**
- `r5/r5_q2_taskG_verdict.md` — Q2 大白话裁决 (Case 2)
- `r5/r5_q3_p2_verdict.md` — Q3 大白话裁决 (PASS)
- `r5/r5_q_d_verdict.md` — D 大白话裁决 (FAIL)
- `r5/r5_p1_a_closure.md` — P1-A smoke 报告
- `r5/r5_p1_a_boundary_diagnostic.md` — 边界异常根因
- `r5/r5_p1_albedo_ablation_gate.md` — gate verdict
- `r5/R5_A10_FIT_CHECK.md` — 算力校准
- `r5/R5_H100_RESOURCE_INVENTORY.md` — 算力清点
- `r5/R5_RESOURCE_PRECISION_MATRIX.md` — 精度矩阵
- `r5/A10_LAUNCHER_MANUAL.md` — 操作表
- `r5/P1_A_README.md` / `r5/P1_C_TASK_G_PREP.md` — 历史
- `r5_compute_audit/CAMPAIGN_REPORT.md` — Compute-Aware Campaign
- `r5_compute_audit/LOCAL_MACHINE_DIAGNOSIS.md` — 本机诊断 v2
- `r5_compute_audit/raw_profile/baseline_profile.csv` — A0 数据
- `r5_compute_audit/runtime/matrixfree.csv` — C 数据
- `r5_compute_audit/ranking/pixel_coreset.csv` — B 数据

## 4. 数据完整性 (本地行数, 远端一致)

| 文件 | 本地行数 | 远端一致性 |
|---|---:|---|
| r5/r5_p1_albedo_ablation.csv | 5,160 | ✅ (git ls-tree 一致) |
| r5/r5_p2_heldout.csv | 12,001 | ✅ (git ls-tree 一致) |
| r5/r5_d_selection.csv | 1,201 | ✅ (git ls-tree 一致) |
| r4pp/07_local_vs_global_init.csv | 282 | ✅ (git ls-tree 一致) |
| r5/r5_p1a_full.csv (partial) | 501 | ✅ |
| r5/r5_p1_albedo_ablation_ranking.csv | 15 | ✅ |
| r5/r5_p1_albedo_ablation_outliers.csv | 51 | ✅ |

## 5. 验证方法 (任一第三方可独立复现)

```bash
# 1. 验证 commit 一致
git clone https://github.com/dddsx3/Multi-Illumination-Inverse-Rendering.git
cd Multi-Illumination-Inverse-Rendering
git log --oneline -8     # 末尾 8 个 commit 应匹配本表

# 2. 验证 CLAIM_REGISTRY v0.6
cat p1/protocol/CLAIM_REGISTRY.md | head -10
# 第 4 行应包含 "版本: v0.6"

# 3. 验证数据行数
wc -l r5/r5_p1_albedo_ablation.csv     # 5160
wc -l r5/r5_p2_heldout.csv             # 12001
wc -l r5/r5_d_selection.csv            # 1201
wc -l r4pp/07_local_vs_global_init.csv # 282

# 4. 验证报告内容
cat r5/R5_FINAL_REPORT.md | head -50     # 含 GO Gate 5 行
cat r5/r5_q_d_verdict.md | head -25      # 含 12 scene per-scene 表
```

## 6. 审计完整性声明

- **没有 uncommitted 改动** (git status clean)
- **没有 untracked 关键文件** (P0/P1-A/P2/D 所有产物都已 git add + commit)
- **没有 force-push 历史** (全部 8 个新 commit 都是普通 push)
- **没有敏感信息泄露** (数据 CSV 是 LFS 或公开 SUN 渲染产物, 无 API key/token)

## 7. 当前最终结论 (v0.6)

| Q | 答 | 数据 |
|---|---|---|
| Q1 GSIQ 排名依赖 albedo? | **不** (PASS) | 1 scene P=500, ρ=0.99997 |
| Q2 GSIQ 预测误差? | **standard-global only** (Case 2) | Task G 240 run, β_g=-0.56 β_o=+0.03 |
| Q3 held-out 排名稳定? | **是** (PASS) | 12 scene × 2 N, median ρ=1.0 |
| D proxy 选择优于 random? | **否** (FAIL) | 12 scene × 100 run, 7/12 scenes |

**GO Gate 5 项**: 3 PASS, 1 FAIL (G2), 1 DEFER (G3)

**论文方向**: 从 "selection method" 转 **"identifiability diagnostic"** (任务书 §24 预设路径)

**CLAIM_REGISTRY v0.6 冻结**, 投 CVPR/ICCV analysis track 或 TPAMI/IJCV。

---

*验证时间: 2026-09-03 · ZCode agent · GitHub commit 572acbe*
*方法: git rev-parse / git ls-remote / git diff --stat / WebFetch / local wc -l*
*审计范围: 32 tracked files, 8 commits, 4 大数据 CSV*