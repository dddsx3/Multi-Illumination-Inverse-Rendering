# Repo Reorganization Manifest · 2026-09-09（整理清单）

> 操作者：执行 agent · **备份先于一切操作**：`REPO_BACKUP_before_reorg_20260909.bundle`
> （git bundle 完整历史，216MB，`git bundle verify` 通过）+ `REPO_SNAPSHOT_before_reorg_20260909.tar.gz`
> （工作树快照，368MB），均存 `D:\MIR_Archive_20260829\`。GitHub 远端本身即第三份备份。
> 原则：**只移动文档，不动任何代码/数据/实验资产路径**；被引用文件核对引用后再移。

## 1. 迁移总表（git mv，内容零修改）

### docs/ 根 → docs/archive/<主题>/（共 41 个 .md）

| 主题 | 文件（原 docs/ 根） |
|---|---|
| `legacy_iccv/` | 论文措辞包_v0.4.md · 论文数字口径说明_v0.1.md · 论文数字口径说明_v0.5.md · outlier_归因分析_top5_20260904.md |
| `handoffs/` | HANDOFF_20260828.md · HANDOFF_20260904.md · HANDOFF_20260907_窗口终局.md |
| `phase_reports/` | Phase1_G1渲染冒烟验收报告.md · Phase1_G4训练冒烟验收记录.md · Phase1_T16_基线评估.md · Phase1_T17_DiLiGenT迁移验证.md · Phase2_6变体对比矩阵模板.md · Phase2_T26_test基线.md · Phase2_结论草稿.md · Phase2_验收报告初稿.md · EX-03_A3-1_FiLM_ablation.md · EX-04_A3-1b_lowSmooth_验收报告.md · EX-05_A3-2_seed123_验收报告.md · 逆渲染Phase0完成报告.md · 逆渲染升级计划-代码级核实报告.md · CHANGES_20260828_后期.md |
| `task_books/` | P1_任务书.md · PRE0_任务书.md · 任务书T_执行条例版v2.md · 任务书T_执行条例版v2.1修订与下一步条例.md · 强制修复书_20260904_开工前必做.md · D14_判据充分性纪律_候选.md · G0_资产清点表.md |
| `ops/` | A10_6h_作战手册.md · V100_6h_一键启动.md · 本机长跑前置清单.md · run_card_howto.md · gpu_ledger.md · backup_log.md · CALIBRATION.md |
| `audits/` | AUDIT_PENDING_20260903.md · AUDIT_REVIEW_20260904.md · T_ARM_续做清单.md |

### 仓库根 → docs/archive/phase_reports/（6 个散落文件）

EXPERT_BRIEFING.md · PRE0_前置证据汇总_20260829.md · 项目交接文档.md ·
逆向渲染项目操作手册.md · README_CLOUD.md · EX-03_verdict.json

### 目录整体迁移（git mv）

- `docs/incidents/`（INC-0001~0010）→ `docs/archive/audits/incidents/`
- `docs/design/` → `docs/archive/audits/design/`
- `docs/verdicts/` → `docs/archive/audits/verdicts/`
- `docs/paper/ICCV2027_SKELETON_v1.md`（头部已标 SUPERSEDED）→ `docs/archive/legacy_iccv/`，空目录 docs/paper/ 删除
- 仓库根 `README.md`（FusionUNet 时代 354 行版）→ `docs/archive/phase_reports/README_FusionUNet_era.md`

## 2. 明确不动的（理由）

| 资产 | 理由 |
|---|---|
| `critical_experiments/`（exp1–exp14 全部，含 json/py/npz） | REPO_MIGRATION B4"原位只读"；CI05/审计引用其路径；诚实负结果证据链 |
| `p1/`、`pre0/`、`r4pp/`、`r5/`、`eval_diligent/`、`fusion_unet.py` 等训练/评估代码 | 同上"原位冻结"；test_datasets_diligent 依赖 critical_experiments 导入 |
| `legacy_redteam/` | 只读归档区（C02 建立），tests 引用其数据语义 |
| `splits/`、`archive/`（旧训练产物目录）、`examples/`、`report_assets/`、`_arms_*.json`、`_bench_*.json`、`_demo_v2.py` 等根目录代码/产物 | 旧管线运行时资产，移交需另行裁决（PARKING_LOT 候选），本次不碰 |
| `_watch_a31_posteval.py` | 前窗口遗留未跟踪脚本，保持 untracked 原状（曾误入 archive commit，已 amend 剔除） |

## 3. 新增

| 文件 | 内容 |
|---|---|
| `docs/archive/README.md` | 归档区索引（六主题表 + 当前主线指针） |
| `README.md`（重写） | 开源科研仓库形态：项目故事 / 仓库布局 / Quick start / 关键结果表 / 方法诚信节 / 文档地图 / Citation |
| `LICENSE` | MIT |
| 本清单 | 全量 old→new 映射 + 判定依据 + 备份位置 |

## 4. 移动后的引用核对（全部通过）

- `STATE.md` / `AGENT_HANDOFF.md` / `paper/manuscript_v0.8.md` 引用的 `docs/` 根文件
  （宪法、主控计划书、REPO_MIGRATION、novelty matrix、memo_*）全部仍在原位；
- `tests/unit/test_datasets_diligent.py` 对 `critical_experiments/exp8r_diligent_discrimination_v3.py`
  的导入路径未受影响（未移动该目录）；
- `docs/archive/README.md` 提供归档区全索引；旧路径断链 = 0（历史文档间的相对引用随文件同迁）。

## 5. 结果对比（整理前 → 后）

| 位置 | 前 | 后 |
|---|---|---|
| 仓库根 .md | 10（含 354 行旧 README、两份操作手册、专家简报） | 3（README 新版 · STATE.md · AGENT_HANDOFF.md · PARKING_LOT.md） |
| docs/ 根 .md | 53 | 17（主线文档 + memo + archive/） |
| docs/archive/ | 不存在 | 6 主题 × 47 文件 + 索引 README |
| 未跟踪杂散 | _watch_a31_posteval.py（保持原状） | 同左（有意不清理，非本次任务资产） |
