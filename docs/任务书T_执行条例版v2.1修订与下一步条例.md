# 任务书 T · 执行条例版 v2.1 修订附录 + 下一步条例（2026-09-04）
> 性质：对《任务书T_执行条例版v2.0.md》的**条例层修订**（v2.0 结构层与 P 书不受影响）。冲突处以本附录为准。
> 依据：审计裁决书_20260903（世代更替裁决）+ 强制修复书 v1.0（FIX-01..07 前置门禁）。
> 范围：本附录 = ① 修订总表（§1，已生效的条例变更）；② 下一步条例 EX-01→EX-03（§2，**FIX 全绿后按序执行**）。

---

## 1. 修订总表（v2.0 → v2.1，即时生效）

| 修订号 | 对象（v2.0 位置） | 变更内容 | 状态 |
|---|---|---|---|
| R-A | §1 OP-2 · RUN_CARD | 升级**三指纹**：`code_commit_sha` + `config_hash` + `data_manifest_sha`（命令见 docs/run_card_howto.md，FIX-05 已落地）；世代切换（任何代码改动后首训）须在账本登记"世代变更行" | ✅ FIX-05 |
| R-B | 全局 | 世代协议冻结：**Gen-A3 = bs4 + 物理约束头 + gray + synthetic_v3 划分 + bf16**。一切 A3 世代实验（A3-1~A3-5、A3-3 FW）锁死此协议，一臂一变量；历史 bs8 世代数字全部 reference-only，禁作对比基准（FIX-01/02 已落地） | ✅ FIX-01/02 |
| R-C | §3 阶段 A · A3 臂表 | 臂清单与排序重排：**A3-0（已完成，Gen-A3 首个 100ep 训练，scene 级 14.887°/32.54/0.0543/phys 0%，INC-0015）→ A3-1 noFiLM → A3-1b lowSmooth（INC-0013(c)，从"备"升正）→ A3-2 3-seed（seed 123/2024）→ A3-4 MeanPool → A3-3 FW 主臂 → A3-5 FW seed2**；备臂 **resC / physcon 删除**（物理约束头已承担 physcon 使命；resC 无叙事位） | ✅ 生效 |
| R-D | 单臂时长锚点 | **9.8–10.3h / 100 epoch（bs4 实测 370.6s/epoch，CALIBRATION）**；旧 4–6h（bs8）作废 | ✅ 生效 |
| R-E | GPU 总账 | A 阶段（含 A3-0 的 11h）：A3-1/1b/4/3/5 ×~10h + A3-2 ×2 臂 20h + 补测 ~3h ≈ **~90h**；全生命周期 **~175–195h**（情景 A 容量 480–600h 富余 2.5×，G1 11-15 不变；白天 8.4h 无人值守模式计入排期） | ✅ 生效 |
| R-F | A3-3 判据包 | 判据 4（albedo 保护）基线：~~1.2×0.0532=0.064~~ → **1.2×A3-0 albedo（scene 级 0.0543）= 0.065**（INC-0015 校准后定稿）；判据 1 参照系：~~F-N5 全平 0.03°~~ → **Gen-A3 n_curve 实测曲线（EX-01 校准版，N1–N5 极差 0.017°，N_min=1 保留）** | ✅ 定稿（EX-01 + INC-0015） |
| R-G | §5 附录判据数字总表 | 三行更新：albedo 保护 ≤0.065（scene 级基准 0.0543）；真实组/冻结纪律行不变；新增行：世代指纹（RUN_CARD 三指纹必须齐全才能验收 run） | ✅ 与 R-F 同批（EX-01/INC-0015） |
| R-H | §2 阶段 0/§3 | A3-0 已提前完成（09-03，非 09-28 起跑）；G0（09-13）验收项追加：本附录 R-A~R-G 落地 + FIX 书全绿 + gate-FIX 标签 | ✅ 生效 |

---

## 2. 下一步条例（FIX 全绿后按序执行）

### 条例 EX-01 · A3-0 n_curve 重测（推理级 · ~1–2h GPU）
- 目标：产出 **Gen-A3 世代 N 曲线**（主图 1"改造前基线"的真身），裁决 N_min 口径。
- 输入：`ckpt/A3-0_f_n5gray_seed42/best_model.pth`（已在手）+ `eval_n_curve.py`。
- 步骤：
  1. `python eval_n_curve.py --checkpoint ckpt/A3-0_f_n5gray_seed42/best_model.pth --data_root D:/data/synthetic_v3 --ns "1,2,3,4,5" --subsets_per_n 3 --out_dir eval_output/A3-0_f_n5gray_seed42_n_curve`（参数以仓库脚本实际签名校准，先 `--help` 核对）；
  2. 产物落库：`eval_output/A3-0_f_n5gray_seed42_n_curve/`（agg + raw json）；
  3. 写 RUN_CARD 补测行（三指纹：同 A3-0 的 code_commit_sha，config_hash = n_curve 参数 json）。
- 产物：n_curve agg/raw json。
- Git 提交：`feat(a3): A3-0 n_curve Gen-A3 世代重测入库 [EX-01]`
- 验收判据（裁决分支）：
  - **曲线平坦**（N=1 与 N=5 正常 MAE 差 ≤0.5°，且 N=1 不劣于 N=5 超过 0.5°）→ S-02 卡冻结 Gen-A3 版（数值更新），N_min=1 声明保留，口径卡同步；
  - **曲线不单调/显著上升**（差 >0.5°）→ N_min 声明改写候选（如 N≥2/N≥3），**停手**，把曲线贴给主智能体裁决（涉及论文主叙事，不自行定稿）；
  - 无论何支：判据包 R-F 回填（判据 1 参照系 = 本曲线）。
- 卡住时：eval_n_curve 参数报错 → 报错原文贴主智能体。

### 条例 EX-02 · A3-0 DiLiGenT zero-shot 重测（推理级 · ~0.5–1h）
- 目标：Gen-A3 世代的真实迁移数字（S-04 回填；40° 可能漂移，如实记录）。
- 输入：A3-0 ckpt（同上）+ `evaluate_diligent.py` + DiLiGenT 数据（D:/ 既有路径）。
- 步骤：
  1. 按仓库既有用法跑 zero-shot（N=5 子集协议与历史一致，先核对 evaluate_diligent.py 的参数语义与 8-25 世代评估时相同——用 `git log` 查该脚本改动历史确认无协议漂移）；
  2. 产物 `eval_output/A3-0_f_n5gray_seed42_diligent/`（json 落库）。
- Git 提交：`feat(a3): A3-0 DiLiGenT zero-shot 入库 [EX-02]`
- 验收：✓ json 入库 + S-04 卡回填（新旧数字并列注明世代）；无预设阈值——若 <35° 属惊喜（上报），>45° 属如实披露（进 Limitations 备料，不改叙事层）。
- 卡住时：DiLiGenT 数据路径缺失 → 停手报告，不自行下载。

### 条例 EX-03 · A3-1 noFiLM 启动（GPU ~10h，**本册完成后的主任务**）
- 目标：判别 FiLM 必要性（表 1 消融行；裁决书排序第一臂）。
- 前置：FIX 全绿 + EX-01/02 完成（S-02/S-04 已有 Gen-A3 数字可对照）。
- 步骤：
  1. 启动前检：`python runtime_safety.py --batch 4`（应 PASS/WARN）；`git status` 干净；账本登记"世代变更行"（Gen-A3 内首臂，写"同世代无代码改动"或列出改动 commit——**A3-1 跑前若改了任何训练代码，必须先行登记**）；
  2. 冒烟：`--disable_film` + 3 epoch，run 名 `A3-1_noFiLM_SMOKE_`（验 loss 下降 + 无 NaN）；
  3. 生产：`run_safe_arms.sh`（run 名 `A3-1_noFiLM`，bs4 默认生效，FIX-06 后无需降配提示）；白天长段或夜跑分块（OP-3 节奏）；
  4. 晨检 + 评估：`python evaluate_model.py --checkpoint ckpt/A3-1_noFiLM.pt --data_root D:/data/synthetic_v3 --split test --split_manifest splits/synthetic_v3.json --out_dir eval_output/A3-1_noFiLM_test`；
  5. RUN_CARD 三指纹齐全（code_commit_sha = 训练启动时 HEAD）；登记 S-06。
- Git 提交：`feat(a3): A3-1 noFiLM 完成指标入库 [EX-03]`
- 验收判据：与 A3-0（Gen-A3 同协议）对照——normal MAE 差 ≤2.0° → FiLM 非关键（如实记录方向，不预设好坏）；albedo si-MAE 差 ≤0.03 → 同结论；phys 违规 0%；**任何"更好/更差超过 2°"或翻转 → INC + 上报主智能体后裁决**。
- 卡住时：INC 流程（OP-5），禁止裸跑（run_safe_arms 强制）。

### 后续排队（EX-03 验收后，逐条在账本排队，无需等待指令）
- **EX-04 · A3-1b lowSmooth**（INC-0013(c)：`--albedo_smooth_stage1 1.0`，其余同 A3-0 协议）——验收：albedo si-MAE 与 albedo range/std（恢复窗口观测：range 0.211 → 若 >0.3 且 si-MAE ≤0.148 记为改善；若 si-MAE 恶化 >0.02 则维持 10.0 并记录裁决）；
- **EX-05 · A3-2 3-seed**（seed 123/2024，两臂；对照 A3-0，差 ≤0.5°）；
- **EX-06 · A3-4 MeanPool**（对照实现；过置换测试后入队）；
- **EX-07 · A3-3 FW 主臂**（A1 设计完成后启动——注意 A1 设计条例（T1-1~T1-6）是 CPU 任务，与 EX-03~06 并行窗口内完成）。

---

## 3. 与上级文档的衔接

- 投稿战略 v1.1：G1（11-15）判定公式不变（实测斜率 ≥60% 理论 → B-T 主线）；判据包数字以本附录 R-F 定稿版为准。
- 强制修复书：本附录在 FIX-07 执行后入库 `docs/`（与 v2.0 任务书同目录）。
- P 书：不受世代更替影响的结构性修订——P 稿的 S 资产引用自动继承新世代数字（S-01 等卡的登记号不变，数值已换）；P 书 v2.0 §OP-P1 数字卫生照常生效（只准从登记表取数，登记表已由 FIX-02 更新）。

*v2.1 附录 · 2026-09-04 · 下一步：FIX-01..07 全绿 → gate-FIX-20260904 → EX-01 → EX-02 → EX-03。*
