# AUDIT_PENDING · 待审计状态交接（2026-09-03）

> 用途：把当前"未决问题 + 本轮行动 + 新增数据"集中提交入库，供作者单独审计与裁决。
> 本文件仅含本仓库（Multi-Illumination-Inverse-Rendering）事项。

## 一、本轮新增数据（已入库）

- A3-0 复现臂（bs4 世代）100/100 epoch 训练完成 + test 评估产物：
  `eval_output/A3-0_f_n5gray_seed42_test/eval_summary.json` + `per_scene_metrics.csv`（commit 见 git log）。
- checkpoint（不入库，位于 `../checkpoints/A3-0_f_n5gray_seed42/`，D5 每 epoch 存档 + best_model）。
- 文档：INC-0014（系统停摆复盘+安全层）、T0-2 资产清点表一稿、T0-3 叙事卡 S-01~S-05、
  T2-1/T2-2 草案（IDENTIFIABILITY_v4 v0.3 + open5 脚本）、CALIBRATION/gpu_ledger/backup_log/run_card_howto。

## 二、未决问题（等待裁决）

1. **A3-0 复现验收偏差（主要）**：

   | 世代 | PSNR | normal MAE | albedo si-MAE |
   |---|---|---|---|
   | A3-0（bs4 复现，本机） | 32.55 | 10.30° | 0.1482 |
   | 历史 F-N5（bs8） | 36.09 | 7.792° | 0.1279 |

   偏差远超阈值 → 建议路线（择一）：(a) 接受 bs4 世代自成基线族，历史 bs8 标 reference-only，
   A3 判据按 bs4 世代重设；(b) A3-0 标"bs8 复现未达成（限制：本机 12GB 不可跑 bs8，INC-0014）"
   写入 limitation；(c) 另找 16GB+ 显存机器复跑 bs8。→ 出 INC + 更新 CLAIM_CARDS S-01 口径。
2. **README §4.1 albedo 0.0532 口径错标**（json 实为 0.1279/0.1304；0.0533 属 albOff_n_curve）——
   待确认是否修正 README；同时任务书 A3-3 判据 4 的 0.0532 引用需改指正确来源。
3. **CLAIM_REGISTRY 版本头**：文件头写 v0.6，正文含 v0.7 段且 git 提交标 v0.7——待统一。
4. G0 裁决书（09-13）未起草；CALIBRATION 室温未补测；历史 bs8 世代数字与 A3 bs4 世代并存的
   口径管理需在 T0-3/README 落定。
5. git 显示 "ahead N" 为本环境跟踪引用显示异常（push 服务端已确认同步），纯显示问题，不影响数据。

## 三、本轮已完成行动摘要

- INC-0014 复盘 + 安全层（runtime_safety/monitor_host/run_safe_arms + trainer/main 熔断 rc42 通道）；
- T0-1 标定、T0-2 清点一稿、T0-3 建卡、T0-4 情景裁定 A、T2-1/T2-2/OPEN-5 草案与数值一稿；
- A3-0 bs4 复现臂 100 epoch 完成并评估（含两次断点续跑、温度墙/内存熔断闭环验证）；
- run_arms/ trainer 收尾健壮性修补（atomic save 韧性、package exist_ok）。

*交接人：执行 agent · 2026-09-03 · 由作者审读裁决。*
