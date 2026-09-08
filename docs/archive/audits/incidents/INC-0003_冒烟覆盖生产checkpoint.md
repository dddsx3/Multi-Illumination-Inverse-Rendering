# INC-0003 · 清单验证冒烟覆盖生产 best checkpoint

- **日期**：2026-08-25
- **发现**：Phase 2 任务书 §0 开工快照（审计者核实）
- **影响**：生产路径 `checkpoints/best_model.pth` 被覆盖为清单冒烟的 epoch-0 产物
  （val 0.1582），Phase 1 基线资产（epoch 95 / val 0.018275）的 best 指针丢失；
  同目录 `checkpoint_epoch_0001–0099.pth` 完好，资产可完整恢复，无永久损失。

## 1. 时间线

| 时刻 | 事件 |
|---|---|
| 前夜 | v2 全量训练完成，best=epoch95 存入 checkpoints/best_model.pth |
| 02:35 | 提交 splits/synthetic_v2.json 划分清单（C1） |
| 02:40 | 清单接线冒烟：1-epoch 训练验证 --split_manifest 接线，**未带独立 checkpoint 目录** |
| 冒烟内 | save_checkpoint 判定首个 val 为新最佳 -> 覆盖 best_model.pth + latest_model.pth |
| 02:45 | commit 1c7a055（含清单代码，未察觉覆盖） |
| 次日晨 | Phase 2 任务书 §0 审计快照标记 🔴 资产被覆盖 |

## 2. 根因分层

**直接原因**：checkpoint 输出目录硬编码于 config.paths（默认 ../checkpoints），
冒烟命令未改该路径 => 冒烟与全量训练写同一目录。

**放大因素**：save_checkpoint 的 is_best 逻辑在每次新进程首 epoch 必然触发覆盖
（首评估必为"历史最佳"），使任何无隔离的冒烟都具备破坏性。

**过程性根因**：incidents README 启动前检查清单已有「关键产物独立存档」条款，
但仅为文字提醒、未落为强制动作（长跑命令模板无 --checkpoint_dir 占位），
执行时清单过了一遍却没有产生行为约束。

## 3. 修复（T2.0）

1. main.py 新增 `--checkpoint_dir` 与 `--viz_dir` 参数化；缺省 run_id 化
   （checkpoints/{run_id}/），单一生产路径不复存在；
2. incidents README「关键产物独立存档」升级为硬检查项：长跑/冒烟命令模板
   强制携带 --checkpoint_dir 占位；
3. 回归证明：带独立目录的冒烟后，生产根目录时间戳零新增（见 G2.0 门禁证据）。

## 4. 资产恢复记录

- `checkpoint_epoch_0095.pth` 只读加载核验：epoch=95, val_loss=0.018275491835083812
  与 `_train_bf16_v2_log.txt` 最佳行逐位一致；
- 已复制为 `phase1_best_recovered.pth`（96,581,535 字节，与源等长）；
- 被覆盖的 best_model.pth（epoch0 现场保留不覆盖），供取证。

## 5. 未尽事项

- 无。关闭条件：G2.0 门禁通过。

## 6. 状态
**已关闭（G2.0 PASS，2026-08-25）。**

关闭证据：
- 作用域缺陷修复后，3-epoch 冒烟（--run_id p2_t00_smoke_20260825，stage 1/1/1）
  exit 0，三阶段切换正常，验证损失有限（0.151 -> 0.085 -> 收敛中）；
- 产物全部落入独立目录 checkpoints/p2_t00_smoke_20260825/（5 个 pth）与
  logs/p2_t00_smoke_20260825/；
- **生产根目录取证：文件数 103 -> 103 零新增**，最新时间戳仍为事故时刻
  （02:40:23），证明隔离机制生效；
- 资产恢复：phase1_best_recovered.pth 与 checkpoint_epoch_0095.pth 等长且
  val_loss 逐位一致。