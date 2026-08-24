# INC-0001 · T1.5 首次全量训练 AMP 梯度爆炸导致永久 NaN

- **日期**：2026-08-24
- **影响**：首次 100 epoch 训练后 65 个 epoch（35-99）全部作废；GPU 时间浪费约 1.5 小时。
  最佳 checkpoint（epoch 31，val 0.0605）在污染前已存档，未损失模型资产。
- **状态**：已修复（NaN 守卫 + 权重校准 + fp32 重训），重训见 `_train_fp32_log.txt`

## 1. 时间线（基于 _train_full_diverged_run1_log.txt）

| Epoch | 事件 |
|---|---|
| 0-23 | 健康：val 0.317 -> 0.085；但 epoch 24 起预裁剪梯度范数已飙至 9.8e3 |
| 24, batch20 | 首个 nan batch（AMP scaler 跳过溢出步，当 epoch 后段恢复） |
| 31 | val 达最优 **0.0605**（阶段1 收尾） |
| 32 | 进入阶段2：权重表切换（albedo_smooth 50->0.1、新增 SH 约束项），val 冲击至 0.297 |
| 33-34 | 恢复中（0.184 / 0.164） |
| **35 起** | **验证/训练损失永久 NaN，直至 epoch 99** |

## 2. 证据链

1. 梯度范数（clip 前）：epoch 24 首批即 9890 / 3483 / 1175 —— 说明梯度在裁剪前
   已爆炸，clip=1.0 只是掩盖了症状（等效学习率骤降但方向已病态）；
2. nan batch 从"偶发"到"永久"的演化：24(偶发) -> 35 起(100% nan)，符合 fp16 下
   权重逐步走向病态区域、最终 loss scale 无法挽救的模式；
3. 排除阶段3/残差解冻：NaN 早于 epoch 60 的解冻点；排除数据问题：同数据集
   fp32 冒烟 3 epoch 干净收敛（0.31 -> 0.05）；
4. 反照率质量指标异常：Albedo Corr=0.9886（反照率几乎等于输入图），说明
   albedo_smooth=50 的极端正则与重建/GT 损失互相拉扯，优化已在多个目标间震荡。

## 3. 根因分层

**直接原因**：AMP fp16 数值域下，多损失项（尤其 albedo_smooth=50 的 TV 正则）叠加
产生溢出梯度；scaler 能跳过 inf 步，但有限却巨大的梯度仍把权重推入病态区域，
最终 total_loss 变 NaN 且不可逆。

**放大因素**：
- `albedo_smooth=50` 是无 GT 自监督时代的历史调参（注释原文"粉碎所有纹理"），
  在有 GT 监督（gt_albedo=0.5）的新范式下属于重复且极端的约束；
- fp16 有效位宽 10 bit，本管线含深度->法线卷积、SH 幂运算、多次除法，数值域天然紧张。

**过程性根因（为什么会带病上线）**：
- G4 冒烟只跑 3 epoch，未覆盖阶段2 切换点后的长期稳定性——切换冲击与梯度累积
  都发生在冒烟视野之外；
- 训练循环无 NaN 守卫与发散自动停机，坏 batch 直接污染权重后继续训练 65 个 epoch；
- 无梯度范数预警阈值（>10^3 即应人工介入）。

## 4. 已实施修复（trainer.py @ commit 0d9a81d+）

1. **非有限损失守卫**：backward 前检查 `torch.isfinite(total_loss)`，失败则
   `zero_grad(set_to_none=True)` + 跳过该 batch + `_skipped_nan` 计数入日志；
2. **权重校准**：阶段1 albedo_smooth 50 -> 10（保留主导正则地位，去除爆炸源）；
3. **fp32 重训**：关闭 --use_amp 作为稳定性基线；AMP 仅在连续一次干净跑通后
   以对照实验方式重新评估。

## 7. 关闭记录（2026-08-25，T2.1 完成时追加）

- **NaN 自动停机**：已实现并经注入测试验证——连续非有限损失达
  nan_abort_streak（收敛值 **10**，声明见 docs/design/t2_1_params.md）抛
  RuntimeError 终止；tests/test_stability_guards.py case1 通过；
- **梯度范数预警/硬停机两级阈值**：>1e3 预警写 TB 标量、>1e4 硬停机，
  case2/case3 通过；集成测试证明守卫经真实 train_epoch 全链路生效；
- **回归冒烟**：fp32 与 BF16 各一次 3-epoch 全管线（独立目录），零 NaN、
  覆盖两次阶段切换。

**状态：已关闭。** 后续观察项保留：fp32 重训若再现梯度范数 >1e3 频发，按
docs/design/t2_1_params.md 升级路径处理（降 LR + warmup）。

## 5. 未尽事项 / 后续观察

- `_skipped_nan` 目前仅计数，未接自动停机（连续 N 次 >0 应终止并告警）；
- 若 fp32 仍出现梯度范数 >10^3，下一步降峰值 LR 至 5e-5 并加 500 step warmup；
- diverged run1 的 best checkpoint（epoch31）保留于 checkpoints_diverged_run1/，
  可用于对比实验但不应作为正式产物。