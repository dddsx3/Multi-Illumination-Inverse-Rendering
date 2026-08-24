# Phase 1 · G4 训练冒烟验收记录（裁决书 C2 补档）

> 本文档补齐裁决书 §1 中 G4「有条件通过」的放行条件 C2：
> 补独立验收记录。原始数字此前仅存在于 INC-0001 引文中。

## 冒烟命令（真实数据 v2 前身 v1 数据集，601 场景）

```
python main.py --mode train --data_root D:/data/synthetic \
  --total_epochs 3 --stage1_epochs 1 --stage2_epochs 1 \
  --batch_size 8 --image_size 256 256 --num_lights 5 --device cuda
```
日志存档：`_smoke_real_log.txt`（已入库）；后续同参数复跑见
`_smoke_bf16_log.txt`（BF16 版本，同样通过）。

## 三 epoch 曲线与阶段切换确认

| Epoch | 阶段 | 验证损失 | 关键事件 |
|---|---|---|---|
| 0 | 阶段1 Geometry | 0.3116 | 残差冻结 ✓ |
| 1 | 阶段2 Material | 0.0835 | 权重表切换 ✓ |
| 2 | 阶段3 Residual | 0.0521 | **残差解冻 ✓**、GT 监督权重归零切换 ✓ |

- 全程无 NaN / 无崩溃 / exit 0；
- GT 各项（depth/albedo/normal）损失入 TensorBoard 且量级合理；
- 数据加载无 GT-图像错位（val 指标随 epoch 正常下降而非发散）。

## 判定
**G4 通过。** 同一冒烟在 BF16 与 fp32 下均复测通过
（`_smoke_bf16_log.txt`），三阶段切换逻辑稳定。