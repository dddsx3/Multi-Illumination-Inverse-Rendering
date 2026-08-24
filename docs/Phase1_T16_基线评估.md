# Phase 1 · T1.6 测试集量化基线（首次）

- **Checkpoint**：best_model.pth = epoch 93（BF16 全量训练 run2，val_loss 0.01427）
- **评估集**：121 个验证场景（确定性划分，未参与训练）
- **协议**：中心裁剪无增强；法线由预测深度经渲染器导出；逐场景 compute_all
- **复现命令**：`python -u evaluate_model.py --checkpoint ../checkpoints/best_model.pth --data_root D:/data/synthetic --split val --batch_size 1`

## 13 项指标（mean ± std，121 场景）

| 指标 | 数值 | G6 参考 | 判定 |
|---|---|---|---|
| image_psnr | **36.58 ± 3.90 dB** | >20dB | 通过 |
| image_ssim | 0.975 ± 0.022 | — | — |
| normal_mae_deg | **9.55 ± 4.65°** | <30° | 通过 |
| normal_median_deg | 2.54 ± 1.21° | — | — |
| normal_acc@11.25° | 88.5% | — | — |
| normal_acc@22.5° | 91.5% | — | — |
| normal_acc@30° | 92.3% | — | — |
| albedo_si_mae | **0.067 ± 0.077** | <0.1 | 通过 |
| albedo_mae（未对齐） | 0.266 ± 0.153 | — | 量规自由度所致 |
| depth_rmse_aligned | **0.260 ± 0.114** | 见注 | 见注 |
| depth_mae_aligned | 0.206 ± 0.085 | — | — |
| depth_rmse（未对齐） | 0.753 ± 0.384 | — | 量规自由度所致 |
| depth_si_rmse（未对齐） | 0.861 ± 0.955 | — | 同上 |

## 关键说明

1. **深度量规自由度**：重建损失不约束绝对深度尺度/偏移（albedo×shading 与
   深度梯度各自可缩放）。未对齐的原始深度指标包含系统性偏置；按单目深度惯例
   增加逐场景最小二乘对齐（scale+offset）后的 aligned 指标作为主报告项。
   对齐后 RMSE 0.26（GT 动态范围约 1.4，相对误差 ≈18%），首版基线合理。
2. **法线表现突出**：MAE 9.55°/中位 2.54°，acc@11.25° 达 88.5%——多光照冗余
   + 物理渲染器监督的直接收益；也是"阶段3 残差解冻带来最大跃升"之外的第二个
   方法论验证点。
3. **反照率**：si-MAE 0.067 通过门禁；未对齐 MAE 0.266 反映与光照的乘积歧义，
   属任务固有性质（A2 每光照反照率升级的动机之一）。

## 训练过程摘要（run2, BF16）

- 100 epoch 零 NaN（INC-0001 守卫生效）；三阶段正常切换，残差解冻带来
  val 0.052 -> 0.024 的最大单步改善（残差网络价值的量化证据）
- 最终 val loss 0.01427；训练曲线单调下降无过拟合迹象

产物：per_scene_metrics.csv / eval_summary.json @ repo/eval_output/