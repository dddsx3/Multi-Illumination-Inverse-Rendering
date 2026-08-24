# Phase 2 · T2.6 冻结 test 集首次正式基线（含锚点事故记录）

- **Checkpoint**：phase1_best_recovered.pth（= Phase 1 epoch95，val_loss
  0.018275491835083812 与训练日志逐位一致）
- **评估集**：冻结 test，**127 场景**（splits/synthetic_v2.json，从未参与训练/调参）
- **复现命令**：`python -u evaluate_model.py --checkpoint ../checkpoints/phase1_best_recovered.pth --data_root D:/data/synthetic_v2 --split test --split_manifest splits/synthetic_v2.json --batch_size 1 --out_dir eval_output/p2_t26a_test_phase1recovered`

## 正式 test 基线（13 项，mean ± std，127 场景）

| 指标 | 数值 |
|---|---|
| image_psnr | 35.4063 ± 5.7856 dB |
| image_ssim | 0.9662 ± 0.0643 |
| normal_mae_deg | **10.2062 ± 7.3912°** |
| normal_median_deg | 3.4022 ± 7.7369° |
| normal_acc@11.25 / 22.5 / 30 | 85.71% / 90.33% / 91.57% |
| albedo_si_mae | 0.0806 ± 0.1057 |
| albedo_mae（未对齐） | 0.3081 ± 0.1576 |
| depth_rmse_aligned | 0.2892 ± 0.4251 |
| depth_mae_aligned | 0.2363 ± 0.3992 |
| depth_rmse / mae / si_rmse（未对齐） | 0.7529 / 0.6626 / 0.7421 |

## 锚点一致性事故与裁定（INC-0004）

任务书预期的锚点（T1.6 报告数字应与本基线逐位一致）**未通过**
（max_deviation=6.52 @normal_median_deg）。排查结论：T1.6 基线评估实际运行在
**v1 数据集的 121 场景验证子集**上（与冻结 test 仅 17 场景交集）——详见
`docs/incidents/INC-0004_T16基线误用未冻结子集.md`。

**裁定**：
1. 本文档数值为**官方 test 基线**；T1.6 报告数字降级为"偶然跨版本泛化测量"，
   其 PSNR 36.58/法线 9.55 作为 v2->v1 跨分布迁移参考点保留；
2. 对外口径更正：法线 MAE 基线 **10.21°**（原传达 9.55° 作废）；
3. evaluate_model.py 已增加 data_root/split_manifest 回显（防再犯）；
4. 可复现性证明：manifest test 与重新计算的 legacy-val 完全一致
   （127=127，集合相等），划分机制本身无缺陷。