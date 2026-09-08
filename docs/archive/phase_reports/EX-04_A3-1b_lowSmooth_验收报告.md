# EX-04 · A3-1b lowSmooth 验收报告（2026-09-04）

> 条例：任务书 T v2.2 §3 EX-04 · INC-0013(c) 判别实验 · 判据 R-J sweep 后（scene 级）。
> 协议：Gen-A3（bs4+物理约束 clamp 头+gray+synthetic_v3+bf16），一臂一变量：`--albedo_smooth_stage1 1.0`（A3-0 为 10.0），其余与 A3-0 完全一致。
> 训练：100/100 epoch，13:31→21:07（墙钟 7.6h，273.7 s/epoch 实测），**全程零 rc42/零中断**（FIX-08-5 加固后首个生产 run，INC-0016 加载窗口未复现）。
> RUN_CARD：FIX-08-4 自动三指纹首例——code_commit_sha=ba6ab76（启动时刻 HEAD，与 git log 一致）/ config_sha256=1ec48ea…/ data_manifest_sha256=853ad79…，train_start=13:31:07 / train_end=21:07:38 / epochs_done=100 自动落盘。

## 1. 判据验收（scene 级，全部 PASS）

| 判据 | 数值 | 阈值 | 结果 |
|---|---|---|---|
| albedo si-MAE | 0.05580（A3-0 0.05432，差 +0.00148） | ≤0.065（判据包 4） | **PASS** |
| normal MAE 差 vs A3-0 | −1.6852°（13.2014 vs 14.8866°） | \|差\|≤2.0° | **PASS** |
| 物理违规率 | 0.0000% | 须为 0 | **PASS** |

## 2. 观测指标：albedo 值域压缩恢复（INC-0013(c) 核心问题）

| 指标（per-scene mean） | A3-0（smooth=10） | A3-1b（smooth=1） | 解读 |
|---|---|---|---|
| phys_albedo_range | 0.1681 | **0.3629** | **>0.30 → 记"压缩恢复改善"**（超弱改善带 0.27–0.30，接近翻倍） |
| phys_albedo_std | 0.0809 | 0.0713 | 略降（mean 上移背景下的结构变化） |
| phys_albedo_mean | 0.5413 | 0.7280 | 输出整体上移（Sigmoid 中区利用更充分） |

**INC-0013(c) 闭环结论**：albedo 值域压缩由 stage1 平滑权重过高主导——权重 10→1 后动态范围恢复（range 0.168→0.363），且估计头指标（normal/albedo si-MAE）均在判据内。**"平滑 10→1 未恢复压缩"分支未发生**；机制为平滑正则把输出压向常数，判别完成。

## 3. 必须披露（与判据并行记录，不合并表述）

- **PSNR 22.8726 dB（−9.67 dB vs A3-0）**：重建保真度显著退化，幅度大于 A3-1 noFiLM 的 −5.98 dB。方向一致：估计头（法线/反照率）不劣化，前向 RGB 重建头依赖被移除的正则强度。
- 排序语境（A3-0 → A3-1 → A3-1b 的 PSNR 序列 32.54 → 26.56 → 22.87）说明 FiLM 与平滑正则对重建头的贡献叠加独立；论文叙事按"估计头 vs 重建头"双头口径分别表述（口径见论文数字口径说明 v0.1 §3）。
- 本臂产物**不进对比矩阵主表**（一臂一变量判别用途），主表仍为 A3-0。

## 4. 后续输入（v2.2 §3/§4 联动）

- **FW（A3-3）设计**：EX-04 已恢复压缩 → "信息加权影响 albedo 动态范围"调查（T1 设计文档 §2.5 预留）转为常规观测项，不进 A1 设计考量紧急清单。
- **臂序**：EX-05（A3-2 seed123）→ EX-06（A3-4 MeanPool）→ EX-07（A3-3 FW，A1 冻结包 T1-1/2/3 已交付，T1-4/5/6 排 EX-06 后）。
- **INC-0016**：本 run 加载窗口未复现长停（启动三指纹 + pf 断言 + 预热加密全生效），接近闭合判据（待 EX-05 再积累一例后闭合）。

## 5. 产物清单

- `eval_output/A3-1b_lowSmooth_test/`：eval_summary.json + per_scene_metrics.csv + RUN_CARD.json（自动三指纹）
- `checkpoints/A3-1b_lowSmooth/`（库外）：checkpoint_epoch_0000..0099 + best_model.pth + latest_model.pth
- S 卡：CLAIM_CARDS S-06·A3-1b（含 PSNR 披露与 range 恢复注记）
- 训练日志：`D:\MIR_Archive_20260829\_arm_A3-1b_run.log`

*验收人：执行 agent · 2026-09-04 21:1x · 判据以任务书 v2.2 EX-04 原文为准。*
