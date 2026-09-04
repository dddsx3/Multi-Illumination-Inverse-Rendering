# EX-03 · A3-1（noFiLM）FiLM 消融对照报告

> 生成时间：2026-09-04_1057 · 自动管线（_watch_a31_posteval.py）
> 口径：Gen-A3 世代 scene 级（INC-0015 校准），A3-0 与 A3-1 同口径可比。

## 判据
- normal MAE 差 ≤ 2.0° 且 albedo si-MAE 差 ≤ 0.03 → FiLM 非关键（消融成立）
- 超阈 → 需 INC + 上报

## 结果

| 指标 | A3-0（冻结） | A3-1（noFiLM） | 差值 | 判据 |
|---|---|---|---|---|
| normal MAE (°) | 14.8866 | 13.5730 | +1.3136 | ≤ 2.0 |
| albedo si-MAE | 0.05432 | 0.05401 | +0.00031 | ≤ 0.03 |
| PSNR (dB) | 32.5424 | 26.5615 | -5.9809 | — |
| 物理违规率 (%) | 0.0000 | 0.0000 | — | 须为 0 |

## 结论
**FiLM 非关键（消融成立）：A3-1 noFiLM 与 A3-0 差异在判据内**

## 三指纹（RUN_CARD）
- A3-1 RUN_CARD.json：未生成（需补）
- config_sha256：N/A

## 来源
- A3-0：eval_output/A3-0_f_n5gray_seed42_test_v2_scenelevel/eval_summary.json
- A3-1：eval_output/A3-1_noFiLM_test/eval_summary.json
