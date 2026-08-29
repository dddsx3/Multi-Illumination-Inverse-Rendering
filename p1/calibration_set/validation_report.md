# Calibration-set Oracle Gate 报告

场景数: 5

Or1 (mesh normal + GT albedo + GT light) mean SI-PSNR: 22.25 dB
Or2 (depth normal + GT albedo + GT light) mean SI-PSNR: 21.50 dB
Or1 - Or2 差: 0.75
Mesh vs depth normal 夹角 mean: 61.86°

## Gate 判读
**WARN** — P 域 oracle 处于 15-25 dB 区间，符合 L=2 SH 截断预期。
**NOTE** — Mesh vs depth normal 夹角 > 10°：论文 GT 应优先 mesh normal。