# Calibration-set Oracle Gate 报告

场景数: 2

Or1 (mesh normal + GT albedo + GT light) mean SI-PSNR: 26.77 dB
Or2 (depth normal + GT albedo + GT light) mean SI-PSNR: 14.25 dB
Or1 - Or2 差: 12.52
Mesh vs depth normal 夹角 mean: 69.54°

## Gate 判读
**PASS** — P 域 oracle > 25 dB，渲染公式成立。
**NOTE** — Mesh vs depth normal 夹角 > 10°：论文 GT 应优先 mesh normal。