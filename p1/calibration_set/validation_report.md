# Calibration-set Oracle Gate 报告

场景数: 4

Or1 (mesh normal + GT albedo + GT light) mean SI-PSNR: 28.25 dB
Or2 (depth normal + GT albedo + GT light) mean SI-PSNR: 14.27 dB
Or1 - Or2 差: 13.98
Mesh vs depth normal 夹角 mean: 65.03°

## Gate 判读
**PASS** — P 域 oracle > 25 dB，渲染公式成立。
**NOTE** — Mesh vs depth normal 夹角 > 10°：论文 GT 应优先 mesh normal。