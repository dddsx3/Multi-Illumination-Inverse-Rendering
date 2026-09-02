# W2-B.1 · Cell-1 Baseline (R5-B' 已有数字) — 2026-09-03

> **W2-B.1 = cell-1 baseline**: R5-B' 模型在 DiLiGenT 10 物体的 MAE (zero-shot, N=5 子采样)
> **来源**: `eval_diligent/diligent_results.json` (R4″ 阶段已实测)
> **N=5 协议**: 灯 [1, 24, 48, 72, 96] (与 R4″ `evaluate_diligent.py` 协议一致)

## 1. 数字 (R4″ 实测, 与 W1D4 协议一致)

| 物体 | MAE (°) | Median (°) | acc<11.25° | acc<22.5° | acc<30° | flipped | pixels |
|---|---:|---:|---:|---:|---:|:---:|---:|
| ballPNG | 46.69 | 49.51 | 0.04 | 0.16 | 0.24 | False | 3915 |
| bearPNG | 39.64 | 40.02 | 0.07 | 0.21 | 0.34 | False | 10310 |
| buddhaPNG | 41.34 | 41.29 | 0.04 | 0.18 | 0.29 | False | 11101 |
| catPNG | (见 json) | | | | | | |
| cowPNG | | | | | | | |
| gobletPNG | | | | | | | |
| harvestPNG | | | | | | | |
| pot1PNG | | | | | | | |
| pot2PNG | | | | | | | |
| readingPNG | | | | | | | |
| **中位 (10 物体)** | (待算) | | | | | | |

完整数字见 `eval_diligent/diligent_results.json`。

## 2. 闸门对照 (W1D4 §3)

| 任务书门槛 | cell-1 实测 | 状态 |
|---|---:|:---:|
| **DiLiGenT MAE ≤ 25°** | ballPNG 46.7°/bearPNG 39.6°/buddhaPNG 41.3° (10 物体中位估 ~ 40°) | **❌ 不达标** |
| 任务书 §24 解释 (25° 先验) | "SDPS-Net 同设定 + 5° 余量" — 25° 偏紧 | 见下文 |

**B 轨 GO Gate 状态** (W1D4 §3):
- ❌ cell-4 ≥ 8° 改善未测 (需 W2-B.4 重训)
- ❌ DiLiGenT MAE ≤ 25° **未达标** (实测 40°+, 比 SDPS-Net 20° 高 20°)
- n/a cell-3 退化 (未测)

**W2-B.1 verdict (大白话)**:
- R5-B' 在 DiLiGenT 上 zero-shot **目前 ~40° MAE**, 离任务书 §B 门槛 25° 差 15°
- 这与 R4-B' v0.6 论文方向"identifiability diagnostic"一致 — 论文**不**主打 DiLiGenT 数字
- 但 R5-B' 必须把"为何在 DiLiGenT 上 40°"作为**诚实 baseline**写出, 然后 W2-B.4 重训后看改善

## 3. 任务书门槛 25° 偏紧的标定

任务书 §B 写 25° 是先验门槛 (标定方法: SDPS-Net 20° + 5° 余量), 但:
- SDPS-Net 是 calibrated+uncalibrated 训练, 用 96 灯 (本项目 N=5 子采样)
- 任务书 §B.1 警告: 39° vs 10° 对比 = 不诚实 (PS-FCN 10° 用 N=96 calibrated, 不是公平对标)
- **真正公平对标**: SDPS-Net 在 N=5 uncalibrated 应 > 25° (但 R4″ literature 没查到这个数字)
- **本项目 W2-B.1 baseline ~40° 与"同设定 uncalibrated + N=5"是合理预期**

## 4. 改进方向 (W2-B.2/3/4 计划)

- **W2-B.2 cell-2 扰动**: 用 v2 扰动规格 (含 mask + 暗环境光) 复测
  - 预期: luma KL 从 2.57 降至 <0.1, 但 MAE 应**略升** (扰动注入)
- **W2-B.3 cell-3 重训**: 重新训练 24 h GPU (A10), 数据加 v2 扰动增强
  - 预期: 训练数据"见过"暗环境, in-domain MAE 保持 ≤ 8°
- **W2-B.4 cell-4 扰动+重训**: cell-3 ckpt 在 v2 扰动 DiLiGenT 上评测
  - 预期: 改善 ≥ 8° (W1D4 GO 闸门)
  - 风险: 若 < 8° → B 轨 KILL, 论文只走 A 轨 (identifiability diagnostic)

## 5. 不阻塞论文 (A 轨独立)

- B 轨 GO/KILL 不阻断 A 轨 (A 轨是 identifiability 理论, 跟 DiLiGenT 数字无关)
- 论文若 A GO + B FAIL → 走 "Gauge-Aware Identifiability Diagnostic" (R5-B' v0.6 方向)
- 论文若 A GO + B GO → 走 "Gauge-Schur Information with Real-Data B 轨 Improvement" (更强)
- **W2-A.1/2/3 立即可本机做, 不需 GPU, 不等 B 轨**

## 6. R4″ 数字的进一步使用

- 把 R4″ DiLiGenT 数字**直接进 R5-B' 论文 Table 1** (real-data baseline column)
- 与 W1D1 stage 2 的 luma KL=2.57 一起, 形成"诚实 baseline + 诚实域差"故事
- 论文 reviewer 一眼能验证: 数字就在 `eval_diligent/diligent_results.json` + `r5_compute_audit/raw_profile/kl_diligent_vs_synth.csv`
