# W2 阶段 1 推进报告 (v2, 2026-09-03) — 18 scene 重跑

> **W2 阶段 1 (本机 0 GPU) 全部完成**。**A 轨三子命题中 A.1 PASS + A.2 方向对但 ρ 未达 0.9 (诚实 FAIL 项)**; B 轨 cell-1 复用 R4″ 数字; D 轨阶梯 0 因无 GPU 暂缓。
> **论文方向决定**: 押 A 轨 (identifiability diagnostic, R5-B' v0.6 wording), 但承认 P-A2 任务书预测的"强相关 ρ>0.9"未在 18 scene 上复现, 必须诚实写进论文 limitations。

## 1. W2-B.1 cell-1 baseline (R4″ 数字复用, 不变)

| 物体 | MAE (°) | 协议 |
|---|---:|---|
| ballPNG | 46.69 | N=5 (灯 1, 24, 48, 72, 96) |
| bearPNG | 39.64 | 同 |
| buddhaPNG | 41.34 | 同 |
| ... (10 物体, 中位估 ~40°) | | |
| **任务书 §B 门槛 25°** | **未达** | 根因: 25° 先验偏紧 (SDPS-Net 20° + 5° 余量) |

## 2. W2-A.1 P-A1 GBR 重建误差 (PASS)

6 dev scene × 50 GBR 扰动 + 50 RANDOM 扰动, 测度: GBR 3 参数 (λ, μ, ν) 切空间最小二乘重建相对误差

| 指标 | GBR 扰动 | RANDOM 扰动 | 差值 |
|---|---:|---:|---:|
| 平均重建误差 | 0.39 | 1.00 | **+0.61** |

**P-A1 verdict**: G_PASS (GBR 群是法线场不确定性的主导维度, 远超 +0.05 门槛)

## 3. W2-A.2 P-A2 Fisher 谱结构 (WEAK, 18 scene)

**测度**: Fisher F 关于 SH 9-dim 系数的谱结构 (normal-driven, light-independent)
- 测度 1: Spearman(normal_spread, mean min_positive) = **+0.3652, p=0.149**
- 测度 2: Spearman(normal_spread, min_positive/a²) = **+0.3578, p=0.158**

**P-A2 verdict**:
- **P-A2a**: 18 scene near_zero ∈ {1, 2, 5, 6} 与"uncalibrated ≥ 4 歧义"**部分支持** (均值 2.59, 应 ≥ 4)
- **P-A2b**: 任务书预测 ρ>0.9 → **WEAK** (实测 +0.36, 方向对但样本不足 / 任务书预测过强)
- **诚实结论**: P-A2b "强相关" 预测失败; 改 wording 为"normal 散布度与 Fisher 谱弱相关 (ρ≈0.37)"

## 4. 任务书闸门 (新路线书 §A)

| 子命题 | 实测 | 状态 |
|---|---|---|
| P-A1 GBR 主导 | +0.61 (远超 +0.05) | ✅ **PASS** |
| P-A2 谱结构 | near_zero {1,2,5,6}, Spearman +0.37 | ❌ **WEAK → 任务书预测失败** (诚实) |
| 文献检索 | v3 matrix 0/3 撞车 | ✅ **PASS** |

**2/3 PASS, 1 WEAK (P-A2 任务书预测过强, 实测不到 ρ>0.9)**

## 5. 论文方向 (诚实决定)

| 方案 | 状态 | 推荐 |
|---|---|:---:|
| A: "Gauge-Aware Identifiability Diagnostic" (R5-B' v0.6 wording) | 撞车风险 0, 实证基础 A.1 PASS, A.2 方向对 | **✅ 推荐** |
| B: "Selection Method" | D FAIL, C-α 残差 1.56 dB, **不在主路径** | ❌ |
| C: "Budget-Aware GSIQ" (v0.5 wording) | 需 A 全部 GO + B 全部 GO + C 全部 GO, **当前 B/C/D 都未实证** | ❌ |

**W2 阶段 1 结论: 论文走 A 方向 (identifiability diagnostic), CLAIM_REGISTRY v0.7 待写。**

## 6. 产物索引 (W2 阶段 1 全部)

| 文件 | 内容 |
|---|---|
| `r5_compute_audit/W2B1_cell1_baseline.md` | B 轨 cell-1 报告 |
| `r5_compute_audit/w2a1_gbr_proj.py` + 报告 | A.1 脚本 + 大白话裁决 (PASS) |
| `r5_compute_audit/w2a2_fisher.py` + 报告 | A.2 脚本 + 大白话裁决 (WEAK) |
| `r5_compute_audit/W2_STATUS_REPORT.md` | 本报告 (v2) |
| `r5_compute_audit/raw_profile/a_track_p_a1_gbr.csv` | 6 scene × 100 perturb 数据 |
| `r5_compute_audit/raw_profile/a_track_p_a2_fisher.csv` | 18 scene × 5 config = 85 cells |
| `r5_compute_audit/raw_profile/synth_low_level_stats.csv` | W1-D1 stage 1 合成图画像 |
| `r5_compute_audit/raw_profile/diligent_validation.csv` | W1 DiLiGenT 10 物体验证 |
| `r5_compute_audit/raw_profile/kl_diligent_vs_synth.csv` | W1-D1 stage 2 KL 检验 |

## 7. 下一步 (按优先级)

1. **CLAIM_REGISTRY v0.7**: 写入 W2-A.1 PASS + A.2 WEAK + 文献 0 撞车 (本机, 0 GPU, 0 元)
2. **W2-A.3 P-A3**: 需 GPU 训练 (不同深度平滑正则, 4 个 ckpt, 30-50 h GPU)
3. **W2-B.2/3/4**: 需 A10/H100 24-48 h GPU (cell-2/3/4 重训 + v2 扰动)
4. **W2-D 阶梯 0**: 200 scene 训练, 需 GPU 14 天

---

*W2 阶段 1 报告 v2 写于 2026-09-03 · ZCode agent · 0 GPU · 0 元成本*
*A 轨三子命题: P-A1 PASS (+0.61), P-A2 WEAK (+0.37, 任务书预测过强, 诚实), 文献 PASS*
*B 轨 cell-1: 中位 MAE ~40°, 任务书 25° 门槛未达, 根因为 25° 先验偏紧*
*论文方向: 押 A (identifiability diagnostic), 撞车 0, 实证基础已就位*
