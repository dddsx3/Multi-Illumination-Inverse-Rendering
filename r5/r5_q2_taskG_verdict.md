# Q2 · Task G Verdict (本机实测, 2026-09-02)

> **Case 2 触发**: β_global = -0.56, β_oracle_local = +0.03
> **论文 claim 调整**: 不再 claim "intrinsic identifiability → error";
>                   改为 "practical optimization recoverability predictor"
>                   (CLAIM_REGISTRY v0.4 §R5-P1-C Case 2 wording 路径, 已冻结)

## 数据
- 来源: r4pp/07_local_vs_global_init.csv (240 unique runs, 6 scene × 2 N × 10 subset × 2 init)
- 拟合: logE_recon = intercept + β × I_gsiq

| init_mode | n | β | pearson r | 解读 |
|---|---:|---:|---:|---|
| global | 120 | **-0.5580** | -0.56 | ✓ R4§ 主结论成立: 信息多 → 误差小 |
| **oracle_local** | 120 | **+0.0285** | +0.05 | ✗ 信息多少与 local 误差无关 |

## 与 R5-B′ §17 对照
- §17 Case 1 (β_g<0 AND β_o<0): 升级 intrinsic identifiability
- §17 Case 2 (β_g<0 AND β_o≥0): **降级到 optimization recoverability** ✓ 命中
- §17 Case 3 (β_g≥0): 重新评估

## 论文 wording 调整
旧 (v0.4 升级目标):
> *At fixed illumination cardinality, Gauge-Schur information quality
>  predicts reconstruction error.* (C3 升级前 wording)

新 (Case 2 降级 wording, v0.5 待写):
> *At fixed illumination cardinality, Gauge-Schur information quality
>  predicts the difficulty of standard (global-initialised) reconstruction
>  and, consequently, the relative quality of selected subsets under
>  such reconstruction pipelines.*

## 进一步动作
- CLAIM_REGISTRY v0.4 → v0.5 (Case 2 wording) — 待做
- 阶段 A 仍要跑: 拿到完整 PASS-A 量化结果
- 阶段 C 仍要跑: 验证 selection 收益在 Case 2 wording 下也成立
- 阶段 D: C3 selection preservation 改为"在 global-initialised solver 下的 selection 收益"
