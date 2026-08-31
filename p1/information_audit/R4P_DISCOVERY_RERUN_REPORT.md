# R4′-D · Discovery Set 复跑报告（v2 稳定性检查；非确认性证据）

> seed=20260831 · N∈[3, 5, 8, 12] × 20 subsets × 4 scenes · primary = full_lam_min_pos_norm（scene/trace 归一 λ_min⁺） · pixel_cap 冻结 = 1000（commit 配额约束；cap1000 vs cap500 稳定性见下）

## 病态检查（主扫描 cap2000/cutoff1e-8）

- gauge residual 最大值：1.663e-05（应 ≲1e-9）
- PSD 余量 min(λmin/trace)：-2.741e-10（应 ≥ −1e-10）
- rank(F_k) 逐场景最小值：{'cube': 2, 'cylinder': 5, 'hemisphere': 9, 'sphere': 9}（低秩=法线多样性限制，pinv 正确处理）
- active_frac 最小：0.181 · ReLU 边界占比最大：0.000e+00
- primary ≤0 的 (scene,subset)：0/320

## 稳定性（primary 的 Spearman 秩相关）

- pixel_cap 1000 vs 500（同子集）：ρ=0.7933（n=319, p=2.72e-70）
- cutoff 1e-8 vs 1e-6（同像素）：ρ=0.7097（n=319, p=3.85e-50）

## 存在性前提（固定 N 内 primary 有宽度）

| N | min | median | max | IQR/median |
|---|---|---|---|---|
| 3 | 1.032e-08 | 2.300e-07 | 8.222e-04 | 3.27 |
| 5 | 1.097e-08 | 5.477e-07 | 5.479e-04 | 1.59 |
| 8 | 1.065e-08 | 9.826e-07 | 6.178e-04 | 0.91 |
| 12 | 1.286e-08 | 1.142e-06 | 6.576e-04 | 1.19 |

## v1(diag-proxy) 视角失真量化

- proxy_lam_min_norm / full_lam_min_pos_norm：median=47.4×，range=[0.4×, 71098.1×]（v1 的逐像素视角系统性高估最坏像素信息）

## 结论

- 本复跑**只用于**：指标列冻结、数值病态排查、预注册参数选择；
- 不得作为 confirmatory 证据（T4′.0 防双重使用）；
- 确认性统计只在 R4′-C 新确认集上做。
