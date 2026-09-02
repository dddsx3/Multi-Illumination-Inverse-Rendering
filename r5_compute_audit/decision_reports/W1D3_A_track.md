# W1-D3 · A 轨命题推导演示

## A-P2 引理 1 (盒约束下 scale gauge 破缺)

- 数值演示: 在 ρ=1 像素上, scale gauge c·ρ 撞约束的最小 c = **1.0**
- 含义: 存在饱和像素 → scale gauge 在可行域内不连续 → 局部可辨识
- 可证伪 (P-A1): 真实场景的子集误差应在 GBR 方向 (含 (λ, μ, ν) 剪切的法线场) 展开

## A-P2 引理 2 (GBR 残余结构性)

- 数值演示: GBR (λ, μ, ν) = (1.2, 0.1, 0.05) 作用于球面法线场 → 平均角度差 5.79°
- 含义: GBR 把法线场'剪切'了, 盒约束 ρ∈[0,1] 不限制法线场 → 残余结构性
- 消除 GBR 需额外先验: 深度平滑 / 光照分布先验
- 解释 N=1 → N=5 的 N-curve 平坦: 残余 GBR 恒定, 全部由学习先验兜底

## A-P3 SH Gram 秩论证

rank(G) = rank(Σ y(ω_i) y(ω_i)^T) ≤ n_light

| N (light) | rank(G) | per-scene identifiability |
|---|---|---|
| 3 | 3 | ❌ (3 < 9) |
| 5 | 5 | ❌ (5 < 9, **N=5 + SH-2 永远 per-scene 不可辨识**) |
| 9 | 9 | ✓ |
| 25 | 9 (饱和) | ✓ |
| 96 | 9 (饱和) | ✓ (PS-FCN 设定) |

**结论**: N=5 + SH-2 (9 dim) per-scene 不可辨识 → 必须 corpus-amortized
→ 与 A-P2 引理一致: 摊销是 per-scene 不可辨识的补偿机制
→ 这是论文"per-scene non-identifiable, corpus-amortized identifiable"的来源

## 对 W2 实施的影响

- **A-P1** (已知, 必须诚实引用, 不当新贡献)
- **A-P2 引理 1+2** (本机可纯推导证明, 1-2 页可严格化, 不需 GPU)
- **A-P3** (Gram 秩论证, 0 GPU, 1 页可写死)
- **W2 实验** (P-A1/P-A2/P-A3 实证): 需合成数据 + GBR 解析 + Fisher 谱
  - 这些实验**本机可做** (CPU 即可, 不需 GPU)
  - 预计耗时: 1-2 天开发 + 1 天跑完 18 scene 合成数据
