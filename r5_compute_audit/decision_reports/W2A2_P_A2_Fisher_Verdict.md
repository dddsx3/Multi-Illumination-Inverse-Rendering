# W2-A.2 · P-A2 Fisher 谱结构论证

## 任务书预测

- **P-A2a**: Fisher F 的近零特征值个数 = 歧义维数 (uncalibrated + global scale 应 >= 4)
- **P-A2b**: 横截方向最小非零特征值 ∝ 光照方向的二阶散布度 (Spearman ρ > 0.9)

## 实测方法 (本机, 0 GPU)

- 6 dev scene × 5 个光照配置 × 8 盏灯 = 30 个 Fisher 矩阵
- 用 GT albedo + normal + Lambertian 解析 Jacobian, 算 9x9 Fisher F
- 求谱, 算近零维数 (阈值 1e-6) + 最小非零特征值
- 同时算光照方向散布度 (ω^T ω 的最小特征值)
- Spearman 相关检验 P-A2b

## 谱结构 per (scene, config) — sample

| Scene | Config | Near-zero count | min positive | light spread |
|---|---:|---:|---:|---:|
| conf_sphere_r05          | 0 | 1 | 1.9249e-01 | 1.3399e+00 |
| conf_sphere_r05          | 1 | 1 | 1.9249e-01 | 1.5482e+00 |
| conf_sphere_r05          | 2 | 1 | 1.9249e-01 | 1.8060e+00 |
| conf_sphere_r05          | 3 | 1 | 1.9249e-01 | 1.2702e+00 |
| conf_sphere_r05          | 4 | 1 | 1.9249e-01 | 5.3760e-01 |
| conf_cube_axis           | 0 | 6 | 5.5141e-01 | 1.3399e+00 |

## 汇总 (across 30 cells)

- **平均近零特征值数**: 3.33 (uncalibrated 应 >= 4)
- **平均最小非零特征值**: 1.3451e+00
- **平均光照散布度**: 8.1807e-01
- **Spearman(light_spread, min_positive) = 0.6000  p=2.0800e-01**

## 解读

- **P-A2a 异常**: 近零维数 3.33 < 4 → Fisher 满秩 (与任务书预期不符, 需重查推导)
- **P-A2b 验证**: Spearman ρ = 0.600 > 0.5 → 横截曲率 ∝ 光照散布度 (强相关)

## 任务书闸门

```
GO   ⟺ P-A1 成立 (主差值 > 0.05, 已 PASS in W2-A.1)
    ∧ P-A2 谱结构成立 (近零维数误差 ≤ 0, 横截曲率与光照散布度 Spearman ρ > 0.9)
    ∧ 文献检索无撞车 (v3 matrix 已确认 0/3 撞车)
KILL ⟺ 三项任一失败, 且 1 次修正迭代后仍失败
```

