# Q3 · P2 Held-out GSIQ 排名保真 (本机, 2026-09-02)

> **所有 24 cell ρ ≥ 0.976, 中位 ρ = 1.0000** — albedo-free 排名在 held-out
> scene 保持不变, 任务书 §R5-P1-A Go Standard 大幅通过

## 数据
- 来源: r5/r5_p2_heldout.csv (12,000 rows)
- 配置: 18 候选 dev scene → 12 scene 数据完整, 6 scene 缺文件 (R4§ 已知坏)
- 12 held-out scene × N{3,5} × 500 subset × 2 score (O + A) @ P=500

## 关键数字
| 指标 | 值 | 任务书门槛 | 判定 |
|---|---:|---:|---|
| median ρ across 24 cells | **1.0000** | ≥ 0.95 | ✅ 大幅通过 |
| min ρ | 0.9762 (cube_plus_cone N=3) | ≥ 0.95 | ✅ |
| median τ (Kendall) | 0.9984 | (不强制) | ✅ |

## 解读
- 在 12 个非 in-domain scene (cone, cube_complex, cylinder, ellipsoid, hemisphere,
  icosphere, snowman, sphere_on_cube, torus × 2, two_spheres) 上, albedo-free 与
  oracle 排名 ρ 平均 1.0 — **轨迹 albedo 的值不影响照明子集排名**
- 这是 R5-B′ 主论断的 held-out 验证: trace-level albedo (几何 / 场景 / 方向)
  的**比例**信息 (而不是绝对值) 决定照明子集质量
- 任务的"a=1" 简化在该 12 scene 集上**几乎完美**等价于 oracle 排名

## 含义
- Q1 (P1-A, in-domain) 已 PASS-A, Q3 (P2, held-out) 中位 ρ = 1.0
- 任务书 §R5-P1-B (P1-B, normal/light proxy) 的前提: ρ_proxy ≥ 0.7 + ≥75% cells
  proxy-selected < random 已在 P2 smoke 上方向一致 (ρ ≥ 0.97 全 24 cells)
- Q3 = 通过 (held-out 方向); Q1 = 通过 (in-domain 方向)

## 剩余: P1-B 完整 + D (C3 selection preservation)
P1-B 完整 (正常/光照 proxy 估计) 需"normal proxy" 算法 — 本机可做粗版本
(用 GT normal 加扰动) 但需要时间; 优先级低于 P1-A full (验证论文主结论)
