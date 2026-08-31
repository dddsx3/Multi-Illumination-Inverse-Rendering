# 05 · Old-Filter Bias Audit（旧收敛判据的选择偏差）

> **来源**：`r4pp/01_master_trial_table.parquet`（988 trial，旧 R4′ 全量）
> **目的**：量化旧 P75×P75 收敛判据的选择偏差，为 Task B 的绝对判据提供基线对照。
> **关联**：`r4pp/CONV_CRITERIA_FROZEN.json`（新判据，已冻结）

---

## 1. 旧判据的三条腿（全部失效）

| 分量 | 定义（旧） | 实测问题 |
|---|---|---|
| `old_success_asrecorded` | `joint_solve` 内部：tail-50 range < 1e-7 且 grad_norm < 1e-3 | **恒为 0**（Discovery 上标定的阈值在确认集上 0% 命中） |
| `old_p75_success_flag` | 分析时组内 P75×P75（loss & grad_norm 双筛） | **0.5840 ≈ 0.5625**（0.75×0.75 构造上限）——不是测量 |
| `old_converged_flag` | `converged`（tail-50 range < conv_tol=1e-7） | **恒为 0** |

## 2. 内生筛选的方向（P0-3 修正版，H1.7 从 master table 挖出）

组内（每 scene,N，Spearman）：

| 量 | median ρ vs reconstruction_error | 负号占比 |
|---|---|---|
| `final_objective` | **+0.131** | 0.42（方向"对"但弱） |
| `grad_norm` | **−0.471** | **0.91（30/33 cell）** |

- `grad_norm` 分量方向完全反了：梯度范数越小 → albedo 误差**越高**。
- 净效果（组内比较，排除跨 cell 汇聚假象）：
  **E[err | 筛入] / E[err | 筛出] = 1.068（median），76% 的 cell > 1**
- **结论：旧判据系统性地筛掉了更准的重建，留下更差的。**

### 推测机制
小梯度范数 = 优化落入平坦/退化方向（含尺度 gauge 平坦方向）；条件数好的问题在
固定迭代预算下仍在下降、残余梯度反而更大。

## 3. 新判据（Task B，已冻结 `CONV_CRITERIA_FROZEN.json`）

```
converged = finite
          AND abs_step200 < 1.0e-06        （末 200 步绝对平均步长）
          AND final_objective < 3.0e-04    （绝对量级上界）
```

**关键设计决定**：
- **`proj_grad_norm` 不参与收敛判定**——pilot 实测
  ρ(proj_grad_norm, error) = **−0.748**（p=2e-18），与旧 grad_norm 同向的
  反向选择依旧存在（gauge 投影消除了部分，但未消除全部）。只记录，不作筛选。
- **`tail_rel_change` 不参与**——ρ = +0.568，弱且分母不稳定。
- 绝对步长（不归一化）与 error 无反向选择（ρ = +0.11），故选它。
- optimization failure **不删除**，作为端点进 hurdle 模型（任务书 §36）。

## 4. 数值标定（pilot 96 runs）

| 量 | median | p90 | max | 冻结阈值 |
|---|---|---|---|---|
| `abs_step200` | 2.81e-07 | 3.64e-07 | 3.93e-07 | **1.0e-06**（≈2.7×p90） |
| `final_objective`（N=2~8 med） | 8.5e-05~1.3e-04 | — | — | **3.0e-04**（≈2×max med） |

## 5. 后续验证（Task C 全量后必须做）

1. 新判据 success rate 应落在 10%~90%（避免 0% 或 100% 的退化）；
2. ρ(新 success_flag, error) 不得为负；
3. 若仍为负，改用纯 finite 判据并把 failure 全部作为端点。

---

*05 · Old-Filter Bias Audit · 2026-08-31 · 依据 R4″ 任务书 §6 Task B*
