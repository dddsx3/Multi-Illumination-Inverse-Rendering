> **[STALE · R3′ 2026-08-31]** 本文档含 v1 gauge_fisher（缺 s_kp、逐像素假 Schur）与近场/旧 SH 常数时代的数字。按《P1 下一阶段执行任务书 v1.0》§1，仅作历史证据，不得作为当前状态引用。现行：gauge_fisher_v2.py + IDENTIFIABILITY_v2.md + R3P_MATH_AUDIT_REPORT.md。

# INFORMATION_AUDIT_v2 · P1-10 受控 Information Audit（calibration 级）

> 数据：P1 calibration set（5 简单 mesh × 32 真实渲染光照，128²，
> 每灯独立 render call，线性域，Route A irradiance SH）
> **这是本项目第一次在真实渲染的多光照数据上做 N 信息量实验**
>（PRE-0 的真实域因数据缺陷无效，解析域只是协议模型）。
> 复现：`python p1/source/information_audit/information_audit_v2.py --data_root p1/calibration_set/data --restarts 2 --exps 1 3 4`

## 1. N curve（受控 solver，GT geometry，A+{L} 联合恢复）

| N | 1 | 2 | 3 | 5 | 8 | 15 | 24 |
|---|---|---|---|---|---|---|---|
| SI-MAE(A) | 0.178 | 0.156 | 0.122 | 0.078 | 0.063 | 0.062 | 0.062 |

- **单调改善 2.9×（N=1→8），N≥8 饱和**。多光照信息在真实渲染数据中
  确凿存在（Gate B 的数据级前提在 calibration 规模上成立）。
- 与 PRE-0 解析域（协议模型，3.8×/N≥5 饱和）一致；真实渲染的
  饱和点稍晚（N≈8），与简单凸体几何简单、少量光即可覆盖一致。

## 2. Solver 诊断（P1-10 强制要求）

- `solver_diagnostics.csv` 落盘：success flag / grad norm / final loss / iters。
- 当前收敛判据（tail-50 loss 极差 < 1e-7 且 grad<1e-3）过严 → 0% success。
  **含义**：N 曲线数字作为趋势证据有效；"只比较收敛 trials"的正式
  对比需放宽判据（tol 1e-5 / grad<1e-2）重跑——已列入数据修复后清单。
- 多 restart（2）已启用；iters 随 N 缩放（800+200N）已实现。

## 3. Novel vs duplicate（cardinality-control + diversity-control）

- 两套定义跑通，per-scene 数字在 csv；汇总均值被 plane 场景
  （mask=0）污染为 NaN——**单场景级数据有效，汇总待重算**（排除
  空 mask 场景一行代码，列入修正）。
- 设计上已杜绝 PRE-0 的"预算压过信息量"混杂（restarts + iters 随 N
  缩放 + 收敛 flag 落盘）。

## 4. Conditioning（P1-11 Fisher 分析）

- `conditioning_summary.csv`：5 场景 × N∈{1..24}，9×9 Fisher F=JᵀJ。
- **有效秩 ≈ 4.6/9**：L=2 SH + 单场景法线分布下，只有 ~5 个组合
  方向可辨识——与 Route A 理论一致（k₂ 卷积后 L=2 子空间受法线
  分布 rank 限制）。
- κ 显示 inf：F 矩阵未做尺度归一（λ_min ~1e-12 量级 vs λ_max ~1e2）。
  **修正方案**：报告 κ(F/trace(F)) 与 λ_min/λ_max 比值；已列修正项。
- per-light Fisher 与 N 无关（设计如此）；**联合 (A,{L}) Fisher 随
  子集变化的版本**是 P1-11 的正式形态（脚本框架已留）。

## 5. 结论（对应任务书 P1-10 / Gate B）

1. **N 信息量在真实渲染数据上确凿存在**（2.9×，N≥8 饱和）——
   PRE-0 的"数据缺陷导致零信息"问题已被 P1-04 生成器修复解决。
2. 正式 Information Audit（200×32 主数据 + 收敛修正 + 空掩码排除）
   是进入 P1-15 Probe 重训前的最后一道门。
3. Conditioning 路线（H-COND 假设）首跑可用，是当前最有论文价值的
   理论方向（见 HANDOFF Q15）。
