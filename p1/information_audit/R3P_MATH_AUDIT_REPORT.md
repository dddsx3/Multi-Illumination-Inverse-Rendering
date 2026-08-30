# R3P_MATH_AUDIT_REPORT · R3′ Gauge-Fisher 数学审计报告

> **裁决：PASS**（2026-08-31 · 基线 commit `c184cdd`）
> 本报告只报 PASS/FAIL，不做任何 R4′ 统计（任务书 §3 T3′.5 / §9 第 6 条）。

## 1. 审计对象与范围

- 对象：GA-ISI 有效 Fisher 的数学正确性（任务书 §0.1 裁决的两个公式级缺陷）；
- 冻结模型：`I_k(p) = a_p·ReLU(Y_pᵀc_k) + ε`，几何已知（最小问题封口）；
- 交付物：
  - `p1/source/information_audit/gauge_fisher_v2.py`（正式实现，含 dense + operator 双路径）
  - `p1/tests/test_gauge_fisher_v2.py`（28 项单测）
  - `p1/protocol/IDENTIFIABILITY_v2.md`（修正后理论，v0.1 §4–§9 作废）
  - 旧 `gauge_fisher.py` / `defining_gate_summary.json`：顶部 **DEPRECATED_EXPLORATORY** 标记

## 2. 修正内容（对照任务书 §0.1）

| # | 缺陷（v1） | 修正（v2） | 验证 |
|---|---|---|---|
| 1 | 交叉块 `F_sℓ(p,k)=a_p·h_kp·Y_pᵀ` 漏 s_kp | `B_k[p,:] = a_p·s_kp·h_kp·Y_pᵀ` | T2e（JᵀJ 互证，max_rel 2.4e-16） |
| 2 | 逐像素标量近似冒充"完整 Schur 补" | `F_eff = F_ss − Σ_k B_k F_ll,k† B_kᵀ`（P×P 稠密 PSD，跨像素耦合）；逐像素版降级为 diag-Schur proxy 分栏 | T3a/b（三条独立路线 relF ≤1.7e-11）、T3d（off-diag/diag_mean≈7.1，结构确实存在）、T3e |
| 3 | gauge 仅靠 a 归一（cosmetic） | 解析恒等式 `F_eff·a = 0` + 投影 Π_g 主策略 + gauge 变换不变性 | T4a（residual 6.5e-14）、T4b（投影正谱不变 2.9e-15）、T4d（(a,C)→(a/2,2C) 指标不变 ≤1e-9） |

## 3. 四类强制单测结果（任务书 T3′.3 冻结阈值）

| 测试 | 要求 | 实测 | 裁决 |
|---|---|---|---|
| Finite-difference Jacobian | rel Frobenius ≤ 1e-5 | 1.2e-10 / 1.1e-10（active / mixed 边界列剔除） | **PASS** |
| Block identity（JᵀJ vs 解析块） | max rel ≤ 1e-8 | 2.4e-16（交叉块）、8.3e-15（F_ll,k）、2.2e-16（F_ss） | **PASS** |
| Schur identity（toy P≤50 显式） | rel ≤ 1e-6 | 1.6e-11（vs JᵀJ 组装）、4.8e-12（vs 投影形式）；PSD λmin=−3.3e-12≥−1e-10·λmax；off-diag 非零 | **PASS** |
| Gauge/null（尺度方向；固定/投影后正谱稳定） | cutoff 1e-8~1e-5 不改变裁决 | residual 6.5e-14（1e-8）/ 4.9e-6（1e-5）；投影正谱 relF 2.9e-15；primary 漂移 spread 4.6e-4 < 1e-3 | **PASS** |

附加（超出任务书最低要求）：重复光 kernel 不变性（T5a）、N=1 秩 = P−9
（T5b，修正 v0.1 命题 P1）、ReLU 边界 z=0 精确语义（T5c）、operator
路径一致性（T6：matvec 3.5e-16 / trace 0 / eigsh λ_min⁺ 5.9e-14）。

完整输出存档：`R3P_test_output_c184cdd.txt`。**总计 28 项 · PASS 28 · FAIL 0。**

## 4. Gate 裁决与效力

- **R3′ MATH GATE = PASS**：允许进入 R4′（R4′-D discovery 复跑 →
  R4′-C 新确认集 → Confirmatory Gate）。
- 效力（任务书 §1 文档优先级）：现有 R4 的固定-N 相关性（ρ=−0.42~−0.86、
  ΔR²=0.002）正式降级为 **exploratory signal**，不得进入主结论、摘要、
  标题或定核 Gate；H-COND 维持 hypothesis。
- 未启动项（持续禁令）：200×32 全量生成、大模型设计/训练、网络架构工作。

## 5. 环境

Python 3.14.2 · numpy/scipy 1.17.0 · Windows git-bash · 单测为 CPU 纯解析，
无 GPU/随机性依赖（种子固定）。

签发：R3′ · ZCode agent · 2026-08-31
