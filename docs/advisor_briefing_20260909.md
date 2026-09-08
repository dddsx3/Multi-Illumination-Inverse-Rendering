# 导师送审简报（1 页）· 2026-09-09

**主题**：Calibration–Confidence Continuum（TCI 投稿）终审请求

## 问题定位（一段话）

光照校准从来不是"已知/未知"的二元开关。我们给出一个**从 calibrated 到 uncalibrated
的连续置信度模型**：把校准不确定度（物理单位 Σ_c）作为随机 nuisance，用白化空间的
混合 CRB/Schur 补得到逐模式的**有效信息连续谱** ΔF(Λ)，并证明 gauge 方向的信息被
校准置信度**精确线性抬升**（Prop 2）直到场景本征地板 λ⋆——λ⋆ 可由场景量闭式预测。
传统 trace 级汇总对此完全失明（calibrated/uncalibrated trace 比 1.01，弱模式条件数 10⁻⁷）。

## 主结果（全部数字可一键复现）

| 结果 | 数字 | 出处 |
|---|---|---|
| 代数正确性（Gate A） | 100/100 identity checks，双路线 2.2e-12 | CI01 |
| λ⋆ 场景量预测 | 54/54 合成场景，median \|log₁₀\| = 0.00000 | CI02 |
| 估计器有限样本校准（Gate B） | 弱模式方差比 0.977–1.009；coverage 68/95 达标 | CI03 |
| 线性化有效域（Gate C） | mask-flip ≤ 0.8–3.0% 内理论误差 <10% | CI03-nl |
| **受控真实 corruption（Gate D）** | OpenIllumination **test 11 对象**冻结清单：Spearman **0.728 [0.668, 0.783]** | CI04 |
| 真实 sanity + ablation | DiLiGenT 10/10 弱模式存在；4 项 within-scene 效应量 | CI05 |

## Claim 边界（已逐句冻结，CLAIMS_REGISTRY 可查）

- 真实数据主张为**秩相关层**（专家锁死句）：预测的逐模式信息损失一致地对受控真实
  corruption 下的经验退化排序；**绝对量级**受二阶效应与未建模校准/模型失配影响（~10²，
  作为 sensitivity band 如实报告，不做绝对量级主张）；
- λ⋆ 合成侧为 quantitative validation；真实侧只写"real-scene diagnostic remains
  finite/meaningful"；
- Λ misspec 只报 −74.9%（单场景二阶，不泛化）；斜率 1.82 只在 Limitations。

## 新颖性（文献闭口已完成，八条最近邻全核实）

1991–2024 的 PS error-analysis 谱系（Jiang & Bunke / Kobayashi / Klaudiny & Hilton /
Chen / Quéau / Gupta TCI 2024 / Cho TPAMI / Brenner 3DV 2024）全部是"**给定**校准误差下"
的 sensitivity 分析；连续置信度 × 逐模式信息量 × 受控真实验证的组合**无撞车**
（docs/novelty_comparison_matrix.md，含 DOI 与检索线索）。TCI 同刊 Gupta et al. 2024
证明选题语境合规。

## Reproducibility 承诺

干净检出 + `bash scripts/reproduce_paper.sh` = 测试 54/54 + 全部正式 run + 9 图重建，
每图带 git SHA + artifact sha256 provenance；clean-room 复现已对账到位级一致
（docs/memo_F3_crossmachine_20260909.md）。

## 请您审阅

1. 主结果与 claim 边界是否可作为论文最终主张（手稿 v0.8 → v0.9 冻结中）；
2. 作者顺序/致谢/基金号；
3. 是否同意按此版本投稿 IEEE TCI（13 页套版中）。
