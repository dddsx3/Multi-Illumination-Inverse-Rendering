# CLAIM_REGISTRY · 论文三句话契约（P1-R6 首件，先于一切实验结果冻结）

> **本文件是论文的宪法**：此后任何实验、图表、章节都必须服务于以下三句话。
> 修改本文件 = 修改论文核心 = 需要 R4 级证据 + 显式版本号。
> **版本**：v0.3（R4″ sprint 完成；6 行 Gate dashboard 见 `r4pp/08_go_no_go_dashboard.md`；
> 终裁 `r4pp/09_R4pp_decision.md` = **PIVOT (B′)**；H-COND 主效应在 Direction
> 弱化版上成立但 Geometry × Information interaction 未观察到稳定趋势）。
> 落盘 commit：`2af42c0`（Task F 完成点）。

## Research Question

> Why do some illumination sets permit substantially better joint
> decomposition than others, even when they contain the same number
> of observations?

（为什么某些光照集合在**相同数量**下允许可观更好的联合分解？）

## Hypothesis（待 R4 定核）

> Recoverability is governed by gauge-aware effective information of the
> illumination set rather than cardinality alone.

（可恢复性由光照集合的 gauge 感知有效信息控制，而非单纯的基数。）

## System Claim

> We study this in feed-forward variable-cardinality uncalibrated inverse
> rendering that jointly estimates canonical reflectance, 2.5D geometry
> and explicit per-image illumination.

（在 feed-forward、可变基数、非定标逆渲染中研究该问题：联合估计
canonical reflectance + 2.5D 几何 + 显式 per-image 光照。）

## 明确不是核心的东西（系统组件，不许写进贡献列表）

- Set Transformer / FiLM / attention（PRE-0 与 P1 任务书双重禁止）
- variable-N 本身（IDArb 已占）
- 9D SH 本身（Basri & Jacobs 系已占）
- active light selection 本身（ReLeaPS 已占）
- 2.5D depth 本身
- "光照质量比数量重要"的直觉（ICCV05 经典 + GeoUniPS 已占）

## 核心判据（R4″ Gate，6 行 dashboard，详见 `r4pp/08_go_no_go_dashboard.md`）

- **Instrument**：bulk 信息度量（M1 log pdet）5/5 stability test PASS。
- **Signal**：low-N R_signal 远高于 repeat noise（全 24 cell > 2，median 27.2）。
- **Direction**：info→error β<0 在多数 scene 成立（β median −0.348, 81% 负号）。
- **Interaction**：controlled geometry pilot 内 G↑⇒|β_G|↑ 趋势 **未稳定**
  （family A N=3 ρ=+0.29, N=5 ρ=+0.72 反向；family B 3 level G 区分度不足）。
- **Saturation**：N=8 R_signal 22.6 且 σ_subset/err 3.2% — 真衰减不是噪声。
- **Externality**：local-vs-global init 实验因本机环境约束 PENDING（详见 decision 报告）。

> **当前状态（v0.3）**：R4″ sprint 6 行 Gate = **Instrument/Signal/Saturation
> PASS, Direction PASS, Interaction FAIL, Externality PENDING**。
> 预注册裁决映射（§44）：**PIVOT (B′)**。
>
> **PIVOT 后的新科学问题**（任务书 §28 B′ 主线原文）：
> *Beyond Cardinality: Effective Information at Fixed Illumination Budget*
> 说明 illumination information 本身有效（Direction + Signal + Instrument 三条支撑），
> 但 geometry × information interaction 没有稳定机制证据（Interaction FAIL）。
>
> 不进入 GO (A2) 也不进入 KILL H-COND —— 前 4 个 Gate 给出足够裁决依据，
> 第 5 个（Externality）受环境硬约束不能执行、但任务书 §28 规定不影响裁决。
>
> 与 R4prime_frozen 旧数据的根本区别（详见 `archive/R4prime_frozen/R4prime_failure_audit.md`
> 与 `r4pp/09_R4pp_decision.md`）：primary metric 已从退化的 `λ_min⁺` 换为 M1 log pdet；
> 收敛判据已从 P75×P75 内生筛选换为 absolute-step200 无内生准则；Solver 加 seed/theta0/
> trace/proj_grad_norm 可复现；新增噪声地板标定。**所有引用旧 R4′ 数字的论文段落
> 必须在更新版删除或重做**。
