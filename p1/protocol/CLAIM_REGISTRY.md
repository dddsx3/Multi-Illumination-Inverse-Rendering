# CLAIM_REGISTRY · 论文三句话契约（P1-R6 首件，先于一切实验结果冻结）

> **本文件是论文的宪法**：此后任何实验、图表、章节都必须服务于以下三句话。
> 修改本文件 = 修改论文核心 = 需要 R4 级证据 + 显式版本号。
> 版本：v0.1（R4 定核 Gate 结果出来前，Hypothesis 以假设形式存在）

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

## 核心判据（R4 Gate，对应 EXPERIMENT_CONTRACT E2/E3）

- **G1**：固定 N 内，GA-ISI 分数（λ⁺_min / logdet⁺ / A-opt）与恢复误差
  有稳定符号的显著相关；
- **G2**：Error ~ logN + score 的解释力显著优于 Error ~ logN（ΔR²）；
- 若通过 → 核心锁定，题目候选
  *Beyond Cardinality: Gauge-Aware Illumination-Set Identifiability
  for Variable-Cardinality Inverse Rendering*；
- 若失败 → 杀掉 H-COND（不许补故事），降级候选 =
  "arbitrary-N single-view joint decomposition with explicit per-light
  illumination + held-out relighting"（弱核心，参考 IDArb/LINO 再定位）。
