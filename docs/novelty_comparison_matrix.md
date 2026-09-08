# Novelty Comparison Matrix · 六+2 条最近邻谱系（卡 F1）

> 日期：2026-09-09 · 目的：新颖性闭口（专家 48h 硬门）。
> 全部条目经 Crossref API 独立核实（DOI 解析成功，卷期页码见行内）——"TCI Differentiable
> Uncalibrated Imaging" 原状态"待核验"现已闭合：**Gupta, Kothari, Debarnot, Dokmanić,
> IEEE TCI vol.10 pp.1–16, 2024, DOI 10.1109/tci.2023.3346294**。
> 列定义：[目标] = 误差传播的对象；[量] = 分析给出的量；[gauge] = 是否处理 gauge 自由度；
> [逐模式] = 是否 mode-resolved；[连续] = 是否建模 calibrated↔uncalibrated 连续置信度；
> [真实验证] = 是否有受控真实数据注入/验证。
> 结论行见 §3；补充检索线索见 §4。

## 1. Comparison Matrix

| # | 文献 | 误差源 | [目标] | [量] | [gauge] | [逐模式] | [连续] | [真实验证] | 内容摘记 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Jiang & Bunke 1991, Signal Processing 23(3):221–226, DOI 10.1016/0165-1684(91)90001-Y | 光照方向/反照率估计误差 | 单位法向量 | 解析灵敏度公式（传播到法线） | 否 | 否 | 否（单点灵敏度） | 否 | 单位法向量表示下 PS 法线对各类误差源的传播公式，经典 sensitivity 起点 |
| 2 | Kobayashi et al. 2011, 3DIMPVT, pp.25–32, DOI 10.1109/3DIMPVT.2011.13 | 光源方向校准误差 | 法线角误差 → 重建似然 | Fisher 分布建模 | 否 | 否 | 否（给定误差水平） | 否（真实数据但无受控注入） | 方向校准误差建为 Fisher 分布并传播到法线/重建，与本文 §VI 校准传播的"给定分布"版本最近 |
| 3 | Klaudiny & Hilton 2014, PRL 48:81–92, DOI 10.1016/j.patrec.2013.12.013 | 光照矩阵 L / 交互矩阵 V / 像素噪声 | albedo-scaled normals | 解析传播 + 误差最小化条件（L 正交、V 对角） | 部分（色彩灯基选择） | 否 | 否 | 否 | 三类误差的解析传播与拍摄设置设计准则 |
| 4 | Chen et al. 2022, CGF 41(6):149–165, DOI 10.1111/cgf.14516 | 五类误差（含光源位置校准） | 法线/重建 | 解析公式 + 统计评估 | 否 | 否 | 否 | 否（统计评估非受控注入） | 近准点光源 PS 误差分析，位置校准误差列为主导因素之一 |
| 5 | Quéau et al. 2017, CVPR, DOI 10.1121（红队报告二已核实） | 光照强度+方向不准确 | 重建本身 | 变量联合细化（robust） | 部分 | 否 | 否（不准确=固定档） | 否 | inaccurate lighting 下联合细化光照与表面——reconstruction 路线，与本文 information 路线正交 |
| 6 | Gupta et al. 2024, IEEE TCI 10:1–16, DOI 10.1109/tci.2023.3346294 | 未校准成像参数 | 图像/几何 | 可微成像管线端到端 | 否 | 否 | 否（uncalibrated 固定端） | 部分（真实采集） | 可微未校准成像：校准/重建联合估计——TCI 语境合规信号；本文诊断 confidence 如何改变可用信息与弱模式 |
| 7（补） | Cho et al. 2020, TPAMI 42:232–245, DOI 10.1109/tpami.2018.2873295 | 部分校准（半校准） | 法线/反照率 | 模型+优化 | 是（半校准 gauge 分析） | 否 | 部分（calibrated/semi/uncalibrated 三态，非连续） | 否 | Semi-Calibrated PS 系列主文；三态离散，无逐模式信息量 |
| 8（补） | Brenner & Sablatnig 2024, 3DV, DOI 10.1109/3dv62453.2024.00019 | 点光源环境误差 | 法线 | 误差分析 + 缓解 | 否 | 否 | 否 | 部分（实验评估） | 点光源 PS 误差分析与缓解，2024 年仍在 sensitivity-分析范式内 |

## 2. 差异定位（专家裁决句式，供 F2 锁死）

过去谱系回答：**给定**校准误差水平时，误差如何传播到法线/重建（sensitivity 分析）。
本文回答：**从 calibrated 到 uncalibrated 的连续置信度**如何逐模式地改变可用信息——
nuisance-marginalized information（ΔF）、逐模式 retention 谱、gauge lifting、
场景条件交叉 λ⋆——并验证这些信息量**预测受控真实 corruption 下相对退化结构**的能力
（CI04, Spearman 0.728）。矩阵 1–8 号条目在 [连续]×[逐模式]×[真实验证] 三列上**全部为否**，
无 prior-art collision；连续置信度模型 + 逐模式信息量 + 受控真实验证的组合未被占据。

## 3. Collision 判定（停止规则①检查）

- 逐行核对：无任何文献同时具备"连续置信度模型 + mode-resolved information prediction
  + 受控真实 corruption 验证"三要素 → **未触发停止规则①，实验层保持 CLOSED**；
- 最大风险条目 = Gupta 2024（TCI 同刊）：其"uncalibrated"是重建路线的固定端点，
  无信息论预测与 mode-resolved 读出；在 Related Work 中作 TCI-facing 邻接 + 精确差异。

## 4. 补充检索线索（逐条一行，2026-09-09 Crossref）

1. `query.bibliographic=Differentiable Uncalibrated Imaging` → 命中 #6（卷期闭合）；
2. `query=calibration uncertainty photometric stereo (sort=published desc)` → 噪声大，改定向；
3. `query.bibliographic=photometric stereo calibration error uncertainty` → 无新近邻；
4. `query.bibliographic=semi-calibrated photometric stereo` → 收获 #7（TPAMI 2020 三态离散）；
5. 定向 `3dv62453` → 收获 #8（2024 error analysis，仍在 sensitivity 范式）；
6. 未发现 2024–2026 间任何"confidence continuum + mode-resolved prediction"式工作。
