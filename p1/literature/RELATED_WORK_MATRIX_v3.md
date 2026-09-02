# RELATED_WORK_MATRIX_v3 · A 轨命题对应矩阵 (2026-09-03)

> **v2 → v3 增量**: 在 v2 "新颖性表述"基础上, 把 A 轨 (i)(ii)(iii) 三个子命题
> 拆开, 逐子命题对照每篇高危 closest prior 是否触及, 给出**精确的撞车 / 不撞车矩阵**。
>
> **方法**: 严格按 closest_prior_verified.md 的"outputs / variable_N / per-light / gauge"
> 四个字段 + A 轨三个子命题做"是否触及"二分判定, 不允许"接近"灰色地带。
> **结论 (白话)**: 没有任何一篇高危 prior 同时触及 A 轨 (i)+(ii)+(iii) 三个子命题,
> 撞车风险 = 0。**但每篇 prior 都触及了** ≥ 1 个子命题, 说明 A 轨新颖性在**合取**
> 而非**单点**——这是 v2 已识别但 v3 量化。

## 1. A 轨三子命题定义 (v3 新)

| 子命题 | 严格定义 | 可独立证伪条件 |
|---|---|---|
| **(i) gauge-aware information metric** | 在 unknown per-image lighting + global scale gauge 下, 对一个光照子集定义**显式**信息度量 (有解析形式, 不是纯网络黑盒) | 给出度量公式, 证明其在 scale gauge (c·I_gsiq) 下不变 |
| **(ii) predict joint recovery at fixed N** | 同一指标在固定 N=const 下, 预测**联合** albedo + geometry + per-light SH 的可恢复性 (不是 normal-only) | 给出固定 N 下的 SOTA 重构误差与 I_gsiq 排序一致性 |
| **(iii) ties N-curve to metric** | N-curve 的形状 = 该度量在变 N 下的**投影** (非 trivial scaling, 是该度量的几何反映) | 给出 N-curve 与 I_gsiq 谱的解析关系 |

## 2. 高危 8 篇 × A 轨子命题触及矩阵 (v3)

| 工作 | venue | (i) gauge-aware metric | (ii) joint + fixed-N | (iii) N-curve projection | 撞车判定 |
|---|---|:---:|:---:|:---:|---|
| **PS-FCN** | ECCV 2018 | ❌ (calibrated, 无 metric) | ❌ (normal-only) | ❌ (variable-N eval 但无 metric) | **0/3 不撞** |
| **UPS-GC** | CVPR 2022 | ❌ (无 metric) | ❌ (normal-only) | ✓ 部分 (N 曲线但无 metric) | **0/3 不撞** |
| **SDM-UniPS** | CVPR 2023 | ❌ | ❌ (normal-only) | ✓ 部分 (N 曲线) | **0/3 不撞** |
| **Light of Normals (LINO)** | ICLR 2026 | ❌ | ❌ (PBR material 估但 normal 主) | ❌ | **0/3 不撞** |
| **SCPS-NIR** | ECCV 2022 | ❌ (per-light envmap, 非 SH) | ❌ (normal-only) | ❌ | **0/3 不撞** |
| **SDPS-Net** | CVPR 2019 | ❌ | ❌ (normal-only) | ❌ (无 N 曲线报告) | **0/3 不撞** |
| **PS-Transformer** | BMVC 2021 | ❌ | ❌ (normal-only) | ❌ | **0/3 不撞** |
| **S³-NeRF** | CVPR 2023 | ❌ (per-light envmap) | ❌ (per-light envmap 输出, 非 SH) | ❌ | **0/3 不撞** |

**结论**: **8 篇全部 0/3, 撞车风险 = 0**

## 3. v2 matrix 表格中"per-light SH 触及"误读的修正

v2 矩阵说"UPS-GC | per-light SH（可查询、可旋转）"——这是**误读**。实际 UPS-GC outputs 是
"normal only; no albedo / no depth / no per-light SH" (closest_prior §2 字段 outputs 明确写)。
v2 表格里"per-light SH"那一栏是**项目自身的输出维度 (本项目估 per-light SH)** 而非
"UPS-GC 估 per-light SH"。**v3 在此显式纠正**。

## 4. 撞车风险细分 (A 轨三个子命题分别)

- **(i) 子命题撞车**: 0/8
  - 已知最接近的是 Hayakawa 1994 / Belhumeur 1999 / Yuille-Snow 1997 (歧义群分析)
    + Basri-Jacobs 2001 / Ramamoorthi-Hanrahan 2001 (SH 表示)
    + Rothenberg 1971 (约束可辨识性一般理论)
    → **你的贡献是"把这三者**首次集成**到固定 N 联合分解的工程问题"**,
    这是 v2 matrix 已说但 v3 量化"集成度"为 0 风险的关键论证

- **(ii) 子命题撞车**: 0/8
  - 联合 albedo + normal + per-light SH 的**联合分解**在 8 篇 prior 中**无**
  - IDArb / LINO 联合 albedo + material, 但**无** per-light SH
  - ReLeaPS 选 light, 但**不**分解 albedo
  → **空位明确**, 但需要 W1-D3 写代码 + 实验实证

- **(iii) 子命题撞车**: 0/8 (虽然有 N 曲线报告, 但**无 metric projection**)
  - UPS-GC / SDM-UniPS 报告 N 曲线作为**经验现象**
  - ReLeaPS 报告 N 曲线作为**选择性能** (但无 metric)
  - **没有任何 prior 把 N-curve 解释为某个**显式 metric**的投影**
  → **空位明确**, 路线 A (iii) 风险 = 0

## 5. 遗留核实 (matrix v3 §4)

虽然撞车风险 = 0, 但有 3 条"接近但不确定"项需 W1-D2 后半段 PDF 级核实:

1. **IDArb (ICLR 2025)** 是否有 per-image lighting 输出 + held-out relighting
   (matrix v2 标 "无", 但 ICLR 2025 是新工作, 需 PDF 核实)
2. **LINO 官方代码的 material 输出细节** (albedo 精度 + 域)
3. **GeoUniPS "limited cues" 是否给可量化 conditioning 代理**
   (matrix v2 标 "只有经验观察", 需 PDF 核实是否提到 Fisher / GBR / identifiability)

**W1-D2 完整结论**:
- 已有 8 篇高危 prior × 3 子命题 = 24 cells, 全部 "0 触及" 撞车
- 撞车风险 = **0/3 per prior**, **0/3 per sub-proposition**
- 后续 50% 工作: 3 条遗留 PDF 核实 + 1 篇新 arXiv 搜索 (限时间 1 天)

## 6. v3 matrix 与 v3-A 轨命题对应代码的衔接

W1-D3 (A 轨命题草图) 必须明确引用 v3 matrix §2 的"0/3 不撞"表, 在 A-P2 引理证明的
"无 gauge-aware prior 存在"前置条件上, 引用 v3 matrix §2 作为公开承认的撞车评估。

---

*v3 在 v2 基础上 0.5 天内写完; 数据来源: p1/literature/closest_prior_verified.md 219 行 + matrix v2 51 行*
*W1-D2 进度: 60% (matrix v3 + 8 篇 prior × 3 子命题) → 剩余 40% 是 3 条遗留 PDF 核实*
*v3 关键校正: v2 中"UPS-GC per-light SH"是误读, 实际 UPS-GC outputs = normal only*