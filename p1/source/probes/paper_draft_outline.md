# Paper Draft 0 · 章节骨架与完成度（P1-R6）

> 依据：专家 §八 建议——60~70% 正文不依赖任何训练结果。
> 本文件跟踪各章状态；写作产物落在 `paper/`（待建）。

| 章 | 状态 | 依赖 |
|---|---|---|
| 1 Introduction | 80% 可写（问题动机 + 数据灾难叙事可作 motivation 小节） | CLAIM_REGISTRY |
| 2 Related Work | 80% 可写 | RELATED_WORK_MATRIX_v2（IDArb/LINO/GeoUniPS/ReLeaPS/ICCV05/Basri） |
| 3 Problem Formulation | **100% 可写** | CLAIM_REGISTRY 三句话 |
| 4 Physical Model | R1 后完成（已达成） | LIGHTING_MODEL.md（Â 修正版）+ DATASET_CONTRACT |
| 5 Illumination-Set Identifiability | R3 后可写（已达成） | IDENTIFIABILITY.md（1-7 节已可写，8-10 部分待 R4 数字） |
| 6 Network | skeleton（架构不设汁，只描述 probe 语义） | P1-15 完成后补 |
| 7 Experiments | 写 protocol 不填结果 | EXPERIMENT_CONTRACT.md（E1-E9 + C0） |
| 8 Results | 空表占位 | R4/Gate 数据 |
| 9 Limitations | 先写（已知 5 条：calibration 规模、solver 收敛偏差、
  normal_depth 边缘、gauge 投影 v1、R 域未生成） | — |
| Abstract | 最后写 | 全部 |

## 图表清单（可先开工的）

- **Fig 1（概念图）**：same-N poor-vs-diverse → different decomposition
  → gauge-aware illumination-set score（可立即画）；
- **Fig 2（系统图）**：variable-N encoder → shared scene + per-light
  lighting → differentiable renderer（可立即画）；
- **Fig 3（灵魂图，预留）**：x = GA-ISI score，y = decomposition error，
  color = N——若成统一曲线，论文故事成立（R4 数据）；
- **Table 1**：related-work matrix（11 维比较，v2 矩阵已给内容）；
- **Table 2**：oracle floor 修正前后对照（14.88 → 22.25 → 28.25 dB 的
  勘误叙事可作附录 transparency 条目）。
