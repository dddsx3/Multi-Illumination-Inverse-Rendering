# W1-D6 · D 轨预注册 pilot 文本 (2026-09-03)

> **来源**: 任务书新路线书 §D 维持原判
> **D 没有可推导的先验, scaling law 只能 empirical**
> **唯一诚实的形式是预注册 pilot, 失败就 KILL**

## 1. 阶梯 (FROZEN)

| 阶段 | 数据量 (scene) | 训练 image 总量 (scene×10) | 预算 GPU-hour |
|---|---:|---:|---:|
| 阶梯 0 (基线) | 200 | 2,000 | 6 |
| 阶梯 1 (中) | 2,000 | 20,000 | 60 |
| 阶梯 2 (决策) | 8,000 (条件) | 80,000 | 240 |

**预注册 GO/KILL**:

```
GO   ⟺ Δ ≥ 3°  (阶梯 0 → 1 的 DiLiGenT zero-shot MAE 下降)
    ∧ log-log 斜率外推到 ×50 数据量可达 ≤ 15° MAE
    ∧ DiLiGenT zero-shot MAE 在阶梯 1 端点 < 20° (SDPS-Net 对标)
KILL ⟺ Δ < 1.5° (平坦, 无 law 苗头)
1.5° < Δ < 3° ⟺ 加阶梯 2 (允许且仅允许这一次延期)
```

## 2. DiLiGenT 数字 (B 轨复述, 用于 D 轨对标)

- **SDPS-Net** (CVPR 2019, uncalibrated+variable-N): DiLiGenT 10 object 中位 MAE ~ 20°
- **UniPS** (CVPR 2022, universal): DiLiGenT 10 object 中位 MAE ~ 15°
- **PS-FCN N=96 重训 baseline**: DiLiGenT 10 object 中位 MAE ~ 10° (calibrated, 不是公平对标)
- **D 轨 zero-shot 目标**: ≤ 20° (SDPS-Net 对标) 或 ≤ 15° (UniPS 对标)

## 3. 数据集与训练

- **数据源**: R5-B′ 合成数据生成 (BlenderProc 渲染, 6 dev scene 已渲染)
  - 阶梯 0 (200 scene): 复用 R5-B′ 6 dev scene + 194 新 scene (新渲染, 1 day GPU 1× A10)
  - 阶梯 1 (2000 scene): 阶梯 0 + 1800 新 scene (5 day GPU 1× A10)
  - **阶梯 2 (8000 scene) 仅在 Δ 模糊时启用**
- **训练**: 同一网络 (R5-B′ 网络架构), 仅改训练数据
- **评估**: DiLiGenT 10 object zero-shot (B 轨下载的 DiLiGenT 数据)

## 4. 与其他轨的关系

- D 轨 **依赖** A 轨 (G1 闸门: practical proxy 与 oracle ranking 一致)
- D 轨 **依赖** B 轨 (B0 协议: cell-4 较 cell-2 改善 ≥ 8° → 重训数据有效)
- D 轨 **依赖** C 轨 GO (K=4 SG 路线若拿到 → 网络架构升级; 否则 C-α 残差)
- **D 轨 GO 是 (A∧B∧C) 全部 GO 之后的最终验证**

## 5. 算力预算 (3 天 GPU 上限)

| 任务 | 阶梯 | GPU-hour |
|---|---|---:|
| 阶梯 0 重训 | 200 scene | 6 h |
| 阶梯 0 评估 | DiLiGenT 10 object | 0.5 h |
| 阶梯 1 重训 | 2000 scene | 60 h |
| 阶梯 1 评估 | DiLiGenT 10 object | 0.5 h |
| 阶梯 2 (条件) | 8000 scene | 240 h |
| **D 轨总预算** | — | **67–307 h (1–12 天 A10)** |

**算力门槛**: ≥ 8×A100 或 1×A10 × 14 天 (任务书路线 D)
- 本机 RTX 5070 Ti Laptop (12GB VRAM) **不够** (A10 24GB 才能 batched train)
- 若没有 GPU 算力, D 轨 KILL, 论文不靠 D 故事 (R5-B′ 已转 identifiability diagnostic, 不需要 scaling 证据)

## 6. 与 ICCV'25 PRM 对比 (任务书 §D 论据)

ICCV'25 PRM (Photorealistic Reconstruction Models):
- 数据规模: **百万级**渲染图 (~1M-10M)
- 算力: 数百 A100-day
- **本项目算力差距**: 数量级
- **结论**: 单纯 scaling 故事在同一赛道**必输** (没有差异化内核的纯规模故事)
- **D 轨的意义**: 不是 "我们能 scaling 到 PRM 体量", 而是 "我们能在 A10 单卡证明 GSIQ 的可分解性 scale with 数据"

## 7. 风险与回退

- **KILL 信号** (Δ < 1.5°): 立即停止 D 轨, 论文仅用 R5-B′ 数据
- **延期信号** (1.5° < Δ < 3°): 申请额外 240 GPU-hour 跑阶梯 2 (1 次延期)
- **GO 信号** (Δ ≥ 3°): 阶梯 1 完成 → 论文主表加 "scaling curve" 图 (Figure 5), 是 nice-to-have 不是 must-have

## 8. 决策树

```
D0 阶梯 0 完成 (200 scene)
   |
   +-- Δ ≥ 3° → 跑阶梯 1 (2000 scene) → 论文加 scaling 章节
   |
   +-- 1.5° < Δ < 3° → 跑阶梯 2 (8000 scene, 1 次延期)
   |
   +-- Δ < 1.5° → D 轨 KILL, 论文不靠 scaling
```

## 9. 与 R5-B' 现状的衔接

- R5-B′ 数据: 6 dev scene (远少于 200), **不足以** 跑 D 轨
- R5-B′ 现有实验**只够** A/B/C 三轨; D 轨**必须**额外算力
- 论文若 A/B/C 三轨全 GO → 可投稿, 不依赖 D 轨 GO
- 论文若 D 轨 GO → 锦上添花, 加 scaling 章节

## 10. FROZEN (不允许 review 后调整)

- 阶梯数据量: 200 / 2000 / (8000 条件)
- 评估集: DiLiGenT 10 object
- 评估指标: zero-shot MAE
- GO/KILL 阈值: 3° / 1.5° (任务书原值)
- 训练数据源: R5-B′ 合成管线 (BlenderProc)

---

*W1-D6 写于 2026-09-03 · ZCode agent · 0 GPU · 0 元成本*
*D 轨门槛最低 (KILL 也接受), 但需要 67-307 GPU-hour*
*本机无 GPU 算力 → D 轨**默认 KILL**, 论文不靠 scaling 故事*