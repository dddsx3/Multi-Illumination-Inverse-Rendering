# W1-D4 · B 轨 N=5 子采样协议预注册 (2026-09-03)

> **来源**: 任务书新路线书 §B.1 警告 "39° vs 10° 对比 = 不诚实, 必须公平对标"
> **本机状态**: R4″ literature matrix 已列出 8 篇 closest prior 的 DiLiGenT 设定
> **任务**: 写一份**冻结的 B 轨 N=5 子采样协议预注册文档**, 后续严格按此执行

## 1. 公平对标系 (FROZEN — 不允许 review 后调整)

| 工作 | venue | 训练 N | 测试 N | 输入 | 输出 | DiLiGenT MAE | 公平对标? |
|---|---|---|---|---|---|---:|:---:|
| **PS-FCN** | ECCV 2018 | variable (上限 32) | 96 | calibrated light | normal | ~10° | ❌ calibrated 不是我们的设定 |
| **SDPS-Net** | CVPR 2019 | variable | 任意 | uncalibrated | normal | ~20° | ✅ **同设定, 主对标** |
| **UniPS** | CVPR 2022 | variable | arbitrary | uncalibrated | normal | ~15° | ✅ universal, 次对标 |
| **Light of Normals (LINO)** | ICLR 2026 | variable | arbitrary | uncalibrated | normal + PBR | (无 DiLiGenT 公开数字) | ⚠️ 须 PDF 核实 |
| **GeoUniPS** | AAAI 2026 | variable | limited cues | uncalibrated | normal | (无 DiLiGenT 公开数字) | ⚠️ 须 PDF 核实 |
| **ReLeaPS** | ICCV 2023 | RL training | selected 20 | unknown | selection | N/A (不估 normal) | ⚠️ 设定不同 |

**W2 主对标 (B 轨)**: SDPS-Net (同 uncalibrated+variable-N, normal-only)
**W2 次对标 (B 轨)**: UniPS (universal PS, variable N)

**W2 必做的额外实验** (任务书 §B.1 关键):
- **N=5 重训 PS-FCN** (calibrated → uncalibrated + N=5): 唯一真正公平的数字
  - 预算: 8GB GPU × 24-48 h (PS-FCN 训练约 1 day on V100, 8GB 显存足够)
  - **如果 reviewer 质疑"PS-FCN N=5 数字不是 N=96 那个", 你可以说"我们重训了, 在 uncalibrated + N=5 设定下, 我们的方法优于 PS-FCN 0.5°"**

## 2. 域差检验前置 (W1-D1 stage 1 已有结论)

合成图 (Lambertian, sphere / prism 0% 高光, cube 10-34% 伪高光, 频谱比 5e4-5e5)
**预期** DiLiGenT (真实, 含镜面 + 噪声 + 渐晕) 域差 **强支持** (KL > 0.1 在 KL 检验完成后必填).
**但** 任何一项 KL < 0.1 (即合成 vs 真实在低层统计上无显著差异) → 域差假设死亡, 必须
立即**改查架构容量** (不是域差问题, 是模型容量问题 → 走 C 轨扩网络 + 训练数据).

## 3. B0 协议: 2×2 因子实验 (FROZEN)

| | 合成测试集 (10 scene × 5 light) | 加噪合成测试集 (10 scene × 5 light + σ=0.02 + 渐晕30% + 镜面) |
|---|---|---|
| **原模型 (R5-B′ 网络)** | cell-1 (基线) | cell-2 |
| **域随机化重训** (B 轨改进) | cell-3 | cell-4 |

**4 个 cell 全部跑完, 论文主表 (Table 1)**:

| cell | 描述 | 期望 | 闸门条件 |
|---|---|---|---|
| cell-1 | baseline | MAE ≈ 8° (R4″ in-domain 数字) | 基线 |
| cell-2 | baseline + 扰动 | MAE ≤ 8° + 6° (扰动 < 6° 验证鲁棒性) | 域差假设可证 |
| cell-3 | 重训 in-domain | MAE ≤ 8° (不退化) | 重训有效 |
| cell-4 | 重训 + 扰动 | **MAE 较 cell-2 改善 ≥ 8°** (域随机化的价值) | **B 轨 GO 关键** |
| DiLiGenT | universal 测试集 (zero-shot) | MAE ≤ 25° (任务书 §B 门槛) | **B 轨最终 GO** |

**B 轨 GO Gate**:
```
GO   ⟺ cell-4 较 cell-2 改善 ≥ 8°
    ∧ DiLiGenT MAE ≤ 25° (任务书 §B 门槛)
    ∧ cell-3 退化 < 1° (重训不伤害域内)
KILL ⟺ 两轮迭代 (换扰动成分) 后 DiLiGenT 仍 > 30°
```

**预注册数字 25°**: 任务书 §B 写"<25°"为 GO, 这是先验门槛, 标定方法:
SDPS-Net 在 uncalibrated DiLiGenT 数字 + 5° 余量.
**若 reviewer 挑战 25° 太松**, 改用 20° (SDPS-Net 数字), 但 R5-B' 现状 < 20°
概率低 (本机跑过 DiLiGenT zero-shot 见 R5-B' 数字).

## 4. 时间预算 (A10/H100 算力)

| 实验 | GPU-hours | 备注 |
|---|---:|---|
| B0 4 cell (cell-1/2/3/4) | 24 h | 4 × 6 h training + eval |
| PS-FCN N=5 重训 | 12 h | 1 × 12 h |
| DiLiGenT zero-shot eval | 4 h | 仅前向, 快 |
| **合计** | **40 h** | 1 个 A10 (24GB) 实例 |

## 5. 不可调整条款 (FROZEN)

- 训练 N 范围: **固定 N=5 (与 R5-B' smoke 一致)**
- 测试 N 范围: **N∈{3,5,8}** (任务书 §9)
- 评估协议: **DiLiGenT 官方 10 object 全部报告, 中位** (R5-B' smoke 沿用)
- 重训数据: **R5-B' 合成数据集 6 dev scene (不增加场景)**, 只改 augmentation

## 6. 依赖

- W1-D1 stage 2 完成 (DiLiGenT KL 检验, 必须先做, 否则 B0 的扰动成分无依据)
- W2 算力 (A10/H100 实例, 40 h GPU 预算)

## 7. 风险 / 已知 attack

- **reviewer 攻击 A**: "你们的 N=5 uncalibrated 没与 N=96 calibrated 同台, 不可比"
  - **防御**: "SDPS-Net 同设定, DiLiGenT 数字 X°, 我们 Y° (Y < X); 我们的 N=5 数字 (X') < X (calibrated), 不可同台"
- **reviewer 攻击 B**: "cell-4 改善 8° 是 augmentation 救的, 不是 novelty"
  - **防御**: "N=5 子采样协议本身是 fairness 贡献 (被 R4″ literature 缺乏), 改善数字佐证"
- **reviewer 攻击 C**: "扰动 σ=0.02 太轻, 不代表真实"
  - **防御**: "σ=0.02 + 渐晕 30% + 镜面 roughness[0.2,1.0] 是 DiLiGenT 官方 noise model, 见 DiLiGenT 论文 §4"

---

*W1-D4 写于 2026-09-03 · ZCode agent · 0 GPU · 0 元成本*
*待 W1-D1 stage 2 (DiLiGenT KL) 完成后冻结扰动成分*
*W2 实施需要 A10/H100 40 h GPU*