# ORACLE_AUDIT · PRE-01 GT 物理一致性审计报告

> 数据：synthetic_v3 test split 全部 124 场景 × 5 光（split hash `98e91fc5…`）。
> 本审计**不使用任何神经网络**：用 GT albedo、GT normal、GT SH 光照走协议渲染器，
> 与真实图像（Cycles 路径追踪产物）对比，量化"物理渲染器能否解释数据"。
> 复现：`python pre0/source/renderer/oracle.py --split test`
> 证据：`oracle_metrics.csv`（逐场景逐光）、`oracle_summary.json`、
> `normal_protocol.csv`、`lighting_convention.csv`、`residual_*.png`、`panel_*.png`。

## 1. 实验 A · GT Oracle Reconstruction（核心数字）

在**线性域**（精确 sRGB 反变换解码后的图像）中，使用完整 GT：

```
Î_k = A^GT ⊙ ReLU(Σ c_ki Y_i(n^GT))
```

| 指标（124 场景均值） | 数值 |
|---|---|
| PSNR | **14.88 dB**（P10 = 8.28 dB） |
| MAE | 0.179 ± 0.110 |
| SSIM | 0.821 |

**解释上限只有 ~15 dB。** 这不是实现误差，而是协议渲染器
（SH-2 远场方向光 + 直接漫反射 + ReLU 截断）与数据生成器
（Cycles 近场点光 + 投射阴影 + 间接反弹 + 彩色 diffuse）之间的结构性差距。
含义：

1. **重建 PSNR 的理论下限约在 15 dB 量级**（residual 全关时）；
   任何显著高于此的重建分数必然来自 mismatch 被网络吸收
   （residual/光照/反照率代偿），不能作为分解质量证据。
2. 与近场点光直接漫反射物理模型（用 GT depth 恢复像素世界坐标、
   含 1/d² 与 cos 因子，无阴影无间接光）对比：PSNR 14.97 dB——
   与 SH 协议模型**几乎相同**。结论：**mismatch 不来自"远场方向光近似"，
   而主要来自投射阴影与间接光（SH-2 与直接漫反射都无法表达）**，
   以及 sRGB 编码域 luma 与彩色 diffuse 合成的非线性。

## 2. 实验 A' · 训练域灾难（双重 gamma 错位）

按旧管线 `data_loader.py` 的解码约定 `(uint8/255)^(1/2.2)` 得到的"训练域"
中，同一 oracle 的 PSNR 只有 **6.31 dB**（MAE 更高一个量级）。

**这是重大协议错位**：磁盘 PNG 本来就是 sRGB 编码（≈ linear^(1/2.2)），
再取 ^(1/2.2) 等于做了**第二次 gamma 压缩**；而 GT albedo、SH 光照、
物理渲染器全部在**线性域**。旧管线的重建损失在拿线性域渲染输出去匹配
双重压缩的图像目标——**网络被迫用输出幅度去补偿一个固定的非线性畸变**。

- 历史影响：旧 FusionUNet 管线的重建损失、反照率监督、渲染监督之间存在
  系统性域冲突，任何"反照率退化/残差代偿"现象都可能与此有关。
- PRE-0 决议：**新协议全部使用线性域**（`scene_loader.img_lin`，
  精确 sRGB 反变换），训练/评估/数据加载三方一致；旧 `^(1/2.2)` 约定废弃。
- 验证器 `validate_dataset.py` C5 用的恰是 `^2.2`（线性域）且因此通过
  （>12dB）——进一步佐证物理一致域是线性域。

## 3. 实验 B · Normal protocol audit

- `normal.npy` 与"从 `depth.npy` 重算的无边缘感知 Sobel 法线"逐像素对比：
  **平均夹角 0.0065°（P90 = 0.028°）——按构造恒等，实测确认**。
- 因此本数据集**不存在**独立于深度的 mesh/render 法线，"render normal
  vs depth normal" 对照**不可行**（这是数据集定义使然，不是遗漏）；
  生成端注释确认：BlenderProc 2.8.0 normals 合成器输出恒零，
  GT 法线统一采用深度导数定义。
- 风险：Cycles 成像实际使用平滑网格法线，与 Sobel 深度导数法线在曲面
  高频处不同——该差异已并入 §1 的 15 dB mismatch，无法单独剥离（如实声明）。
- 协议冲突项：训练渲染器默认 `use_edge_aware=True`，数据生成是
  `edge_aware=False`。**PRE-0 决议：一切渲染/评估统一 `use_edge_aware=False`**。

## 4. 实验 C · Lighting convention sanity（帧审计）

从 `sh_coeffs.npy` 反解每盏光的隐含方向（球面网格 argmax⟨c,Y(d)⟩，网格分辨率
~2°），与两个期望系对比：

| 对比对象 | 平均角差 | 结论 |
|---|---|---|
| 反解方向 vs **世界系**光方向（生成端语义） | **0.92°** | sh_coeffs 存的是**世界系**方向 |
| 反解方向 vs **相机系**期望（渲染器求值语义） | 69.6° | 渲染器按相机系法线求值 → 每光方向被误解 ~44–70° |

- 即：**存在真实的帧语义错位**——`sh_coeffs` 是世界系（Blender Z-up）系数，
  而 `physics_renderer` 用相机系法线求值，等效于把每盏光的方向误读了
  ~44°（相机与世界夹角）。
- 影响量化：把系数正确旋到相机系后，oracle PSNR 仅从 14.88 → 13.97 dB
  （反而略差）。即**帧错位的影响是二阶的**，总 mismatch 被阴影/间接光
  主导（§1）。但它是必须钉死的语义地雷：任何依赖光照方向真值的
  分析（如 diversity 统计、光照误差评估）必须先声明参照系。
- PRE-0 决议：**光照误差一律在世界系中评估**（生成端语义）；渲染器求值
  保持现协议（相机系法线 × 落盘系数），其帧错位作为已知二阶效应记录。
- 单光独立反演诊断：固定 GT 几何+反照率，从单张图反演 9 维 SH，
  系数幅度漂移 27~1300×——**光照 GT 无法从图像独立验证**，只能由
  协议构造定义（进一步证明重建类指标不能当分解证据）。

## 5. 失败案例

`panel_*.png`（12 场景 × 图像/oracle/残差/掩码四联图）+ `residual_*.png`。
最差场景（PSNR P10 以下）：`12aef634…`、`060c9e87…`、`2dfa3a4d…` 等
（完整列表 `oracle_summary.json.worst_scenes`）。残差空间分布显示：
阴影边界与凹陷区为系统性误差集中区（与 §1 归因一致）。

## 6. 裁决（对应任务书 §3）

> 当前 physics renderer 的不可解释误差有多大、主要来自哪里、
> reconstruction loss 的理论下限大约在哪里？

- 不可解释误差：**线性域 ~15 dB PSNR / MAE 0.18**（全员 GT 情况下）。
- 主要来源（按证据排序）：投射阴影与间接光 ≫ 帧错位（二阶）≈ 编码域非线性。
- 理论下限：residual 全关的重建 PSNR ≤ ~15 dB；**旧管线在双重 gamma
  训练域的下限更是 ~6.3 dB**。
- 是否暂停网络实验？mismatch 存在系统性来源**但均有解释与量化**
  （阴影/间接光/帧/编码），不属于"明显系统误差却没有解释"。
  PRE-03 probe 实验允许进行，但**重建 PSNR 不得作为分解质量指标**，
  一律以 albedo SI-MAE / normal 角误差 / lighting 误差 + PRE-05
  held-out relighting 为准。
