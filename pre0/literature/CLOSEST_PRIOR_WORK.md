<!--
```yaml
protocol_status: superseded
valid_for_multi_illumination_claims: false
reason: duplicated illumination frames
supersedes: synthetic_v3 (commit 2c23026, before 2026-08-29)
verdict: docs/verdicts/PRE0_VERDICT.md
next_valid_artifact: p1/calibration_set/, p1/physics_clean/
```
-->

# CLOSEST PRIOR WORK — 最接近工作与 Novelty 风险地图（PRE-06）

配套文件：`literature_matrix.csv`（43 行文献矩阵）。本文件回答 PRE-0 Gate D 的核心问题：
**给定本项目的问题设定（固定单相机单视角；每场景 N 张不同光照图像，N 可变 1~15；光照未知不校准；输出联合分解 canonical 标量 albedo + depth/normal + 每光二阶球谐 9 系数；permutation-invariant mean 聚合；合成 256×256 灰度训练；DiLiGenT zero-shot 评估），哪 5~10 篇工作最可能被审稿人用来攻击 novelty，以及我们有什么客观证据反驳。**

引用核实说明（先说清三处"任务书引用可疑"的地方，避免后面被抓）：
- **"Neural Photometric Stereo (Kaya 2018)" 不存在**：多轮检索（arXiv/ECCV/CVF/Semantic Scholar）未找到 Kaya 于 2018 年发表的名为 "Neural Photometric Stereo" 的论文。Kaya 的相关工作实为 CVPR 2021 "Uncalibrated Neural Inverse Rendering for Photometric Stereo of General Surfaces"（arXiv:2012.06777）；2018 年名为 "Neural Photometric Stereo Reconstruction..." 的是 Taniai & Maehara 的 arXiv 预印本，正式发表为 ICML 2018 "Neural Inverse Rendering for General Reflectance Photometric Stereo"。两者均已收录进矩阵。
- **"SGRNet" 不是光度立体方法**：SGRNet = "Shadow Generation for Composite Image in Real-world Scenes"（Hong et al., AAAI 2022, arXiv:2104.10338），是阴影生成网络。未找到任何名为 SGR-Net 的 PS 论文（已从 ICLR 2026 universal PS 论文参考文献反查确认不存在）。
- **"Light-Net" 无法核实**：未找到任何以 Light-Net 命名的 uncalibrated PS 论文；相关字段在 CSV 中标注 uncertain，不做编造。

---

## 1. PS-FCN（Chen, Han, Wong · ECCV 2018 · arXiv:1807.08696）—— 可变 N 聚合的鼻祖

**它做了什么**：深度全卷积网络，输入任意数量（测试时 1~32）固定相机、已知光方向 illuminations 的图像，逐图特征提取后经 **max-pooling** 顺序不变聚合，输出 per-pixel normal。训练用 100 个合成物体；在 DiLiGenT 上评估。

**审稿人可能的攻击**："你们说的 variable-N 证据累积，PS-FCN 八年前就做了，连 'arbitrary number of input images' 的措辞都一样。"

**客观反驳证据（formulation 级）**：
1. **校准假设**：PS-FCN 每张输入图必须附带 **已校准的 per-image 光方向**。本项目光照完全未知、不校准，且**没有每光的已知方向向量可供输入**——这在输入空间上与 PS-FCN 不同域（PS-FCN 的 input 是 (image, light-dir) 对，我们是 (image) 集合）。
2. **输出因子**：PS-FCN 只输出 normal。本项目联合输出 canonical albedo + depth/normal + **每光 9 维 SH**，且每光 SH 是显式输出（可用 held-out 光做可检验预测）。
3. **cross-N 泛化是被"测试手段"而非"研究对象"**：PS-FCN 用 max-pool 使 N 可变只是工程灵活性，论文没有研究"误差如何随 N 下降"的证据累积曲线，也没有 N=1 的退化情形分析；我们把 N∈[1,15] 的 cross-N 行为本身作为实验对象与贡献。
4. **灰度/RGB 域**：PS-FCN 为 RGB 合成训练；我们 256×256 灰度 + DiLiGenT zero-shot。

**剩余暴露面**：max-pool vs mean 聚合的对比是审稿人会要求的第一基线；我们必须把 PS-FCN（用其训练协议或我们的合成数据重训）作为"可变 N + 校准上界"基线，否则差异只是校准假设的差异。

**Novelty 风险：高**

---

## 2. UPS-GC（Ikehata · CVPR 2022 · arXiv:2206.02452）—— "Universal PS"：未知光 + 可变 N + 单相机

**它做了什么**：提出 universal photometric stereo 任务：仅需 shading 图像 + mask，**光照任意未知**，**每物体图像数量/顺序任意**，通过 global lighting context 池化聚合，输出 normal；合成+真实混合训练，DiLiGenT 评估。

**审稿人可能的攻击**："uncalibrated + variable-N + 单相机 + DiLiGenT 评估，你们的四要素它占了三个半。"

**客观反驳证据**：
1. **输出因子（最硬的一条）**：UPS-GC **只输出 normal**。它既不输出 albedo，也不输出任何 per-image 光照参数（lighting context 只是内部表征，不是可检验的 9 维 SH 预测）。因此它**无法做 oracle-query-light 的 held-out relighting 协议**——我们的第四个输出因子构成了它结构上缺失的预测目标。
2. **无几何输出**：没有 depth/3D 因子，无法做联合几何-材质-光照分解的误差链分析（如 albedo 错误对 SH 的耦合）。
3. **无 joint decomposition 的 ground-truth 监督**：它没有 per-factor 损失（albedo/lighting/geometry 各自的 GT 误差报告）；我们的 formulation 每个因子都可单独定量评估。
4. **物理分解假设不同**：UPS-GC 明确"不对物理光照模型做假设"（不建模 directional light），因此它与二阶 SH 辐射度假设不同——SH 假设是可证伪的物理设定，报告 per-light 方向/强度误差即是证据。

**剩余暴露面**：如果我们的 normal-only 消融（去掉 albedo/SH 头）在 DiLiGenT 上打不过 UPS-GC/SDM-UniPS，攻击成立。必须给出 normal 通道可比数字。

**Novelty 风险：高（当前最危险）**

---

## 3. SDM-UniPS（Ikehata · CVPR 2023）与 Light of Normals（ICLR 2026 · arXiv:2506.18882）—— universal PS 的现状天花板

**它们做了什么**：SDM-UniPS 把 universal PS 推到 mask-free、可变输入数、高分辨率、未知任意光照（transformer 聚合）；Light of Normals（ICLR 2026）用统一特征表示处理"任意视角×光照数"输入，在 441 个 held-out 真实场景上评估。两者都只输出 normal。

**审稿人可能的攻击**："2026 年的 universal PS 已经把 unknown lighting + arbitrary cardinality + 真实数据 held-out 评估做完了，你们只是加了几个输出头。"

**客观反驳证据**：
1. **"加输出头"恰恰是可证伪的 formulation 差异**：per-light 9 维 SH 输出可以被直接证伪（拿 DiLiGenT 已知光方向当 GT 算 per-light 方向误差），而 universal PS 系论文没有这个输出、也没有这类评估。这不是"它没用某模块"，而是"它的 problem statement 不含这个可检验预测"。
2. **联合分解的耦合约束**：我们在共享 latent 上同时受 albedo、geometry、per-light SH 三个因子的监督约束，与 normal-only 模型的解空间不同；可通过"去掉任一因子头导致其余因子误差上升"的消融给出因果证据。
3. **协议差异**：它们没有 oracle-query-light relighting 协议（用估计的光照渲染 held-out 光下图像并算 relighting 误差）。这是新的评估贡献，与模型贡献独立成立。
4. **数据域**：它们大规模混合真实+合成；我们刻意用灰度 256 合成并在真实基准 zero-shot，研究的是"信息量受限的可变 N 证据累积"，评估维度不同（cross-N 曲线）。

**剩余暴露面**：Light of Normals 是 ICLR 2026、时间上极新，写作时必须显式引用并逐条对比；若只字不提会被视为漏洞。

**Novelty 风险：高**

---

## 4. SCPS-NIR：Self-calibrating Photometric Stereo by Neural Inverse Rendering（Li et al. · ECCV 2022）

**它做了什么**：自监督神经逆渲染做 uncalibrated PS：显式建模 **per-image 光照 + 法线 + 漫反射 albedo + 高光残差**，DiLiGenT 评估。

**审稿人可能的攻击**："联合分解（albedo+normal+光照，光照未知）它已经做了，你们只是把 SH 写得更显式。"

**客观反驳证据**：
1. **可变基数未被研究**：SCPS-NIR 没有把 N 的可变性/1~15 的 cross-N 泛化作为研究对象，没有证据累积分析；其协议固定在常规 DiLiGenT 全集输入上。
2. **无几何（depth）因子**：其分解是 2.5D 的（normal + 反照率 + 光 + 高光），没有 depth 输出，因此不是完整的三因子联合分解。
3. **光照表示与可检验性**：其光照是低维方向/强度或残差图，不是每光 9 维 SH 的显式可渲染表示；held-out 光 relighting 协议不可直接套用。
4. **监督来源不同**：SCPS-NIR 主要靠自监督重建损失 + 高光先验，per-factor GT 误差（albedo/lighting）报告有限；我们用全 GT 因子监督并分别定量。

**剩余暴露面**：若审稿人要求 SCPS-NIR 基线对比 albedo/lighting 误差，必须能复现或引用其报告数字。

**Novelty 风险：高**

---

## 5. SDPS-Net（Chen et al. · CVPR 2019 · arXiv:1903.07366）

**它做了什么**：uncalibrated deep PS 双网络：LCNet 逐图回归光方向+强度，NENet 用估计光照回归法线；合成 5.4M 图训练，DiLiGenT 评估。

**审稿人可能的攻击**："未知光的逐图估计（light + normal）它 2019 年就做了，你们只是把方向向量换成 SH。"

**客观反驳证据**：
1. **无 albedo 输出、无联合分解**：SDPS-Net 的分解只到 (light, normal)，albedo 被吸收进网络先验里，没有 canonical albedo 因子与相应的 GT 监督。
2. **SH 与 (dir, intensity) 不是同构替换**：二阶 SH 表达环境光/软光与部分非定向能量，能渲染 held-out 光照图；SDPS-Net 的输出无法渲染任意查询光下的图像——oracle 协议对它不成立。
3. **可变 N 非其研究对象**：它测试时 N 可变（结构上支持），但论文没有 cross-N 泛化或 N 极限情形的系统研究。
4. **无 geometry（depth）输出**。

**剩余暴露面**：SDPS-Net 是必引必比基线；我们的 NENet-normal 通道必须与之可比（zero-shot 协议下要说明训练域差异）。

**Novelty 风险：中-高**

---

## 6. PS-Transformer（Ikehata · BMVC 2021 · arXiv:2211.11386）

**它做了什么**：标题即 "Learning **Sparse** Photometric Stereo Network using Self-Attention"：用可学习自注意力在稀疏（少量）校准光照图像集合上聚合，输出法线。

**审稿人可能的攻击**："'sparse photometric stereo + 集合注意力聚合'两个词都在它的标题里。"

**客观反驳证据**：
1. **校准假设**：PS-Transformer 输入包含**已知光方向**（calibrated sparse PS）；我们的光照未知。
2. **输出因子**：仅 normal；无 albedo/lighting/depth，无 relighting 能力。
3. **聚合与方差分析**：它没有研究聚合因子的统计性质（permutation invariance 的形式化、N 变化时的误差-样本复杂度曲线）。
4. **任务定义**：它是"用更少输入恢复同质量 normal"的效率研究，不是"未校准条件下随 N 增长的证据累积 + 联合因子恢复"。

**剩余暴露面**：注意术语——论文里不要独占 "sparse photometric stereo" 一词而不引 PS-Transformer / SPLINE-Net，否则被视为文献疏漏。

**Novelty 风险：中**

---

## 7. MIRR：Neural Inverse Rendering for General Reflectance Photometric Stereo（Taniai & Maehara · ICML 2018 · arXiv:1802.10328）

**它做了什么**：CNN 自编码器（normal→illumination 一致性），输入任意 N 张未知**空间变化光照**图像，联合估计法线与每图光照图；常以 arXiv 旧题 "Neural Photometric Stereo Reconstruction for General Reflectance Surfaces" 被误引。

**审稿人可能的攻击**："未知光照 + 可变 N + 联合（法线+光照）估计，2018 年就齐了。"

**客观反驳证据**：
1. **albedo/depth 因子缺失**：MIRR 不输出 canonical albedo（其反照率假设隐含/未单独建模），无 depth 输出。
2. **光照是每图 2D 图而非低维可查询表示**：其 per-image illumination map 无法在"查询一个新光"的意义下外推——oracle-query-light 协议对它不成立；9 维 SH 是有限维、可预测、可证伪的表示。
3. **评估规模与协议**：小规模实验室数据，无 DiLiGenT zero-shot、无 cross-N 曲线。
4. **几何完整性**：仅 normal，无 depth/3D 一致性监督。

**剩余暴露面**：其"unknown spatially-varying lighting"比我们的假设更宽（我们假设可被 9 维 SH 近似），需在 limitation 中明确这是"更强先验换取可检验性"的 trade-off。

**Novelty 风险：中**

---

## 8. S³-NeRF：Neural Reflectance Field from Shading and Shadow under a Single Viewpoint（Yang et al. · NeurIPS 2022 · arXiv:2210.08936）

**它做了什么**：单视角、多张变化点光源图像 → NeRF 式联合重建几何与反射场（利用 shading+shadow 线索），DiLiGenT 等数据评估。

**审稿人可能的攻击**："单相机单视角 + 多光照 + 联合几何/反射恢复，它就是你们问题的 NeRF 版。"

**客观反驳证据**：
1. **逐场景优化 vs 前馈集合网络**：S³-NeRF 是 per-scene optimization（每场景训练一个场），无跨场景泛化、无 zero-shot 评估主张；我们是单次前馈推理 + 跨场景泛化（DiLiGenT zero-shot）。
2. **光照模型**：其为已知（或低自由度）点光设定（光参数非 9 维 SH 显式输出；具体校准程度论文设定需在复现时确认，CSV 标 uncertain），无 per-light SH 预测与 held-out 光协议。
3. **无 canonical albedo 因子与灰度域**：输出为反射场（BRDF 场），非标量 albedo + 光照的显式分解。
4. **无 permutation-invariant 集合结构**：输入顺序/N 变化不是其研究对象。

**剩余暴露面**：若审稿人把"前馈 vs 优化"视为工程差异，需要用"同 budget 下 N=1~15 的 cross-N 曲线 + 速度/泛化"来支撑，而不是只主张架构不同。

**Novelty 风险：中**

---

## 9. Joint Material and Illumination Estimation from Photo Sets in the Wild（Wang, Ritschel, Mitra · 3DV 2018 · arXiv:1710.08313）

**它做了什么**：对**一组野外照片**联合估计材质（含 diffuse/specular）与锐利光照：CNN 提供初值 + 优化精修。

**审稿人可能的攻击**："photo set → joint material + lighting（光照未知），你的问题陈述的早期版本。"

**客观反驳证据**：
1. **无几何因子**：不输出 normal/depth；分解只有 (material, lighting) 两因子，且面向"物体材质分类/重打光"，非 PS 式逐像素法线。
2. **非端到端集合模型**：CNN 仅做初始化，主体是逐场景优化；无 permutation-invariant 前馈集合结构、无跨场景泛化。
3. **无可控 N 的证据累积**：photo set 数量与质量不可控（网络图），无 N∈[1,K] 受控实验。
4. **数据域**：野外互联网照片，无合成监督 + 真实基准 zero-shot 的受控协议。

**剩余暴露面**：低引用量工作，但问题陈述相似度高，必须在 related work 中点名对比，避免"未审到 3DV 2018"的批评。

**Novelty 风险：中**

---

## 10. 单图像联合分解参照系：Sengupta ICCV 2019 与 NVIDIA Indoor IR（ICCV 2021）

**它们做了什么**：Sengupta et al.（IRN，ICCV 2019）单张室内图自监督分解 albedo+normal+环境光；Li et al.（NVIDIA, ICCV 2021）单图输出 albedo+depth+normal+3D 空变光照（含 9 系数量级的低阶光照表示传统）。

**审稿人可能的攻击**："joint albedo+normal+depth+lighting 的因子清单它们都有，你们 N>1 而已。"

**客观反驳证据**：
1. **N=1 是我们 problem 的退化点而非主结果**：它们的输入没有多光照证据，光照不确定性只能靠先验正则；我们在 N≥2 时利用光照间变化作监督信号（同一 albedo/geometry、不同 per-light SH），这是可实验验证的信息来源差异（N=1 vs N=2 vs N=15 的因子误差下降曲线）。
2. **每光光照而非单环境光**：它们的 lighting 是一张环境贴图/一个 3D 光照体积（场景级一次曝光）；我们输出 N 个 per-light SH——因子数量随输入增长，这是不同的预测目标结构。
3. **协议**：室内 RGB 场景数据，非物体级 PS 基准；无 DiLiGenT zero-shot。

**Novelty 风险：中-低**（作为"联合分解不新鲜"的引用弹药，需逐因子对比表回应）

---

## 11. 协议/数据重叠（提级防守）：Murmann ICCV 2019 与 OpenIllumination NeurIPS 2023

- **Murmann et al., ICCV 2019**（MIT Multi-Illumination，1000+ 真实场景 × 25 定向光）与 **OpenIllumination**（NeurIPS 2023 D&B，64 真实物体、108K 图、点光 + GT 几何/材质）证明"多光照采集 + 逆渲染评估"是活跃协议。任务书中的 "MiLCD" 名称无法独立核实，可核实条目即 Murmann ICCV 2019 数据集。
- **反驳点**：它们是数据集/协议工作，不提供可变 N 前馈模型；我们可把两者列为补充真实评估（OpenIllumination 的校准点光还正好可用于检验我们的 per-light SH 预测），反而强化而非削弱 novelty。
- **Novelty 风险：低-中**（协议先例存在 → 主张措辞要写成"协议的系统化 + 模型"，不是"首创多光照评估"）。

---

## Novelty 风险评级汇总与 Gate D 判断

| # | 工作 | 年份/Venue | 核心重叠轴 | 缺失轴（我们的差异立足点） | 风险 |
|---|------|-----------|-----------|--------------------------|------|
| 1 | PS-FCN | ECCV 2018 | 可变 N 聚合、DiLiGenT | 校准假设、联合分解、证据累积研究、relighting | 高 |
| 2 | UPS-GC | CVPR 2022 | 未知光 + 可变 N + 单相机 + DiLiGenT | 仅 normal；无 albedo/depth/per-light SH；无 relighting 协议 | 高（最危险） |
| 3 | SDM-UniPS / Light of Normals | CVPR 2023 / ICLR 2026 | universal PS 天花板、held-out 真实评估 | 输出因子与可证伪的 SH 预测、联合监督、cross-N 曲线 | 高 |
| 4 | SCPS-NIR | ECCV 2022 | 未知光联合分解（albedo+normal+light） | 无 depth、无可变 N 研究、SH 可查询表示、held-out 光协议 | 高 |
| 5 | SDPS-Net | CVPR 2019 | 未知光逐图估计 + normal | 无 albedo/depth、(dir,int)≠SH、无 relighting | 中-高 |
| 6 | PS-Transformer | BMVC 2021 | "sparse PS" 术语、集合注意力 | 校准、仅 normal、无聚合统计研究 | 中 |
| 7 | MIRR (Taniai) | ICML 2018 | 未知空间变光 + 可变 N + 联合(法线+光) | 无 albedo/depth、光照不可查询、无受控基准 | 中 |
| 8 | S³-NeRF | NeurIPS 2022 | 单视角多光联合几何/反射 | per-scene 优化 vs 前馈泛化、无 SH/协议 | 中 |
| 9 | Wang 3DV 2018 | 3DV 2018 | photo set 联合材质+光照 | 无几何因子、优化式、N 不可控 | 中 |
| 10 | Sengupta 2019 / NVIDIA 2021 | ICCV 2019/2021 | 联合因子清单（单图） | per-light 多光照证据、N 退化分析 | 中-低 |
| 11 | Murmann 2019 / OpenIllumination 2023 | 数据集 | 多光照评估协议先例 | 非模型、无可变 N 前馈 | 低-中 |

**Gate D 判断：当前 formulation 仍留有清晰、可实验验证的差异，但窗口在收窄。**

可实验验证的差异（每条都有明确指标，不依赖架构命名）：
1. **per-light 9 维 SH 的可证伪预测**：在 DiLiGenT（真实已知光方向 GT）上报告 per-light 方向/强度误差与 held-out 光 relighting 误差（oracle-query-light 协议）。现有 4 篇高危工作（UPS-GC/SDM-UniPS/Light of Normals/PS-Transformer）**均无此输出与指标**。
2. **cross-N 证据累积曲线**：同一模型在 N∈{1,2,3,5,8,15} 的因子误差（albedo/normal/depth/lighting）曲线 + N=1 退化一致性；没有任何一篇近作把它作为研究对象。
3. **联合分解的耦合证据**：去掉任一输出头导致其余因子误差上升的消融；高危工作中只有 SCPS-NIR 接近（缺 depth）。
4. **zero-shot 域迁移**：灰度 256 合成训练 → DiLiGenT 真实 zero-shot，报 normal + albedo + lighting 三因子误差。

必要防守动作（不完成则风险上探为"高/拒绝"）：
- 相关工作必须点名并逐条对比 UPS-GC、SDM-UniPS、Light of Normals（ICLR 2026）、PS-Transformer、SCPS-NIR、Geometry Meets Light（AAAI 2026，"limited multi-illumination cues"与我们 sparse 端直接相邻）；术语上避免独占 "sparse photometric stereo"。
- 至少把 UPS-GC（或其开源权重）与 SDPS-Net 作为 zero-shot 基线复现在同一 DiLiGenT 协议上；normal 通道必须可比。
- 在写作中把贡献定位为"**可变 N 证据累积下的联合物理分解 + 可查询光照协议**"这一组合 formulation，而非任何单一组件（组件层面每一条都已有先例：可变 N→PS-FCN；未知光→SDPS-Net/UPS-GC；联合分解→SCPS-NIR/Taniai；注意力集合→PS-Transformer）。

**结论**：三个最危险对手为 **UPS-GC（CVPR 2022）**、**SDM-UniPS/Light of Normals（CVPR 2023/ICLR 2026，universal PS 系）**、**SCPS-NIR（ECCV 2022）**。它们分别覆盖了"可变 N + 未知光"与"未知光联合分解"，但没有任何已核实工作同时给出：(a) 可变基数作为研究对象（cross-N 曲线）、(b) 含 depth 的三因子联合监督、(c) per-light 低维可查询光照（SH）及 held-out 光 relighting 评估。本 formulation 的 PRE-0 Gate D 判定：**通过（有条件）**——前提是上述四项实验全部落地并在文中与 11 项风险工作逐条对表。
