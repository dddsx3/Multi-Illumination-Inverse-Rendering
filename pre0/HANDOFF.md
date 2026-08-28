# PRE-0 HANDOFF · 前置证据获取任务交付（2026-08-29）

> 交付 commit：见仓库最新 commit（本文件所在版本）。
> 执行：ZCode agent · 单机 RTX 5070 Ti Laptop 12GB · PyTorch 2.12/CUDA。
> 一句话总裁决：**数据灾难 Gate B FAIL——synthetic_v3 的"多光照"维度无效
> （5 张图 = 同一图像），一切 evidence accumulation 主故事暂停，先修数据。**
> 协议/估计器/评估器/文献地图已就绪且经实跑验证；解析补光域给出了
> "数据修好后预期成立"的参考信息量结论。

---

## 十二问逐答

### Q1 Synthetic renderer 与训练 physics renderer 是否一致？差异是多少？

**不一致，且另有数据级灾难。** 量化（`oracle_renderer/ORACLE_AUDIT.md`）：

- 线性域 GT oracle（全员 GT 走协议渲染器）PSNR **14.88 dB** / MAE 0.179 /
  SSIM 0.82（124 test 场景）→ 重建 loss 的理论下限 ~15 dB；
- 近场点光直接漫反射物理模型同口径 14.97 dB → mismatch 主因是**投射阴影
  与间接光**（两种模型都表达不了），不是远场近似本身；
- 旧训练管线解码域 `^(1/2.2)` 为**双重 gamma**：同口径 oracle 仅 **6.31 dB**
  ——旧管线重建目标域与物理域系统性错位（PRE-0 已废除此约定，全链路改线性域）；
- 帧语义：`sh_coeffs` 为**世界系**方向，渲染器按**相机系**法线求值，
  每光方向被误解 ~44°（影响二阶，已钉死记录）；
- **数据灾难**：5 张"不同光照"图实为同一图像（±1~2 灰阶噪声，
  见 Q4）。实际生效光照不明（light-5 oracle 略优 16.08 vs 14.5 dB）。

### Q2 depth 与 normal 的物理定义分别是什么？

见 `protocol/DATASET_CONTRACT.md`（逐项物理意义）。要点：depth = 视空间 z
（沿视线距离，世界单位，场景最长边归一 1.6）；normal = **深度导数定义**
（无边缘感知 Sobel，n = normalize([−∂z/∂x, −∂z/∂y, 1])，相机系，z>0 面朝相机），
**不是** mesh 法线；训练渲染器默认 edge_aware=True 与数据定义冲突，
PRE-0 起统一 False。

### Q3 depth-derived normal 与 render/mesh normal 是否一致？

**无法对照——数据集中不存在独立的 mesh/render normal。**
`normal.npy` 本身就是深度导数（生成端 BlenderProc normals 输出恒零，
改用深度导数，`render_dataset.py` 注释确认）。实测 normal.npy vs 重算
Sobel：平均夹角 0.0065°（按构造恒等）。真实网格法线对照需要重渲
（自定义 AOV），列入数据修复清单。

### Q4 没有神经网络时，增加 N 是否改善 factor recovery？

**真实数据上：N 不提供任何信息——因为 5 张图是同一张图。**

- 直接证据：抽检 20+ 场景，`light_001..005.png` 两两平均灰差
  0.000~0.049/255（99.9% 像素逐位相同，差异 = Cycles 采样噪声）；
  DiLiGenT 真实换光同口径 3.8/255。`sh_coeffs` 5 行确实不同
  （生成端参数正确、渲染帧动画失效）。
- 后果：真实域 GT 几何下 A/L 联合恢复（PRE-02 exp1）对 N∈{1,2,3,5}
  的结果逐位相同；novel-vs-duplicate 的 Δ_new = Δ_dup = 0（精确成立）；
  probe 的 N 曲线对 N 完全平坦（Q7）。
- **解析补光域（协议模型重打光，数据修复后的参考预期）**：
  修复 N 曲线 SI-MAE(A) 从 N=1 的 ~0.058 降至 N=5 的 ~0.017（3.4×，
  bootstrap CI 见 `information_audit/`），证明协议模型与估计器本身
  能从真多光照数据中提取信息——瓶颈确证在数据，不在方法。

### Q5 diversity 是否比 N 本身更能解释误差？

在解析域（唯一有效域）中：误差随 N 下降主要由子集覆盖的**光照方向
spread**驱动；固定 N=5 环形子集（real 语义）spread 恒定。解析 15 光的
等距子集内，N 与 spread 高度共线，PRE-0 规模下无法把二者统计分离
（数据修复后在真实 Cycles 图像上重做才能回答）。诚实结论：**未解决，
需要正确数据**。工具已就绪（`diversity_results.csv`：angular spread /
covariance 谱 / κ(Σ A²YYᵀ) 逐场景落盘）。

### Q6 novel illumination 是否比 duplicate 提供更多收益？

- 真实数据：Δ_new = Δ_dup = 0（同一张图，问题不存在）。
- 解析域（噪声 σ=0.005）：Δ_new ≈ +0.001~0.010 > Δ_dup ≈ 0
  （`novel_vs_duplicate.csv`、图 `novel_vs_duplicate.png`）；
  量级小是因为解析域无噪声时双线性恢复本已近精确——**该实验的
  区分力要在真实噪声+失配下才有意义**，即需修复后的数据重做。

### Q7 三个 Probe 中哪一个 N curve 最合理？

**三者全部平坦（N∈{1,2,3,5} 输出逐位不变）——在当前数据上这是
"理性"行为：数据里多图无信息，均值聚合网络学到忽略冗余输入。**
跨输入直接验证：同一场景只换第 1 张图，三个 probe 输出几乎不变
（encoder 特征跨 5 图 std ≈ 1e-6）；跨场景输出有微弱差异
（albedo 跨场景均差 0.0036 vs GT 0.13）——probe 退化为
"近常数 + 弱场景调制"。因此 **Q7 无法回答，Gate C 无从评估**
（不是 attention 能解决的，是数据问题）。

| Probe（best ckpt，test 124 场景） | SI-MAE(A) | depth L1 | normal 角 | recon PSNR |
|---|---|---|---|---|
| A MeanSpatial | 0.0545 | 0.298 | 9.56° | 8.53 dB |
| B MeanVarSpatial | 0.0545 | 0.331 | 9.55° | 8.53 dB |
| C GlobalSet | 0.0547 | 0.292 | 9.55° | 10.90 dB |

（三者 SI-MAE 几乎相同：当前监督结构下方差特征与全局聚合都不带来
材质增益——同为数据缺陷下的产物，仅作记录。）

### Q8 spatial aggregation 是否明显优于 global aggregation？

**没有明显差异**（上表：A/B/C test SI-MAE 差 <0.001）。在此数据与
预算（~0.71M 参数 × 40 epoch，统一不调参）下无法区分——与 Q7 同源。
这也意味着：**旧项目"spatial fusion 优于 global"类结论在修复数据前
都不可信**。

### Q9 不同 illumination subsets 恢复的 canonical albedo/geometry 是否一致？

完全一致（D_albedo=0.0000，D_normal=0.00°）——但这是数据缺陷的
产物（子集内容相同）。协议本身（cross-subset 一致性检验，
`cross_subset.csv`）已验证可运行，待正确数据后此检验才有分辨力。

### Q10 oracle-light held-out relighting 是否随 N 改善？

不随 N 改善（HO-PSNR 对 N 平坦：A/B ~15.5 dB，C ~16.9 dB）——
**在本数据上 held-out 是伪命题**（query 光 = support 光 = 同一图像；
评估器提供的 L_q^GT 与 support 完全相关）。PRE-05 协议实现
（oracle-query-light，residual 全关，`heldout_relighting/`）可运行，
待正确数据后重测。C 的 HO-PSNR 略高值得注意（global 聚合在
单光退化场景更稳），仅作记录。

### Q11 最接近本项目的 5–10 篇工作是什么？

见 `literature/CLOSEST_PRIOR_WORK.md`（43 篇矩阵
`literature/literature_matrix.csv`）。最危险三篇：
1. **UPS-GC（CVPR22）**：未知光 + 可变 N + 单相机 + DiLiGenT——但
   normal-only，无 albedo/depth/SH 分解与 relighting 协议；
2. **universal PS 系（SDM-UniPS CVPR23 / Light of Normals ICLR26）**：
   可变基数+未知光已成人性天花板，全部 normal-only；
3. **SCPS-NIR（ECCV22）**：uncalibrated 下 albedo+normal+光照联合分解
   （自监督），缺 depth、缺可变 N 研究、缺可查询光照表示。

### Q12 根据现有证据，最值得继续验证的三个方法假设是什么？

（只写假设，不宣布主架构；且全部以**修复数据**为前提）
1. **假设 H1（证据累积可学性）**：在真实多光照合成数据上，
   spatial mean 聚合 + per-light SH 头的最小模型即可呈现
   Δ_new > Δ_dup 与 cross-subset 一致性随 N 收敛——若最小模型
   仍不能，问题在融合位置/监督方式，而非容量。
2. **假设 H2（光照表示的可查询性）**：per-light 低维 SH（可证伪、
   可旋转、可查询）作为光照表示，配合 oracle-query-light held-out，
   比 implicit/全局光照更能把"分解质量"与"重泛化能力"解耦。
3. **假设 H3（域一致性优先）**：修复数据时同步统一生成端与训练端
   （线性域 + 帧一致 + edge_aware=False），预期在**不引入任何新模块**
   的前提下显著缩小 GT-oracle 与可学习上限的差距（当前 15 dB 天花板
   大部分来自阴影/间接光，需评估 residual 的必要性而非默认引入）。

---

## 总门禁裁决（任务书 §12）

| Gate | 裁决 | 依据 |
|---|---|---|
| A Physics Validity | **有条件 PASS** | 语义冲突全部量化并钉死（Q1/Q2/Q3）；但"15dB 天花板"意味着当前协议渲染器只能支撑受控分析，不足以直接当最终训练渲染器 |
| B Information Validity | **FAIL** | 多光照信息在数据中不存在（Q4/Q6）；**暂停 evidence accumulation 主故事，先修数据** |
| C Learnability | **BLOCKED（随 B）** | 三 probe 行为正确但无分辨力（Q7/Q8/Q9/Q10） |
| D Novelty Viability | **PASS（有条件）** | 文献地图（Q11）确认存在可实验验证的 formulation 级空白；防守条件见 CLOSEST_PRIOR_WORK.md |

**下一步（优先级序）**：
1. 修复 `render_dataset.py` 光照帧动画（bproc `Light.set_location(…, frame=k)`
   验证逐帧生效）+ 新增生成端门禁：跨帧结构差异下限、sh 系数-图像方向一致性；
2. 重渲 synthetic_v3（或 v4）→ 重跑 PRE-01（预期 oracle PSNR 结构改变）→
   重跑 PRE-02 真实域（N 曲线/Δ_new 应呈现真信息）；
3. 数据就位后再谈主模型架构（任务书 §0 纪律不变）。

## 证据索引（对应任务书 §10 目录结构）

| 任务书要求 | 位置 |
|---|---|
| pre0_protocol.yaml / DATASET_CONTRACT.md / split_manifest | `pre0/protocol/` |
| oracle_renderer/ | `pre0/oracle_renderer/`（ORACLE_AUDIT.md = 最重要的单文件） |
| information_audit/ | `pre0/information_audit/` |
| probe_results/ + checkpoints/ + logs/ | `pre0/probe_results/`、`pre0/checkpoints/`、`pre0/logs/` |
| evidence_accumulation/ | `pre0/evidence_accumulation/` |
| heldout_relighting/ | `pre0/heldout_relighting/` |
| literature/ | `pre0/literature/` |
| benchmark/ | `pre0/benchmark/`（合同+260 固定子集+评估器） |
| source/ | `pre0/source/`（dataset/renderer/probe_models/train/evaluate） |
| 环境依赖 | 根 `requirements.txt`（本机另需 scipy；torch 2.12+cu128 实跑通过） |
