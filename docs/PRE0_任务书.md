# Multi-Illumination Inverse Rendering
# 前置证据获取任务书 PRE-0 v1.0

**状态**：立即生效  
**性质**：研究方向决策前置任务，不属于正式论文消融实验  
**目标**：在重新设计主模型以前，以最低技术债获取足够证据，回答“问题是否成立、物理协议是否成立、信息来自哪里、最接近的已有工作是谁、什么架构方向值得下注”。

---

## 0. 总原则

本阶段**不以提高指标为目标**，不追求 SOTA，不训练最终模型。

唯一目标是消除五类不确定性：

1. 数据与物理模型是否自洽；
2. 多增加一张不同光照图片是否真的增加了可利用的信息；
3. N 增大带来的收益究竟来自新增光度证据，还是简单的统计平均；
4. 最简单的 permutation-invariant 模型是否已经足够；
5. 文献中是否已经存在与本项目核心设定实质等价的工作。

只有完成本任务书以后，才允许确定新的主模型架构。

---

# 1. 本阶段禁止事项

在 PRE-0 验收以前：

- 禁止恢复旧 FusionUNet 作为默认主模型；
- 禁止加入 FiLM、Set Transformer、PMA、appearance residual、global/local residual 等复杂模块作为正式设计；
- 禁止进行 100 epoch 级正式 arm sweep；
- 禁止做多 seed 正式训练；
- 禁止以 reconstruction PSNR 单独判断分解质量；
- 禁止使用旧的 “N=1～5 几乎不变 = 鲁棒” 结论；
- 禁止使用旧 checkpoint 证明新方法有效；
- 禁止提前写“我们的创新是 Set Transformer / 任意 N”等论文结论。

旧代码允许作为参考实现，但不得默认继承旧语义。

---

# 2. PRE-00 · 建立唯一实验协议与最小仓库

## 目标

保证以后产生的任何实验结果都能回答：

> 它到底在哪个数据、renderer、normal 定义、split、N 分布和代码版本下产生？

## 必做

建立：

`protocol/pre0_protocol.yaml`

至少包含：

```yaml
protocol_id:
git_commit:
random_seed:

image:
  modality:
  resolution:
  linear_or_srgb:
  normalization:

camera:
  model:
  intrinsics:
  coordinate_system:

depth:
  definition:
  unit:
  valid_range:
  normalization:

normal:
  definition:
  coordinate_system:
  source:

albedo:
  definition:
  valid_range:
  color_space:

lighting:
  representation:
  sh_order:
  coordinate_system:
  calibrated_in_training:
  calibrated_in_evaluation:

dataset:
  dataset_id:
  scene_split_hash:
  lights_per_scene:

renderer:
  renderer_id:
  equation:
```

同时建立：

`docs/DATASET_CONTRACT.md`

逐项解释每个量的**物理意义，而不只是 tensor shape**。

## 交付

- `pre0_protocol.yaml`
- `DATASET_CONTRACT.md`
- canonical train/val/test scene ID
- split hash
- 环境依赖文件
- 最小仓库目录树

## PASS 条件

不存在：

- depth 单位未知；
- normal 坐标系未知；
- RGB 是否 linear 未知；
- SH 坐标系未知；
- albedo 值域未知；
- train/test scene 重叠未知。

有任何一项“不确定”，PRE-00 不通过。

---

# 3. PRE-01 · Ground-truth 物理一致性审计

这是整个前置阶段优先级最高的实验。

## 核心问题

训练使用的 differentiable renderer 到底能否解释生成数据？

如果连 GT：

\[
A^{GT},\quad n^{GT},\quad L^{GT}
\]

都无法很好解释：

\[
I^{GT},
\]

那么神经网络以后一定会被迫把 renderer mismatch 塞进 albedo、lighting 或 residual。

这种情况下再讨论网络架构没有意义。

## 必做实验 A：GT Oracle Reconstruction

不使用网络。

直接计算：

\[
\hat I_k
=
A^{GT}\odot S(n^{GT},L_k^{GT})
\]

然后比较：

\[
I_k^{GT}
\quad vs\quad
\hat I_k.
\]

逐场景记录：

- MAE
- MSE
- PSNR
- SSIM
- residual mean/std
- residual spatial map

必须保存失败案例。

## 必做实验 B：Normal protocol audit

如果存在 mesh/render normal 与 depth-derived normal 两套定义，比较：

\[
n_\text{render}
\quad vs\quad
n_\text{depth}.
\]

报告：

- angular MAE
- median
- P90
- 每场景分布

如果只有 depth-derived normal，也必须明确说明无法做这项对照。

## 必做实验 C：Lighting convention sanity

选取至少三种容易人工判断的法线：

\[
(0,0,1),\ (1,0,0),\ (0,1,0)
\]

验证 SH lighting 旋转、轴方向和图像亮区是否符合生成器定义。

## 交付

`pre0/oracle_renderer/`

包含：

- `oracle_metrics.csv`
- `oracle_summary.json`
- `normal_protocol.csv`
- 至少 10 个 residual visualization
- `ORACLE_AUDIT.md`

## 裁决

不预设 PSNR 必须达到某个任意阈值。

但必须回答：

> 当前 physics renderer 的不可解释误差有多大、主要来自哪里、以后 reconstruction loss 的理论下限大约在哪里？

如果存在明显系统误差却没有解释：

**暂停所有网络实验。**

---

# 4. PRE-02 · 不依赖神经网络的 N 信息量实验

## 目标

先回答：

> “更多光照有帮助”究竟是数据本身的事实，还是我们希望网络学出来的故事？

## 实验对象

只使用合成数据 GT。

至少测试：

\[
N\in\{1,2,3,5,7,10,15\}
\]

若当前数据不足 15 光，先对少量场景补渲染到 ≥15 光即可；不得因此重新随机划分 scene split。

## 实验 1：GT geometry 条件下恢复 material/light

固定：

\[
n=n^{GT}
\]

只优化/求解：

\[
A,\{L_k\}.
\]

从不同 N 的子集恢复 A。

测：

\[
\text{SI-MAE}(A,A^{GT})
\]

以及 lighting error。

## 实验 2：GT albedo 条件下恢复 geometry/light

固定：

\[
A=A^{GT}
\]

优化：

\[
n,\{L_k\}
\]

可在低分辨率/小规模场景进行。

报告 normal angular error 随 N 的变化。

## 实验 3：illumination diversity

每个 subset 同时计算：

- N；
- angular spread；
- light-vector covariance；
- condition number 或其他明确写出定义的 conditioning measure。

画：

\[
error\ vs\ N
\]

和：

\[
error\ vs\ diversity.
\]

## 实验 4：新证据 vs 重复证据

对：

\[
S=\{I_1,I_2,I_3\}
\]

分别添加：

\[
I_\text{new}
\]

以及重复的：

\[
I_1.
\]

比较：

\[
\Delta_\text{new}
=
E(S)-E(S\cup I_\text{new})
\]

\[
\Delta_\text{dup}
=
E(S)-E(S\cup I_1).
\]

真正的 evidence accumulation 应当主要来自：

\[
\Delta_\text{new}
\]

而不是简单增加 cardinality。

## 交付

`pre0/information_audit/`

至少包含：

- `subset_results.csv`
- `diversity_results.csv`
- `novel_vs_duplicate.csv`
- N-error 图
- diversity-error 图
- novel-vs-duplicate 图
- `INFORMATION_AUDIT.md`

所有曲线报告 bootstrap 置信区间或重复采样 mean±std。

禁止凭肉眼说“明显提升”。

---

# 5. PRE-03 · 三个最小 Probe Model

本任务**不是寻找最终网络**。

目的是判断“什么类型的信息融合值得继续研究”。

所有模型使用：

- 相同 per-image encoder；
- 相同 decoder；
- 相同参数预算数量级；
- 相同训练数据；
- 相同 loss；
- 不使用 residual；
- 不使用 FiLM；
- 不使用 attention；
- 不使用高级 backbone。

只改变集合融合。

## Probe-A · MeanSpatial

逐图得到：

\[
F_k\in\mathbb R^{C\times H\times W}
\]

然后：

\[
F_{set}
=
\frac1N\sum_kF_k.
\]

这是最重要的基线。

## Probe-B · MeanVarSpatial

计算：

\[
\mu_F=\operatorname{mean}_k F_k
\]

和：

\[
\sigma_F^2=\operatorname{var}_kF_k
\]

输入 decoder：

\[
[\mu_F,\sigma_F].
\]

目的是测试：

> 光照间变化本身是否提供 geometry/material cue。

## Probe-C · GlobalSet

每图先：

\[
z_k=\operatorname{GAP}(F_k)
\]

再进行最简单 mean aggregation：

\[
z=\operatorname{mean}_kz_k
\]

用 z 条件化共享 decoder。

本模型用于判断：

> global descriptor aggregation 是否已经损失了关键的 pixel-aligned photometric evidence。

**本阶段不要上 Set Transformer。**

## 输出

先保持最小：

- canonical albedo；
- geometry；
- per-light lighting。

appearance residual 全部关闭。

## 训练纪律

只做统一预算的 probe training。

不调参竞赛。

任何一个模型如果因为工程问题无法训练，记录失败，不为它单独大量调参。

---

# 6. PRE-04 · Evidence Accumulation 诊断协议

对 PRE-03 的三个 Probe 使用完全相同的评价。

这是未来主架构是否值得继续的核心门槛。

## A. N curve

测试：

\[
N=1,2,3,5,7,10,15.
\]

报告：

- normal error
- albedo error
- depth/geometry error
- lighting error

N=1 只作为 stress test。

主分析从 N≥3 开始。

## B. Cross-subset consistency

同一场景随机取：

\[
S_a,\quad S_b.
\]

得到：

\[
(A_a,n_a),\quad(A_b,n_b).
\]

计算：

\[
D_A(A_a,A_b)
\]

和：

\[
D_n(n_a,n_b).
\]

如果恢复的是共享场景属性，不同 light subset 应逐渐趋于一致。

## C. Novel-light gain

重复 PRE-02：

\[
\Delta_\text{new}
\quad vs\quad
\Delta_\text{dup}.
\]

但这次测神经模型。

## D. Permutation test

固定集合，只改变顺序。

要求输出差异处于纯数值误差量级。

该测试只能证明 permutation invariance，**不得解释为 evidence accumulation**。

## E. Fusion sensitivity

记录：

\[
\left\|
\frac{\partial y}{\partial F_k}
\right\|.
\]

确认每个 illumination 的 feature 对最终结果确实有非零影响。

不要自行发明“>0.01 才算有效”等阈值。

看分布、相对量以及不同 N/不同模型之间的比较。

---

# 7. PRE-05 · Held-out illumination 协议定稿

当前项目已经把 held-out relighting 作为识别“真分解 vs reconstruction cheating”的重要证据，这个方向保留。现有顶层设计也已要求将它进入核心评价协议。

但 PRE-0 必须把协议定义干净。

## 主协议：Oracle-query-light

support：

\[
S=\{I_1,\ldots,I_N\}
\]

仅用于估计：

\[
A,n.
\]

query image：

\[
I_q
\]

**绝对不进入模型。**

评估器提供：

\[
L_q^{GT}.
\]

然后：

\[
\hat I_q
=
A\odot S(n,L_q^{GT}).
\]

报告：

- HO-PSNR
- HO-SSIM
- HO-MAE

所有 residual 必须关闭。

## 目的

这个指标回答：

> 从 support illumination 恢复出的共享 scene factors，能否解释一盏模型从未看到的灯？

## 额外要求

必须明确区分：

1. oracle-query-light held-out；
2. predicted-query-light held-out。

二者禁止混在一个指标里。

PRE-0 主指标使用第一种。

---

# 8. PRE-06 · 外部文献与创新性地图

这一项与实验同等重要。

在确定新模型前完成。

## 搜索主题

至少覆盖以下关键词组合：

- sparse photometric stereo
- few-shot photometric stereo
- uncalibrated photometric stereo
- variable-number photometric stereo
- arbitrary-number input photometric stereo
- set-based photometric stereo
- permutation invariant inverse rendering
- multi-illumination inverse rendering
- unknown lighting inverse rendering
- transformer photometric stereo
- joint albedo normal lighting estimation
- sparse multi-light inverse rendering

## 文献范围

建立：

`literature/literature_matrix.csv`

至少覆盖：

- 经典 photometric stereo；
- uncalibrated PS；
- deep PS；
- sparse/few-light PS；
- inverse rendering；
- set/attention aggregation；
- 最近与本问题最接近的工作。

不要只收“支持我们创新”的论文。

必须主动找**可能杀死 novelty 的论文**。

## 每篇记录

```text
title
year
venue
url/doi/arxiv
input_N
fixed_or_variable_N
calibrated_light
single_or_multi_view
RGB_or_gray
output_geometry
output_albedo
output_lighting
joint_decomposition
permutation_invariant
training_data
real_benchmark
core_method
closest_similarity_to_ours
critical_difference
```

## 必须单独输出

`CLOSEST_PRIOR_WORK.md`

只讨论最接近的 5–10 篇。

每篇写：

> 如果审稿人说“你们就是这篇工作的变体”，我们是否有客观证据反驳？

禁止使用：

> “它没有用 Set Transformer”

作为主要差异。

优先考察：

- problem formulation；
- input budget；
- calibration assumption；
- variable cardinality；
- joint factor recovery；
- cross-N generalization；
- physical decomposition。

---

# 9. PRE-07 · DiLiGenT / 真实 benchmark 合同

这一阶段只建评估器，不追求成绩。

## 必须确认

对 DiLiGenT 建立：

`benchmark/DILIGENT_CONTRACT.md`

明确：

- 哪些量作为模型输入；
- 哪些量只作为 evaluator GT；
- lighting 是否允许模型看到；
- normal GT 的定义；
- mask；
- image normalization；
- subset selection；
- calibrated / uncalibrated 协议；
- matched-N 比较方法。

## Subset protocol

每个 N 使用固定的随机种子集合：

\[
S_{N,1},\ldots,S_{N,K}.
\]

所有模型必须使用完全相同的 subsets。

不得每个方法随机抽自己的图。

---

# 10. PRE-0 最终交付包

完成后只向外部审计者交一个 ZIP：

`PRE0_HANDOFF_<commit>.zip`

目录必须如下：

```text
PRE0_HANDOFF/
├── HANDOFF.md
├── protocol/
│   ├── pre0_protocol.yaml
│   ├── DATASET_CONTRACT.md
│   └── split_manifest.*
├── source/
│   ├── renderer/
│   ├── dataset/
│   ├── probe_models/
│   ├── train/
│   └── evaluate/
├── oracle_renderer/
├── information_audit/
├── probe_results/
├── evidence_accumulation/
├── heldout_relighting/
├── literature/
│   ├── literature_matrix.csv
│   └── CLOSEST_PRIOR_WORK.md
├── benchmark/
│   └── DILIGENT_CONTRACT.md
├── configs/
├── logs/
└── checkpoints/
```

checkpoints 只需：

- Probe-A best；
- Probe-B best；
- Probe-C best。

不需要交所有 epoch checkpoint。

---

# 11. HANDOFF.md 必须回答的 12 个问题

不要写成长篇汇报，只逐题回答并给证据路径。

1. Synthetic renderer 与训练 physics renderer 是否一致？差异是多少？
2. depth 与 normal 的物理定义分别是什么？
3. depth-derived normal 与 render/mesh normal 是否一致？
4. 在**没有神经网络**的情况下，增加 N 是否改善 factor recovery？
5. diversity 是否比 N 本身更能解释误差？
6. novel illumination 是否比 duplicate illumination 提供更多收益？
7. 三个 Probe 中哪一个 N curve 最合理？
8. spatial aggregation 是否明显优于 global aggregation？
9. 不同 illumination subsets 恢复的 canonical albedo/geometry 是否一致？
10. oracle-light held-out relighting 是否随 N 改善？
11. 最接近本项目的 5–10 篇工作是什么？
12. 根据现有证据，你认为最值得继续验证的三个方法假设是什么？

第 12 题只能写**假设**，不能自行宣布下一版主架构。

---

# 12. PRE-0 总门禁

只有满足下面四个条件，才进入正式方法设计：

### Gate A · Physics Validity

数据生成、GT 定义和 differentiable renderer 的关系已经被量化，没有未解释的坐标系/normal/depth/lighting 语义冲突。

### Gate B · Information Validity

至少能够回答：

\[
N\uparrow
\]

在数据本身是否增加可恢复信息，以及这一收益和 illumination diversity 的关系。

如果 oracle 层面增加 N 都没有信息收益：

**暂停“evidence accumulation”主故事，先修数据/问题设定。**

### Gate C · Learnability

至少一个简单 Probe 表现出：

\[
\Delta_\text{new}
>
\Delta_\text{duplicate}
\]

的统计趋势，并且 cross-subset scene-factor consistency 随更多有效观测改善。

如果三个简单 Probe 全部无法学习这一现象：

**先研究融合位置/监督方式，不引入更复杂 attention。**

### Gate D · Novelty Viability

CLOSEST_PRIOR_WORK 审计后仍然存在至少一个清晰、可实验验证的问题差异。

如果已有工作已经完整覆盖：

- sparse；
- variable cardinality；
- uncalibrated；
- joint geometry/albedo/lighting；
- same-view multi-illumination；

则不得继续使用当前 formulation 作为核心 novelty，必须先重新定位。

---

# 13. 本阶段唯一允许形成的研究结论

PRE-0 结束时不需要知道“最终网络是什么”。

我们只需要能够严谨回答：

> **这个问题是否值得做；多光照证据是否真的存在；最简单的方法在哪一步失效；现有文献留下的真正空白在哪里。**

下一阶段的方法设计必须从这些结果推导，而不能从个人偏好、已有旧代码或者某个热门模块反推实验。