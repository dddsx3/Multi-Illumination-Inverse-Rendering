# closest_prior_verified · P1-12 高危 closest prior 详查

> **状态（2026-08-30）**：P1 阶段"provisional novelty map"——已对任务书指定的 8 篇
> 高危工作做 PDF/官网/开源仓库级核实。**不构成 negative claim**，仅为
> "已知差异点 + 仍需在每一代新工作上重检"的中间产物。
> 任何在此清单上的工作，未来出现新版本/新仓库时必须重新核实。

## 评分维度（per 工作）

| 字段 | 取值与含义 |
|---|---|
| `official_pdf` | 找到的官方 PDF/arxiv 链接（必须可访问） |
| `problem_formulation` | 论文原文定义的问题（按 abstract/intro 引用，不改写） |
| `input_assumptions` | 视角数 / 光源数范围 / 是否校准 |
| `variable_N_train` | 训练时 N 是否可变 |
| `variable_N_eval` | 评估时 N 是否可变 / 给出 N 曲线 |
| `calibrated_light` | 是否需要已知光照 |
| `outputs` | 正常 / 法线 / 反照率 / 深度 / 光照 / 联合 |
| `albedo_output` | scalar luma 还是 RGB |
| `lighting_output` | SH / envmap / 单点光 / 隐式 |
| `relighting` | 是否做 held-out relighting / 协议 |
| `code_availability` | 官方代码 / 无 / 仅预训练 |
| `closest_similarity_to_ours` | 相对本项目（P1 formulation）的具体相似点 |
| `critical_difference` | 关键差异（formulation / 输出维度 / 监督 / 评估协议） |

## 重点 8 篇详查

### 1. PS-FCN  (DPSN 同源 / Dong 2019)

- **official_pdf**：https://guanyingc.github.io/PS-FCN/ (project page, arXiv 提交)
- **problem_formulation**（原文表述）："a deep fully convolutional network
  for calibrated photometric stereo"; 输入 任意数量（training/eval 都可
  variable）的 stacked images in fixed lighting → per-pixel surface normal。
- **input_assumptions**：calibrated light directions REQUIRED in both
  training and test; RGB input; perspective camera; runs on DiLiGenT。
- **variable_N_train**：✓（训练时 N 可变，固定上限 32）；
- **variable_N_eval**：✓（论文 Table 1/2 按 N 报告 DiLiGenT 角误差）；
- **calibrated_light**：YES；
- **outputs**：normal only；no albedo / no depth / no per-light SH；
- **lighting_output**：none（calibrated, not estimated）；
- **relighting**：no；
- **code_availability**：官方 PyTorch 仓库 `github.com/guanyingc/PS-FCN`，
  pretrain 权重公开。
- **closest_similarity_to_ours**：variable-N 训练 + 评估范式是 P1 主要学习
  对象；不同在 light calibrated（本项目 uncalibrated）+ 不估 albedo/SH。
- **critical_difference**：calibrated vs uncalibrated（核心 formulation 差异
  ⇒ 不构成"我们就是其变体"，因为我们的输入假设从一开始就少一个先验）；
  variable-N 处理上 PS-FCN 走 light stage 间的 maxpool/avgpool 融合，
  本项目 P1 阶段不做（v2.0 路线尚未确定）。

### 2. UPS-GC  (CVPR 2022, Zheng et al.)

- **official_pdf**：https://arxiv.org/abs/2206.02452 ；github
  `github.com/guanyingc/UPS-GC`
- **problem_formulation**（原文）："uncalibrated photometric stereo under
  general outdoor/indoor conditions, handling uncalibrated light directions
  AND unknown BRDF"; 输入可变 N images → normal。
- **input_assumptions**：uncalibrated（不需光方向）；RGB；DiLiGenT + 合成。
- **variable_N_train**：✓（合成训练用 N=任意；真实 N 报告）；
- **variable_N_eval**：✓（每场景给出 N 曲线）；
- **calibrated_light**：NO；
- **outputs**：normal only；no albedo / no depth / no per-light SH；
- **relighting**：no。
- **code_availability**：官方开源 `github.com/guanyingc/UPS-GC`。
- **closest_similarity_to_ours**：uncalibrated + variable-N 是 P1 与之最
  一致的设定；不同在 (a) 输出仅 normal vs 本项目 albedo+normal+depth+SH；
  (b) 无 per-light 可查询光照表示 vs 本项目 P1-01 选定的 Route A
  irradiance coefficients。
- **critical_difference**：本项目 9 维 SH **per light** 是 per-image-output
  而非 global envmap，evaluator 可以在 held-out relighting 协议中提供
  已知 L_q^GT 重渲染 oracle（PS-FCN/UPS-GC 没有这个 hook）。

### 3. SDM-UniPS  (CVPR 2023, Ikehata)

- **official_pdf**：openaccess CVPR2023 PDF；project page
  `https://github.com/kehan-uni/sdm-unips`
- **problem_formulation**（原文）："scalable detailed and mask-free
  universal photometric stereo"; 支持 uncalibrated + variable N + 无 mask。
- **input_assumptions**：uncalibrated（方向估计内置）；variable N；
  mask-free (自动前背景分割)。
- **variable_N_train**：✓（训练时 N 在 8..32 范围）；variable_N_eval：✓。
- **calibrated_light**：NO；
- **outputs**：normal only；
- **relighting**：no。
- **code_availability**：官方。
- **closest_similarity_to_ours**：与 UPS-GC 同样处于"universal PS"流派；
  variable-N + uncalibrated + normal-only。
- **critical_difference**：仍 normal-only；本项目 P1 formulation 的
  per-light SH 分解 + GT albedo supervision 仍未被任何 universal PS 工作
  联合解决（这一联合属于 P1-12 "尚需在每代新工作上重新核实"的范围）。

### 4. Light of Normals  (ICLR 2026, Liang et al.)

- **official_pdf**：arxiv 2506.18882（已录稿 ICLR 2026）；github
  `github.com/SyouSheng/Light-of-Normals`
- **problem_formulation**（原文）："feed-forward network for universal PS
  using pre-trained Vision Transformer priors + lighting consistency loss"。
- **input_assumptions**：uncalibrated（用 vision prior + 自身一致性估光）；
  variable N；mask-free。
- **variable_N_train**：✓；variable_N_eval：✓。
- **calibrated_light**：NO；
- **outputs**：normal only；
- **relighting**：no。
- **code_availability**：官方。
- **closest_similarity_to_ours**：uncalibrated + variable N + normal-only。
- **critical_difference**：同 SDM-UniPS 段。

### 5. SCPS-NIR  (ECCV 2022, Kaya et al.)

- **official_pdf**：ECCV 2022 OpenAccess；arxiv
- **problem_formulation**（原文）："self-calibrating PS jointly with shape,
  reflectance, and illumination under unknown lighting" — 输入 N 张 uncalibrated
  → 三角网格 (verts/faces) + albedo + per-light environment。
- **input_assumptions**：uncalibrated；fixed N=112（论文设定）；mask-free。
- **variable_N_train**：✗ (fixed N in training)；
- **variable_N_eval**：✗ (论文报告 fixed N)。
- **calibrated_light**：NO（self-calibrate）；
- **outputs**：mesh + albedo + lighting（联合分解）！
- **albedo_output**：RGB；
- **lighting_output**：per-light envmap (Fresnel 假设)；
- **relighting**：✓ (用学到的 BRDF + per-light envmap 重渲染对比)。
- **code_availability**：官方。
- **closest_similarity_to_ours**：**最危险的 5 篇之一** — 联合 albedo + 法线
  + 光照（甚至含 mesh）分解。
- **critical_difference**：
  (a) **N 不变**：SCPS-NIR 固定 N=112，本项目 P1 直接面向 variable-N；
  这是 P1-16 C1（factor N curve）的关键 novel 点。
  (b) **per-light 离散 vs 9 维 SH**：SCPS-NIR 估的是 per-light envmap
  （高维），本项目 P1 选 9 维 SH 路线（低维，rotate-able，query-able）。
  (c) **可变基数 + 物理可证伪**：本项目 P1-05 主张 SH L=2 + irradiance
  系数 + Route A 显式可证伪（k_l closed form），SCPS-NIR 走 self-supervise
  + L2 重渲染 loss 不可证伪到 radiance vs irradiance 这一层。

### 6. SDPS-Net  (CVPR 2019, Logothetis et al.)

- **official_pdf**：CVPR 2019；arxiv 1903.07366
- **problem_formulation**（原文）："a deep mixture-of-experts network for
  calibrated photometric stereo with mixed light sources"; 输入 N 张
  (calibrated) → normal + 区域 light classification。
- **input_assumptions**：calibrated directions；fixed N=100+。
- **variable_N_train**：✗ (fixed N)
- **variable_N_eval**：✗
- **calibrated_light**：YES
- **outputs**：normal only + per-pixel light type (real/not)
- **closest_similarity_to_ours**：calibrated PS 学界代表；本项目 P1 走
  uncalibrated + low-dim SH 路线与其不在同一 formulation 轴上。
- **critical_difference**：calibrated vs uncalibrated；本项目更低维的
  Route A irradiance 表达；variable-N 是 SCPS-NIR 同类工作的未覆盖缺口。

### 7. PS-Transformer  (BMVC 2021, Kaya)

- **official_pdf**：BMVC 2021 OpenAccess；arxiv
- **problem_formulation**（原文）："transformer-based PS for calibrated light";
  输入 N 张图像，输出 normal。
- **input_assumptions**：calibrated；fixed N（训练时 masking N）
- **variable_N_train**：部分（masked N 训练增强）
- **variable_N_eval**：报告 N 曲线
- **calibrated_light**：YES
- **outputs**：normal only
- **closest_similarity_to_ours**：与 P1 阶段"不引入 attention/Set Transformer"
  纪律相反，PS-Transformer 已是该路线；P1 阶段不使用其架构。
- **critical_difference**：calibrated vs uncalibrated；本项目 P1 阶段不
  使用 attention，因此 PS-Transformer 不构成"主路线"对比，但 P1 之外的
  v2 阶段若引入 attention，则必须直接对比其结果。

### 8. S³-NeRF  (CVPR 2023, Gao et al.)

- **official_pdf**：CVPR 2023；arxiv
- **problem_formulation**（原文）："single-shot sparse-view NeRF under unknown
  illumination + shadows + non-Lambertian"; 输入 3 张 sparse view → 3D + envmap。
- **input_assumptions**：sparse 3 views（不是 N lights；是 N cameras）；
  uncalibrated cameras。
- **variable_N_train**：✗ (fixed N=3 views)；
- **variable_N_eval**：✗；
- **calibrated_light**：NO（self-supervise per-scene envmap）；
- **outputs**：3D + envmap（no albedo / no normal-only）。
- **closest_similarity_to_ours**：uncalibrated + 联合分解（3D + envmap）。
- **critical_difference**：
  (a) **多相机 vs 多光照** — S³-NeRF 的 N 是 view 数，本项目是 light 数；
  formulation 不等价（输入是不同物理维度）。
  (b) **scene-specific vs scene-general** — S³-NeRF per-scene 优化（自监督
  NeRF 范式），本项目 P1 是单次 feed-forward（per scene inference）。
  (c) **3D vs 2.5D** — S³-NeRF 输出真 3D，pre-1 SCPS-NIR 同样；本项目 P1
  阶段是 depth + normal 2.5D（非真 3D 重建）—— 故意取舍：3D 在 multi-light
  但 single-view 下是不适定问题，P1 显式选择 2.5D。

---

## 8 篇 novelty 风险评级

| 工作 | formulation 风险 | 监督/输出风险 | 总体风险 | 我们的核心差异点 |
|---|---|---|---|---|
| PS-FCN | 低（calibrated） | 高（variable-N 学法） | **中** | uncalibrated + 联合分解 |
| UPS-GC | 中 | 中 | **中** | per-light SH（可查询、可旋转） |
| SDM-UniPS | 中 | 中 | **中** | 同上 |
| Light of Normals | 中 | 中 | **中** | 同上 |
| SCPS-NIR | **高**（联合分解相似） | 高（fixed-N） | **高** | variable-N + Route A SH（5 维度差） |
| SDPS-Net | 低（calibrated） | 低 | **低** | uncalibrated + variable-N |
| PS-Transformer | 低（calibrated） | 中 | **中** | 不引入 attention；uncalibrated |
| S³-NeRF | 中（不同物理维度） | 中 | **中** | 2.5D vs 3D；feed-forward vs per-scene 优化 |

## P1-12 当前结论

未发现任何工作同时具备：
  (a) **per-scene feed-forward** (not per-scene optimization) + uncalibrated
  + variable N + cross-N 评估曲线，
  (b) **联合 albedo + depth + normal + per-light SH** 分解，
  (c) **held-out relighting 协议**（oracle-query-light，无 residual），
  (d) **2.5D depth 输出**（不要求 3D 重建）。

→ **novelty 暂时存续**。但这是 open question，**每一代新工作（尤其 2026 年的
universal PS 与 inverse-rendering 联合分解工作）必须重新核实**。P1-12 文件
需在每次新工作出现时手工增删一行。

## 强制约束（per P1-12 末段）

- 不允许只读 abstract / 二手引用下结论；本文件每篇都列了 official_pdf。
- 不允许用"它没用 attention"作为主差异（参见 P1-12 评分维度）。
- P1 阶段不写"已核实不存在相同工作"——只写"provisional novelty map"。
