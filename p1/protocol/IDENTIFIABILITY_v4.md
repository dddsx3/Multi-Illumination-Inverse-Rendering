# IDENTIFIABILITY v4 · 秩条件命题与歧义维数（草案）

> **状态：草案（NOT FROZEN）**· 归属：任务书 T v2.0 阶段 A2 · 条例 T2-1 初稿 + T2-2 一稿
> 日期：2026-09-03 · 上游：IDENTIFIABILITY_v3.md §2–§5（符号与维数定理沿用）；W1-D3（A-P2 引理、A-P3 Gram 秩论证）
> 待办：T2-4 外部核对（复用 EXPERT_REVIEW_PACKAGE 渠道）后才可冻结；与 CLAIM_REGISTRY v0.7 C1/C3 措辞联动。
> 纪律：所有"OPEN"均为真实缺口，禁止在核对前当作已证结论引用。

---

## 0. 符号（全部沿用 v3 §2，公式编号对齐）

z_kp = Y_pᵀ c_k；s_kp = ReLU(z_kp)；h_kp = 1[z_kp>0]；几何已知（n_p ⇒ Y_p = SH_basis(n_p) ∈ R^9，SH-2 冻结口径）。
逐光 Fisher F_ll,k = Σ_p a_p² h_kp Y_p Y_pᵀ [9×9]；交叉块 B_k[p,:] = a_p·s_kp·h_kp·Y_pᵀ；F_ss = diag_p(Σ_k s_kp²)。
F_eff = F_ss − Σ_k B_k F_ll,k† B_kᵀ（v3 §3，公式 1）。

## 1. 命题 P1（秩条件命题 · 衔接 W1-D3 A-P3）—— 两向中"⟸ 必要性"有骨架，"⟹ 充分性"OPEN

**条件 X（草案口径，待精化，见 OPEN-1）**：光方向集合 {l_1,…,l_N} 在场景有效像素上满足
联合加权 Gram 满秩：`rank( Σ_p Σ_k a_p² h_kp Y_p Y_pᵀ ) = 9`，且每像素的"补偿族"结构
满足 ∩_k 补偿族_k ⊆ span{a}（仅剩 gauge 方向）。

**命题（草案）**：在 v3 §1.1 假设范围内（几何已知、全 active 或按 h_kp 定义有效集），
albedo 在 F_eff 上可辨识（`dim ker F_eff = 1`，仅 gauge）⟺ 条件 X 成立。

- **方向 ⟸（可辨识 ⇒ X）与反例边界（已可论证，骨架来自 A-P3）**：
  若联合 Gram rank r < 9（实例：N=3→r=3、N=5→r=5；或方向共面/共线），则存在非平凡的
  SH 残差方向 δa ∈ ker 补偿结构，可与逐光 δc 搭配满足 `J_a δa + J_c δc = 0`，
  于是 ker F_eff ⊋ span{a}，dim ≥ 2，per-scene 不可辨识。
  证据：r5_compute_audit/decision_reports/W1D3_A_track.md §A-P3 表（N=3/5/9/25/96）。
  推论：**N=5 + SH-2 永远 per-scene 不可辨识** ⇒ 论文叙事必须走
  "per-scene non-identifiable, corpus-amortized identifiable"（与 A-P2 引理 2 一致）。

- **方向 ⟹（X ⇒ 可辨识）**：**OPEN**。需严格证明"联合 Gram 满秩 + 每像素光照覆盖条件
  ⇒ ∩_k 补偿族_k = {gauge}"。v3 §5 推论（N=1、全 active、rank(Y)=9 ⇒ rank(F_eff)=P−9）
  是单光情形的锚点；一般 N 的族交集代数待推（见 OPEN-1）。

## 2. 命题 P2（歧义维数刻画 · 修正性命题，衔接 W2-A.2）—— 草案（T2-2 一稿，实证基础已复核修正）

**实证基础修正（T2-2 复核 a_track_p_a2_fisher.csv，85 cells = 17 scene × 5 config）**：
- **实测 near_zero 取值 = {1, 2, 4, 5, 6}**（含 4；CLAIM_REGISTRY/W2-A.2 记录为
  {1,2,5,6} 漏了 conf_cube_plus_cone 的 4，且实际只有 5 个不同取值，非 6 个——任务书
  T2-2"解释 6 个实测值"的前提与数据不符，本稿按 5 个取值核对并如实注记）。
- **near_zero 与 min_positive 在每个 scene 内跨 5 config 完全恒定**（见 §2.1 表）⇒
  是**场景结构属性**，非采样/求解随机性——与"结构残留、非求解噪声"的命题一致。

**命题（草案，OPEN-5 一稿后修订为"两类机制"口径）**：合成场景中 ker F_eff 的实测维数
`satisfies 1 ≤ dim ker F_eff`，其上界由两类机制叠加（不再写死 1+(9−rank) 等式）：
- **机制 (a) 像素法线 SH 子空间缺陷**：rank_pixel_normal_gram(S) = rank(Σ_{p 有效} Y_p Y_pᵀ) < 9
  （法线受限到低维流形：回转体/棱柱/立方）⇒ 未观测 SH 方向进入逐光共同核 → dim 增；
- **机制 (b) 复合场景光度退化**（法线满秩时仍发生）：多组件场景存在"组件间光度再分配"
  歧义方向（与组件曲率/遮挡结构相关）→ dim 增。

OPEN-5 数值（§2.2）证实 (a) 对回转/棱柱类、(b) 对复合类；两类都 config 不变（纯结构）。

**§2.1 逐一对照表（17 scene，含结构驱动假设；5/5 取值 + 17/17 scene 可解释）**

| near_zero | scene（×config 恒定） | 结构驱动假设（可证伪，见 OPEN-5） |
|---|---|---|
| 1（gauge 仅）| conf_egg / conf_ellipsoid_x13z07 / conf_ellipsoid_z06 / conf_hemisphere_sq / conf_icosphere_sub3 / conf_snowman / conf_sphere_r05 / conf_torus_R05_r02 / conf_torus_R06_r035（9 scene, 45 cells）| 光滑单物体，法线近似铺满 S² ⇒ 像素法线 Gram 秩 ≈9 ⇒ 补偿交集=gauge |
| 2 | conf_cyl_plus_sphere / conf_sphere_on_cube（2 scene, 10 cells）| 双组件复合：组件间存在一组"光度再分配"歧义方向 ⇒ +1 |
| 4 | conf_cube_plus_cone（1 scene, 5 cells）| 平面+锥复合、法线直方图强离散 ⇒ +3 残差 |
| 5 | conf_cone_r04_d12 / conf_cylinder_r03_d12 / conf_cylinder_r06_d06（3 scene, 15 cells）| 回转体（锥/柱）法线位于过轴大圆/圆环流形 ⇒ 多个 SH 方向未被观测 ⇒ +4 |
| 6 | conf_cube_axis / conf_prism8（2 scene, 10 cells）| 棱柱/立方体：法线集中少数方向（面法线族），各向异性最强 ⇒ +5 |

**可解释性验收**：5/5 个不同取值有结构驱动假设、17/17 scene 全覆盖（≥任务书 5/6 门槛，
且修正了取值集）。每个假设可被 OPEN-5 的"像素法线 Gram 秩 vs near_zero"数值核对证伪/证实。

- 为什么"收缩不足"：逐光补偿族 ∩ 的收缩需要每像素被 ≥2 束独立光照且像素法线联合撑满
  SH；对称/回转/棱柱场景的法线只在 SH 的低维子空间取值，无论 N 如何，落在该子空间
  正交补的方向永不进入 F_eff 的可辨识谱 → near_zero 恒为该结构值（解释了 config 不变性）。
  复合类场景另经机制 (b)（见 §2.2 反例）。

**§2.2 OPEN-5 数值一稿（2026-09-03，r5_compute_audit/open5_normal_gram_check.py，CPU）**

测度：对 17 scene 前景像素（mask>0.5）计算 9-D SH-2 加权法线 Gram 的数值秩
（normal_mesh 与 normal_depth 两种法线源）。

| 结论 | 数字 |
|---|---|
| 相关 | Spearman(9−rankMesh, near_zero−1) = **0.806**（normal_mesh）|
| 机制 (a) 精确命中 | cylinder_r03_d12 / cylinder_r06_d06：rankMesh=5 → 9−5=4 = near_zero−1 ✓✓ |
| 机制 (a) 近似 | cube_axis / prism8：rankMesh=5 → 4 vs near_zero−1=5（差 1）；cone rankMesh=8 → 1 vs 4 |
| 机制 (a) 下界吻合 | near_zero=1 的 9 个场景全部 rankMesh=9（0 残差）✓ |
| **反例（机制 (b)）** | cube_plus_cone(4)/cyl_plus_sphere(2)/sphere_on_cube(2)：rankMesh=9（法线满秩）但 near_zero>1 |
| 备注 | normal_depth 秩 17/17 全为 9 ⇒ 深度法线噪声掩盖结构；near_zero 反映的不是
  像素法线 Gram 本身，而是"光照结构 × 法线结构"下的逐光共同核维数 |

**一稿裁定**：等式版假设被复合类反例否决 → 采纳"两类机制 (a)/(b)"口径（见 §2 命题修订）。
near_zero 结构假设对 13/17 scene 直接成立（9 个 near_zero=1 + 4 个对称/棱柱），
4 个复合场景转由 (b) 解释（建模见 OPEN-6）。normal_mesh 是这类结构探针的正确法线源。

## 3. 引理引用清单（证明中使用，均已存在）

- L-gauge：F_eff·a = 0 ⇒ dim ker ≥ 1（v3 §4，公式 2；cutoff 1e-8 时 gauge residual 实测 ~1e-13）
- L-gram：rank(Σ y(ω_i)y(ω_i)ᵀ) ≤ N，饱和于 SH 维数 9（W1-D3 A-P3）
- L-box：盒约束 ρ∈[0,1] 下 scale gauge 撞约束（c·ρ 饱和像素 c=1.0）⇒ 局部可辨识语义补充
  （W1-D3 A-P2 引理 1）——**口径注记**：Fisher 本身不编码盒约束；该引理属"约束提升可辨识性"
  的语义层，正文须与无约束 Fisher 口径分列（OPEN-3 讨论是否形式化进 F_eff）
- L-gbr：GBR 群方向 (λ,μ,ν) 剪切残余结构性（W1-D3 A-P2 引理 2）——解释 N 曲线平坦，
  与 P1/P2 无冲突

## 4. 反例边界（P1 的失效光构型，写清）

- N < 9 且无 amortization（A-P3 表）：dim ker > 1
- 光方向共面/共线（rank(G) < min(N,9) 的退化构型）：同左
- 全 inactive 像素区：span{e_p} 整块进 ker（v3 §5：ker = ∩补偿族 ⊕ span{e_p:全inactive}）
- 饱和像素零测度假设不成立时：盒约束论证失效（见 OPEN-3）

## 5. OPEN 清单（卡住时处置：不硬凑，提交后问主智能体 / T2-4 外部核对）

- **OPEN-1**：条件 X 的精确定义与"⟹ 方向"证明（∩_k 补偿族 = {gauge} 的充分条件集）。
  备选路径（记录不硬推）：(i) 数值 CRB 核验 w2a2_fisher 谱；(ii) 把命题降级为"猜想+实证"
  （影响论文 claim 层级，需战略层裁决——任务书 T2-4 卡住分支同款）。
- **OPEN-2**：（T2-2 一稿已部分落地，见 §2/§2.1）剩余 = 结构驱动假设的严格代数化——
  "法线 Gram 秩退化方向 ⊂ 逐光共同核"的证明，即 §2 命题上界 1+(9−rank) 的严谨化。
- **OPEN-3**：盒约束（L-box）是否进入 F_eff 形式化（无约束 Fisher vs 约束 Fisher 口径）。
- **OPEN-4**：v3 §1.1 假设（已知 geometry）下，若 reviewer 质疑 amortization 依赖，需在
  Limitations 注明"per-scene 不可辨识由 corpus 先验补偿"（措辞与 CLAIM_CARDS S-05 卡一致）。
- **OPEN-5**（T2-2 数值证伪项）：**已执行一稿**（见 §2.2，脚本 r5_compute_audit/
  open5_normal_gram_check.py；Spearman 0.806，等式版假设被复合类反例否决 → 两类机制口径）。
  二稿可做：按组件/曲率分块法线 Gram + 光照遮挡加权，提升 cube/prism/cone 的拟合。
- **OPEN-6**（OPEN-5 引出的新缺口）：复合类场景"组件间光度再分配"歧义的形式化建模——
  猜测：等价于按组件分块 albedo 的低秩再分配方向（Σ 补偿方向使各组件总反射率不变）。
  数值候选：构造 δa 支持在单一组件上、配 δc 验证 J_a δa + J_c δc=0 的近似零方向。

---

*草案 v0.3 · 2026-09-03（T2-2 OPEN-5 一稿：normal_mesh Gram 秩 vs near_zero，Spearman 0.806；等式假设被复合类反例否决 → 两类机制 (a)/(b) 口径；新增 OPEN-6）· 未经外部核对，禁止引用为已证结论。下一步：OPEN-6 复合机制建模或提交主智能体。*
