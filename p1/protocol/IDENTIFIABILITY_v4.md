# IDENTIFIABILITY v4 · 秩条件命题与歧义维数（草案）

> **状态：草案（NOT FROZEN）**· 归属：任务书 T v2.0 阶段 A2 · 条例 T2-1/T2-2 初稿
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

## 2. 命题 P2（歧义维数刻画 · 修正性命题，衔接 W2-A.2）—— 草案

**背景**：W2-A.2 实测 near_zero ∈ {1,2,5,6}（a_track_p_a2_fisher.csv，18 scene × 5 config），
与"单参数猜测 ≥4"不符；测度 1/2 Spearman 仅 ≈ +0.37（任务书预测 ρ>0.9，WEAK）。

**命题（草案）**：合成场景中 ker F_eff 的实测维数满足
`1 ≤ dim ker F_eff ≤ 1 + Σ_p (被照明 SH 维数 − 光覆盖独立维数)_p 的上界收紧`，
且逐光补偿族交集在合成场景中"收缩不足"（部分补偿方向残留），使 dim 落在 {1,2,5,6}
而非理论最紧下界 1。解释：W2-A.2 的 near_zero 大值来自结构残留，非求解噪声。

- 可解释性目标：逐一对照 6 个实测值（任务书 T2-2 验收：≥5/6 可解释）——**当前 3/6
  有初步解释（1=gauge；2/5/6 待逐 scene 对照补偿族谱）**，标 OPEN-2。

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
- **OPEN-2**：P2 的 6 个实测 near_zero 逐 scene 对照（需 18 scene × 5 config 谱数据逐行核对）。
- **OPEN-3**：盒约束（L-box）是否进入 F_eff 形式化（无约束 Fisher vs 约束 Fisher 口径）。
- **OPEN-4**：v3 §1.1 假设（已知 geometry）下，若 reviewer 质疑 amortization 依赖，需在
  Limitations 注明"per-scene 不可辨识由 corpus 先验补偿"（措辞与 CLAIM_CARDS S-05 卡一致）。

---

*草案 v0.1 · 2026-09-03 · 未经外部核对，禁止引用为已证结论。下一步：T2-1 验收自查
（⟸/⟹ 两向标注、反例边界齐）→ OPEN-1 走数值预演或提交主智能体。*
