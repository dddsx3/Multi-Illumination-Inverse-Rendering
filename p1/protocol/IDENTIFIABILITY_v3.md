# IDENTIFIABILITY v3 · Gauge-Schur Information Quality (GSIQ) — R5-B′ 冻结数学文档

> **版本**：v3.0 · 落盘于 R5-P0 阶段 · 取代 v2.0 的 §7「无量纲指标」表与命名
> **依据**：`R5-B'_PUBLICATION_LOCK.md` §1–§2（C1–C4 + structural-null gate）
> **实现**：`p1/source/information_audit/gauge_fisher_v2.py`（无算法改动；命名 / 报告口径变更）
> **关键变化（vs v2）**：
> 1. 主指标 **M1** 改名 **Gauge-Schur Information Quality (GSIQ)** / 别名 **Nuisance-Marginalized Spectral Information**，不再称 "log pdet"；
> 2. geometry / normals 显式标记为 **已知**（C1 数学构造的硬性收紧）；
> 3. M1 仅作为 **trace-normalized spectral quality**，明确不衡量 absolute information magnitude；
> 4. 新增 **structural-null gate**：`d_expected = P − n_dead − 1`、`d_pos`、`d_extra-null`；
> 5. 所有 positive-spectrum 指标必须与 structural-null 状态同时报告，d_extra-null>0 标记 structurally deficient；
> 6. 数值 / spectral / Schur / 超参 / 冻结性五项理由冻结 M1 为 primary（M2/M5 降为 sensitivity）。

---

## 1. 冻结模型（C1 · Mathematical construction）

固定针孔相机、**已知几何**（分阶段论证；本轮 n_p 视为已知），线性域像素观测：

```
I_k(p) = a_p · ReLU(Y_pᵀ c_k) + ε_k(p),    Y_p = SH_basis(n_p) ∈ R⁹
```

未知参数：

```
θ = (a, c_1, …, c_N)
```

其中：

- a ∈ R^P：canonical albedo（P = |Ω|）；
- c_k ∈ R^9：per-image 9D irradiance SH（Route A，卷积语义在 LIGHTING_MODEL.md 冻结）；
- ε_k(p)：**homoscedastic Gaussian** 假设下，Fisher = JᵀJ（Gauss–Newton Fisher，**局部**信息；非全局 Bayes posterior）；
- **geometry / normals**：本轮视为已知；不得写 joint geometry–albedo–lighting Fisher；
- **gauge**：全局尺度 (δa = εa, δc = −εc) 在 F_eff 上表现为解析 null 方向（Π_g = I − ââᵀ 投影处理）。

> **C1 数学构造冻结**：我们构造的是 *gauge-aware, per-light-nuisance-marginalized Gauss–Newton / Fisher information*；未知参数 θ = (a, c_1, …, c_N)。

### 1.1 假设范围（必须显式声明）

- 噪声同方差：当前 Fisher 推导假设 ε_k(p) ~ N(0, σ²)；非异方差情形下 M1 不直接适用；
- 局部近似：JᵀJ 是 θ₀ 邻域内的局部信息；不能直接解读为"全 θ 空间的总信息量"；
- 几何已知：n_p 已知 ⇒ Y_p = SH_basis(n_p) 已知；这也是为什么 F_eff 是关于 albedo a 的 Fisher，而非关于 albedo+normal 的联合 Fisher；
- 线性化：ReLU 处理为分段线性；边界像素 |z|<1e-6 单独统计（boundary_frac），不参与导数近似路径。

---

## 2. Jacobian 与 Fisher 分块（沿用 v2 §2–§3，无算法改动）

```
z_kp = Y_pᵀ c_k,  s_kp = ReLU(z_kp),  h_kp = 1[z_kp > 0]
∂I_k(p)/∂a_q  = s_kp · δ_pq
∂I_k(p)/∂c_jm = a_p · h_kp · Y_pm · δ_kj

F_ss   = diag_p( Σ_k s_kp² )                    [P×P 对角]
F_ll,k = Σ_p a_p² h_kp Y_p Y_pᵀ                 [9×9 逐光；跨光块 ≡ 0]
B_k[p,:] = a_p · s_kp · h_kp · Y_pᵀ             [P×9 交叉块]
```

---

## 3. Full Schur 补（v2 §4 不变）

```
F_eff = F_ss − Σ_k B_k F_ll,k† B_kᵀ            [P×P 稠密、对称、PSD]
```

三条要点：

1. F_ll,k 是全体像素共享的 9×9 ⇒ 消去 c_k 把所有被光 k 照亮的像素耦合起来；
2. 等价投影形式：`F_eff = J_aᵀ (I − P_col(J_c)) J_a`，J_c = [diag(a·h_1)Y, …, diag(a·h_N)Y]；
3. 逐像素对角近似只许叫 diag-Schur proxy。

---

## 4. Gauge nullspace（v2 §5 不变）

```
F_eff · a = 0        （伪逆精确时严格成立）
```

处理策略：

- 主策略 = 投影：Π_g = I − ââᵀ，正谱指标在 Π_g F_eff Π_g 下不变；
- gauge fixing（a 归一 RMS）仅是换基；
- gauge residual ‖F_eff â‖/‖F_eff‖ 作数值健康检查（cutoff=1e-8 时实测 ~1e-13）。

---

## 5. 维数定理（v2 §6 不变）

```
ker F_eff = (∩_k 补偿族_k) ⊕ span{e_p : p 全 inactive},   dim ≥ 1（gauge）
```

推论（与 v2 同）：

- N=1、全 active、rank(Y)=9 ⇒ rank(F_eff) = P − 9；
- 重复光不缩小歧义族；
- N≥2 且方向分散时，逐光补偿族的交 generically 收缩到 gauge 一维。

---

## 6. 主指标 GSIQ / M1（C2 · Metric interpretation）

### 6.1 定义

对 F_eff 取 trace 归一 λ̃ = λ/trace(F_eff)，定义：

```
I_GS ≡ (1/d⁺) Σ_{i: λ̃_i > spec_cutoff} log(λ̃_i)
```

其中 spec_cutoff = 1e-8（沿用 v2）。

**正式命名**：**Gauge-Schur Information Quality (GSIQ)**，别名 **Nuisance-Marginalized Spectral Information**。
**实现符号**（不重命名代码）：`logdet_pos_norm`。

### 6.2 它不衡量什么（C2 硬性收紧）

**I_GS 不衡量 absolute information magnitude**。它衡量：

> identifiable subspace 上的信息谱平衡 / bulk conditioning quality。

这意味着：

- I_GS 的单调上升 ⇔ F_eff 正谱部分的特征值更"平坦"（D-最优风格），不是"信息总量更多"；
- I_GS 与 absolute information 之间**没有量纲对应**；trace 归一已经把绝对量纲消去；
- 不同 scene 之间可以比较 I_GS，**前提是同一 F_eff 定义、同一 spec_cutoff、同一 noise model**；不能跨 noise level 比较。

### 6.3 与 M2 / M5 的关系

| 符号 | 定义 | 在 v3 中的角色 |
|---|---|---|
| M1 / GSIQ | `mean log λ̃⁺` | **primary**（C2 命名 + §6.4 五条理由） |
| M2 | `mean 1/λ̃⁺`（A-最优，按 d⁺ 归一） | sensitivity（A-最优口径） |
| M5 | `min λ̃⁺` | sensitivity（λ_min⁺ 视角；已不再 primary） |
| λ_max | `max λ̃⁺` | diagnostic only |
| d_pos | `#{λ̃⁺ > spec_cutoff}` | **structural-null gate 一部分**（§7） |

### 6.4 为什么 GSIQ / M1 是 primary（C2 + 任务书 §2 T0.3）

冻结理由（五条，不允许根据新实验再选）：

1. **数值稳定**：`r4pp/03_metric_stability.csv` 已 PASS 5/5 stability test；
2. **有标准 spectral-volume interpretation**：与 D-optimal design literature 直接对应；
3. **与 full Schur 数学结构直接对应**：M1 = mean log λ̃⁺ 是 det(F_eff/trace)^{1/d⁺} 的对数，结构上完全紧扣 F_eff；
4. **无额外超参数**：除 cutoff=1e-8 与 spec_cutoff=1e-8（v2 已冻结）外无调节；
5. **已在 R4″ 预先冻结**：`archive/R4prime_frozen/` 与 CLAIM_REGISTRY v0.3 已记录这一选择。

**禁止**：根据下一批 error 再重新选择 primary（M2/M5 留作 sensitivity）。

---

## 7. Structural-null gate（C1+C2 配套 + 任务书 §2 T0.2）

### 7.1 为什么需要

M1 / GSIQ **只对正特征值取平均**。如果 F_eff 因结构性原因在 identifiable subspace 上有大量零特征值（例如重复光、stacked-co-linear 光簇、几何已知导致 Y 列空间降秩），M1 仍然能给一个看似合理的数 — 但此时它实际衡量的是"少量非零方向的均衡度"，**不**是"该 subset 是否真的有可辨识信息"。

因此 §7 要求：M1 必须与 structural-rank 状态同时报告。

### 7.2 定义

```
P               : 总像素数
n_dead          : 全 inactive 像素数（F_ss_diag[p] = 0 ⇒ pixel 永远被任何光 k 打到 0）
d_expected      : P − n_dead − 1        （= 解析期望的 dim(Π_g F_eff Π_g) − null 维度之外）
d_pos           : #{λ̃_i > spec_cutoff}  （= spectrum_metrics 现有输出）
d_extra_null    : d_expected − d_pos
```

### 7.3 Gate 判定

| d_extra-null | 状态 | 报告要求 |
|---|---|---|
| = 0 | **structurally full** | M1 可用；可直接报告 I_GS |
| > 0 | **structurally deficient** | M1 必须与 `(d_expected, d_pos, d_extra_null)` 同时报告；正文不得单独引用 I_GS |
| < 0 | **spectrum-flip**（数值异常） | 触发 review；当前实测未见 |

### 7.4 实现位置

- `d_expected`：在 `gauge_fisher_v2.py` 的 `spectrum_metrics` / `ga_isi_v2_scores` 路径中加入，需要 `P` 与 `n_dead`（已有 `F_ss_diag` 可派生）；
- `d_pos`：已有；
- `d_extra_null`：派生；
- 输出列加 `d_expected` / `d_extra_null` / `structural_status ∈ {full, deficient, flip}`。

### 7.5 评分口径（"structural status + M1" 两部分报告）

每个 (scene, subset) 必须输出：

```
I_GS, d_pos, d_expected, d_extra_null, structural_status
```

不允许只报告 `I_GS`。这是**结构性修订**，不是换 metric。

---

## 8. 数值策略（v2 §8 不变）

- F_ll,k†：eigh 伪逆 + 相对截断（cutoff=1e-8）；
- P≤3000：dense P×P eigh 全谱；
- P>3000：LinearOperator matvec + eigsh 移位反演求 λ_min⁺；
- cutoff 1e-8~1e-5 扫描 primary 漂移 ≤4.6e-4，**裁决不变**。

---

## 9. Fixed-budget wording（C3 · Fixed-budget effect）

R5-B′ 任务书 §1 C3 正式 wording：

> *At fixed illumination cardinality, higher Gauge-Schur information quality is associated with lower reconstruction error.*

升级路径：

- 在 R5 held-out selection **通过**后允许升级为 *"predicts reconstruction quality"* 或 *"enables subset selection"*；
- 升级前**不得**在正文使用 "predict" / "select"。

---

## 10. High-N wording（C4 · High-N result）

N≥8 的 high-N 现象不再称 *"noise-limited saturation"*。正式用语：

> **selection-leverage compression** / **relative subset-sensitivity saturation**

要求同时报告：

```
CV_subset(N) = σ_subset / Ē
R_signal(N)  = σ_subset / σ_repeat
```

论文结论冻结 wording：

> *Subset choice remains measurable at high N, but its practical leverage rapidly compresses.*

---

## 11. 与实验的接口（v2 §9 更新）

- R5-P1：Oracle→Proxy availability audit 的五种 score 共享同一 GSIQ 实现（不重写）；
- R5-P2（条件 PASS）：Held-out selection benchmark 必须输出 §7.4 全列；
- R5-P3：Task G local-vs-global 与 external estimator 共用 structural-null gate 输出；
- R5-P4：N-curve figure 同时画 CV_subset 与 R_signal，不再使用 "noise floor" / "render noise floor" 表述。

---

## 12. 与 v2 的术语差异一览

| v2 用语 | v3 用语 | 出处 |
|---|---|---|
| log pdet / M1 logdet | GSIQ / Gauge-Schur Information Quality | C2 |
| primary metric uniquely stable | primary metric chosen for 5 frozen reasons | T0.3 |
| joint recoverability | （删；geometry 已知） | T0.4 |
| noise-floor saturation at N=8 | selection-leverage compression | C4 |
| M1 is the only stable metric | M1 is one of several stable metrics; chosen for 5 reasons | T0.3 |
| render noise floor | solver-repeat noise / repeatability floor | §20 |
| N curve is projection of conditioning | （删；非冻结结论） | T0.4 |