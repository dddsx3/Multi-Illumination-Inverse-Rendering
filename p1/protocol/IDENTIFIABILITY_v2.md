# IDENTIFIABILITY v2 · Gauge-Aware Illumination-Set Information（R3′ 数学封口版）

> **版本**：v2.0（R3′）· 2026-08-31 · 取代 v0.1 的 §4–§9 数学内容。
> **依据**：`P1_NEXT_STAGE_EXECUTION_TASKBOOK v1.0` §3（T3′.1–T3′.5）。
> **实现**：`p1/source/information_audit/gauge_fisher_v2.py`（唯一正式实现）；
> 旧 `gauge_fisher.py` = **DEPRECATED_EXPLORATORY**（交叉块缺 s_kp + 逐像素
> 标量近似冒充 Schur 补），其派生的 R4 定核数字只能作 exploratory signal。
> **验证**：`p1/tests/test_gauge_fisher_v2.py`，28 项全 PASS
> （阈值 = 任务书冻结值：FD Jacobian ≤1e-5 / block ≤1e-8 / Schur ≤1e-6 /
> gauge-null cutoff 1e-8~1e-5 稳定）。

---

## 1. 冻结模型（最小问题，T3′.1）

固定针孔相机、已知几何（分阶段论证；本轮 n_p 视为已知），线性域像素观测：

```
I_k(p) = a_p · ReLU(Y_pᵀ c_k) + ε_k(p),    Y_p = SH_basis(n_p) ∈ R⁹
```

- θ_s = (a_p)_{p∈Ω}：canonical albedo（P = |Ω| 个未知量）；
- θ_ℓ = (c_1, …, c_N)：per-image 9D irradiance SH（Route A，Â=[π,2π/3,π/4]
  卷积已含在 c 的语义里，见 LIGHTING_MODEL.md）；
- **本轮不把 depth/normal 放进未知参数**（先封口最小可辨识问题）；
- 噪声 ε 高斯同方差（Gauss-Newton Fisher；JᵀJ）。

## 2. 解析 Jacobian

```
z_kp = Y_pᵀ c_k,  s_kp = ReLU(z_kp),  h_kp = 1[z_kp > 0]
∂I_k(p)/∂a_q  = s_kp · δ_pq
∂I_k(p)/∂c_jm = a_p · h_kp · Y_pm · δ_kj
```

## 3. Fisher 分块（R3′ 修正点 1：交叉块必须含 s_kp）

```
F_ss   = diag_p( Σ_k s_kp² )                    [P×P 对角]
F_ll,k = Σ_p a_p² h_kp Y_p Y_pᵀ                 [9×9 逐光；跨光块 ≡ 0]
B_k[p,:] = a_p · s_kp · h_kp · Y_pᵀ             [P×9 交叉块]
```

v0.1/v1 的 `F_sℓ(p,k) = a_p·h_kp·Y_pᵀ` **漏掉了 s_kp**，使"信息被光照
解释掉"的扣除项在暗像素处被系统性高估（暗像素 s_kp≈0 时本应几乎不扣）。

## 4. Full Schur 补（R3′ 修正点 2：P×P，存在跨像素耦合）

消去全部 per-light nuisance 后的有效信息（pseudo-inverse 逐光）：

```
F_eff = F_ss − Σ_k B_k F_ll,k† B_kᵀ            [P×P 稠密、对称、PSD]
```

三条要点：

1. **F_ll,k 是全体像素共享的 9×9** ⇒ 消去 c_k 把所有被光 k 照亮的像素
   耦合起来。F_eff 的 off-diagonal ≠ 0 是结构性存在（toy 实测
   max|off-diag| 可达对角均值的 ~7 倍）。v1 的逐像素标量只是
   diag(F_eff) 的一个**错误**近似（连对角都算错，见 §3）。
2. 等价投影形式（测试独立第三路线）：
   `F_eff = J_aᵀ (I − P_col(J_c)) J_a`，J_c = [diag(a·h_1)Y, …, diag(a·h_N)Y]。
   直观：albedo 扰动 δa 可辨识 ⟺ 其图像签名 s_k⊙δa 落在光照参数
   无法解释的补空间里。
3. **逐像素对角近似只许叫 diag-Schur proxy**（= diag(F_eff)，与 full-Schur
   在 CSV/表格中分栏报告），不得再称"完整 Schur 补"。

## 5. Gauge nullspace（解析结果）

**命题（尺度 gauge）**：`B_kᵀ a = F_ll,k c_k` 且 `B_k c_k = a⊙s_k²`，故

```
F_eff · a = 0        （伪逆精确时严格成立）
```

即消去光照后，全局尺度规范 (δa=εa, δc=−εc) 表现为 F_eff 的解析 null
方向。处理策略（v2 定稿）：

- **主策略 = 投影**：Π_g = I − ââᵀ，正谱指标在 Π_g F_eff Π_g 下不变；
- gauge fixing（a 归一 RMS）仅是换基；无量纲指标必须且实测在
  (a,C)→(a/2,2C) 变换下严格不变（测试 T4d，误差 <1e-9）；
- 报告 gauge residual ‖F_eff â‖/‖F_eff‖ 作数值健康检查
  （cutoff=1e-8 时实测 ~1e-13）。

## 6. 维数定理（修正 v0.1 的命题 P1）

**命题（kernel 结构）**：ker(F_eff) = {δa : ∀k, s_k⊙δa ∈ col(diag(a·h_k)Y)}，
即"能被每盏光分别补偿的 albedo 扰动"之交，外加全 inactive 像素：

```
ker F_eff = (∩_k 补偿族_k) ⊕ span{e_p : p 全 inactive}，  dim ≥ 1（gauge）
```

**推论（P1 修正）**：N=1、全 active、rank(Y)=9 时 rank(F_eff) = P − 9。
单图 albedo 局部不可辨识到 9 维歧义族（其中 1 维是 gauge；v0.1 的
"逐像素 = 0"陈述错误，测试 T5b 实证 rank=P−9）。

**推论（P2 修正）**：重复光不缩小歧义族：ker(F_eff^{S∪dup(k)}) =
ker(F_eff^{S})（测试 T5a）。重复观测只放大特征值（噪声平均），
不增加可辨识结构。

**推论（互补光）**：N ≥ 2 且各光 SH 方向足够分散时，逐光补偿族的交
generically 收缩到 gauge 一维 ⇒ 可辨识性由**子集的方向构成**决定，
与基数 N 解耦——这是 H-COND 的数学载体（"相同 N、不同质量"的来源：
第一项 Σ_k s_kp² 是 naive 信息，Σ_k B_k F_k† B_kᵀ 是"能被重新优化
光照解释掉"的部分；两光同向时后者吃掉前者）。

## 7. 无量纲指标（T3′.4，跨场景可比）

对 F_eff 取 trace 归一 λ̃ = λ/trace(F_eff)（trace 闭式：F_ss.sum() −
Σ_k tr(F_k†·Yᵀdiag(w_k²)Y)，w_k = a·s_k·h_k）：

| 指标 | 定义 | 角色 |
|---|---|---|
| `lam_min_pos_norm` | min{λ̃⁺} | **primary**（R4′ 预注册冻结） |
| `logdet_pos_norm` | mean log λ̃⁺ | secondary（D-最优） |
| `a_opt_pos_norm` | mean 1/λ̃⁺ | secondary（A-最优，按 d⁺ 归一） |
| `d_pos` | #{λ̃⁺ > spec_cutoff} | 有效维数 |

ReLU 边界像素（|z| < 1e-6）单独统计（`boundary_frac`），不参与任何
导数近似路径；全 inactive 像素是 F_eff 的精确零行列（`active_frac`）。

## 8. 数值策略（T3′.4）

- F_ll,k† 一律 eigh 伪逆 + 相对截断（cutoff × λmax(F_k)，默认 1e-8），
  输出 rank(F_k)/cutoff/ridge（默认 0，仅诊断）；
- P ≤ 3000：dense P×P eigh 全谱（预注册像素策略 pixel_cap=2000 在此域内）；
  P > 3000：LinearOperator matvec（O(P·9N)/光）+ 闭式 trace + eigsh 移位
  反演求 λ_min⁺（核维数 1+n_dead 解析已知，σ=−ε 推核到变换谱最大处），
  全谱二级指标在大 P 下不输出（NaN，标 path=operator）；
- operator 与 dense 在 P=200 toy 上互证：matvec relF ≤3.5e-16，
  trace 相对差 ≤1e-10 量级实测 0，λ_min⁺ 相对差 5.9e-14；
- cutoff 1e-8~1e-5 扫描：primary 漂移实测 ≤4.6e-4（阈值 1e-3），
  **裁决不变**；预注册固定 cutoff=1e-8。

## 9. 与实验的接口（不变 + 更新）

- R4′-D：4 个 SUN discovery scene 复跑 = 实现稳定性检查（非确认证据）；
- R4′-C/R4′：新确认集（20–30 scene ×32 SUN）+ 预注册统计
  （E2 same-N / G2 beyond-N scene-grouped / E3 matched-conditioning）；
- C0 Gate：probe 误差随 GA-ISI v2 primary 改善（EXPERIMENT_CONTRACT）；
- 排查顺序（若固定 N 内相关弱）：gauge 投影 → solver 收敛偏差 →
  ReLU 信息真空（active_frac/boundary_frac）。
