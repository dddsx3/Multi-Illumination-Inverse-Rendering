# R4′ 诊断取证包（INC-002 附件）

> 生成脚本：`p1/source/information_audit/r4p_diagnostics.py`（纯观测，不改任何度量/阈值）
> 输出目录：`p1/information_audit/diagnostics/`
> 复现的 subset 序列与 scores/solve 阶段**严格同 rng 消费序**（seed 20260902，
> pixel_cap 1000，cutoff 1e-8，NS={3,5,8}，30 subsets/N）。

---

## 交付物 1 · 每 trial 完整 eigen-spectrum 摘要

**文件**：`diagnostics/r4p_trial_eigenspectrum.csv`（1620 行 = 18 scene × 3 N × 30 subset）

对每个 (scene, N, subset) 重算 `F_eff = F_ss − Σ_k B_k F_ll,k† B_kᵀ`（P=1000 dense），
取 `np.linalg.eigvalsh` 全谱 `w`（升序，含 fp 负噪声），归一化 `wn = w / trace`。

| 列族 | 列 | 含义 |
|---|---|---|
| 标识 | `scene, N, subset, P, pixel_seed, cutoff, trace` | 可与 scores/trials 表按 (scene,N,subset) join |
| **归一化全谱分位** | `eig_norm_q{0,0.1,0.5,1,2,5,10,25,50,75,90,95,99,100}` | `np.percentile(wn, q)`；q0 = 最小（可能为负），q100 = 最大 |
| **尾部计数（退化核心）** | `n_above_{0,1e-12,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4}` | `(wn > thr).sum()`；判断"正谱"有多少是真实的、多少贴在 cutoff 上 |
| 负谱 | `n_negative, eig_raw_min, eig_raw_max, neg_mass_over_trace` | fp 误差量级；`neg_mass_over_trace = −Σ_{w<0} w / trace` |
| **bulk 候选度量** | `logdet_pos_mean_log` | `log(wn⁺).mean()`（= 现 secondary `full_logdet_pos_norm`） |
| | `eig_norm_geomean_pos` | `exp(log(wn⁺).mean())`，几何均值 |
| | `eff_rank_entropy` | `exp(−Σ wn⁺ log wn⁺)`，谱熵有效秩 |
| | `participation_ratio` | `(Σwn)² / Σwn²`，参与比 |
| | `cond_p1_p99` | `q99 / q1`，抗离群条件数 |
| gauge | `gauge_residual` | `‖F_eff·â‖ / ‖F_eff‖₂`（解析应为 0） |
| **逐光 F_k** | `rank_Fk_min, rank_Fk_max, rank_Fk_mean` | 见交付物 4 的定义 |
| | `Fk_eigmax_min, Fk_minpos_min` | 逐光 9×9 的 λmax 最小值 / 最小正特征值最小值 |
| ReLU/像素 | `active_frac_min, active_frac_mean, boundary_frac_max` | 光照命中率与 ReLU 边界带 |
| | `F_ss_diag_min, F_ss_diag_median, n_dead_pixels` | `F_ss_diag[p]=Σ_k s_kp²`；dead = `≤1e-12·max` |

**用途**：判定 `full_lam_min_pos_norm`（= `min{λ̃ > spec_cutoff}`）是否被
数值近核尾部与人为 cutoff 主导。

### 已从本表得出的决定性结论（1617 trial join 后）

| 量 | median | p10 | p90 |
|---|---|---|---|
| `primary` = min{λ̃ > 1e-8} | **+3.085e-07** | +1.646e-08 | +1.524e-06 |
| `eig_norm_q0`（绝对最小特征值） | −1.377e-17 | −7.00e-15 | +2.07e-15 |
| `eig_norm_q0.1`（≈第 1–2 小） | +5.733e-08 | −3.45e-19 | +1.227e-06 |
| `eig_norm_q1`（≈第 10 小） | **+4.107e-05** | +4.645e-07 | +2.032e-04 |
| `eig_norm_q5`（≈第 50 小） | +1.330e-04 | +8.42e-06 | +4.934e-04 |

- **谱的形状**：1 个 gauge 零方向落在 fp 噪声（`|q0|` median = 2.53e-16，
  56% 的 trial 该方向为负），紧接着一段 **~1e-8 到 1e-5 的近核尾部**（约第 2–9 小），
  然后 bulk 从 ~1e-5 起。`n_above_1e-8` median = 999、`n_above_1e-6` median = 996
  ⇒ 只有 3 维落在 1e-8~1e-6。
- **primary 拾取的位置**：`primary / q1` 的 median = **1.11e-02**，即 primary
  比真实谱边缘（1% 分位）低 **90 倍**；`primary < q0.1` 的比例为 0.000。
  ⇒ primary 实际是**第 2 到第 9 小特征值区间内的极值序统计量**。
- **cutoff 直接决定取值的比例**：`primary ∈ [1e-8, 2e-8)`（紧贴 spec_cutoff
  两倍内）占 **13.6%**。这部分样本的 primary 数值由人为 cutoff 而非物理决定。

**因此**：`full_lam_min_pos_norm` 不是"光照子集的有效信息"的稳健度量，而是
"P=1000 矩阵近核尾部的极值序统计量 + 人为截断的交互产物"。它对单个最差方向
（通常是单个最差像素）敏感，且 1/8 的样本贴在 cutoff 上。这是**测量效度失效
（measurement invalidity）**，不是假设被否证 —— 退化的仪器无法否证假设。

对照：bulk 统计量在同一批数据上的 per-scene ρ 中位数（post-hoc 诊断，非裁决）
—— `logdet_pos_norm` 在 N=3/5 给出 −0.347/−0.332，优于 primary 的 −0.252/−0.260。

---

## 交付物 2 · 未筛选 raw trial 表

**文件**：`diagnostics/r4p_raw_trials_joined.csv`（988 行，随后台 solve 增长；28 列）

`r4p_confirmatory_trials.csv` ⊕ `r4p_confirmatory_scores.csv` 按 (scene,N,subset) 内连接。
**未做任何 success / 收敛 / 有限性筛选**（988/988 全部 scores 匹配）。

| 前缀 | 列 |
|---|---|
| — | `scene, N, subset` |
| `solver_` | `final_loss, grad_norm, tail_range, restart, iters, si_mae_A, ho_psnr` |
| `solver_*_asrecorded` | `success_asrecorded, converged_asrecorded`（**solver 当场按 Discovery-P75 阈值写的原始判定，在确认集上恒为 0**，见下方警告） |
| `score_` | `P, cutoff, path, full_lam_min_pos_norm, full_lam_max_norm, full_logdet_pos_norm, full_a_opt_pos_norm, full_d_pos, full_trace, full_min_eig, full_gauge_residual, full_offdiag_max, rank_Fk_min, active_frac_min, boundary_frac_max` |
| — | `scores_matched` |

### ⚠ 关于 success 的两套定义（必须分清）

1. **`solver_success_asrecorded`**：`joint_solve` 内部按 pilot 在 **Discovery**
   上标定的固定阈值（`grad_norm < 3.879e-4` 且 tail-loss 判据）写入。
   在确认集上**恒为 0**（27/27 → 0%，复合 mesh 自阴影使优化更难）。
2. **分析时重算的 `success`**：`r4p_confirmatory_gate._load_trials()` 用
   **每 (scene,N) 组内自适应 P75 双筛**：
   `success = (final_loss < P75_loss) AND (grad_norm < P75_grad)`。

**这是一个已确认的方法论缺陷**：`0.75 × 0.75 = 0.562`，实测 success 比例
0.584 —— 所谓"58% 收敛率"是**阈值定义的产物，不是测量结果**。真实收敛率
从未被测量（缺 run-to-run 噪声地板标定，任务书 §7 待办 #3 被跳过）。
本表提供 raw `final_loss`/`grad_norm`，任何重新定义收敛的分析都应从这里出发。

---

## 交付物 3 · 每 scene 的 normal / SH Gram spectrum

**文件**：`diagnostics/r4p_scene_gram_spectrum.csv`（18 行）

全掩码像素（P_full，不下采样）上计算：

| 列族 | 定义 |
|---|---|
| `normal_cov_eig{1,2,3}` | 中心化法线协方差 `Cn = nᶜᵀnᶜ/P` 的特征值（降序） |
| `normal_cov_trace/anisotropy/eff_rank` | `Σλ` / `λ₁/λ₃` / `(Σλ)²/Σλ²` |
| `normal_2ndmoment_eig{1,2,3}` | 未中心化二阶矩 `nᵀn/P`（球面覆盖） |
| `sh_gram_eig{1..9}` | **未加权** SH Gram `G = YᵀY/P` 的 9 个特征值（降序） |
| `sh_gram_trace/min_over_max/logdet/rank_1em8/eff_rank` | `λ₉/λ₁`、`Σlogλ`、`#{λ>1e-8λ₁}`、参与比 |
| `sh_gram_a2_eig{1..9}` | **albedo² 加权** Gram `Yᵀdiag(a²)Y`（= `F_ll,k` 在 h≡1 时的形态） |
| `sh_gram_a2_min_over_max/rank_1em8` | 同上比值与秩 |
| `Fk_rank_{min,median,max}_allK` | 对全 32 灯逐个算真实 `F_ll,k`（含 ReLU 的 h）的秩 |
| `Fk_rank_lt9_frac` | 32 灯中秩 < 9 的比例 |
| `active_frac_{min,median}_allK` | 逐光命中像素占比 |
| `albedo_{min,median}` | albedo 范围 |

### 已观测到的分层（关键）

| scene | normal_eff_rank | sh_gram_rank | Fk_rank_min |
|---|---|---|---|
| conf_cube_axis | **1.00** | **4** | **2** |
| conf_prism8 | 1.18 | **5** | **2** |
| conf_cylinder_r03_d12 | 1.20 | **6** | **5** |
| conf_cylinder_r06_d06 | 1.17 | **6** | **5** |
| conf_cone_r04_d12 | 1.24 | 9 | **7** |
| conf_sphere_on_cube | 1.14 | 9 | 9 |
| conf_cube_plus_cone | 1.17 | 9 | 9 |
| conf_cyl_plus_sphere | 1.67 | 9 | 9 |
| conf_ellipsoid_z06 | 2.05 | 9 | 9 |
| conf_ellipsoid_x13z07 | 2.11 | 9 | 9 |
| conf_egg | 2.16 | 9 | 9 |
| conf_torus_R05_r02 | 2.17 | 9 | 9 |
| conf_hemisphere_sq | 2.19 | 9 | 9 |
| conf_torus_R06_r035 | 2.20 | 9 | 9 |
| conf_snowman | 2.27 | 9 | 9 |
| conf_icosphere_sub3 | 2.27 | 9 | 9 |
| conf_sphere_r05 | 2.26 | 9 | 9 |
| conf_two_spheres_row | 2.30 | 9 | 9 |

- **`sh_gram_rank < 9` 的 4 个场景**（cube_axis 4、prism8 5、cylinder×2 6）
  是**几何本身**使 SH 基在该法线分布上退化 —— 不是光照子集的问题。
  这 4 个场景的 `F_ll,k` 永远低秩，`F_ll,k†` 的伪逆截断始终在起作用。
- 这与前面观测到的 ρ 符号分层吻合：cluster 法线（rank<9）→ ρ 弱或翻正；
  smooth 法线（rank=9）→ ρ 强负。
- **`normal_eff_rank` 全部 < 2.31**：所有场景的法线分布都高度各向异性
  （单视角 + 30° 俯角相机只能看到半球的一部分），最好的 two_spheres_row
  也只有 2.30/3.0。

---

## 交付物 4 · `rank_Fk_min` 的确切定义与代码

### 数学定义

对光照子集 S 中每盏光 k：

```
F_ll,k = Σ_p a_p² · h_kp · Y_p Y_pᵀ        [9×9, PSD]
  其中 h_kp = 1[Y_pᵀ c_k > 0]（ReLU 指示），a_p = 该像素 albedo（已 gauge 归一到单位 RMS）
       Y_p  = SH_basis(n_p) ∈ R⁹，p 遍历下采样后的 P=1000 个掩码像素

rank(F_ll,k) := #{ λ_i(F_ll,k) : λ_i > cutoff · λ_max(F_ll,k) },  cutoff = 1e-8

rank_Fk_min := min_{k ∈ S} rank(F_ll,k)
```

即：**光照子集中"最退化的那一盏光"的 9×9 Fisher 块的数值秩**。
它同时受两件事影响 —— (a) 几何：`Y` 在掩码法线上的 Gram 是否满秩；
(b) 该盏光的 ReLU 阴影：`h_kp = 0` 的像素被完全剔除，使有效像素集变小。

### 代码（逐字，来自 `p1/source/information_audit/gauge_fisher_v2.py`）

**常量**：
```python
DEFAULT_CUTOFF = 1e-8        # F_k 伪逆相对截断（× λmax(F_k)）
BOUNDARY_TOL = 1e-6          # ReLU 边界带 |z| < tol
```

**`F_ll,k` 的构造**（`fisher_blocks`）：
```python
    Z, S, H = shading_field(Y, C)
    N, P = S.shape
    F_ss_diag = (S ** 2).sum(axis=0)                       # Σ_k s_kp²
    w = a[None, :] * S * H                                 # a·s·h  [N,P]
    B = [w[k][:, None] * Y for k in range(N)]              # [P,9] ×N
    Fk = [(Y * (a ** 2 * H[k])[:, None]).T @ Y for k in range(N)]
    bd = np.abs(Z) < BOUNDARY_TOL
    diag = dict(
        N=N, P=P,
        active_frac=H.mean(axis=1),                        # 每光命中像素占比
        boundary_frac=bd.mean(axis=1),                     # ReLU 边界带占比
        Fk_eigmax=np.array([float(np.linalg.eigvalsh(Fk[k]).max()) for k in range(N)]),
    )
```

**秩的计算**（`pinv_psd`，`rank` 是其第 2 个返回值）：
```python
def pinv_psd(F, cutoff=DEFAULT_CUTOFF):
    """PSD 矩阵的 eigh 伪逆（相对截断 cutoff×λmax）。返回 (F†, rank, λmax)。"""
    w, V = np.linalg.eigh(F)
    lam_max = float(max(w.max(), 0.0))
    if lam_max <= 0.0:
        eye = np.eye(F.shape[0])
        return np.zeros_like(F), 0, 0.0
    keep = w > cutoff * lam_max
    winv = np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0)
    Finv = (V * winv[None, :]) @ V.T
    return Finv, int(keep.sum()), lam_max
```

**聚合到 `rank_Fk_min`**（`schur_full` 收集 → `ga_isi_v2_scores` 取 min）：
```python
def schur_full(bl, cutoff=DEFAULT_CUTOFF):
    P = bl["P"]
    F_eff = np.diag(bl["F_ss_diag"])
    pinv_info = []
    for k in range(bl["N"]):
        Fk_inv, rank, lmax = pinv_psd(bl["Fk"][k], cutoff)
        F_eff -= bl["B"][k] @ Fk_inv @ bl["B"][k].T
        pinv_info.append(dict(k=k, rank=rank, lam_max=lmax))
    F_eff += F_eff.T          # 转置视图原地累加（numpy 检测重叠自动缓冲）
    F_eff *= 0.5
    bl.setdefault("diag", {})["pinv_info"] = pinv_info
    return F_eff
```
```python
            rank_Fk_min=min(i["rank"] for i in bl["diag"]["pinv_info"]),
```

### 三点需要注意的语义

1. **`cutoff` 是相对的**（`cutoff · λmax(F_k)`），逐光独立。所以两盏光的
   rank 相同不代表它们的绝对条件数相当 —— 需配合 `Fk_eigmax` / `Fk_minpos_min` 读。
2. **`rank_Fk_min` 是 min over subset**，是极值统计量，与 primary
   `λ_min⁺` 一样对单个离群元素敏感。交付物 1 同时给了
   `rank_Fk_mean` / `rank_Fk_max` 以便做 bulk 对照。
3. **交付物 3 的 `Fk_rank_*_allK` 用全掩码像素（P_full，不下采样）+ 全 32 灯**，
   与交付物 1 的 `rank_Fk_*`（P=1000 下采样 + 子集内的光）**不是同一口径**。
   前者刻画场景固有几何退化，后者刻画具体 trial 的实际数值秩。二者可能不同，
   下采样会进一步降低有效秩。

---

## 已知失效项索引（详见 INC-002）

| # | 问题 | 证据所在 |
|---|---|---|
| P0-1 | primary `λ_min⁺` 退化：13.6% 贴数值地板（<2e-8），34.4% <1e-7，55.7% 的 `d_pos = P−1` | 交付物 1 的 `n_above_*` 列族 |
| P0-2 | 误差动态范围随 N 坍塌：scene 内 IQR/median 0.508 → 0.141 → **0.058**（N=3/5/8） | 交付物 2 的 `solver_si_mae_A` |
| P0-3 | 收敛率从未测量：`0.75×0.75=0.562` vs 实测 0.584 | 交付物 2 的两套 success 定义 |
| P1-4 | G2 函数形式错：primary 跨 4.9 个数量级，线性回归 ΔR²_oos = −54.7 | 交付物 2 的 `score_full_lam_min_pos_norm` 分布 |
| P1-5 | E3 分箱恒不可行：30 subsets / 10 分位 = 3 < 要求的 8 | 代码 `e3_stats` |
| P2-6 | 统计功效连续 5 次削减（scene 20–30→18、subsets 100→30、N 去掉 12、pixel_cap 2000→1000、solver 换串行） | `R4P_PREREGISTRATION.md` |
| P3-7 | 科学线索：`sh_gram_rank < 9` 的 4 个场景 ρ 弱/翻正，rank=9 的场景 ρ 强负 | 交付物 3 的分层表 |
