"""P1-R3' · Gauge-Aware Illumination-Set Information v2（full-Schur 正式版）。

本模块是 R3' 数学审计的正式实现（P1_NEXT_STAGE_EXECUTION_TASKBOOK v1.0 §3）。
旧 `gauge_fisher.py` 已标 **DEPRECATED_EXPLORATORY**：其交叉块漏掉 s_kp 因子、
且把逐像素标量近似冒充"完整 Schur 补"。R4 的固定-N 相关性因此只能作
exploratory signal（任务书 §0.1 裁决）。

=======================================================================
冻结模型（T3'.1：本轮只封口最小问题）
=======================================================================
  I_k(p) = a_p · ReLU(Y_p^T c_k) + eps,     Y_p = SH_basis(n_p) ∈ R^9
  几何 n_p 视为已知（分阶段论证）；未知参数 θ = (a ∈ R^P, c_1..c_N ∈ R^9)。
  全局尺度 gauge（解析已知 null 方向）：(a, c_k) -> (λa, c_k/λ)。

=======================================================================
解析 Jacobian（行序 k·P+p；列序 [a(P), c_1(9), ..., c_N(9)]）
=======================================================================
  z_kp = Y_p^T c_k
  s_kp = ReLU(z_kp),  h_kp = 1[z_kp > 0]
  ∂I_k(p)/∂a_q   = s_kp · δ_pq
  ∂I_k(p)/∂c_jm  = a_p · h_kp · Y_pm · δ_kj

=======================================================================
Fisher 分块（Gauss-Newton / J^T J；全部含 s_kp —— R3' 修正点 1）
=======================================================================
  F_ss      = diag_p( Σ_k s_kp² )                      [P×P 对角]
  F_ll,k    = Σ_p a_p² h_kp Y_p Y_p^T                  [9×9 逐光；跨光块=0]
  B_k[p,:]  = a_p · s_kp · h_kp · Y_p^T                [P×9 交叉块]  ← v1 缺 s_kp

=======================================================================
Full Schur 补（消去全部 per-light nuisance —— R3' 修正点 2）
=======================================================================
  F_eff = F_ss − Σ_k B_k F_ll,k† B_k^T                 [P×P 稠密、对称、PSD]

  因为 F_ll,k 是全部像素共享的 9×9，消去 c_k 会把所有被光 k 照亮的像素
  耦合起来 —— off-diagonal ≠ 0 是结构性存在（测试 T3 断言）。
  等价投影形式（测试用独立第三路线）：
      F_eff = J_a^T (I − P_col(J_c)) J_a,   J_c = [diag(a·h_1)Y, ..., diag(a·h_N)Y]

  diag-Schur proxy（逐像素标量，仅diag(F_eff)，明确降级命名，与 full 分栏）：
      diag(F_eff)[p] = F_ss[p] − Σ_k B_k[p,:] F_ll,k† B_k[p,:]^T
  注意：v1 的逐像素公式连 diag 都不对（交叉块缺 s_kp）。

=======================================================================
Gauge 处理
=======================================================================
  解析恒等式（伪逆精确时严格成立）：
      B_k^T a = F_ll,k c_k   且   B_k c_k = a ⊙ s_k²   ⇒   F_eff · a = 0。
  即消去光照后的尺度 gauge 表现为 F_eff 的解析 null 方向 δa = a。
  处理策略（v2 主策略 = 投影）：
      Π_g = I − â â^T,  F_eff^proj = Π_g F_eff Π_g。
  正谱指标（λ⁺ 系列）在投影前后不变（T4 断言）；报告 gauge residual
  ||F_eff â|| / ||F_eff|| 作为数值健康检查。gauge fixing（a 归一 RMS）
  仅是换基，正谱指标必须 gauge 变换不变（T4 断言）。

=======================================================================
无量纲主指标（T3'.4：跨场景必须无量纲）
=======================================================================
  λ̃ = eig(F_eff) / trace(F_eff)（trace = Σλ 闭式可算）
  primary   : lam_min_pos_norm = min{ λ̃⁺ }          （λ̃⁺ = 正谱）
  secondary : logdet_pos_norm  = mean_p⁺ log λ̃⁺     （mean log-eigenvalue）
              a_opt_pos_norm   = mean_p⁺ 1/λ̃⁺       （按有效维数 d⁺ 归一）
              d_pos            = #{ λ̃⁺ > cutoff }

=======================================================================
数值策略
=======================================================================
  小 P（≤ dense_max_p，默认 3000）：显式 P×P eigh（精确全谱）。
  大 P：LinearOperator matvec（O(P·9N)）+ Woodbury logdet + eigsh 正谱。
  F_ll,k† 一律 eigh 伪逆 + 相对截断 cutoff（默认 1e-8 × λmax(F_k)），
  输出 rank(F_k)/cutoff/active_frac/ReLU 边界占比等诊断。
  ReLU 边界像素（|z| < 1e-6）单独统计，不参与任何"近似导数"路径。
"""
import argparse
import csv
import gc
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "physics")))
from sh import sh_basis_npy  # noqa: E402

DEFAULT_CUTOFF = 1e-8        # F_k 伪逆相对截断（× λmax(F_k)）
BOUNDARY_TOL = 1e-6          # ReLU 边界带 |z| < tol
DENSE_MAX_P = 3000           # 超过则走 operator 路径


# ======================================================================
# 前向模型与解析 Jacobian（测试用 oracle）
# ======================================================================
def shading_field(Y, C):
    """Y [P,9], C [N,9] -> z [N,P], s [N,P] (ReLU), h [N,P] (indicator)."""
    Z = (Y @ C.T).T                                   # [N,P]
    S = np.maximum(Z, 0.0)
    H = (Z > 0.0).astype(np.float64)
    return Z, S, H


def model_images(a, Y, C):
    """无噪声前向模型 I [N,P] = a ⊙ ReLU(Y c_k)。"""
    _, S, _ = shading_field(Y, C)
    return S * a[None, :]


def jacobian_full(a, Y, C):
    """解析 full Jacobian [N*P, P+9N]（测试 oracle；不做逐像素近似）。"""
    P, N = a.size, C.shape[0]
    _, S, H = shading_field(Y, C)
    J = np.zeros((N * P, P + 9 * N))
    for k in range(N):
        rows = slice(k * P, (k + 1) * P)
        J[rows, :P] = np.diag(S[k])                                  # ∂/∂a
        J[rows, P + 9 * k: P + 9 * (k + 1)] = (a * H[k])[:, None] * Y  # ∂/∂c_k
    return J


# ======================================================================
# 解析 Fisher 块（含 s_kp —— R3' 修正点 1）
# ======================================================================
def fisher_blocks(a, Y, C):
    """解析三块 Fisher + 诊断量。

    返回 dict:
      a [P], Y [P,9], N, P
      z/s/h [N,P]
      F_ss_diag [P]           对角
      B   list[N] of [P,9]    B_k[p,:] = a_p·s_kp·h_kp·Y_p^T
      Fk  list[N] of [9,9]    F_ll,k = Σ_p a²h Y Y^T
      diag  dict              active_frac/boundary_frac/rank 等逐光诊断
    """
    a = np.asarray(a, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
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
    return dict(a=a, Y=Y, C=C, z=Z, s=S, h=H, F_ss_diag=F_ss_diag,
                B=B, Fk=Fk, P=P, N=N, diag=diag)


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


# ======================================================================
# Full Schur（R3' 修正点 2）与 diag proxy
# ======================================================================
def schur_full(bl, cutoff=DEFAULT_CUTOFF):
    """F_eff = diag(F_ss) − Σ_k B_k F_k† B_k^T，P×P 稠密对称（fp 对称化，原地）。

    R5-P1 smoke 修复（2026-09-01）：在累加循环中显式 del 中间矩阵，避免
    Windows commit 配额紧张时 numpy 的 numpy._core._exceptions._ArrayMemoryError。
    """
    P = bl["P"]
    F_eff = np.diag(bl["F_ss_diag"])
    pinv_info = []
    # snapshot B list locally so we can del the per-iter arrays without breaking bl["B"] length
    B_list = bl["B"]
    Fk_list = bl["Fk"]
    N = bl["N"]
    for k in range(N):
        Fk_inv, rank, lmax = pinv_psd(Fk_list[k], cutoff)
        Bk = B_list[k]
        # F_eff -= Bk @ Fk_inv @ Bk.T
        G = Fk_inv @ Bk.T
        update = Bk @ G
        F_eff -= update
        del Fk_inv, G, update
        gc.collect()
        pinv_info.append(dict(k=k, rank=rank, lam_max=lmax))
    F_eff += F_eff.T          # 转置视图原地累加（numpy 检测重叠自动缓冲）
    F_eff *= 0.5
    bl.setdefault("diag", {})["pinv_info"] = pinv_info
    return F_eff


def schur_diag_proxy(bl, cutoff=DEFAULT_CUTOFF):
    """diag-Schur proxy：diag(F_eff) 逐像素标量（正确对角；非完整 Schur）。"""
    d = bl["F_ss_diag"].copy()
    for k in range(bl["N"]):
        Fk_inv, _, _ = pinv_psd(bl["Fk"][k], cutoff)
        d -= (bl["B"][k] * (bl["B"][k] @ Fk_inv)).sum(axis=1)
    return d


# ======================================================================
# Gauge
# ======================================================================
def gauge_unit(a):
    """尺度 gauge 方向（单位化）。"""
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def gauge_project(F_eff, a):
    """Π_g F_eff Π_g, Π_g = I − ââ^T。"""
    g = gauge_unit(a)
    Proj = np.eye(F_eff.shape[0]) - np.outer(g, g)
    return Proj @ F_eff @ Proj


def gauge_residual(F_eff, a):
    """||F_eff â|| / ||F_eff||₂（解析应为 0；cutoff 伪逆下应极小）。"""
    denom = max(np.linalg.norm(F_eff, 2), 1e-300)
    return float(np.linalg.norm(F_eff @ gauge_unit(a)) / denom)


# ======================================================================
# 无量纲谱指标
# ======================================================================
def spectrum_metrics(F_eff, cutoff=DEFAULT_CUTOFF, spec_cutoff=1e-8):
    """全谱指标（trace 归一 → 无量纲）。

    spec_cutoff：正谱判定阈（相对 trace 归一化后的 λ̃），与 F_k 伪逆
    cutoff 语义不同（后者作用于 9×9 块）。

    R5-P0 additive: 输出 `n_at_cutoff` 字段（无量纲 λ̃ 落在
    [spec_cutoff, 100×spec_cutoff] 区间的 eigenvalue 数量），用于 surface
    boundary-granularity 敏感度（IDENTIFIABILITY_v3 §7 与 r5_p1_a_boundary_diagnostic.md）。
    """
    w = np.linalg.eigvalsh(F_eff)                # 输入已对称（schur_full 保证）
    tr = float(w.sum())
    if tr <= 0:
        return dict(lam_min_pos_norm=0.0, lam_max_norm=0.0, logdet_pos_norm=float("-inf"),
                    a_opt_pos_norm=float("inf"), d_pos=0, min_eig=float(w.min()),
                    trace=0.0, n_at_cutoff=0)
    lam_n = w / tr                                       # 无量纲谱
    pos = lam_n > spec_cutoff
    d = int(pos.sum())
    # boundary window: λ̃ in (spec_cutoff, 100*spec_cutoff]
    upper = 100.0 * spec_cutoff
    n_at_cutoff = int(((lam_n > spec_cutoff) & (lam_n <= upper)).sum())
    if d == 0:
        return dict(lam_min_pos_norm=0.0, lam_max_norm=float(lam_n.max()),
                    logdet_pos_norm=float("-inf"), a_opt_pos_norm=float("inf"),
                    d_pos=0, min_eig=float(w.min()), trace=tr, n_at_cutoff=n_at_cutoff)
    lp = lam_n[pos]
    return dict(
        lam_min_pos_norm=float(lp.min()),
        lam_max_norm=float(lp.max()),
        logdet_pos_norm=float(np.log(lp).mean()),
        a_opt_pos_norm=float((1.0 / lp).mean()),
        d_pos=d,
        min_eig=float(w.min()),                          # PSD 检查用（应 ≥ −fp tol）
        trace=tr,
        n_at_cutoff=n_at_cutoff,
    )


def diag_proxy_metrics(d, spec_cutoff=1e-8):
    """diag-Schur proxy 的同构指标（P 个标量当"谱"，仅供分栏对照）。"""
    d = np.asarray(d, dtype=np.float64)
    tr = float(d.sum())
    if tr <= 0:
        return dict(proxy_lam_min_norm=0.0, proxy_logdet_norm=float("-inf"),
                    proxy_a_opt_norm=float("inf"), proxy_valid_frac=0.0)
    dn = d / tr
    pos = dn > spec_cutoff
    n = int(pos.sum())
    if n == 0:
        return dict(proxy_lam_min_norm=0.0, proxy_logdet_norm=float("-inf"),
                    proxy_a_opt_norm=float("inf"), proxy_valid_frac=0.0)
    return dict(proxy_lam_min_norm=float(dn[pos].min()),
                proxy_logdet_norm=float(np.log(dn[pos]).mean()),
                proxy_a_opt_norm=float((1.0 / dn[pos]).mean()),
                proxy_valid_frac=float(pos.mean()))


# ======================================================================
# Operator 路径（大 P；不显式构造 P×P）
#   v ↦ F_eff v = diag(F_ss)⊙v − Σ_k w_k ⊙ (Y (F_k† (Yᵀ(w_k⊙v))))
#   w_k = a·s_k·h_k；每光 O(P·9)。trace 有闭式；λ_min⁺ 用 eigsh 移位反演。
# ======================================================================
def schur_operator(bl, cutoff=DEFAULT_CUTOFF):
    """返回 (apply, trace, meta)。apply: callable v[P] -> F_eff v[P]。"""
    a, Y = bl["a"], bl["Y"]
    N, P = bl["N"], bl["P"]
    w = [a * bl["s"][k] * bl["h"][k] for k in range(N)]
    Fk_inv = []
    pinv_info = []
    for k in range(N):
        Finv, rank, lmax = pinv_psd(bl["Fk"][k], cutoff)
        Fk_inv.append(Finv)
        pinv_info.append(dict(k=k, rank=rank, lam_max=lmax))
    bl.setdefault("diag", {})["pinv_info"] = pinv_info

    def apply(v):
        out = bl["F_ss_diag"] * v
        for k in range(N):
            u = Fk_inv[k] @ (Y.T @ (w[k] * v))
            out -= w[k] * (Y @ u)
        return out

    trace = float(bl["F_ss_diag"].sum())
    for k in range(N):
        # tr(B_k F_k† B_kᵀ) = tr(F_k† · Yᵀ diag(w_k²) Y)
        BtB = (Y * (w[k] ** 2)[:, None]).T @ Y
        trace -= float(np.trace(Fk_inv[k] @ BtB))
    meta = dict(rank_Fk_min=min(i["rank"] for i in pinv_info),
                active_frac_min=float(bl["diag"]["active_frac"].min()),
                boundary_frac_max=float(bl["diag"]["boundary_frac"].max()))
    return apply, trace, meta


def lambda_min_pos_eigsh(bl, apply, trace, k_extra=8, shift_rel=1e-6,
                         spec_cutoff=1e-8, cg_rtol=1e-10):
    """eigsh 移位反演求最小正特征值（主指标路径，任意 P）。

    F_eff ⪰ 0，核 = gauge 方向 â + 全 inactive 像素（解析已知：
    dead ⟺ F_ss_diag[p]=0，维数 n_null = 1 + n_dead）。σ = −ε 移位后核
    被推到变换谱最大处，紧随其后的就是最小正特征值。
    返回 dict(lam_min_pos, lam_min_pos_norm, n_null, n_eig_returned)。
    """
    from scipy.sparse.linalg import LinearOperator as SLO, eigsh, cg
    P = bl["P"]
    g = gauge_unit(bl["a"])
    dead = bl["F_ss_diag"] <= dead_tol_abs(bl)
    n_null = 1 + int(dead.sum())
    shift = shift_rel * max(trace / max(P, 1), 1e-300)

    def Amv(v):
        v = v * (~dead)
        out = apply(v)
        out -= g * (g @ out)
        return out * (~dead)

    A = SLO((P, P), matvec=Amv, dtype=np.float64)
    eps = max(shift, 1e-300)

    def OPinv_mv(b):
        Ashift = SLO((P, P), matvec=lambda t: Amv(t) + eps * t, dtype=np.float64)
        x, _ = cg(Ashift, b, rtol=cg_rtol, atol=0.0)
        return x

    OPinv = SLO((P, P), matvec=OPinv_mv, dtype=np.float64)
    k = min(n_null + k_extra, P - 1)
    vals = eigsh(A, k=k, sigma=-eps, which="LM", OPinv=OPinv,
                 mode="normal", return_eigenvectors=False)
    vals = np.sort(np.asarray(vals, dtype=np.float64))
    pos = vals[vals > spec_cutoff * max(trace, 1e-300)]
    out = dict(lam_min_pos=float(pos.min()) if pos.size else 0.0,
               n_null=n_null, n_eig_returned=int(k))
    out["lam_min_pos_norm"] = out["lam_min_pos"] / max(trace, 1e-300)
    return out


def dead_tol_abs(bl):
    """全 inactive 判定阈：F_ss_diag[p] ≤ 1e-12·max(F_ss)。"""
    return 1e-12 * max(float(bl["F_ss_diag"].max()), 1e-300)


def n_dead_count(bl):
    """全 inactive 像素数（任何光都打不到 → F_ss_diag[p] = 0）。"""
    return int((bl["F_ss_diag"] <= dead_tol_abs(bl)).sum())


def structural_null_gate(bl, d_pos_observed=None):
    """R5-P0 · T0.2 structural-null gate（IDENTIFIABILITY_v3.md §7）。

    Returns
    -------
    dict with keys
      P            : int
      n_dead       : int   (全 inactive 像素数)
      d_expected   : int   (= P - n_dead - 1，gauge 投影后 Π_g F_eff Π_g 的解析期望 dim)
      d_pos        : int   (#{λ̃_i > spec_cutoff}; d_pos_observed 提供时用其值，
                            否则 NaN 占位 — 在 dense 路径外不可用)
      d_extra_null : int   (= d_expected - d_pos; > 0 标记 structurally deficient)
      structural_status : {'full', 'deficient', 'flip', 'unknown'}
    """
    P = int(bl["P"])
    n_dead = n_dead_count(bl)
    d_expected = P - n_dead - 1
    if d_pos_observed is None or (isinstance(d_pos_observed, float) and np.isnan(d_pos_observed)):
        status = "unknown"
        d_pos = -1
        d_extra_null = -1
    else:
        d_pos = int(d_pos_observed)
        d_extra_null = d_expected - d_pos
        if d_extra_null == 0:
            status = "full"
        elif d_extra_null > 0:
            status = "deficient"
        else:
            status = "flip"
    return dict(P=P, n_dead=n_dead, d_expected=d_expected,
                d_pos=d_pos, d_extra_null=d_extra_null,
                structural_status=status)


# ======================================================================
# 端到端：dense 路径（P ≤ DENSE_MAX_P）/ operator 路径（大 P）
# ======================================================================
def ga_isi_v2_scores(a, Y, C, cutoff=DEFAULT_CUTOFF, want_proxy=True,
                     path="auto"):
    """给定像素集合的 (a, Y, C) → 全套 v2 指标。

    path="auto"：P ≤ DENSE_MAX_P 走 dense 全谱（精确）；
    否则 operator + eigsh（primary λ_min⁺ 精确求取；logdet/A-opt 等全谱
    二级指标在大 P 下返回 NaN 并标注 path，预注册像素策略 ≤ DENSE_MAX_P
    时二级指标不受影响）。
    返回 dict（CSV 列，full_* 与 diagproxy_* 分栏）。
    """
    bl = fisher_blocks(a, Y, C)
    use_op = (path == "operator") or (path == "auto" and bl["P"] > DENSE_MAX_P)
    if not use_op:
        F_eff = schur_full(bl, cutoff)
        P = bl["P"]
        # 内存精简：offdiag 分块求 max（避免 3×P² 临时矩阵；本机 commit 配额紧张）
        d_sl = np.diag(F_eff)
        offdiag_max = 0.0
        for r0 in range(0, P, 256):
            r1 = min(r0 + 256, P)
            blk = F_eff[r0:r1].copy()
            blk[np.arange(r0, r1) - r0, np.arange(r0, r1)] = 0.0
            if blk.size:
                offdiag_max = max(offdiag_max, float(np.abs(blk).max()))
            del blk
        m = spectrum_metrics(F_eff)                 # F_eff 已对称，eigvalsh 直接用
        row = dict(
            P=P, N=bl["N"], cutoff=cutoff, path="dense",
            full_lam_min_pos_norm=m["lam_min_pos_norm"],      # ← primary
            full_lam_max_norm=m["lam_max_norm"],
            full_logdet_pos_norm=m["logdet_pos_norm"],
            full_a_opt_pos_norm=m["a_opt_pos_norm"],
            full_d_pos=m["d_pos"],
            full_trace=m["trace"],
            full_min_eig=m["min_eig"],
            full_gauge_residual=gauge_residual(F_eff, bl["a"]),
            full_offdiag_max=offdiag_max,
            rank_Fk_min=min(i["rank"] for i in bl["diag"]["pinv_info"]),
            active_frac_min=float(bl["diag"]["active_frac"].min()),
            boundary_frac_max=float(bl["diag"]["boundary_frac"].max()),
            full_n_at_cutoff=m["n_at_cutoff"],            # R5-P0 additive: boundary granularity
        )
        # R5-P0 · T0.2 structural-null gate
        sn = structural_null_gate(bl, d_pos_observed=m["d_pos"])
        row.update(sn)
        del F_eff
        gc.collect()
    else:
        apply_f, trace, meta = schur_operator(bl, cutoff)
        eigs = lambda_min_pos_eigsh(bl, apply_f, trace)
        row = dict(
            P=bl["P"], N=bl["N"], cutoff=cutoff, path="operator",
            full_lam_min_pos_norm=eigs["lam_min_pos_norm"],   # ← primary
            full_lam_max_norm=float("nan"),
            full_logdet_pos_norm=float("nan"),
            full_a_opt_pos_norm=float("nan"),
            full_d_pos=float("nan"),
            full_trace=trace,
            full_min_eig=0.0,
            full_gauge_residual=float("nan"),
            full_offdiag_max=float("nan"),
            rank_Fk_min=meta["rank_Fk_min"],
            active_frac_min=meta["active_frac_min"],
            boundary_frac_max=meta["boundary_frac_max"],
            eigsh_n_null=eigs["n_null"],
            full_n_at_cutoff=float("nan"),                # operator 路径无全谱
        )
        # R5-P0 · T0.2 structural-null gate（operator 路径 d_pos 未知 → status='unknown'）
        sn = structural_null_gate(bl, d_pos_observed=float("nan"))
        row.update(sn)
    if want_proxy:
        d = schur_diag_proxy(bl, cutoff)
        row.update(diag_proxy_metrics(d))
    return row


# ======================================================================
# 数据加载（与 v1 同格式；v2 仅替换度量核心）
# ======================================================================
def load_scene(path):
    sc = {"dir": path, "name": os.path.basename(path)}
    K = len([f for f in os.listdir(path) if f.startswith("light_") and f.endswith("_lin.npy")])
    sc["K"] = K
    sc["imgs_lin"] = np.stack([np.load(os.path.join(path, f"light_{k+1:03d}_lin.npy"))
                               for k in range(K)])
    sc["sh_irr"] = np.load(os.path.join(path, "sh_coeffs_irradiance.npy"))
    sc["albedo"] = np.load(os.path.join(path, "albedo.npy"))[0]
    sc["n_mesh"] = np.load(os.path.join(path, "normal_mesh.npy"))
    sc["mask"] = np.load(os.path.join(path, "mask.npy"))[0].astype(bool)
    return sc


def scene_arrays(sc, subset, pixel_cap=2000, seed=0, fix_gauge=True):
    """场景 dict + 光照子集 → (a, Y, C)。像素超上限时随机下采样（seed 落盘由调用方负责）。"""
    mask = sc["mask"]
    idx = np.argwhere(mask)
    if len(idx) > pixel_cap:
        rng = np.random.default_rng(seed)
        idx = idx[rng.choice(len(idx), pixel_cap, replace=False)]
    a = sc["albedo"][idx[:, 0], idx[:, 1]].astype(np.float64)
    if fix_gauge:
        # gauge fixing（换基，不影响无量纲谱指标；T4 断言）
        rms = np.sqrt((a * a).mean())
        a = a / max(rms, 1e-9)
    n = sc["n_mesh"].transpose(1, 2, 0)
    n_pts = n[idx[:, 0], idx[:, 1]]                       # [P,3]
    Y = sh_basis_npy(n_pts)                               # [P,9]
    C = sc["sh_irr"][np.asarray(subset, dtype=int)]       # [N,9]
    return a, Y, C


# ======================================================================
# CLI（与 v1 同接口 + v2 新旗标；R4'-D 复跑直接可用）
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="GA-ISI v2 (full-Schur) — R3' 正式版")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--ns", nargs="+", type=int, default=[3, 5, 8, 12])
    ap.add_argument("--subsets_per_N", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--pixel_cap", type=int, default=2000)
    ap.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF)
    ap.add_argument("--no_proxy", action="store_true", help="不输出 diag proxy 分栏")
    args = ap.parse_args()

    scenes = [os.path.join(args.data_root, d) for d in sorted(os.listdir(args.data_root))
              if os.path.isdir(os.path.join(args.data_root, d))
              and os.path.isfile(os.path.join(args.data_root, d, "sh_coeffs_irradiance.npy"))]
    rng = np.random.default_rng(args.seed)
    rows = []
    for sd in scenes:
        sc = load_scene(sd)
        for N in args.ns:
            if N > sc["K"]:
                continue
            for si in range(args.subsets_per_N):
                sub = sorted(rng.choice(sc["K"], N, replace=False).tolist())
                a, Y, C = scene_arrays(sc, sub, args.pixel_cap, seed=rng.integers(1 << 31))
                r = ga_isi_v2_scores(a, Y, C, cutoff=args.cutoff,
                                     want_proxy=not args.no_proxy)
                r.update(scene=sc["name"], N=N, subset=",".join(map(str, sub)))
                rows.append(r)
        print(f"  {sc['name']}: done ({len(args.ns)} N × {args.subsets_per_N} subsets)",
              flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"[ga-isi-v2] {args.out_csv} rows={len(rows)}")
    # 存在性前提摘要：full primary 分数在固定 N 内应有宽度
    import numpy as np2
    for N in args.ns:
        v = np2.array([r["full_lam_min_pos_norm"] for r in rows
                       if r["N"] == N and r["full_lam_min_pos_norm"] > 0])
        if v.size:
            print(f"  N={N}: full λ̃⁺_min 分布 min={v.min():.3e} med={np2.median(v):.3e} "
                  f"max={v.max():.3e}（有宽度=子集质量确实不同）")


if __name__ == "__main__":
    main()
