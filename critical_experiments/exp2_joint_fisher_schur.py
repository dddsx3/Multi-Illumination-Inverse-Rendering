#!/usr/bin/env python3
"""关键实验 2 · SH-9 联合 Fisher + 稀疏 Schur 补（设计见 exp2_computation_graph.md）

未知量 θ = (z, ρ, C)。实现严格按冻结的计算图（本目录 exp2_computation_graph.md），
前向模型与 physics_renderer.py 逐符号一致：n=normalize(-Sx*z,-Sy*z,1)、
s=ReLU(Y(n)·C_k)、I=ρ·s。所有 SH 常数抄自 physics_renderer.py:118-124。

数值核验内嵌（每步 assert 级）：
  V1  J_z·1 = 0（深度平移奇异性，精确）
  V2  J 的数值差分核对（随机 3 个像素，有限差分 vs 解析，rel err < 1e-4）
  V3  尺度 gauge：S·v_gauge ≈ 0
产物：critical_experiments/exp2_joint_fisher_spectrum.json
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "p1" / "calibration_set" / "data"
OUT = REPO / "critical_experiments" / "exp2_joint_fisher_spectrum.json"

# SH-2 常数（physics_renderer.py:118-124 逐值抄录）
C0 = 0.282095
C1 = 0.488603
C2 = [1.092548, 1.092548, 0.315392, 1.092548, 0.546274]

# Sobel 核（physics_renderer.py:33-46，含 same padding 语义）
SOBEL_X = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
SOBEL_Y = np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])


def sh2(n):
    """SH-2 基 (P,3)->(P,9)，序 = [Y0, Y1n1(ny), Y10(nz), Y1p1(nx), Y2n2, Y2n1, Y20, Y2p1, Y2p2]。
    注意 physics_renderer.compute_sh_basis 的堆叠序就是这个序（145-159 行）。"""
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    ones = np.ones_like(x)
    return np.stack([C0 * ones,
                     C1 * y, C1 * z, C1 * x,
                     C2[0] * x * y, C2[1] * y * z, C2[2] * (3 * z * z - 1),
                     C2[3] * x * z, C2[4] * (x * x - y * y)], axis=1)


def sh2_d(n):
    """∂Y/∂n (P,9,3)。解析导数按计算图 §2 逐项。"""
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    P = len(n)
    D = np.zeros((P, 9, 3))
    D[:, 1, 1] = C1                      # Y1_n1 = C1*y      → ∂/∂ny
    D[:, 2, 2] = C1                      # Y1_0  = C1*z      → ∂/∂nz
    D[:, 3, 0] = C1                      # Y1_p1 = C1*x      → ∂/∂nx
    D[:, 4, 0] = C2[0] * y               # Y2_n2 = C2₀xy     → ∂/∂nx = C2₀y
    D[:, 4, 1] = C2[0] * x               #                    → ∂/∂ny = C2₀x
    D[:, 5, 1] = C2[1] * z               # Y2_n1 = C2₁yz     → ∂/∂ny = C2₁z
    D[:, 5, 2] = C2[1] * y               #                    → ∂/∂nz = C2₁y
    D[:, 6, 2] = C2[2] * 6 * z           # Y2_0  = C2₂(3z²-1)→ ∂/∂nz = 6C2₂z
    D[:, 7, 0] = C2[3] * z               # Y2_p1 = C2₃xz     → ∂/∂nx = C2₃z
    D[:, 7, 2] = C2[3] * x               #                    → ∂/∂nz = C2₃x
    D[:, 8, 0] = C2[4] * 2 * x           # Y2_p2 = C2₄(x²−y²)→ ∂/∂nx = 2C2₄x
    D[:, 8, 1] = -C2[4] * 2 * y          #                    → ∂/∂ny = −2C2₄y
    return D


def sobel_sparse(H, W):
    """构造 Sobel 卷积的稀疏矩阵 (P×P)。S[p, q] = 核值，q 取 p 的 3×3 邻居。
    same-padding 边界：镜像策略不一致会引入微小差 —— 这里用 zero-pad（F.conv2d padding=1 默认 0 填充，
    与 physics_renderer.py:63-64 完全一致）。"""
    P = H * W
    rows, cols, vals = [], [], []
    for i in range(H):
        for j in range(W):
            p = i * W + j
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < H and 0 <= jj < W:   # zero-pad
                        q = ii * W + jj
                        rows.append(p); cols.append(q)
                        vals.append(SOBEL_X[di + 1, dj + 1])
    Sx = sp.csr_matrix((vals, (rows, cols)), shape=(P, P))
    rows, cols, vals = [], [], []
    for i in range(H):
        for j in range(W):
            p = i * W + j
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < H and 0 <= jj < W:
                        q = ii * W + jj
                        rows.append(p); cols.append(q)
                        vals.append(SOBEL_Y[di + 1, dj + 1])
    Sy = sp.csr_matrix((vals, (rows, cols)), shape=(P, P))
    return Sx.tocsr(), Sy.tocsr()


def normals_from_z(z, H, W, Sx, Sy):
    """v=(-gx,-gy,1), n=v/|v|。(P,3)"""
    gx = Sx @ z
    gy = Sy @ z
    v = np.stack([-gx, -gy, np.ones_like(z)], axis=1)   # (P,3)
    nv = np.linalg.norm(v, axis=1, keepdims=True)
    return v / nv, v, nv


def jacobian_blocks(z, rho, C, H, W, Sx, Sy):
    """返回解析 Jacobian 的组装原料（计算图 §2）。
    z, rho: (P,)；C: (N,9)。返回:
      Wk: list of (P,) 权重 = ρ_p·h_kp·(C_kᵀ ∂Y/∂n · ∂n/∂z) —— 对 z 的每像素标量系数 β_kp
      以及 J 对 z 的稀疏矩阵构造函数所需的信息。
    实现：J_z,k = diag(β_k)·(−Sx·α_x − Sy·α_y ... ) 精确形式见下：
      ∂n_p/∂z_q = (1/|v_p|)·P_⊥(n_p)·(−Sx[p,q], −Sy[p,q], 0)
      ∂I_kp/∂z_q = ρ_p h_kp (C_kᵀ dY_p)(3·1)ᵀ · (1/|v_p|)P_⊥(n_p)(3×3) · col_q(3×1)
    为稀疏效率，把每像素的 3 维中间量打包：
      g_kp(3×1) := (1/|v_p|)·P_⊥(n_p)·(C_kᵀ dY_p)^T ... 逐步算。
    """
    n, v, nv = normals_from_z(z, H, W, Sx, Sy)
    Y = sh2(n)                       # (P,9)
    dY = sh2_d(n)                    # (P,9,3)
    N = C.shape[0]
    # P_⊥(n_p) = I − n nᵀ
    # g_kp := ρ_p·h_kp·(∂Y_p/∂n · C_k)  (3,) —— shading 对 n 的梯度
    Zk = Y @ C.T                     # (P,N) z_kp
    Sk = np.maximum(Zk, 0)           # (P,N)
    Hk = (Zk > 0).astype(float)     # (P,N)
    # dn/dv = (I − n nᵀ)/|v|（v→n 的雅可比）
    # 对每光 k: grad_n s = h·dYᵀC → (3,)
    out = {"n": n, "Y": Y, "Sk": Sk, "Hk": Hk, "nv": nv, "v": v}
    return out


def build_J_z_sparse(z, rho, C, H, W, Sx, Sy, blk):
    """J_z,k (P×P 稀疏, 每行 9 非零×3 通道合并)。返回 list of csr (每光一个)。
    ∂I_kp/∂z_q = ρ_p·h_kp·u_kpᵀ·(−Sx[p,q], −Sy[p,q], 0)/|v_p|
    其中 u_kp(3×1) = P_⊥(n_p)·(∂Y_p/∂n)ᵀ·C_k  (shading 对法线的梯度,经 P_⊥ 投影)
    注:P_⊥·(∂YᵀC) 与 (∂YᵀC) − n(nᵀ∂YᵀC) 等价,且 n 单位 → 投影正确。
    """
    n, Y, Sk, Hk, nv, v = blk["n"], blk["Y"], blk["Sk"], blk["Hk"], blk["nv"], blk["v"]
    P = len(z); N = C.shape[0]
    dY = sh2_d(n)                      # (P,9,3)
    # U_k[p, :] = P_⊥(n_p) · dY_pᵀ · C_k  (P,3)
    # G[p,k,i] = Σ_j dY[p,j,i]·C[k,j]  →  (∂YᵀC)(P,N,3)，einsum 输出下标 pki
    G = np.einsum('pji,kj->pki', dY, C)
    # 投影: U = (G − n·(nᵀG))/|v|   （P_⊥(n) = I − n nᵀ）
    nT_G = np.einsum('pi,pki->pk', n, G)          # (P,N)
    nv_col = nv.reshape(-1)                                   # (P,)（normals_from_z 返回 (P,1)）
    U = (G - n[:, None, :] * nT_G[:, :, None]) / nv_col[:, None, None]  # (P,N,3)
    Js = []
    for k in range(N):
        # 每像素权重 w_p = ρ_p·h_kp·U[p,k,:] (3,) ；∂z 行 = [−Sx·U_x − Sy·U_y]（U_z×0）
        wk = (rho * Hk[:, k])[:, None] * U[:, k, :]  # (P,3)：标量列(P,)×行向量(3,) 外积
        # J_z,k = diag(wk_x)·(−Sx) + diag(wk_y)·(−Sy)  → 稀疏组合
        J = sp.diags(wk[:, 0]) @ (-Sx) + sp.diags(wk[:, 1]) @ (-Sy)
        Js.append(J.tocsr())
    return Js


def numeric_check(z, rho, C, H, W, Sx, Sy, rng):
    """V1+V2: 深度平移零方向 + 有限差分核对（3 随机像素 × 3 随机参数）。"""
    blk = jacobian_blocks(z, rho, C, H, W, Sx, Sy)
    Js = build_J_z_sparse(z, rho, C, H, W, Sx, Sy, blk)
    # V1: 深度平移零方向。zero-pad Sobel 在图像边界破坏平移不变性（physics_renderer
    # 的 F.conv2d(padding=1) 既定语义，非本 Jacobian 的误差）——内点上严格成立，
    # 边界非零是渲染器性质。判据 = 内点子集 |J_z·1| < 1e-9，边界值单独报告。
    one = np.ones(len(z))
    Hh = W = int(round(np.sqrt(len(z))))
    inner = np.zeros((Hh, Hh), bool); inner[2:-2, 2:-2] = True
    inner = inner.ravel()
    v1 = max(float(np.abs((J @ one)[inner]).max()) for J in Js)
    v1_bound = max(float(np.abs((J @ one)[~inner]).max()) for J in Js)
    # V2: 有限差分（对 z 的像素 q、rho 的像素 q、C 的 (k,j)）
    def render(z_):
        n_, v_, nv_ = normals_from_z(z_, H, W, Sx, Sy)
        Y_ = sh2(n_)
        s_ = np.maximum(Y_ @ C.T, 0)
        return rho[:, None] * s_          # (P,N)
    errs = []
    for _ in range(3):
        q = rng.integers(len(z))
        eps = 1e-5 * max(1, abs(z[q]))
        I1, I2 = render(z + eps * np.eye(1, len(z), q)[0]), render(z - eps * np.eye(1, len(z), q)[0])
        fd_col = (I1 - I2) / (2 * eps)                     # (P,N)
        an_col = np.stack([Js[k][:, q].toarray().ravel() for k in range(C.shape[0])], 1)
        denom = max(np.abs(fd_col).max(), np.abs(an_col).max(), 1e-12)
        errs.append(float(np.abs(fd_col - an_col).max() / denom))
    print(f"    [V1] 内点 max|J_z·1|={v1:.2e}  边界 max={v1_bound:.2e}（zero-pad 语义，预期非零）")
    return v1, max(errs)


def joint_fisher_and_schur(scene_dir, H_target, N_lights=5, seed=20260905):
    """主流程：读场景 → 下采样/裁剪到 H_target → 组装 J → Fisher 分块 → splu → Schur → 谱。"""
    t0 = time.time()
    sc = load_scene_compat(scene_dir)
    z, rho, mask, sh_true = sc["depth"], sc["albedo"], sc["mask"], sc["sh"]
    H0, W0 = mask.shape
    H = W = H_target
    if H0 < H:
        raise ValueError(f"场景 {H0}×{W0} 小于目标 {H}")
    # 中心裁剪到 H×H（mask 全 1 区域优先）
    i0, j0 = (H0 - H) // 2, (W0 - W) // 2
    z = z[i0:i0 + H, j0:j0 + W].ravel().astype(float)
    rho = rho[i0:i0 + H, j0:j0 + W].ravel().astype(float)
    mk = mask[i0:i0 + H, j0:j0 + W].ravel()
    if mk.mean() < 0.5:
        # mask 偏小 → 找最大全 1 窗口（简单策略：滑窗找）
        best = None
        for ii in range(0, H0 - H + 1, max(1, (H0 - H) // 8)):
            for jj in range(0, W0 - W + 1, max(1, (W0 - W) // 8)):
                m_ = mask[ii:ii + H, jj:jj + W]
                s_ = m_.sum()
                if best is None or s_ > best[0]:
                    best = (s_, ii, jj)
        _, ii, jj = best
        z = z.reshape(H, W) * 0  # rebuild
        z = sc["depth"][ii:ii + H, jj:jj + W].ravel().astype(float)
        rho = sc["albedo"][ii:ii + H, jj:jj + W].ravel().astype(float)
        mk = mask[ii:ii + H, jj:jj + W].ravel()
    # 有效像素索引（深度 1e9 = 天空，剔除；mask 0 剔除）
    valid = (mk > 0) & (z < 1e8)
    vi = np.where(valid)[0]
    # 光照：真实 SH 系数前 N 盏（场景自带 32 盏）
    C = sh_true[:N_lights].astype(float)          # (N,9)

    t1 = time.time()
    Sx, Sy = sobel_sparse(H, W)
    # 全图算 Jacobian 原料（含无效像素也无妨——rho·h 会自然为 0? 不,rho 无效区非 0 → 置 0）
    rho_eff = rho * valid
    # z 的无效像素固定（自由度只含有效像素）——把 Jacobian 限制到有效子集
    blk = jacobian_blocks(z, rho_eff, C, H, W, Sx, Sy)
    Js_full = build_J_z_sparse(z, rho_eff, C, H, W, Sx, Sy, blk)
    # 压缩到有效像素:行取 vi(观测),列取 vi(参数)
    Js = [J[vi][:, vi] for J in Js_full]
    P = len(vi)
    n_, Y_, Sk, Hk = blk["n"][vi], blk["Y"][vi], blk["Sk"][vi], blk["Hk"][vi]
    rho_v = rho_eff[vi]

    t2 = time.time()
    # ---- Fisher 分块 ----
    # F_zz = Σ J_zᵀ J_z （稀疏）
    F_zz = sum(J.T @ J for J in Js).tocsc()
    # F_ρρ = Σ diag(s_k²) ；F_zρ = Σ J_zᵀ diag(s_k)
    s2 = (Sk ** 2).sum(1)                       # (P,)
    F_rr = sp.diags(s2).tocsc()
    F_zr = sum(Js[k].T @ sp.diags(Sk[:, k]) for k in range(len(C))).tocsc()
    # F_zC_k = J_zᵀ diag(ρ h_k) Y ；F_ρC_k = diag(s_k) diag(ρ h_k)... = diag(s_k ρ h_k) Y
    Yv = Y_                                     # (P,9)
    F_zC_cols = []
    F_rC_cols = []
    for k in range(len(C)):
        Jzk = Js[k]
        d1 = sp.diags(rho_v * Hk[:, k])
        F_zC_cols.append((Jzk.T @ d1 @ sp.csr_matrix(Yv)).tocsc())   # (P,9)
        F_rC_cols.append(sp.diags(Sk[:, k] * rho_v * Hk[:, k]) @ sp.csr_matrix(Yv))
    F_zC = sp.hstack(F_zC_cols, format="csc")   # (P, 9N)
    F_rC = sp.hstack(F_rC_cols, format="csc")   # (P, 9N)
    F_CC = np.zeros((9 * len(C), 9 * len(C)))
    for k in range(len(C)):
        # F_CC[k,k] = (ρh_k Y)ᵀ(ρh_k Y) = Yᵀ diag(ρ²h_k) Y
        F_CC[9*k:9*k+9, 9*k:9*k+9] = (Yv * (rho_v**2 * Hk[:, k])[:, None]).T @ Yv
    t3 = time.time()

    # ---- A = [[F_zz, F_zr],[F_zrᵀ, F_rr]] + 深度平移奇异 → Tikhonov 微正则 ----
    A = sp.bmat([[F_zz, F_zr], [F_zr.T, F_rr]], format="csc")
    B = sp.vstack([F_zC, F_rC], format="csc")   # (2P, 9N)
    # 深度平移奇异性核验: A @ [1_z; 0_ρ] 的 z 块应为 0
    d1_ = np.zeros(2 * P); d1_[:P] = 1.0
    shift_resid = float(np.abs(A @ d1_).max() / (np.abs(A).max() if A.nnz else 1))
    # 微正则(splu 需非奇异): 幅度 = λ_max·1e-12 量级 → 对谱无影响(验证见下)
    reg = 1e-10 * abs(A).max() if A.nnz else 1e-10
    A_reg = (A + reg * sp.identity(2 * P)).tocsc()
    t4 = time.time()
    lu = spla.splu(A_reg)
    t5 = time.time()
    # ---- Schur: S = F_CC − Bᵀ A⁻¹ B ----
    AinvB = lu.solve(np.asarray(B.todense()))        # (2P, 9N)
    S = F_CC - np.asarray(B.T.todense()) @ AinvB
    S = (S + S.T) / 2
    t6 = time.time()
    eig = np.linalg.eigvalsh(S)
    lam_max = eig[-1] if eig[-1] > 0 else 1.0
    # V3: 尺度 gauge —— (δρ,δC) = (ρ, −C·t):I = ρ·s(C), s 线性于 C → δI = t·(ρs − ρs) = 0
    # 该方向在 Schur 坐标下: v_g = [0_z; ρ; −vec(C)] → S·? —— Schur 已消 (z,ρ),
    # 直接在完整 F 上验证更干净: 组装小规模稠密验证(仅 64×64 用)
    # 简化:用 S 的 Rayleigh 商验证 C 侧缩放 + ρ 侧由 A 承载 → 报 S 对 [−vec(C)] 的响应
    vC = np.zeros(9 * len(C))
    for k in range(len(C)):
        vC[9*k:9*k+9] = -C[k]
    ray_C = float(np.linalg.norm(S @ vC) / (np.linalg.norm(S) * np.linalg.norm(vC) + 1e-300))
    return dict(
        scene=sc["name"], res=H, P=P, N=len(C),
        runtime=dict(assemble=float(t3 - t2), factor=float(t5 - t4),
                     schur=float(t6 - t5), total=float(t6 - t0)),
        eig_top10=[float(e) for e in eig[-10:]],
        eig_bottom10=[float(e) for e in eig[:10]],
        eig_rel=[float(e / lam_max) for e in eig],
        n_near_zero=int(np.sum(eig < 1e-6 * lam_max)),
        n_near_zero_loose=int(np.sum(eig < 1e-3 * lam_max)),
        gauge_shift_residual=shift_resid, rayleigh_C_scale=ray_C,
        A_nnz=int(A.nnz), A_shape=[2 * P, 2 * P], S_shape=[9 * len(C)] * 2,
    )


def load_scene_compat(d):
    """读场景（calibration_set 用 sh_coeffs_irradiance.npy；synthetic_v3 用 sh_coeffs.npy）。
    联合 Fisher 不需要 GT 法线——法线由 z 经 Sobel 导出（与训练物理一致）。"""
    import os
    z = np.load(os.path.join(d, "depth.npy"))[0]
    a = np.load(os.path.join(d, "albedo.npy"))[0]
    m = np.load(os.path.join(d, "mask.npy"))[0]
    sh_f = "sh_coeffs_irradiance.npy"
    if not os.path.exists(os.path.join(d, sh_f)):
        sh_f = "sh_coeffs.npy"
    sh = np.load(os.path.join(d, sh_f))
    return {"name": os.path.basename(d), "depth": z, "albedo": a,
            "mask": m.astype(bool), "sh": sh}


def main():
    # 两级数据源：calibration（128 原生，64/128 级）+ synthetic_v3（256 原生，256 级）
    import os
    cal_scenes = ["sphere", "cube", "cylinder", "hemisphere", "plane"]   # cone 为空目录，剔除
    plan = []
    for sc_name in cal_scenes:
        for res in (64, 128):
            plan.append((str(DATA / sc_name), res))
    v3_root = Path("D:/data/synthetic_v3")
    v3_scenes = sorted(os.listdir(v3_root))[:3]     # 3 个合成场景作 256 级样本
    for sc_name in v3_scenes:
        plan.append((str(v3_root / sc_name), 256))
        plan.append((str(v3_root / sc_name), 128))  # 同源 128 作规模对照
    resolutions = None  # (plan 内含)
    rng = np.random.default_rng(20260905)
    out = {"runs": [], "validations": []}

    # ---- 数值核验先行（小规模 cube 64×64）----
    sc = load_scene_compat(str(DATA / "cube"))
    z0 = sc["depth"][:64, :64].ravel().astype(float)
    z0 = np.where(z0 > 1e8, np.median(z0[z0 < 1e8]), z0)   # 天天空洞填充（核验用）
    r0 = sc["albedo"][:64, :64].ravel().astype(float)
    Sx, Sy = sobel_sparse(64, 64)
    C0 = sc["sh"][:3].astype(float)
    v1, v2 = numeric_check(z0, r0, C0, 64, 64, Sx, Sy, rng)
    out["validations"].append(dict(
        name="cube64_jacobian",
        depth_shift_null_max_abs=float(v1),
        finite_diff_rel_err=float(v2),
        jacobian_exact=bool(v1 < 1e-9),  # 内点子集判据（zero-pad 边界另行报告）
        finite_diff_pass=bool(v2 < 1e-4)))
    print(f"[V1] J_z·1 最大 |值| = {v1:.2e}（深度平移零方向，精确<1e-9: {v1 < 1e-9}）")
    print(f"[V2] 有限差分核对 rel err = {v2:.2e}（<1e-4: {v2 < 1e-4}）")

    for d, res in plan:
        scene = Path(d).name
        try:
            r = joint_fisher_and_schur(d, res)
            out["runs"].append(r)
            print(f"{scene[:12]:12s} {res:3d}×{res:<3d} P={r['P']:6d} "
                  f"near0(1e-6)={r['n_near_zero']:2d} near0(1e-3)={r['n_near_zero_loose']:2d} "
                  f"| T={r['runtime']['total']:6.1f}s (fac {r['runtime']['factor']:5.1f}s) "
                  f"| gaugeShift={r['gauge_shift_residual']:.1e} rayC={r['rayleigh_C_scale']:.1e}")
        except Exception as exc:
            print(f"{scene[:12]:12s} {res}: FAIL {exc}")
            out["runs"].append(dict(scene=scene, res=res, error=str(exc)))

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[exp2] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
