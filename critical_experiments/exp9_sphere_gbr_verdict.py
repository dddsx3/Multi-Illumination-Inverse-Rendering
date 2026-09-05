#!/usr/bin/env python3
"""关键实验 9(卡 A)· 球体 GBR 三判别 + 核消融 + 预测(ii) —— 干净实现

设计:exp9_computation_graph.md(冻结)。实现采用【观测残差式】:
  生成元切向量的观测像 δI_kp = ρ_p h_kp (C_kᵀ dY_p)·δn_p + s_kp·δρ_p(几何部分)
  δn_p = P_⊥(n_p)·δv_p/‖v_p‖,δv 解析(GBR 一阶):
     λ: δv=(v_x,0,0)  μ: δv=(-8,0,0)  ν: δv=(0,-8,0)   [8=Sobel 线性增益]
     δρ_p = ρ_p(v_p·δv_p)/‖v_p‖²
  δC 口径乙(最优补偿):每光独立 9×9 LSQ min_{δC_k} Σ_p(δI^geom_kp + ρ h Y_pᵀδC_k)²
  Rayleigh = Σ(残差²)/‖δθ‖²,‖δθ‖² = ‖δz‖²+‖δρ‖²+‖δC‖²(参数空间)

判别:
  (a) sphere 三生成元相对 Rayleigh + 底部 4 维特征子空间主夹角(参数空间, LOBPCG)
  (b) 残差逐像素能量 × 轮廓带标记 + 32/64/128 分辨率扫描
  (c) 核消融 toy 球体: 标准核 GBR Rayleigh(预期机器零——SH-2 卷积定理下 GBR 精确)
      vs l=0 段×5 修改核(模仿数据核, 预期上升)——直接定位破缺来源
预测(ii): 合成方向光配置 η=(E0²+E2²)/(E1²) vs GBR Rayleigh 相关

核验(预注册):
  V3 生成元切向量有限差分核验: 扰动渲染 vs δI 一阶预测, rel<1e-6
  V4 全局尺度方向 Rayleigh(已知精确零)作阴性对照
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))

from exp2_joint_fisher_schur import (  # noqa: E402
    DATA, load_scene_compat, sobel_sparse, sh2, sh2_d,
    jacobian_blocks, build_J_z_sparse,
)

OUT = HERE / "exp9_sphere_gbr_verdict.json"
SCENES = ["sphere", "cube", "cylinder", "hemisphere"]
RES = 64
N_LIGHTS = 5
SEED = 20260905
SOBEL_GAIN = 8.0          # Sobel 对线性函数的响应(代码口径)
K_STD = np.array([np.pi, 2*np.pi/3, 2*np.pi/3, 2*np.pi/3] + [np.pi/4]*5)  # 标准 clamped-cosine 核


def sh2_d_raw(n):
    """∂Y2/∂n (P,9,3)。"""
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    D = np.zeros((len(n), 9, 3))
    D[:, 1, 1] = 0.488603
    D[:, 2, 2] = 0.488603
    D[:, 3, 0] = 0.488603
    D[:, 4, 0] = 1.092548 * y
    D[:, 4, 1] = 1.092548 * x
    D[:, 5, 1] = 1.092548 * z
    D[:, 5, 2] = 1.092548 * y
    D[:, 6, 2] = 0.315392 * 6 * z
    D[:, 7, 0] = 1.092548 * z
    D[:, 7, 2] = 1.092548 * x
    D[:, 8, 0] = 0.546274 * 2 * x
    D[:, 8, 1] = -0.546274 * 2 * y
    return D


def make_sc(z, rho, valid, C, H, W, name="synthetic"):
    """从场构造 scene dict(全图口径, 与 scene_full 输出同构)。"""
    Sx, Sy = sobel_sparse(H, W)
    v = np.stack([-(Sx @ z), -(Sy @ z), np.ones_like(z)], axis=1)
    nv = np.linalg.norm(v, axis=1)
    n = v / nv[:, None]
    vi = np.where(valid)[0]
    return dict(z=z, rho=rho, valid=valid, vi=vi, C=C, H=H, W=W, Sx=Sx, Sy=Sy,
                v=v, nv=nv, n=n, contour=None, name=name)


def scene_full(scene, res):
    """全图场景量(含无效像素)——生成元切向量需在全图 Sobel 口径下构造。"""
    sc = load_scene_compat(str(DATA / scene))
    z, rho, mask = sc["depth"], sc["albedo"], sc["mask"]
    H0, W0 = mask.shape
    H = W = res
    i0, j0 = (H0 - H) // 2, (W0 - H) // 2
    z = z[i0:i0+H, j0:j0+W].ravel().astype(float)
    rho = rho[i0:i0+H, j0:j0+W].ravel().astype(float)
    mk = mask[i0:i0+H, j0:j0+W].ravel()
    valid = (mk > 0) & (z < 1e8)
    vi = np.where(valid)[0]
    C = sc["sh"][:N_LIGHTS].astype(float)
    s = make_sc(z, rho, valid, C, H, W, name=sc["name"])
    # 轮廓带: 3×3 窗口含无效邻居的有效像素
    mv = valid.reshape(H, W).astype(float)
    nb = np.zeros_like(mv)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            nb += np.roll(np.roll(mv, di, 0), dj, 1)
    contour = valid.reshape(H, W) & (nb < 9)
    s["contour"] = contour.ravel()
    return s


def gen_delta_v(name, v):
    if name == "lambda":
        # z' = (1+t)z ⇒ v'_x=(1+t)v_x, v'_y=(1+t)v_y, v'_z=1 ⇒ δv=(v_x,v_y,0)
        return np.stack([v[:, 0], v[:, 1], np.zeros(len(v))], axis=1)
    if name == "mu":
        return np.stack([-SOBEL_GAIN * np.ones(len(v)), np.zeros(len(v)),
                         np.zeros(len(v))], axis=1)
    if name == "nu":
        return np.stack([np.zeros(len(v)), -SOBEL_GAIN * np.ones(len(v)),
                         np.zeros(len(v))], axis=1)
    raise ValueError(name)


def delta_fields(sc, name):
    """生成元的 (δz 场, δv, δn, δρ) 全图解析。"""
    z, rho, v, nv, n = sc["z"], sc["rho"], sc["v"], sc["nv"], sc["n"]
    H, W = sc["H"], sc["W"]
    if name == "lambda":
        dz = z.copy()
    elif name == "mu":
        xg, _ = np.meshgrid(np.arange(W), np.arange(H))
        dz = xg.ravel().astype(float)
    elif name == "nu":
        _, yg = np.meshgrid(np.arange(W), np.arange(H))
        dz = yg.ravel().astype(float)
    dv = gen_delta_v(name, v)
    # dn = (δv − n(n·δv))/‖v‖,  n·δv = (v·δv)/‖v‖ (除一次!)
    dn = (dv - n * ((v * dv).sum(1) / nv)[:, None]) / nv[:, None]   # P_⊥δv/‖v‖
    drho = rho * (v * dv).sum(1) / nv ** 2
    return dict(dz=dz, dv=dv, dn=dn, drho=drho)


def nv2_safe(nv):
    return np.maximum(nv, 1e-300) ** 2


def gbr_generators(z, rho, valid, H, W, Sx, Sy):
    """三生成元 (δz, δρ) 全图场(薄包装, 复用 delta_fields)。"""
    sc = dict(z=z, rho=rho, valid=valid, H=H, W=W, Sx=Sx, Sy=Sy,
              v=None, nv=None, n=None, C=None, name="tmp")
    # delta_fields 需要 v/nv/n —— 现场构造:
    v = np.stack([-(Sx @ z), -(Sy @ z), np.ones_like(z)], axis=1)
    nv = np.linalg.norm(v, axis=1)
    n = v / nv[:, None]
    sc.update(v=v, nv=nv, n=n)
    out = {}
    for name in ("lambda", "mu", "nu"):
        d = delta_fields(sc, name)
        out[name] = dict(dz=d["dz"], drho=d["drho"], dz_all=d["dz"], drho_all=d["drho"])
    return out


def delta_observation(sc, dfd, mode="geom"):
    """观测残差 δI (P_all, N): 几何部分 + 可选 C 最优补偿。
    δI^geom_kp = ρ h_kp (C_kᵀ dY_p)·δn_p + s_kp δρ_p"""
    rho, C, v, nv, n = sc["rho"], sc["C"], sc["v"], sc["nv"], sc["n"]
    Sx, Sy = sc["Sx"], sc["Sy"]
    z = sc["z"]
    valid = sc["valid"]
    # 渲染中间量(全图)
    Y = sh2(n)                                     # (P_all,9)
    Zk = Y @ C.T                                   # (P_all,N)
    Sk = np.maximum(Zk, 0)
    Hk = (Zk > 0).astype(float)
    dY = sh2_d_raw(n)                              # (P_all,9,3)
    CdY = np.einsum('kj,pji->pki', C, dY)          # (P,N,3) = C_kᵀ dY_p
    dn = dfd["dn"]; drho = dfd["drho"]
    # 注意: 无效像素的 n 由天空 z(1e8) 决定 → 垃圾; 但 h·ρ 权重: ρ 是全图原始 rho!
    # 与 exp2 管线一致: Fisher 线性化点的 h 用全图 n(轮廓带被污染——这正是判别(b)对象)
    deltaI_geom = (rho[:, None] * Hk) * np.einsum(
        'pki,pki->pk', CdY, np.broadcast_to(dn[:, None, :], CdY.shape)) \
        + Sk * drho[:, None]
    if mode == "geom":
        return deltaI_geom, None
    # C 最优补偿(每光独立 9×9):
    N = C.shape[0]
    dC = np.zeros_like(C)
    Wmat = rho[:, None] * Hk                       # (P,N) 观测权重
    for k in range(N):
        w = Wmat[:, k]
        A9 = ((Y * w[:, None]).T @ Y)              # (9,9) = Σ w² Y Yᵀ
        b9 = -(Y * w[:, None]).T @ deltaI_geom[:, k]
        dC[k] = np.linalg.solve(A9 + 1e-12 * np.trace(A9) / 9 * np.eye(9), b9)
    deltaI_C = np.stack(
        [(Y * (rho * Hk[:, k])[:, None]) @ dC[k] for k in range(C.shape[0])],
        axis=1)
    return deltaI_geom + deltaI_C, dC


def gen_rayleigh(sc, name, mode="opt"):
    """单生成元 Rayleigh 商(全 ‖δθ‖ 参数范数含 δz 场)。"""
    dfd = delta_fields(sc, name)
    valid = sc["valid"]
    dI, dC = delta_observation(sc, dfd, mode=mode)
    resid = dI[valid]                              # 有效观测
    ray_obs = float((resid ** 2).sum())
    gnorm2 = float((dfd["dz"] ** 2).sum() + (dfd["drho"] ** 2).sum())
    if dC is not None:
        gnorm2 += float((dC ** 2).sum())
    return dict(rayleigh=ray_obs / gnorm2, gnorm2=gnorm2,
                obs_energy=float(ray_obs), dC_norm=float(np.linalg.norm(dC) if dC is not None else 0.0))


def scale_check_rayleigh(sc):
    """全局尺度方向 (δz=0, δρ=ρ, δC_k=−C_k) —— 已知精确零(阴性对照)。"""
    rho, C, v, nv, n = sc["rho"], sc["C"], sc["v"], sc["nv"], sc["n"]
    Y = sh2(n)
    Zk = Y @ C.T
    Sk = np.maximum(Zk, 0)
    Hk = (Zk > 0).astype(float)
    dC = np.stack([-C[k] for k in range(len(C))], 0)      # (N,9)
    deltaI_C = np.stack(
        [(Y * (rho * Hk[:, k])[:, None]) @ dC[k] for k in range(len(C))],
        axis=1)                                            # (P,N): ρ h_kp 逐 (p,k)
    deltaI_geom = Sk * rho[:, None]                        # J_ρ·ρ
    resid = (deltaI_geom + deltaI_C)[sc["valid"]]
    gnorm2 = float((rho ** 2).sum() + (dC ** 2).sum())
    return float((resid ** 2).sum() / gnorm2)


def lam_max_est(sc):
    """λmax(F) 参考: 观测能量界 ‖J‖₂² ≤ ‖J‖_F²; 用幂迭代在 F 上太贵 → 用
    max over pixels 的行范数和(保守上界)与实测 Rayleigh 的参照。这里给
    ‖J‖_F² = Σ_kp (J_kp·)² 估计——直接由 δI 对随机探针的期望(Hutchinson)。"""
    rng = np.random.default_rng(SEED)
    ests = []
    for _ in range(12):
        # 参数空间随机探针(只打有效自由度)
        dz = rng.normal(size=len(sc["z"])) * sc["valid"]
        drho = rng.normal(size=len(sc["z"])) * sc["valid"]
        dC = rng.normal(size=sc["C"].shape)
        sc2 = dict(sc)
        dfd = dict(dz=dz, dv=np.zeros_like(sc["v"]), dn=np.zeros_like(sc["n"]),
                   drho=drho)
        # 数值 J·δθ: 用 exp2 的 J 矩阵(有效子集)——直接调用 delta_observation 不行
        # (它是生成元专用); 这里走 exp2 管线的 J:
        pass
    return None   # λmax 参考改用 exp3 管线的 A 幂迭代(见 rayleigh_table)


def rayleigh_table():
    """主表: 四场景 × {λ,μ,ν} × {opt,none} + 尺度对照。"""
    out = []
    for scene in SCENES:
        sc = scene_full(scene, RES)
        row = {"scene": scene, "gens": {}, "scale_rayleigh": scale_check_rayleigh(sc)}
        for name in ("lambda", "mu", "nu"):
            dfd = delta_fields(sc, name)
            dIo, dC = delta_observation(sc, dfd, mode="opt")
            dIn, _ = delta_observation(sc, dfd, mode="geom")
            gnorm2 = float((dfd["dz"] ** 2).sum() + (dfd["drho"] ** 2).sum() +
                           (dC ** 2).sum())
            row["gens"][name] = dict(
                ray_opt=float((dIo[sc["valid"]] ** 2).sum() / gnorm2),
                ray_none=float((dIn[sc["valid"]] ** 2).sum() / gnorm2),
                dC_norm=float(np.linalg.norm(dC)))
        out.append(row)
        print(f"{scene:10s} scale={row['scale_rayleigh']:.2e} | " +
              "  ".join(f"{g}: opt={row['gens'][g]['ray_opt']:.2e} "
                        f"bare={row['gens'][g]['ray_none']:.2e}"
                        for g in ("lambda", "mu", "nu")))
    return out


def principal_angles_bottom4(scene, res=RES, n_lights=N_LIGHTS):
    """判别 (a): F 底部 4 维特征子空间(参数空间, LOBPCG) vs
    span{三 GBR 生成元切向量(含 δC_opt), 全局尺度} 的主夹角。"""
    sc = scene_full(scene, res)
    z, rho, valid, vi, C = sc["z"], sc["rho"], sc["valid"], sc["vi"], sc["C"]
    N = len(C)
    Sx, Sy = sc["Sx"], sc["Sy"]
    rho_eff = rho * valid
    blk = jacobian_blocks(z, rho_eff, C, sc["H"], sc["W"], Sx, Sy)
    Js_full = build_J_z_sparse(z, rho_eff, C, sc["H"], sc["W"], Sx, Sy, blk)
    Js = [J[vi][:, vi] for J in Js_full]
    P = len(vi)
    Sk, Hk, Yv = blk["Sk"][vi], blk["Hk"][vi], blk["Y"][vi]
    rho_v = rho_eff[vi]
    F_zz = sum(J.T @ J for J in Js).tocsc()
    F_rr = sp.diags((Sk ** 2).sum(1)).tocsc()
    F_zr = sum(Js[k].T @ sp.diags(Sk[:, k]) for k in range(N)).tocsc()
    F_zC = sp.hstack([(Js[k].T @ sp.diags(rho_v * Hk[:, k]) @ sp.csr_matrix(Yv)).tocsc()
                      for k in range(N)], format="csc")
    F_rC = sp.hstack([sp.diags(Sk[:, k] * rho_v * Hk[:, k]) @ sp.csr_matrix(Yv)
                      for k in range(N)], format="csc")
    F_CC = np.zeros((9 * N, 9 * N))
    for k in range(N):
        F_CC[9*k:9*k+9, 9*k:9*k+9] = (Yv * (rho_v**2 * Hk[:, k])[:, None]).T @ Yv
    A = sp.bmat([[F_zz, F_zr], [F_zr.T, F_rr]], format="csc")
    B = sp.vstack([F_zC, F_rC], format="csc")
    n2P, n_tot = 2 * P, 2 * P + 9 * N

    def F_mv(x):
        x = np.asarray(x).reshape(-1)
        return np.concatenate([A @ x[:n2P] + B @ x[n2P:],
                               B.T @ x[:n2P] + F_CC @ x[n2P:]])
    Op = spla.LinearOperator((n_tot, n_tot), matvec=F_mv, dtype=np.float64)

    # 生成元切向量(参数空间, δC 用最优补偿 —— 复用 delta_observation 的每光解)
    gens = gbr_generators(z, rho, valid, sc["H"], sc["W"], Sx, Sy)
    F_CC_pinv = np.linalg.pinv(F_CC, rcond=1e-10)
    vecs = []
    for name in ("lambda", "mu", "nu"):
        g = gens[name]
        dz_all, drho_all = g["dz_all"], g["drho_all"]
        u_zr = np.concatenate([
            np.asarray(Js_full[k] @ dz_all).ravel()[vi] + Sk[:, k] * g["drho_all"][vi]
            for k in range(N)])
        b = np.zeros(9 * N)
        for k in range(N):
            b[9*k:9*k+9] = -(Yv * (rho_v * Hk[:, k])[:, None]).T @ u_zr[k*P:(k+1)*P]
        dC = F_CC_pinv @ b
        vecs.append(np.concatenate([g["dz"][vi], g["drho_all"][vi], dC]))
    vecs.append(np.concatenate([np.zeros(P), rho_v,
                                np.concatenate([-C[k] for k in range(N)])]))   # 全局尺度
    Q, _ = np.linalg.qr(np.stack(vecs).T)
    rng = np.random.default_rng(SEED)
    X0 = rng.normal(size=(n_tot, 6))
    # 注: 不投影掉生成元子空间——主夹角要测的正是底部特征向量与它的对齐
    w, U = spla.lobpcg(Op, X0, largest=False, maxiter=600, tol=1e-10)
    order = np.argsort(w)
    w, U = w[order], U[:, order]
    U4 = U[:, :4]
    sv = np.linalg.svd(U4.T @ Q, compute_uv=False)
    return dict(lobpcg_eigs=[float(x) for x in w],
                principal_angles_deg=[float(np.degrees(np.arccos(np.clip(x, 0, 1)))) for x in sv])


def contour_scan():
    """判别 (b)①: 残差(裸几何) 逐像素能量 × 轮廓带。"""
    out = {}
    for scene in SCENES:
        sc = scene_full(scene, RES)
        valid = sc["valid"]; contour = sc["contour"]
        row = dict(contour_area_frac=float(contour[valid].sum() / valid.sum()),
                   gens={})
        for name in ("lambda", "mu", "nu"):
            dfd = delta_fields(sc, name)
            dIn, _ = delta_observation(sc, dfd, mode="geom")
            e = (dIn ** 2).sum(1)                       # (P_all,) 逐像素能量
            e_valid = e[valid]; e_ct = e[contour]
            e_in = e[valid & ~contour]
            row["gens"][name] = dict(
                contour_energy_frac=float(e_ct.sum() / max(e_valid.sum(), 1e-300)),
                energy_vs_area=float((e_ct.sum() / max(e_valid.sum(), 1e-300)) /
                                     max(contour[valid].sum() / valid.sum(), 1e-12)),
                med_inside=float(np.median(e_in)) if len(e_in) else None,
                med_contour=float(np.median(e_ct)) if len(e_ct) else None)
        out[scene] = row
        print(f"{scene:10s} 轮廓面积 {row['contour_area_frac']:.3f} | " +
              "  ".join(f"{g}: 能量/面积={row['gens'][g]['energy_vs_area']:.1f}"
                        for g in ("lambda", "mu", "nu")))
    return out


def res_scan(scene="sphere", ress=(32, 64, 128)):
    """判别 (b)②: 分辨率扫描(乙口径相对 Rayleigh —— 相对量需要除尺度)。
    这里报告绝对 Rayleigh 与"Rayleigh/‖F‖ 参考"两个口径。"""
    rows = []
    for res in ress:
        sc = scene_full(scene, res)
        row = dict(res=res, gens={})
        # λ 参考: 用 μ 生成元同款幂迭代太重 → 用 δI 能量对随机探针的比率。
        # 简化且严格: 报告 opt 口径的 Rayleigh 与"裸几何 Rayleigh"的比值(无量纲),
        # 加报告残差能量相对 Σ(I_true²) 的比例(量纲自然)。
        Y = sh2(sc["n"]); Zk = Y @ sc["C"].T
        I_true = (sc["rho"][:, None] * np.maximum(Zk, 0))[sc["valid"]]
        I_ref = float((I_true ** 2).sum())
        for name in ("lambda", "mu", "nu"):
            dfd = delta_fields(sc, name)
            dIo, dC = delta_observation(sc, dfd, mode="opt")
            ray = float((dIo[sc["valid"]] ** 2).sum())
            row["gens"][name] = dict(
                ray_abs=ray, ray_rel_I=ray / I_ref,
                gnorm2=float((dfd["dz"] ** 2).sum() + (dfd["drho"] ** 2).sum() +
                             (dC ** 2).sum()))
        rows.append(row)
        print(f"  {scene} {res}²: " + "  ".join(
            f"{g} ray/I={row['gens'][g]['ray_rel_I']:.3e}" for g in ("lambda", "mu", "nu")))
    return rows


def toy_core_kernel(base_kernel, res=48, n_lights=3):
    """toy 球体 + 指定核的 (ρ,C) GBR Rayleigh(几何已知口径 + 联合 z 版本太贵,
    这里几何已知: 法线固定真值, 参数 (ρ,C), 生成元只剩 δρ(GBR 的 ρ 补偿)——
    δz 通道不存在 → GBR 不变性表现为 δρ 补偿后 δI=0。核消融下测残差。"""
    # 解析球: z(x,y) = sqrt(R²−x²−y²)+z0, R=0.8, 中心网格
    H = W = res
    xg, yg = np.meshgrid(np.linspace(-0.7, 0.7, W), np.linspace(-0.7, 0.7, H))
    R = 0.8
    z = np.sqrt(np.maximum(R**2 - xg**2 - yg**2, 1e-9)) + 1.5
    valid = (xg**2 + yg**2) < (0.92 * R**2)
    z = z.ravel(); rho = np.full(H*W, 0.5); valid = valid.ravel()
    vi = np.where(valid)[0]
    # 法线(解析): n = ((x)/R? 正交投影球: n = (−x/R, −y/R, z_part)/…) 用 Sobel 管线保持一致
    Sx, Sy = sobel_sparse(H, W)
    v = np.stack([-(Sx @ z), -(Sy @ z), np.ones_like(z)], axis=1)
    nv = np.linalg.norm(v, axis=1)
    n = v / nv[:, None]
    rng = np.random.default_rng(SEED)
    dirs = rng.normal(size=(n_lights, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    # 核: base_kernel 为 (9,) 或 ("scaled", alpha0, alpha1, alpha2)
    if isinstance(base_kernel, tuple) and base_kernel[0] == "scaled":
        _, a0, a1, a2 = base_kernel
        k = np.array([a0, a1, a1, a1, a2, a2, a2, a2, a2])
    else:
        k = np.asarray(base_kernel)
    C = np.stack([I * k * sh2(d[None])[0] for d, I in
                  zip(dirs, 4.49 * np.ones(n_lights))], 0)
    Y = sh2(n)
    Sk = np.maximum(Y @ C.T, 0)
    Hk = (Y @ C.T > 0).astype(float)
    # GBR λ 生成元(几何已知): δρ = ρ n_x² (δv=(v_x,0,0) → δρ=ρ v·δv/‖v‖²=ρ n_x²)
    v_sub = v[valid]; n_sub = n[valid]; nv_sub = nv[valid]
    # δI_kp = ρ h (C dY)·δn + s δρ, δn = P_⊥(v_x,0,0)/‖v‖
    dY = sh2_d_raw(n_sub)
    CdY = np.einsum('kj,pji->pki', C, dY)
    dvx = np.stack([v_sub[:, 0], np.zeros(len(v_sub)), np.zeros(len(v_sub))], 1)
    dn = (dvx - n_sub * ((v_sub * dvx).sum(1) / nv_sub**2)[:, None]) / nv_sub[:, None]
    drho = rho[valid] * n_sub[:, 0] ** 2
    dI_geom = (rho[valid][:, None] * Hk[valid]) * np.einsum(
        'pki,pki->pk', CdY, np.broadcast_to(dn[:, None, :], CdY.shape)) \
        + Sk[valid] * drho[:, None]
    # δC 最优补偿(每光)
    resids = []
    dC_tot = 0.0
    for kk in range(n_lights):
        w = rho[valid] * Hk[valid][:, kk]
        A9 = ((Y[valid] * w[:, None]).T @ Y[valid])
        b9 = -(Y[valid] * w[:, None]).T @ dI_geom[:, kk]
        dCk = np.linalg.solve(A9 + 1e-12 * np.trace(A9) / 9 * np.eye(9), b9)
        dC_tot += float(np.linalg.norm(dCk))
        resids.append(dI_geom[:, kk] + (Y[valid] * w[:, None]) @ dCk)
    resid = np.concatenate(resids)
    gnorm2 = float((drho ** 2).sum() + dC_tot)
    # 相对量: 除以 ‖δI_geom‖(裸几何残差)
    geom_norm = float((dI_geom ** 2).sum())
    return dict(ray_after_comp=float((resid ** 2).sum() / gnorm2),
                geom_energy=geom_norm,
                comp_ratio=float((resid ** 2).sum() / max(geom_norm, 1e-300)))


def prediction_eta():
    """预测(ii): 合成方向光配置 η=(l=0²+l=2²)/l=1² vs 补偿后 GBR Rayleigh。"""
    rows = []
    rng = np.random.default_rng(SEED + 1)
    H = W = 48
    xg, yg = np.meshgrid(np.linspace(-0.7, 0.7, W), np.linspace(-0.7, 0.7, H))
    R = 0.8
    z = np.sqrt(np.maximum(R**2 - xg**2 - yg**2, 1e-9)) + 1.5
    valid = ((xg**2 + yg**2) < 0.92 * R**2).ravel()
    z = z.ravel(); rho = np.full(H*W, 0.5)
    Sx, Sy = sobel_sparse(H, W)
    v = np.stack([-(Sx @ z), -(Sy @ z), np.ones_like(z)], axis=1)
    nv = np.linalg.norm(v, axis=1); n = v / nv[:, None]
    Y = sh2(n)
    for ci in range(10):
        dirs = rng.normal(size=(5, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        # 每灯的 η: 方向光 SH 展开的段能量(l=0, l=2 vs l=1) —— 纯方向光下
        # E0 = π·C0, E1 = (2π/3)·C1, E2 = (π/4)·|Y2(d̂)|
        etas = []
        C = np.stack([4.49 * K_STD * sh2(d[None])[0] for d in dirs], 0)
        for d in dirs:
            yl = sh2(d[None])[0]
            e0 = (np.pi * 0.282095) ** 2
            e1 = ((2*np.pi/3) * 0.488603 * np.linalg.norm([yl[1], yl[2], yl[3]])) ** 2
            e2 = ((np.pi/4) * np.linalg.norm(yl[4:9])) ** 2
            etas.append((e0 + e2) / max(e1, 1e-30))
        eta = float(np.mean(etas))
        # GBR λ 补偿后 Rayleigh(复用 toy 管线, 单生成元)
        Sk = np.maximum(Y @ C.T, 0); Hk = (Y @ C.T > 0).astype(float)
        vsub = v[valid]; nsub = n[valid]; nvsub = nv[valid]
        dY = sh2_d_raw(nsub)
        CdY = np.einsum('kj,pji->pki', C, dY)
        dvx = np.stack([vsub[:, 0], np.zeros(len(vsub)), np.zeros(len(vsub))], 1)
        dn = (dvx - nsub * ((vsub * dvx).sum(1) / nvsub**2)[:, None]) / nvsub[:, None]
        drho = rho[valid] * nsub[:, 0] ** 2
        dI_geom = (rho[valid][:, None] * Hk[valid]) * np.einsum(
            'pki,pki->pk', CdY, np.broadcast_to(dn[:, None, :], CdY.shape)) \
            + Sk[valid] * drho[:, None]
        resids = []
        for kk in range(5):
            w = rho[valid] * Hk[valid][:, kk]
            A9 = ((Y[valid] * w[:, None]).T @ Y[valid])
            b9 = -(Y[valid] * w[:, None]).T @ dI_geom[:, kk]
            dCk = np.linalg.solve(A9 + 1e-12 * np.trace(A9) / 9 * np.eye(9), b9)
            resids.append(dI_geom[:, kk] + (Y[valid] * w[:, None]) @ dCk)
        ray = float((np.concatenate(resids) ** 2).sum())
        rows.append(dict(config=ci, eta=eta, rayleigh_comp=ray))
    from scipy.stats import spearmanr
    rho_s, p_s = spearmanr([r["eta"] for r in rows], [r["rayleigh_comp"] for r in rows])
    return dict(per_config=rows, spearman=dict(rho=float(rho_s), p=float(p_s)))


def main():
    out = {"meta": dict(seed=SEED, res=RES, n_lights=N_LIGHTS,
                        sobel_gain=SOBEL_GAIN)}
    print("[exp9] 主表(乙=最优C补偿 / bare=裸几何; 单位=观测能量/参数范数²)")
    out["main_table"] = rayleigh_table()

    print("\n[exp9-a] sphere 底部 4 维主夹角(判别 a)")
    try:
        pa = principal_angles_bottom4("sphere")
        out["a_principal_angles"] = pa
        print("  lobpcg eigs:", [f"{x:.3e}" for x in pa["lobpcg_eigs"]])
        print("  主夹角(°):", [f"{x:.2f}" for x in pa["principal_angles_deg"]])
    except Exception as exc:
        import traceback; traceback.print_exc()
        out["a_principal_angles"] = dict(error=str(exc))

    print("\n[exp9-b] 轮廓带能量诊断(判别 b①)")
    out["b_contour"] = contour_scan()

    print("\n[exp9-b2] sphere 分辨率扫描(判别 b②)")
    out["b_res_scan"] = res_scan("sphere")

    print("\n[exp9-c] 核消融 toy 球体(判别 c)")
    std = toy_core_kernel(K_STD)
    # 数据型核: 段幅度实测 (4.4900, 3.8890, 1.0090) 相对标准 (π·C0=0.8862, 2π/3·C1=1.0233, π/4)
    scaled = toy_core_kernel(("scaled", 4.4900, 3.8890, 1.0090 * np.pi / 4 / (np.pi / 4)))
    out["c_kernel_ablation"] = dict(
        standard_kernel=std,
        data_like_kernel=scaled)
    print(f"  标准核: 补偿后/裸 = {std['comp_ratio']:.3e} (≈0 → SH-2 内 GBR 精确)")
    print(f"  数据型核(l0×5.07,l1×3.80): 补偿后/裸 = {scaled['comp_ratio']:.3e}")

    print("\n[exp9-ii] 预测(ii) η vs Rayleigh(合成配置)")
    out["prediction_eta"] = prediction_eta()
    print("  Spearman:", out["prediction_eta"]["spearman"])

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[exp9] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
