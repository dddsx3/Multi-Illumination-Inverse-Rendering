#!/usr/bin/env python3
"""exp9 收尾 · (a) 主夹角(生成元作初值的 LOBPCG) + (c) 联合口径核消融

(a) 修正: 初版 LOBPCG 把生成元子空间投影掉(主夹角恒 90° 平凡)。本版直接以
    生成元切向量为初值——λ 已知机器零(7e-19), LOBPCG 应收敛到含它的底部特征子空间。
(c) 修正: 初版 toy 用了几何已知 (ρ,C) Fisher——GBR 的 δz 分量不在参数空间,
    测的不是 GBR 破缺。本版用联合 (z,ρ,C) Fisher(exp2 管线), toy 解析球体,
    标准核 vs 数据型核(段幅度 α=(5.066, 3.800, 1.009)) 的 λ 生成元 Rayleigh。
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

from exp9_sphere_gbr_verdict import (  # noqa: E402
    RES, SEED, DATA, delta_fields, delta_observation, make_sc, scene_full,
    sh2, sh2_d_raw,
)
import scipy.sparse as sp  # noqa: E402

OUT = HERE / "exp9b_finish.json"


def fisher_blocks_joint(sc, N):
    """联合 Fisher 块(A, B, F_CC)——与 exp3 build_S_dense 同构。"""
    z, rho, valid, vi, C = sc["z"], sc["rho"], sc["valid"], sc["vi"], sc["C"]
    Sx, Sy = sc["Sx"], sc["Sy"]
    rho_eff = rho * valid
    blk = jacobian_blocks_safe(z, rho_eff, C, sc["H"], sc["W"], Sx, Sy)
    Js_full = build_J(z, rho_eff, C, sc["H"], sc["W"], Sx, Sy, blk)
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
    return A, B, F_CC, P


def jacobian_blocks_safe(z, rho, C, H, W, Sx, Sy):
    from exp2_joint_fisher_schur import jacobian_blocks
    return jacobian_blocks(z, rho, C, H, W, Sx, Sy)


def build_J(z, rho, C, H, W, Sx, Sy, blk):
    from exp2_joint_fisher_schur import build_J_z_sparse
    return build_J_z_sparse(z, rho, C, H, W, Sx, Sy, blk)


def generator_tangents(sc, N, Js_full, P, Sk, Hk, Yv, rho_v):
    """三 GBR 生成元 + 全局尺度的参数空间切向量(δC 最优补偿)。"""
    gens = {}
    for name in ("lambda", "mu", "nu"):
        dfd = delta_fields(sc, name)
        gens[name] = dfd
    # δC 最优补偿需要逐光残差——重算 u_zr
    tangents = []
    details = {}
    from exp9_sphere_gbr_verdict import SOBEL_GAIN
    for name in ("lambda", "mu", "nu"):
        dfd = gens[name]
        dz_all, drho_all = dfd["dz"], dfd["drho"]
        v, nv, n = sc["v"], sc["nv"], sc["n"]
        dv = {"lambda": np.stack([v[:, 0], v[:, 1], np.zeros(len(v))], 1),
              "mu": np.stack([-8.0*np.ones(len(v)), np.zeros(len(v)), np.zeros(len(v))], 1),
              "nu": np.stack([np.zeros(len(v)), -8.0*np.ones(len(v)), np.zeros(len(v))], 1)}[name]
        u_zr = np.concatenate([
            np.asarray(Js_full[k] @ dz_all).ravel()[vi] + Sk[:, k] * drho_all[sc["vi"]]
            for k in range(N)])
        # δC 最优补偿(每光 9×9)
        dC = np.zeros((N, 9))
        for k in range(N):
            w = rho_v * Hk[:, k]
            A9 = ((Yv * w[:, None]).T @ Yv)
            b9 = -(Yv * w[:, None]).T @ u_zr[k*P:(k+1)*P]
            dC[k] = np.linalg.solve(A9 + 1e-12*np.trace(A9)/9*np.eye(9), b9)
        # 参数空间切向量: [δz_valid; δρ_valid; δC_stacked]
        # δρ_valid: drho_all[sc["vi"]]
        tangents.append(np.concatenate([dfd["dz"][sc["vi"]],
                                        dfd["drho"][sc["vi"]],
                                        dC.reshape(-1)]))
        details[name] = dict(dC_norm=float(np.linalg.norm(dC)))
    # 全局尺度
    tangents.append(np.concatenate([np.zeros(P), rho_v,
                                    np.concatenate([-sc["C"][k] for k in range(N)])]))
    return tangents, details


def principal_angles_fixed(scene="sphere", res=64):
    sc = scene_full(scene, res)
    N = len(sc["C"])
    A, B, F_CC, P = fisher_blocks_joint(sc, N)
    n2P = 2 * P
    n_tot = n2P + 9 * N

    def F_mv(x):
        x = np.asarray(x).reshape(-1)
        return np.concatenate([A @ x[:n2P] + B @ x[n2P:],
                               B.T @ x[:n2P] + F_CC @ x[n2P:]])
    Op = spla.LinearOperator((n_tot, n_tot), matvec=F_mv, dtype=np.float64)

    Sk = np.maximum(sh2(sc["n"]) @ sc["C"].T, 0)
    Hk = (sh2(sc["n"]) @ sc["C"].T > 0).astype(float)
    Yv = sh2(sc["n"])
    rho_v = sc["rho"] * sc["valid"]
    tangents, details = generator_tangents(sc, N, *_jigsaw(sc, N))
    V = np.stack(tangents)                       # (4, n_tot)
    Q, _ = np.linalg.qr(V.T)
    # LOBPCG 以 Q 为初值(收敛到含生成元方向的底部特征子空间)
    w, U = spla.lobpcg(Op, Q, largest=False, maxiter=800, tol=1e-12)
    order = np.argsort(w)
    w, U = w[order], U[:, order]
    U4 = U[:, :4]
    sv = np.linalg.svd(U4.T @ Q, compute_uv=False)
    # 生成元各自的 Rayleigh(收敛后)
    rays = [float(u @ (Op @ u) / max(u @ u, 1e-300)) for u in U.T]
    return dict(converged_eigs=[float(x) for x in w],
                principal_angles_deg=[float(np.degrees(np.arccos(np.clip(x, 0, 1)))) for x in sv],
                rayleigh_of_converged=[float(r) for r in rays],
                generator_dc_norms=details)


def _jigsaw(sc, N):
    """返回 (Js_full, P, Sk, Hk, Yv, rho_v) —— principal_angles 的中间量。"""
    z, rho, valid, vi, C = sc["z"], sc["rho"], sc["valid"], sc["vi"], sc["C"]
    Sx, Sy = sc["Sx"], sc["Sy"]
    rho_eff = rho * valid
    blk = jacobian_blocks_safe(z, rho_eff, C, sc["H"], sc["W"], Sx, Sy)
    Js_full = build_J(z, rho_eff, C, sc["H"], sc["W"], Sx, Sy, blk)
    P = len(vi)
    Sk = np.maximum(sh2(sc["n"]) @ C.T, 0)[vi]
    Hk = (sh2(sc["n"]) @ C.T > 0).astype(float)[vi]
    Yv = sh2(sc["n"])[vi]
    return Js_full, P, Sk, Hk, Yv, rho_eff[vi]


def kernel_ablation_joint(res=48, n_lights=3):
    """判别 (c) 联合口径: toy 解析球 + {标准核, 数据型核} 的 λ 生成元 Rayleigh。"""
    H = W = res
    xg, yg = np.meshgrid(np.linspace(-0.9, 0.9, W), np.linspace(-0.9, 0.9, H))
    R = 0.8
    z = np.sqrt(np.maximum(R**2 - xg**2 - yg**2, 1e-9)) + 1.8
    valid = ((xg**2 + yg**2) < 0.92 * R**2).ravel()
    z = z.ravel(); rho = np.full(H*W, 0.5)
    rng = np.random.default_rng(SEED)
    dirs = rng.normal(size=(n_lights, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    out = {}
    for tag, alpha in (("standard", (1.0, 1.0, 1.0)),
                       ("data_like", (5.066, 3.800, 1.009))):
        k = np.concatenate([[alpha[0]*np.pi], [alpha[1]*(2*np.pi/3)]*3,
                            [alpha[2]*(np.pi/4)]*5])
        C = np.stack([1.0 * k * sh2(d[None])[0] for d in dirs], 0)
        sc = make_sc(z, rho, valid, C, H, W, name=f"toy_{tag}")
        dfd = delta_fields(sc, "lambda")
        dI, dC = delta_observation(sc, dfd, mode="opt")
        ray = float((dI[valid] ** 2).sum())
        gnorm2 = float((dfd["dz"] ** 2).sum() + (dfd["drho"] ** 2).sum() + (dC ** 2).sum())
        # 参考: ‖J‖² 量级(Hutchinson)
        est = 0.0
        for _ in range(8):
            x = rng.normal(size=len(z))
            x = x * sc["valid"]
            # 观测能量: J·x ≈ δI(x 场=0 需走 J 矩阵) —— 简化: 用 I 能量做参照
            pass
        I_ref = float(((rho[:, None] * np.maximum(sh2(sc["n"]) @ C.T, 0))**2).sum())
        out[tag] = dict(ray_joint=ray / gnorm2, ray_abs=ray,
                        ray_over_I=ray / max(I_ref, 1e-300))
    out["ratio_data_over_std"] = (out["data_like"]["ray_joint"] /
                                  max(out["standard"]["ray_joint"], 1e-300))
    return out


def main():
    out = {}
    print("[exp9b-a] sphere 底部 4 维主夹角(生成元初值 LOBPCG)")
    try:
        out["a"] = principal_angles_fixed("sphere")
        print("  收敛特征值:", [f"{x:.3e}" for x in out["a"]["converged_eigs"]])
        print("  主夹角(°):", [f"{x:.2f}" for x in out["a"]["principal_angles_deg"]])
        print("  收敛后 Rayleigh:", [f"{x:.3e}" for x in out["a"]["rayleigh_of_converged"]])
    except Exception as exc:
        import traceback; traceback.print_exc()
        out["a"] = dict(error=str(exc))

    print("\n[exp9b-c] 联合口径核消融(toy 解析球)")
    try:
        out["c"] = kernel_ablation_joint()
        for tag in ("standard", "data_like"):
            print(f"  {tag:10s}: ray_joint={out['c'][tag]['ray_joint']:.3e} "
                  f"ray/I={out['c'][tag]['ray_over_I']:.3e}")
        print(f"  数据核/标准核 比 = {out['c']['ratio_data_over_std']:.2f}")
    except Exception as exc:
        import traceback; traceback.print_exc()
        out["c"] = dict(error=str(exc))

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[exp9b] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
