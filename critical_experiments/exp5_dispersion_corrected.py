#!/usr/bin/env python3
"""关键实验 5 · 修正的散布度相关检验（设计文档原文语义）

路线(ii)框架下：
  1. 对 (z, ρ) 块做 Schur 补，得到几何/反照率块的有效信息矩阵；
  2. 提取该块的最小非零特征值 λ_min⁺；
  3. 计算光照系数矩阵 C（N×9）的最小奇异值 σ_min；
  4. 对多场景×多光照配置，算 Spearman(λ_min⁺, σ_min²)。

对象纠正（设计文档"为什么"）：之前 P-A2b 测的是"光照块 vs 光照方向二阶矩"（对象错）；
本次测"几何块 vs 光照系数矩阵奇异值"——经典理论预期：光照系数的病态（σ_min 小）
直接削弱对 (z,ρ) 的可辨识性（Schur 补 S_geo = F_geo − Bᵀ F_CC⁺ B 中 B 携带 C 通道）。

实现（复用实验 2 管线，(z,ρ) 侧 Schur——注意方向对偶）：
  完整 F 分块 [[A(z,ρ), B],[Bᵀ, F_CC(C)]]
  几何/反照率有效信息 = A − B·F_CC⁺·Bᵀ  （消去 C）
  —— A 是稀疏 (2P)×(2P)，C 侧 F_CC 是小稠密 9N×9N，伪逆容易；
     B·F_CC⁺·Bᵀ 仍稀疏（B 稀疏）。最小非零特征值用 scipy.sparse eigsh(which='SM')
     或先投影掉 gauge（深度平移 1 维 + 尺度 1 维）后取最小。
光照配置：每场景随机 K 组 N 盏灯（与 w2a2 相同的球面均匀采样协议），σ_min(C) 变化
  → 相关检验的散点。calibration 场景的 32 盏真实 SH 系数作为光源池（真实光照族）。

产物：critical_experiments/exp5_dispersion_corrected.json（含散点原始数据）
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from exp2_joint_fisher_schur import (  # noqa: E402
    DATA, jacobian_blocks, load_scene_compat, sobel_sparse, build_J_z_sparse,
)

OUT = HERE / "exp5_dispersion_corrected.json"
SCENES = ["sphere", "cube", "cylinder", "hemisphere"]
RES = 64
N_CONFIGS = 8          # 每场景 8 组光照配置
N_LIGHTS = 5
SEED = 20260905


def geometry_schur(scene_dir, C, res=RES):
    """A_eff = A − B F_CC⁺ Bᵀ（消去 C 的 (z,ρ) 有效信息），返回 (A_eff_sparse, gauge_vecs, P)。"""
    sc = load_scene_compat(scene_dir)
    z, rho, mask, sh_pool = sc["depth"], sc["albedo"], sc["mask"], sc["sh"]
    H0, W0 = mask.shape
    H = W = res
    i0, j0 = (H0 - H) // 2, (W0 - H) // 2
    z = z[i0:i0 + H, j0:j0 + W].ravel().astype(float)
    rho = rho[i0:i0 + H, j0:j0 + W].ravel().astype(float)
    mk = mask[i0:i0 + H, j0:j0 + W].ravel()
    valid = (mk > 0) & (z < 1e8)
    vi = np.where(valid)[0]
    C = np.asarray(C, float)
    N = len(C)
    Sx, Sy = sobel_sparse(H, W)
    rho_eff = rho * valid

    blk = jacobian_blocks(z, rho_eff, C, H, W, Sx, Sy)
    Js_full = build_J_z_sparse(z, rho_eff, C, H, W, Sx, Sy, blk)
    Js = [J[vi][:, vi] for J in Js_full]
    P = len(vi)
    Sk, Hk, Yv = blk["Sk"][vi], blk["Hk"][vi], blk["Y"][vi]
    rho_v = rho_eff[vi]

    F_zz = sum(J.T @ J for J in Js).tocsc()
    s2 = (Sk ** 2).sum(1)
    F_rr = sp.diags(s2).tocsc()
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

    # F_CC⁺（eigh 伪逆, 相对截断 1e-8）——只保留谱因子, 不显式组装 A_eff
    wC, vC = np.linalg.eigh(F_CC)
    cut = 1e-8 * max(wC[-1], 1e-300)
    keep = wC > cut
    Yp = vC[:, keep] / np.sqrt(wC[keep])            # 谱因子: F⁺ = Yp Ypᵀ

    # gauge 方向（解析）：深度平移 (1_z, 0) 与尺度 (0, ρ)（在有效自由度坐标 vi 上）
    d_shift = np.zeros(2 * P); d_shift[:P] = 1.0     # z 平移（内点严格；边界在 valid 裁剪下多为内点）
    d_scale = np.zeros(2 * P); d_scale[P:] = rho_v  # ρ 尺度 gauge
    # 返回隐式算子原料（2026-09-05 根因修正：显式 A_eff 的 B F⁺ Bᵀ 项 nnz ~ P² 爆炸
    # → 16GB RAM 换页挂死 74min；改 LinearOperator 隐式乘积，数学完全等价）
    return dict(A=A, B=B, Yp=Yp, d_shift=d_shift, d_scale=d_scale, P=P), d_shift, d_scale, P, valid


def min_positive_eig(eff, d_shift, d_scale, P, tol_rel=1e-8, n_iter=300):
    """隐式 A_eff 的最小非零特征值：gauge 投影 + LOBPCG。

    A_eff·v = A·v − B·(Yp·(Ypᵀ·(Bᵀ·v)))（隐式 Schur，无显式稠密化）
    gauge 投影 P_g = I − GᵀG（G=[d̂_shift, d̂_scale] 正交化）作用于 matvec 内。
    LOBPCG 从随机初值求最小几个特征值；gauge 二方向投影后为精确 ~0，
    最小非零 = 排除前 2 个近零后的第一个。
    """
    A, B, Yp = eff["A"], eff["B"], eff["Yp"]
    n = 2 * P
    G = np.stack([d_shift, d_scale])
    # Gram-Schmidt 正交化
    q, _ = np.linalg.qr(G.T)                     # (n, 2)
    rng = np.random.default_rng(20260905)

    def mv(v):
        v = np.asarray(v).reshape(-1)
        v = v - q @ (q.T @ v)                    # gauge 投影
        Av = A @ v
        z1 = B.T @ v                             # (45,)
        z2 = Yp.T @ z1                           # (k,)
        z3 = Yp @ z2                             # (45,)
        z4 = B @ z3                              # (n,)
        out = Av - z4
        return out - q @ (q.T @ out)             # 双侧投影(对称化)
    Op = spla.LinearOperator((n, n), matvec=mv, dtype=np.float64)

    X = rng.normal(size=(n, 6))                  # 求最小 6 个
    X, _ = np.linalg.qr(X - q @ (q.T @ X))       # 初值也投影
    w, _ = spla.lobpcg(Op, X, largest=False, maxiter=n_iter, tol=1e-8)
    w = np.sort(np.real(w))
    lam_max = float(np.abs(A.diagonal()).max())   # Gershgorin 上界参考
    # gauge 方向投影后 ~0（数值 1e-12 量级），取超过 tol_rel·‖A‖ 的第一个
    pos = w[w > tol_rel * lam_max * 1e-3]
    # 更稳的口径: 排除与 gauge/数值零合并的前 2 个后取第 1 个
    if len(w) >= 3:
        cand = w[2:] if abs(w[0]) < 1e-6 * lam_max and abs(w[1]) < 1e-6 * lam_max else w
        lam = float(cand[0]) if len(cand) else float('nan')
    else:
        lam = float(pos[0]) if len(pos) else float('nan')
    return lam, w


def main():
    rng = np.random.default_rng(SEED)
    out = {"runs": [], "meta": dict(n_configs=N_CONFIGS, n_lights=N_LIGHTS, res=RES,
                                    object="geometry/(z,ρ) block effective info (Schur over C) vs σ_min(C)²",
                                    correction_note="P-A2b 初版测光照块 vs 光照方向二阶矩(对象错); 本版按设计文档改为几何块 vs 光照系数奇异值")}
    for scene in SCENES:
        d = DATA / scene
        if not d.is_dir():
            continue
        sc = load_scene_compat(str(d))
        sh_pool = sc["sh"]                    # (32, 9) 真实光照族
        for ci in range(N_CONFIGS):
            # 光照配置: 从真实 32 盏池随机抽 N 盏 (σ_min 有自然变化)
            sel = rng.choice(len(sh_pool), N_LIGHTS, replace=False)
            C = sh_pool[np.sort(sel)].astype(float)
            sig = np.linalg.svd(C, compute_uv=False)
            sigma_min = float(sig[-1])
            try:
                A_eff, d_shift, d_scale, P, valid = geometry_schur(str(d), C)
                lam_min, w = min_positive_eig(A_eff, d_shift, d_scale, P)
                out["runs"].append(dict(scene=scene, config=ci,
                                        sigma_min=sigma_min, sigma_min_sq=sigma_min**2,
                                        lam_min_pos=lam_min, P=P,
                                        w6=[float(x) for x in w]))
                print(f"{scene:10s} cfg{ci} σ_min={sigma_min:.4f} σ_min²={sigma_min**2:.5f} "
                      f"λ_min⁺={lam_min:.4e} (P={P})")
            except Exception as exc:
                print(f"{scene:10s} cfg{ci}: FAIL {exc}")
                out["runs"].append(dict(scene=scene, config=ci, error=str(exc)))

    # Spearman（全样本 + 逐场景）
    ok = [r for r in out["runs"] if "error" not in r]
    x = [r["sigma_min_sq"] for r in ok]
    y = [r["lam_min_pos"] for r in ok]
    if len(set(x)) > 3 and np.std(y) > 0:
        rho_all, p_all = spearmanr(x, y)
    else:
        rho_all = p_all = float("nan")
    per_scene = {}
    for scene in SCENES:
        xs = [r["sigma_min_sq"] for r in ok if r["scene"] == scene]
        ys = [r["lam_min_pos"] for r in ok if r["scene"] == scene]
        if len(set(xs)) > 3 and np.std(ys) > 0:
            r_, p_ = spearmanr(xs, ys)
            per_scene[scene] = dict(rho=float(r_), p=float(p_), n=len(xs))
    out["spearman_all"] = dict(rho=float(rho_all), p=float(p_all), n=len(ok))
    out["spearman_per_scene"] = per_scene
    print(f"\nSpearman(λ_min⁺(几何块), σ_min²) 全样本 = {rho_all:.3f} (p={p_all:.3e}, n={len(ok)})")
    for s, v in per_scene.items():
        print(f"  {s:10s}: ρ={v['rho']:.3f} (p={v['p']:.3e}, n={v['n']})")

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[exp5] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
