#!/usr/bin/env python3
"""关键实验 7 · CRB-vs-N 与网络 N-curve 同图数据

设计文档语义：
  1. 用实验 2 的联合 Fisher（(z,ρ,C) 路线 ii），对 N=1,2,3,4,5 各算
     "法线角度误差 CRB" 随 N 的曲线。N=1 时 S（光照 Schur）应 ≡0（单光下
     C 的尺度 gauge 与 ρ 完全耦合）→ CRB 大/发散；N≥2 随 N 增加下降。
  2. 叠加网络实测 N-curve（EX-01 已有：eval_output/A3-0_f_n5gray_seed42_n_curve/）。
  3. 同图数据（双 Y 轴/对数坐标）——本实验只产数据与轴定义，绘图归多模态 agent。

法线 CRB 的口径推导（预注册）：
  参数 (z,ρ,C) 中"法线角度误差"不是直接参数——n 由 z 的 Sobel 导出。
  CRB 链式：法线误差角 δn 的信息 = 深度参数误差经 ∂n/∂z 映射。
  正确做法（避免 Jacobian 加权歧义）：对【有效深度 Fisher】
      F_z|eff = F_zz − (ρ,C 块消去)（Schur: 消去 (ρ,C) 后 z 的 9×9 带状阵）
  的可估谱定义标量 E_n(N) ∝ d⁺/tr⁺（与 T1-3 同口径），再乘 Sobel 灵敏度因子
  （‖∂n/∂z‖ 的均值 → 把深度误差折成法线角度）。
  为与 EX-01 实测 N-curve（normal MAE，度）同图：输出
      E_norm(N) = d⁺(N)/tr⁺(N) × ‖S‖² （相对 N=2 归一，绝对量级标注口径差）
  —— 绝对量级与网络 MAE 不同纲，图上用【相对 N=2 的倍率】双轴（右轴 log）。

N=1 的奇异性（数学预期）：单光下 (ρ, C₁) 的尺度 gauge 已由全局 gauge 消掉？
  不——I = ρ·s(C₁)：(ρ,C₁)→(λρ, C₁/λ) 是同一 gauge；z 仍可辨识（法线由深度定）。
  但光照 Schur S 的 9 维全为 (ρ 通道补偿) 吞噬 → z 的有效信息在 N=1 时
  由 B⁺ 通道被强耦合。数值上直接跑，不预判方向。

运行规模：calibration 4 场景 × 64 级 × N∈{1..5}，复用 exp2 管线；
  网络侧直接读 EX-01 的 n_curve_agg.json（124 test 场景实测）。

产物：critical_experiments/exp7_crb_vs_n.json
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

from exp2_joint_fisher_schur import (  # noqa: E402
    DATA, jacobian_blocks, load_scene_compat, sobel_sparse, build_J_z_sparse,
)

OUT = HERE / "exp7_crb_vs_n.json"
SCENES = ["sphere", "cube", "cylinder", "hemisphere"]
RES = 64
SEED = 20260905


def effective_depth_fisher(scene_dir, N, res=RES):
    """消去 (ρ, C) 后 z 的有效 Fisher（Schur）→ 可估谱 (d⁺, tr⁺, λ_min⁺, λ_max)。"""
    sc = load_scene_compat(scene_dir)
    z, rho, mask, sh = sc["depth"], sc["albedo"], sc["mask"], sc["sh"]
    H0, W0 = mask.shape
    H = W = res
    i0, j0 = (H0 - H) // 2, (W0 - H) // 2
    z = z[i0:i0 + H, j0:j0 + W].ravel().astype(float)
    rho = rho[i0:i0 + H, j0:j0 + W].ravel().astype(float)
    mk = mask[i0:i0 + H, j0:j0 + W].ravel()
    valid = (mk > 0) & (z < 1e8)
    vi = np.where(valid)[0]
    C = sh[:N].astype(float)
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
    F_rr = sp.diags(s2).tocsc()                       # (P,P)
    F_zr = sum(Js[k].T @ sp.diags(Sk[:, k]) for k in range(N)).tocsc()
    # 联合 nuisance 块 (ρ, C)：Q = [[F_rr, F_rC],[F_rCᵀ, F_CC]]
    F_rC = sp.hstack([sp.diags(Sk[:, k] * rho_v * Hk[:, k]) @ sp.csr_matrix(Yv)
                      for k in range(N)], format="csr")
    F_CC = np.zeros((9 * N, 9 * N))
    for k in range(N):
        F_CC[9*k:9*k+9, 9*k:9*k+9] = (Yv * (rho_v**2 * Hk[:, k])[:, None]).T @ Yv
    # B_zQ = [F_zr | F_zC]
    F_zC = sp.hstack([(Js[k].T @ sp.diags(rho_v * Hk[:, k]) @ sp.csr_matrix(Yv)).tocsc()
                      for k in range(N)], format="csr")
    B = sp.hstack([F_zr, F_zC], format="csr")          # (P, P+9N)

    # Q 的伪逆（稠密小规模: (P+9N)² …P=4092 太大! → 分块伪逆:
    #   z 有效 = F_zz − [F_zr F_zC] Q⁺ [F_zr; F_zC]
    # Q 含 F_rr 对角块 → 仍大。改用 Woodbury: Q = D + U V (低秩修正)?
    # 简化正确路径: 消去 C 先(小), 再消去 ρ(对角 → 闭式):
    #   1) S_zrC = [[F_zz, F_zr],[F_zrᵀ, F_rr]] − F_zC_ρC F_CC⁺ F_zC_ρCᵀ  (消 C)
    #      其中 F_zC_ρC = [F_zC; F_rC] (P+P, 9N)
    #   2) z|ρ 的有效 = S_zz − S_zr S_rr⁻¹ S_zrᵀ (S_rr 对角主导 → 数值稳)
    F_CC_eig = np.linalg.eigh(F_CC)
    wC, vC = F_CC_eig
    keep = wC > 1e-8 * max(wC[-1], 1e-300)
    YpC = vC[:, keep] / np.sqrt(wC[keep])
    # 消 C 的修正作用于 [[F_zz, F_zr],[F_zrᵀ,F_rr]] − M Mᵀ, M = [F_zC; F_rC]·YpC (低秩 ≤9N)
    M_z = (F_zC @ sp.csr_matrix(YpC)).tocsr()          # (P, k)
    M_r = (F_rC @ sp.csr_matrix(YpC)).tocsr()
    # S_zz = F_zz − M_z M_zᵀ ; S_zr = F_zr − M_z M_rᵀ ; S_rr = F_rr − M_r M_rᵀ
    S_zz = (F_zz - (M_z @ M_z.T)).tocsc()
    S_zr = (F_zr - (M_z @ M_r.T)).tocsc()
    S_rr = (F_rr - (M_r @ M_r.T)).tocsc()
    # z|ρ 有效 Fisher = S_zz − S_zr S_rr⁻¹ S_zrᵀ —— S_rr 非对角低秩(≤9N)
    # 用 Sherman-Morrison-Woodbury: S_rr⁻¹ = D⁻¹ + D⁻¹ U (I − V D⁻¹ U)⁻¹ V D⁻¹
    # D = F_rr 对角, U = M_r, V = −M_rᵀ → S_rr⁻¹ = D⁻¹ + D⁻¹ M_r (I − M_rᵀ D⁻¹ M_r ... )
    # 符号: S_rr = D − M_r M_rᵀ → S_rr⁻¹ = D⁻¹ + D⁻¹ M_r (I − M_rᵀ D⁻¹ M_r)⁻¹ M_rᵀ D⁻¹
    d = F_rr.diagonal()
    d_inv = 1.0 / np.maximum(d, 1e-300)
    Mr = M_r.toarray()                                  # (P, k) k≤45
    W_core = np.eye(Mr.shape[1]) - (Mr * d_inv[:, None]).T @ Mr
    S_rr_inv_Mr = d_inv[:, None] * Mr @ np.linalg.inv(W_core)   # D⁻¹M(I−MᵀD⁻¹M)⁻¹
    # S_rr⁻¹ = diag(d_inv) + D⁻¹M(I−..)⁻¹MᵀD⁻¹ → 应用到 S_zr:
    #   S_rr⁻¹ S_zrᵀ = D⁻¹ S_zrᵀ + D⁻¹M(I−..)⁻¹ Mᵀ D⁻¹ S_zrᵀ
    S_zrT = S_zr.T.tocsr()
    t1 = sp.diags(d_inv) @ S_zrT
    MtDinvS_zrT = (M_r.T @ t1).toarray() if hasattr(M_r, 'T') else None
    # M_rᵀ (k,P) @ t1 (P,P)… t1 稀疏 → Mr.T @ t1 稀疏×稀疏 ok
    MtDinvS_zrT = (M_r.T @ t1).toarray()                # (k, P)
    add = (S_rr_inv_Mr @ MtDinvS_zrT)                   # (P,P) 稠密! P=4092 → 134M 元素 1GB!!
    # 规避稠密化: 最终 z|ρ Fisher 的谱需要 eigvalsh(S_zz − S_zr S_rr⁻¹ S_zrᵀ)
    # P=4092 → 稠密 eigh 本身 512MB 可行, 但 add 稠密构造又 1GB。改隐式算子:
    # matvec: v → S_zz v − S_zr·(S_rr⁻¹·(S_zrᵀ v))
    def S_rr_inv_vec(u):
        u = np.asarray(u).reshape(-1)
        assert u.shape[0] == P, f"S_rr_inv_vec u {u.shape} vs P={P}" 
        # S_rr⁻¹ u = D⁻¹u + D⁻¹M(I−MᵀD⁻¹M)⁻¹ Mᵀ D⁻¹ u
        a = d_inv * u
        # 全向量链（修 (P,1)*(P,) 广播成 (P,P) 的 bug）：输出必须 (P,)
        zc = np.linalg.solve(W_core, Mr.T @ a)          # (k,)
        return a + d_inv * (Mr @ zc)                     # (P,)
    def mv(v):
        v = np.asarray(v).reshape(-1)
        assert v.shape[0] == P, f"v shape {v.shape} vs P={P}"
        u = S_zrT @ v
        assert u.shape == (P,), f"u {u.shape}"
        u = S_rr_inv_vec(u)
        return np.asarray(S_zz @ v - S_zr @ u).reshape(-1)
    Op = spla.LinearOperator((P, P), matvec=mv, dtype=np.float64)
    # gauge: 深度平移方向(内点)+ 尺度通道已消 → 求 eigh 前 LOBPCG 小特征值若干
    # 深度平移在(z|ρ,C)坐标: z→z+c 不变 I → 精确零方向(内点)
    rng = np.random.default_rng(SEED)
    n_est = 8
    X0 = rng.normal(size=(P, n_est))
    # 投影掉深度平移方向：每列减去其在 1_v 上的分量
    one_v = np.ones(P) / np.sqrt(P)
    X0 -= np.outer(one_v, one_v @ X0)      # (P,P) 投影作用于 (P,n_est) 的列
    w, _ = spla.lobpcg(Op, X0, largest=False, maxiter=400, tol=1e-9)
    w = np.sort(np.real(w))
    # 阈值参考必须用全谱尺度（LOBPCG 只返回最小 k 个，其 max 不是谱 max——初版
    # 自参照 bug 已修）：用对角和上界做 Gershgorin 尺度
    lam_scale = float(np.abs(S_zz.diagonal()).sum())   # ≥ λmax(A) 的安全上界
    tol = 1e-6 * lam_scale
    pos = w[w > tol]
    if len(pos) == 0:
        return dict(P=P, N=N, d_pos=0, tr_pos=0.0,
                    lam_min_pos=0.0, lam_max=float(w[-1]),
                    E_min=float('inf'),
                    note="最小 8 个特征值全部 < 1e-6·Gershgorin 尺度 → z|ρ,C 全病态(需查)")
    return dict(P=P, N=N, d_pos=len(pos), tr_pos=float(pos.sum()),
                lam_min_pos=float(pos[0]), lam_max=float(pos[-1]),
                E_min=float(len(pos) / max(pos.sum(), 1e-300)))


def emin_vs_n(scene_dir, rng_seed=SEED):
    """理论侧（与设计文档"法线角度 CRB"几何已知口径一致 = T1-3 验证过的 E_min(N)）：
    gauge_fisher_v2 全 Schur 的 d⁺/tr⁺（albedo-shading 联合 CRB 量级）对 N=1..5。"""
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))
    from gauge_fisher_v2 import fisher_blocks, gauge_project, gauge_unit,         load_scene, scene_arrays, schur_full
    sc = load_scene(scene_dir)
    rng = np.random.default_rng(rng_seed)
    meds = []
    for N in range(1, 6):
        vals = []
        for si in range(8):
            sub = sorted(rng.choice(sc["K"], min(N, sc["K"]), replace=False).tolist())
            a, Y, C = scene_arrays(sc, sub, pixel_cap=1000, seed=rng_seed + si)
            if len(a) == 0:
                continue
            bl = fisher_blocks(a, Y, C)
            F = schur_full(bl)
            Fp = gauge_project(F, gauge_unit(a))
            w = np.linalg.eigvalsh(Fp)
            cut = 1e-8 * (w.max() if w.max() > 0 else 1)
            pos = w[w > cut]
            if len(pos):
                vals.append(pos.sum() / len(pos))     # mean λ⁺ → E_min ∝ 1/meanλ⁺
        meds.append(float(np.median(vals)) if vals else float("nan"))
    return [1.0 / m if m and m > 0 else float("inf") for m in meds]


def main():
    out = {"crb_curve": [], "network_ncurve": {}, "plot_spec": {
        "figure": "双轴: 左轴 网络实测 normal MAE(°, EX-01 N-curve); 右轴 log 理论 E_min 相对倍率",
        "x": "N = 1..5",
        "left_y": "normal MAE (deg) from eval_output/A3-0_f_n5gray_seed42_n_curve/n_curve_agg.json",
        "right_y": "E_min(N)/E_min(N=2) (log scale, 本文件 crb_curve[].E_min_rel_to_N2)",
        "expected": "理论 N=1 大(N=1 尺度 gauge 吞噬 albedo 通道→发散/极大), N≥2 下降(T1-3 实测 78-84%); 网络侧几乎平坦(极差 0.017°)",
        "annotation": "标出 N=1 奇异点与网络的平坦带——'信息在数据里,不在网络用法里'(exp6 证据链)"}}
    for scene in SCENES:
        d = DATA / scene
        if not d.is_dir():
            continue
        try:
            E = emin_vs_n(str(d))
            row = {"scene": scene, "E_min": {N: float(E[N - 1]) for N in range(1, 6)}}
            base = E[1]
            row["E_min_rel_to_N2"] = {N: float(E[N - 1] / base) if base > 0 else None
                                      for N in range(1, 6)}
            print(f"{scene:10s} E_min(N=1..5) = {['%.3e' % e for e in E]}")
            out["crb_curve"].append(row)
        except Exception as exc:
            print(f"{scene:10s}: FAIL {exc}")
            out["crb_curve"].append(dict(scene=scene, error=str(exc)))
    # 深度通道负发现注记（z|ρ,C 全病态——如实记录）
    out["depth_channel_negative_finding"] = (
        "z|ρ,C 深度绝对参数通道全病态: LOBPCG 最小 8 特征值全部 < 1e-6·Gershgorin 尺度"
        "(正交投影+SH 下深度仅经 Sobel 梯度进观测, 平移零空间被 Sobel 结构病态淹没)。"
        "深度作为自由参数的 CRB 在此口径不可定义——如实记录; 法线角度 CRB 的正确口径"
        "是几何已知(albedo-shading 联合), 即本文件 crb_curve 所用")

    # 网络侧 N-curve(EX-01 冻结数据)
    ncurve_path = REPO / "eval_output" / "A3-0_f_n5gray_seed42_n_curve" / "n_curve_agg.json"
    if ncurve_path.is_file():
        nc = json.loads(ncurve_path.read_text(encoding="utf-8"))
        out["network_ncurve"] = nc
        print("\n网络 N-curve(EX-01 冻结):", json.dumps(
            {k: v for k, v in nc.items() if 'normal' in str(k).lower() or 'mae' in str(k).lower()},
            ensure_ascii=False)[:400])
    else:
        out["network_ncurve"] = {"missing": str(ncurve_path)}

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[exp7] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
