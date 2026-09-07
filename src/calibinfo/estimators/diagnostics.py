"""B2 · diagnostics：诊断参考实现（exp8S 变体 A 正确代数）。

来源：critical_experiments/exp8s_oracle_schur_audit.py 变体 A（严格全耦合 Schur）。
v3.3 已核验为正确代数（两处代数错误的修正版）：
  1. ρ 边缘化 = **逐像素对角** 精确 Schur：F_ρρ 是对角阵（每像素一行），
     S_λλ = F_λλ − F_λρ · diag(1/F_ρρ) · F_ρλ —— **不是**全局标量 (Σx)²/Σd 口径；
  2. α 强度维 = 全局 3×3 块联合 pinv 边缘化（S_aa 块，docstring 声称=实现一致，红线 #9）。

红线 #8：入库时过随机小规模稠密对照（test_diagnostics_dense_contrast.py，rel<1e-10）。
下游：CI03/CI04 的诊断层一律 import 本实现，禁止再手写 Schur（公式单源化）。
"""

from __future__ import annotations

import numpy as np


def tangent_basis(dirs):
    """每光方向切向正交基 (t1, t2)：与 exp8R/exp8S diagnose_trace 同定义。

    ref 轴选择与 legacy 一致：|l_z|<0.9 用 z 轴，否则用 x 轴（避免共线退化）。
    """
    N = dirs.shape[0]
    t1s = np.zeros((N, 3))
    t2s = np.zeros((N, 3))
    for k in range(N):
        ref = np.array([0.0, 0.0, 1.0]) if abs(dirs[k, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        t1 = np.cross(dirs[k], ref)
        t1 /= np.linalg.norm(t1)
        t1s[k] = t1
        t2s[k] = np.cross(dirs[k], t1)
    return t1s, t2s


def noise_weights(I_sub, a, b):
    """异方差噪声权重 w = 1/(a + b·I)，行序 = 光主序 (N, P)。

    来源：exp8r/exp8s 的 (a, b) 噪声模型口径（calibrate 输出）。
    """
    return 1.0 / np.maximum(a + b * np.maximum(I_sub, 0), 1e-6)


def direction_fisher_schur(n_gt, rho, dirs, w):
    """全耦合方向 Fisher 的精确 Schur（消 ρ，逐像素对角）+ α 联合边缘化。

    输入：n_gt (P,3) 已知法线；rho (P,) 反照率（真值或拟合值——口径由调用方声明）；
    dirs (N,3) 光方向；w (N,P) 噪声权重（α 已含入权重与梯度的按 legacy 口径：
    调用方若按 α=1 校准请先并入 w，见 diagnose_trace_v31）。
    返回 dict：S_dir(2N,2N)（消 ρ、消 α 后的方向块）、S_aa(3,3)、blocks[k] = 每光 2×2 块、
    trace = Σ_k (F11+F22)/det、info = Σ_k det/(F11+F22)（变体 C 口径）。
    """
    N = dirs.shape[0]
    P = len(n_gt)
    nl = np.clip(n_gt @ dirs.T, 0, None)            # (P,N)
    h = (nl > 0).astype(float)
    t1s, t2s = tangent_basis(dirs)
    # 参数序（变体 A）：[α_k, t1_k, t2_k]×N（n_l = 3N）
    n_l = 3 * N
    Fpp = (((h * nl) ** 2) * w.T).sum(1)            # (P,) F_ρρ 逐像素对角
    Fpp = np.maximum(Fpp, 1e-300)
    F_ll = np.zeros((n_l, n_l))
    F_lr = np.zeros((n_l, P))
    for k in range(N):
        wt = w[k] * h[:, k]
        g_a = rho * nl[:, k]
        g_1 = rho * (n_gt @ t1s[k])
        g_2 = rho * (n_gt @ t2s[k])
        F_ll[3 * k + 0, 3 * k + 0] = (wt * g_a * g_a).sum()
        F_ll[3 * k + 1, 3 * k + 1] = (wt * g_1 * g_1).sum()
        F_ll[3 * k + 2, 3 * k + 2] = (wt * g_2 * g_2).sum()
        F_ll[3 * k + 0, 3 * k + 1] = F_ll[3 * k + 1, 3 * k + 0] = (wt * g_a * g_1).sum()
        F_ll[3 * k + 0, 3 * k + 2] = F_ll[3 * k + 2, 3 * k + 0] = (wt * g_a * g_2).sum()
        F_ll[3 * k + 1, 3 * k + 2] = F_ll[3 * k + 2, 3 * k + 1] = (wt * g_1 * g_2).sum()
        F_lr[3 * k + 0] = wt * nl[:, k] * g_a
        F_lr[3 * k + 1] = wt * nl[:, k] * g_1
        F_lr[3 * k + 2] = wt * nl[:, k] * g_2
    S = F_ll - F_lr @ (F_lr.T / Fpp[:, None])       # 精确 Schur（消 ρ，逐像素对角）
    alpha_idx = [3 * k for k in range(N)]
    dir_idx = [i for i in range(n_l) if i % 3 != 0]
    S_aa = S[np.ix_(alpha_idx, alpha_idx)]
    S_dir = S[np.ix_(dir_idx, dir_idx)] \
        - S[np.ix_(dir_idx, alpha_idx)] @ np.linalg.pinv(S_aa) @ S[np.ix_(alpha_idx, dir_idx)]
    blocks, trace, info = [], 0.0, 0.0
    for k in range(N):
        blk = S_dir[2 * k:2 * k + 2, 2 * k:2 * k + 2]
        det = blk[0, 0] * blk[1, 1] - blk[0, 1] * blk[1, 0]
        blocks.append(blk)
        if det > 0:
            trace += (blk[0, 0] + blk[1, 1]) / det
            ssum = blk[0, 0] + blk[1, 1]
            if ssum > 0:
                info += det / ssum
    return dict(S_dir=S_dir, S_aa=S_aa, blocks=blocks, trace=float(trace),
                info=float(info))


def diagnose_trace_v31(n_gt, I_sub, rho, dirs, alphas, a, b):
    """exp8R v3.1 诊断量（变体 A 移植，@拟合值口径）。

    与 exp8S 变体 A 的唯一差别：α 不取 1，而是把 alphas 并入权重与 ρ 的乘积口径
    （legacy v3.1 口径：g 项 = α_k·ρ·…，F_ρρ 项 = (α h n·l)²·w）。
    返回 float 迹（度²量纲由调用方负责，此处为原始 Fisher 单位）。
    """
    N = dirs.shape[0]
    nl = np.clip(n_gt @ dirs.T, 0, None)
    h = (nl > 0).astype(float)
    w = 1.0 / np.maximum(a + b * np.maximum(I_sub, 0), 1e-6)     # (N,P)
    t1s, t2s = tangent_basis(dirs)
    n_l = 3 * N
    Fpp = (((alphas[None, :] * h * nl) ** 2) * w.T).sum(1)
    Fpp = np.maximum(Fpp, 1e-300)
    F_ll = np.zeros((n_l, n_l))
    F_lr = np.zeros((n_l, len(n_gt)))
    for k in range(N):
        wt = w[k] * h[:, k]
        g_a = rho * alphas[k] * nl[:, k]
        g_1 = rho * alphas[k] * (n_gt @ t1s[k])
        g_2 = rho * alphas[k] * (n_gt @ t2s[k])
        F_ll[3 * k + 0, 3 * k + 0] = (wt * g_a * g_a).sum()
        F_ll[3 * k + 1, 3 * k + 1] = (wt * g_1 * g_1).sum()
        F_ll[3 * k + 2, 3 * k + 2] = (wt * g_2 * g_2).sum()
        F_ll[3 * k + 0, 3 * k + 1] = F_ll[3 * k + 1, 3 * k + 0] = (wt * g_a * g_1).sum()
        F_ll[3 * k + 0, 3 * k + 2] = F_ll[3 * k + 2, 3 * k + 0] = (wt * g_a * g_2).sum()
        F_ll[3 * k + 1, 3 * k + 2] = F_ll[3 * k + 2, 3 * k + 1] = (wt * g_1 * g_2).sum()
        F_lr[3 * k + 0] = wt * alphas[k] * nl[:, k] * g_a
        F_lr[3 * k + 1] = wt * alphas[k] * nl[:, k] * g_1
        F_lr[3 * k + 2] = wt * alphas[k] * nl[:, k] * g_2
    S = F_ll - F_lr @ (F_lr.T / Fpp[:, None])
    alpha_idx = [3 * k for k in range(N)]
    dir_idx = [i for i in range(n_l) if i % 3 != 0]
    S_aa = S[np.ix_(alpha_idx, alpha_idx)]
    S_dir = S[np.ix_(dir_idx, dir_idx)] \
        - S[np.ix_(dir_idx, alpha_idx)] @ np.linalg.pinv(S_aa) @ S[np.ix_(alpha_idx, dir_idx)]
    tr = 0.0
    for k in range(N):
        blk = S_dir[2 * k:2 * k + 2, 2 * k:2 * k + 2]
        det = blk[0, 0] * blk[1, 1] - blk[0, 1] * blk[1, 0]
        if det > 0:
            tr += (blk[0, 0] + blk[1, 1]) / det
    return float(tr)
