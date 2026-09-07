"""B2 绑定测试 · 诊断参考实现过随机小规模稠密对照（红线 #8 首次正式执行）。

来源：REPO_MIGRATION B2 行 2 —— exp8S 变体 A 代数 → estimators/diagnostics.py。
对照路线：本测试独立构造全耦合 Jacobian J（Jᵀ·diag(w)·J = 完整 Fisher），走通用
稠密分块代数（F_ρρ 用稠密逆而非逐像素对角捷径；α 块 pinv 边缘化），与结构化实现
逐元素比对 rel < 1e-10。该对照正是红队报告三攻击五要抓的"两处险恶错误"
（F_ρρ 形状口径错位 / 边缘化声称≠实现）的机械防线。

覆盖口径：α=1（exp8S 变体 A oracle 口径，direction_fisher_schur）与
通用 α（exp8R v3.1 @拟合值口径，diagnose_trace_v31）两条路径。
观测余量：随机小规模下结构化与稠密两条路线在 ~1e-13 量级一致，阈值 1e-10 留 3+ 个数量级。
"""

import numpy as np
import pytest

from calibinfo.estimators import diagnostics as diag

SEED = 20260907


def _random_case(rng, P, N):
    n_gt = rng.normal(size=(P, 3))
    n_gt /= np.linalg.norm(n_gt, axis=1, keepdims=True)
    n_gt[:, 2] = np.abs(n_gt[:, 2]) + 0.15
    n_gt /= np.linalg.norm(n_gt, axis=1, keepdims=True)
    dirs = rng.normal(size=(N, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs[:, 2] = np.abs(dirs[:, 2]) + 0.15
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rho = np.exp(rng.uniform(np.log(0.3), np.log(1.0), size=P))
    return n_gt, dirs, rho


def _dense_reference(n_gt, dirs, rho, w, alphas, P, N):
    """独立稠密路线：Jᵀ diag(w) J 全耦合 Fisher → 稠密 Schur/pinv 分块代数。

    参数序：[ρ(P) | (α_k, t1_k, t2_k)×N]；行序 = 光主序 (k,p)。
    """
    nl = np.clip(n_gt @ dirs.T, 0, None)            # (P,N)
    h = (nl > 0).astype(float)
    t1s, t2s = diag.tangent_basis(dirs)
    n_l = 3 * N
    J = np.zeros((N * P, P + n_l))
    for k in range(N):
        r = slice(k * P, (k + 1) * P)
        J[r, :P] = -np.diag(alphas[k] * h[:, k] * nl[:, k])
        J[r, P + 3 * k + 0] = -rho * alphas[k] * h[:, k] * nl[:, k]
        J[r, P + 3 * k + 1] = -rho * alphas[k] * h[:, k] * (n_gt @ t1s[k])
        J[r, P + 3 * k + 2] = -rho * alphas[k] * h[:, k] * (n_gt @ t2s[k])
    W = w.reshape(-1)                                # (N,P) 行主序 = 行序 (k,p)
    F = J.T @ (J * W[:, None])
    Fll, Flr = F[P:, P:], F[P:, :P]
    S = Fll - Flr @ np.linalg.inv(F[:P, :P]) @ Flr.T    # 稠密逆（非结构化捷径）
    # S 已是光照块 (n_l, n_l)：索引不带 P 偏移
    alpha_idx = [3 * k for k in range(N)]
    dir_idx = [3 * k + i for k in range(N) for i in (1, 2)]
    S_aa = S[np.ix_(alpha_idx, alpha_idx)]
    S_dir = S[np.ix_(dir_idx, dir_idx)] \
        - S[np.ix_(dir_idx, alpha_idx)] @ np.linalg.pinv(S_aa) @ S[np.ix_(alpha_idx, dir_idx)]
    return S_dir


def _trace_from_blocks(S_dir, N):
    tr = 0.0
    for k in range(N):
        blk = S_dir[2 * k:2 * k + 2, 2 * k:2 * k + 2]
        det = blk[0, 0] * blk[1, 1] - blk[0, 1] * blk[1, 0]
        assert det > 0                              # 随机泛型数据下方向块 PD
        tr += (blk[0, 0] + blk[1, 1]) / det
    return tr


@pytest.mark.parametrize("N", [2, 3])
def test_variant_a_alpha1_trace_dense_contrast(N):
    """α=1 口径（exp8S 变体 A）：结构化 trace == 独立稠密路线，rel<1e-10。"""
    rng = np.random.default_rng(SEED)
    P = 30
    n_gt, dirs, rho = _random_case(rng, P, N)
    w = rng.uniform(0.5, 2.0, size=(N, P))          # 正权重（α=1 路线直接吃 w）
    tr_struct = diag.direction_fisher_schur(n_gt, rho, dirs, w)["trace"]
    S_dense = _dense_reference(n_gt, dirs, rho, w, np.ones(N), P, N)
    rel = abs(tr_struct - _trace_from_blocks(S_dense, N)) / abs(_trace_from_blocks(S_dense, N))
    assert rel < 1e-10


@pytest.mark.parametrize("N", [2, 3])
def test_v31_fitted_alpha_trace_dense_contrast(N):
    """通用 α 口径（v3.1 @拟合值）：结构化 trace == 独立稠密路线，rel<1e-10。"""
    rng = np.random.default_rng(SEED)
    P = 30
    n_gt, dirs, rho = _random_case(rng, P, N)
    alphas = rng.uniform(0.7, 1.4, size=N)
    I_sub = rng.uniform(0.05, 1.0, size=(N, P))
    a_, b_ = 0.05, 0.5
    w = diag.noise_weights(I_sub, a_, b_)           # 与 v3.1 内部同式
    tr_struct = diag.diagnose_trace_v31(n_gt, I_sub, rho, dirs, alphas, a_, b_)
    S_dense = _dense_reference(n_gt, dirs, rho, w, alphas, P, N)
    tr_dense = _trace_from_blocks(S_dense, N)
    rel = abs(tr_struct - tr_dense) / abs(tr_dense)
    assert rel < 1e-10


def test_variant_a_sdir_matrix_dense_contrast():
    """不只对迹：整块 S_dir 逐元素对照（α=1 口径，rel<1e-10），
    防"迹对但矩阵错"的假阴性通过。"""
    rng = np.random.default_rng(SEED)
    P, N = 40, 3
    n_gt, dirs, rho = _random_case(rng, P, N)
    w = rng.uniform(0.5, 2.0, size=(N, P))
    S_struct = diag.direction_fisher_schur(n_gt, rho, dirs, w)["S_dir"]
    S_dense = _dense_reference(n_gt, dirs, rho, w, np.ones(N), P, N)
    rel = np.linalg.norm(S_struct - S_dense) / np.linalg.norm(S_dense)
    assert rel < 1e-10
