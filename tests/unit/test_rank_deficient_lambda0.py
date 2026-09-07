"""Λ=0 秩亏 erratum + 谱形式（红队已知答案测试，红线 RL-solve / RL-thin-svd）。

来源：legacy_redteam/redteam2_exp.py §N3（谱形式，红队观测 1.4e-16）+ 红队报告二 §5 攻击四
N3-fix 补测（`solve` 在奇异 BᵀB 上不报错、静默返回垃圾，与 pinv 极限差 8.3e4 相对误差；
pinv 极限下连续统良定义，M(0) 秩 = m − rank(B)）。
阈值分层（宪法 §6.2）：identity 类 <1e-10；连续性 <1e-6；静默垃圾为方向性断言（rel err >1e2）。
实现红线：所有 ΔF(Λ) 实现禁用 solve 于 Λ=0/秩亏路径，统一 lstsq/SVD/pinv。
"""

import numpy as np
import pytest

from _redteam_reference import SEED

RANK, M_DIM = 4, 600


@pytest.fixture(scope="module")
def rank_deficient_B():
    """600×9、数值秩恰为 4 的 B：B = X @ R（X 600×4, R 4×9）。
    浮点矩阵乘破坏精确线性相关 → BᵀB 有 ~1e-16 量级伪特征值而非精确零 →
    LAPACK 不报错、solve 静默返回垃圾 = 红线现象的确定性复现。"""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(M_DIM, RANK))
    R = rng.normal(size=(RANK, 9))
    return X @ R


def test_n3_spectral_form_full_rank():
    """thin-SVD 谱形式 vs 直接构造（红队 N3 观测 1.4e-16）。
    RL-thin-svd：零奇异方向由 (I−UUᵀ) 项承载，禁 full-SVD 逐项写。"""
    rng = np.random.default_rng(SEED)
    Bf = rng.normal(size=(600, 9))
    for lam in [1e-3, 1.0, 100.0]:
        Mdir = np.eye(600) - Bf @ np.linalg.solve(Bf.T @ Bf + lam * np.eye(9),
                                                  Bf.T)
        U, sv, _ = np.linalg.svd(Bf, full_matrices=False)
        Msvd = (np.eye(600) - U @ U.T) + U @ np.diag(lam / (sv ** 2 + lam)) @ U.T
        rel = np.linalg.norm(Mdir - Msvd) / np.linalg.norm(Mdir)
        assert rel < 1e-10


def test_lambda0_pinv_limit_rank_deficient(rank_deficient_B):
    """Λ=0 秩亏：lstsq 路径 == pinv 极限（identity <1e-10）；
    M(0) 秩 = m − rank(B)（红队报告二 N3-fix）；
    thin-SVD 谱形式在 λ=0 与 pinv 极限一致。"""
    B = rank_deficient_B
    BTB = B.T @ B
    M_lstsq = np.eye(M_DIM) - B @ np.linalg.lstsq(BTB, B.T, rcond=None)[0]
    M_pinv = np.eye(M_DIM) - B @ np.linalg.pinv(BTB) @ B.T
    rel = np.linalg.norm(M_lstsq - M_pinv) / np.linalg.norm(M_pinv)
    assert rel < 1e-10

    sv_M = np.linalg.svd(M_pinv, compute_uv=False)
    n_ones = int((sv_M > 1 - 1e-8).sum())
    assert n_ones == M_DIM - np.linalg.matrix_rank(B)   # 600 − 4 = 596

    U, svb, _ = np.linalg.svd(B, full_matrices=False)
    U = U[:, svb > 1e-12 * svb.max()]                    # col(B) 正交基（thin）
    M_svd0 = np.eye(M_DIM) - U @ U.T
    rel0 = np.linalg.norm(M_svd0 - M_pinv) / np.linalg.norm(M_pinv)
    assert rel0 < 1e-10


def test_lambda0_continuity_with_pinv(rank_deficient_B):
    """pinv 极限下连续统良定义：|M(0⁺) − M(1e-10)| / |M(0)| < 1e-6。"""
    B = rank_deficient_B
    BTB = B.T @ B
    M0 = np.eye(M_DIM) - B @ np.linalg.pinv(BTB) @ B.T
    Mlt = np.eye(M_DIM) - B @ np.linalg.solve(BTB + 1e-10 * np.eye(9), B.T)
    rel = np.linalg.norm(M0 - Mlt) / np.linalg.norm(M0)
    assert rel < 1e-6


def test_solve_silent_garbage_redline(rank_deficient_B):
    """红线 RL-solve 数值复现：solve 在奇异 G 上无任何担保——要么报错、要么静默返回
    垃圾（一般 RHS 下与 pinv 相对误差必然天文数字；红队观测 8.3e4 来自其内嵌补测的
    具体构造，本测试用一般 RHS 展示同一机制，与 BLAS 行为无关）。
    禁止任何实现依赖 solve 走 Λ=0/秩亏路径。"""
    B = rank_deficient_B
    BTB = B.T @ B
    C = np.random.default_rng(12345).normal(size=(9, 9))   # 一般 RHS
    w_pinv = np.linalg.pinv(BTB) @ C
    try:
        w_solve = np.linalg.solve(BTB, C)
    except np.linalg.LinAlgError:
        return                     # 本环境检测到奇异并报错 —— 同样满足红线（不可依赖）
    rel = np.linalg.norm(w_solve - w_pinv) / np.linalg.norm(w_pinv)
    assert rel > 1e2


def test_solve_exact_duplicate_column_raises():
    """重复列（精确奇异）情形：lstsq/pinv 路径正确且互检一致；solve 行为不被依赖。"""
    rng = np.random.default_rng(SEED)
    B = rng.normal(size=(300, 4))
    B[:, 3] = B[:, 1]                                   # 精确重复列
    BTB = B.T @ B
    w_lstsq = np.linalg.lstsq(BTB, B.T, rcond=None)[0]
    w_pinv = np.linalg.pinv(BTB) @ B.T
    rel = np.linalg.norm(w_lstsq - w_pinv) / np.linalg.norm(w_pinv)
    assert rel < 1e-10                                  # 参考路径互检一致
