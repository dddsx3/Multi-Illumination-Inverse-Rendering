"""
open5_normal_gram_check.py · OPEN-5 数值证伪（IDENTIFIABILITY_v4 §2.2 / T2-2 一稿后续）

假设（结构驱动）：near_zero − 1 ≈ 9 − rank(Σ_{p 有效} Y_p Y_pᵀ)，即法线 Gram 秩越低，
SH-2 正交补中未被观测的方向越多，per-scene 歧义维数越高（补偿族交集收缩不足）。

用法（CPU）：python r5_compute_audit/open5_normal_gram_check.py
数据源：p1/calibration_set/data_sun_confirmatory/<scene>/（normal_mesh|normal_depth + mask）
对照：r5_compute_audit/raw_profile/a_track_p_a2_fisher.csv 的 near_zero（config 不变，取 c0）
"""

import csv
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "p1" / "calibration_set" / "data_sun_confirmatory"
CSV_PATH = REPO / "r5_compute_audit" / "raw_profile" / "a_track_p_a2_fisher.csv"


def sh2_basis_columns(normals):
    """9-D real SH-2 列（未归一常数；rank 对每列常数缩放不变）。normals (P,3)。"""
    x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
    return np.stack([
        np.ones_like(x),        # l0
        y, z, x,                # l1
        x * y, y * z, z * x,    # l2 (off-diag)
        x * x - y * y,          # l2
        2.0 * z * z - x * x - y * y,  # l2
    ], axis=1)  # (P,9)


def gram_rank(normal_path, mask_path, valid_thr=0.5, albedo=None):
    n = np.load(normal_path)
    m = np.asarray(np.load(mask_path)).squeeze()     # 去全部单例轴
    # 兼容 (3,H,W) 通道在前
    if n.ndim == 3 and n.shape[0] == 3 and n.shape[-1] != 3:
        n = np.transpose(n, (1, 2, 0))
    while n.ndim > 3:
        n = n[0]
    ok = m > valid_thr
    nv = n[ok]
    if albedo is not None:
        a = albedo[ok].astype(np.float64) ** 2
    else:
        a = np.ones(nv.shape[0])
    ln = np.linalg.norm(nv.reshape(nv.shape[0], -1), axis=1)
    nv = nv.reshape(nv.shape[0], -1) / np.maximum(ln[:, None], 1e-12)
    B = sh2_basis_columns(nv.astype(np.float64))
    G = (B * a[:, None]).T @ B          # 加权像素法线 Gram，9×9
    r = np.linalg.matrix_rank(G, tol=1e-9 * max(1.0, np.linalg.norm(G, 2)))
    s = np.linalg.svd(G, compute_uv=False)
    return r, s


def main():
    near = {}
    for row in csv.DictReader(open(CSV_PATH, encoding="utf-8")):
        if int(row["config"]) == 0:
            near[row["scene"]] = int(row["near_zero"])

    print(f"{'scene':22s} {'near0':>5s} {'rankMesh':>8s} {'rankDepth':>9s}  "
          f"{'9-rankMesh':>10s}  near0-1")
    rows = []
    for scene, nz in sorted(near.items()):
        d = DATA_ROOT / scene
        rm, sm = gram_rank(d / "normal_mesh.npy", d / "mask.npy")
        rd, sd = gram_rank(d / "normal_depth.npy", d / "mask.npy")
        rows.append((scene, nz, rm, rd))
        print(f"{scene:22s} {nz:5d} {rm:8d} {rd:9d} {9 - rm:10d} {nz - 1:7d}")

    # 简化相关：rank 与 near_zero 的序数一致性（Spearman 手算）
    def spearman(xs, ys):
        def rankify(v):
            s = sorted(v)
            return [s.index(x) + 1 for x in v]
        rx, ry = rankify(xs), rankify(ys)
        n = len(xs)
        mx, my = np.mean(rx), np.mean(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = np.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        return num / den

    xs = [9 - r[2] for r in rows]
    ys = [r[1] - 1 for r in rows]
    print(f"\nSpearman(9-rankMesh, near_zero-1) = {spearman(xs, ys):.3f}")
    xs_d = [9 - r[3] for r in rows]
    print(f"Spearman(9-rankDepth, near_zero-1) = {spearman(xs_d, ys):.3f}")


if __name__ == "__main__":
    main()
