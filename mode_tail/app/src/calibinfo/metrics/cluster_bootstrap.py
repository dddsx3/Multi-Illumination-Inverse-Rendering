"""Object-cluster bootstrap（T6.3/T7.5；B=10000，seed=20260908）。

抽样单元 = 整对象（保留对象内全部 levels×modes）；禁止 330-point 独立 bootstrap。
compute_stat(objs) 接收 [(obj_id, payload), ...]（含重复项），返回标量统计量。
"""
from __future__ import annotations

import numpy as np


def cluster_bootstrap(objects: list, compute_stat, B: int, seed: int):
    """返回 (point, ci95, boots)。

    objects: [(obj_id, payload)]；payload 任意（compute_stat 解释）。
    point = compute_stat(objects)（全样本）；bootstrap 有放回整对象重抽 n 次。
    """
    point = compute_stat(objects)
    rng = np.random.default_rng(seed)
    n = len(objects)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        boots.append(float(compute_stat([objects[i] for i in idx])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), (float(lo), float(hi)), boots
