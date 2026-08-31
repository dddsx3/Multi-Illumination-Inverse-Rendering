"""P1-R4′ · solver 收敛判据 pilot 标定（Discovery 上执行，非确认性证据）。

预注册 §2：批量 solver 在 Discovery 4 scene × N=5 × 6 subsets = 24 trials 上
记录 tail_range / grad_norm 分布 → 取 P75 冻结为收敛判据 → 写
p1/information_audit/r4p_conv_thresholds.json（冻结后不得更改）。

用法：python p1/source/information_audit/r4p_pilot_calibration.py
"""
import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "information_audit"))

from solver_batched import joint_solve_batched  # noqa: E402
from information_audit_v2 import load_scene  # noqa: E402

DATA_ROOT = os.path.join(_REPO, "p1", "calibration_set", "data_sun")
OUT_JSON = os.path.join(_REPO, "p1", "information_audit", "r4p_conv_thresholds.json")
N = 5
SUBSETS = 6
SEED = 20260906


def main():
    rng = np.random.default_rng(SEED)
    scenes = sorted([os.path.join(DATA_ROOT, d) for d in os.listdir(DATA_ROOT)
                     if os.path.isfile(os.path.join(DATA_ROOT, d, "sh_coeffs_irradiance.npy"))])
    tails, grads, losses = [], [], []
    for sd in scenes:
        sc = load_scene(sd)
        subs = [sorted(rng.choice(sc["K"], N, replace=False).tolist())
                for _ in range(SUBSETS)]
        res = joint_solve_batched(sc, subs, restarts=3, N=N if False else None) \
            if False else joint_solve_batched(sc, subs, restarts=3)
        for r in res:
            tails.append(r["tail_range"])
            grads.append(r["grad_norm"])
            losses.append(r["final_loss"])
        print(f"  {sc['name']}: done", flush=True)
    tails = np.array(tails); grads = np.array(grads)
    out = dict(
        n_trials=int(tails.size), N=N, subsets_per_scene=SUBSETS, seed=SEED,
        tail_range=dict(P25=float(np.percentile(tails, 25)), P50=float(np.percentile(tails, 50)),
                        P75=float(np.percentile(tails, 75)), P90=float(np.percentile(tails, 90)),
                        max=float(tails.max())),
        grad_norm=dict(P25=float(np.percentile(grads, 25)), P50=float(np.percentile(grads, 50)),
                       P75=float(np.percentile(grads, 75)), P90=float(np.percentile(grads, 90)),
                       max=float(grads.max())),
        final_loss_med=float(np.median(losses)),
        frozen=dict(tail_range_max=float(np.percentile(tails, 75)),
                    grad_norm_max=float(np.percentile(grads, 75))),
        rule="trial 收敛 iff tail_range < frozen.tail_range_max 且 grad_norm < frozen.grad_norm_max",
    )
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"[pilot] frozen -> {OUT_JSON}")


if __name__ == "__main__":
    main()
