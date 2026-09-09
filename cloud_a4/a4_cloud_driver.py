"""A4 cloud driver: parallel, resumable, Linux-native (frozen grid, audited code).

Reads the frozen config from repo-new; overrides ONLY data paths (env) and the
parallel worker count (arg). The grid, seeds, policies, endpoint and statistics
are identical to the audited commit and may not be changed.

Usage:
  python a4_cloud_driver.py --repo /path/to/repo-new [--workers N] [--resume]
  python a4_cloud_driver.py --precheck          # the 7 pre-run seal checks
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO_DEFAULT = Path.cwd() / "repo-new"
DATA_ROOT = os.environ.get("MRC_DATA_ROOT", "./data/OpenIllumination")
DATA_META = os.environ.get("MRC_DATA_META", "./data/OpenIllumination_meta")

BUDGETS = [0.1, 0.2, 0.4, 0.6, 0.8]
BUDGET_COUNTS = [14, 28, 57, 85, 114]      # k = round(b * 142)
REGIMES = [10, 100]
POLICIES = ["mode_aware", "e_opt", "a_opt", "d_opt", "random"]
DET_POLICIES = POLICIES[:-1]
N_RANDOM_PERMS = 5
LEVELS = [0.2, 0.5, 1.0]
SEEDS_PER_LEVEL = 10
COHORT = ["obj_03_pumpkin", "obj_04_dolphin", "obj_07_pumpkin2", "obj_09_ball",
          "obj_10_pumpkin3", "obj_11_pine", "obj_13_mushroom",
          "obj_16_friends_cup", "obj_17_pumpkin5", "obj_18_fabric_hat",
          "obj_19_cylinder"]
AUDITED_SHA = "ddd44b5332de03337720ad9b1095b4a9d7ecdf6f"


def _setup_repo(repo: Path):
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))


def precheck(repo: Path):
    """The 7 pre-run seal checks (final task book, section 2)."""
    os.chdir(repo)
    checks = {}
    ok = True
    h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True).stdout.strip()
    checks["head"] = h
    checks["head_is_audited"] = (h == AUDITED_SHA)
    tag = subprocess.run(["git", "rev-list", "-1", "allocation-prerun-audited"],
                         capture_output=True, text=True).stdout.strip()
    checks["tag_points_to_head"] = (tag == h)
    ok &= checks["head_is_audited"] and checks["tag_points_to_head"]

    st = subprocess.run(["git", "status", "--porcelain"],
                        capture_output=True, text=True).stdout
    checks["worktree_clean"] = (st.strip() == "")
    ok &= checks["worktree_clean"]

    mf = repo / "results/openillumination/allocation/prerun_manifest.sha256"
    expected = {}
    for line in mf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        hsh, name = line.split(None, 1)
        expected[name.lstrip("*")] = hsh
    mismatches = [n for n, w in expected.items()
                  if hashlib.sha256((repo / n).read_bytes()).hexdigest() != w]
    checks["f5_manifest_match"] = (not mismatches)
    checks["f5_files_checked"] = len(expected)
    checks["f5_mismatches"] = mismatches
    ok &= checks["f5_manifest_match"]

    c = subprocess.run(["sha256sum", "-c", "checksums.sha256"],
                       capture_output=True, text=True)
    n_bad = sum(1 for l in c.stdout.splitlines()
                if l.endswith("FAILED") or "WARNING" in l)
    checks["checksum_failures"] = n_bad
    ok &= (n_bad == 0)

    p = subprocess.run([sys.executable, "-m", "pytest"], capture_output=True,
                       text=True)
    tail = ([l for l in p.stdout.splitlines() if "passed" in l] or [""])[-1]
    checks["pytest_summary"] = tail
    ok &= ("failed" not in tail and "error" not in tail and "passed" in tail)

    res = repo / "results/openillumination/allocation"
    allowed = {"prerun_manifest.sha256"}
    extra = sorted(f.name for f in res.iterdir() if f.name not in allowed)
    checks["results_dir_clean"] = (not extra)
    checks["results_dir_extra"] = extra
    ok &= checks["results_dir_clean"]

    checks["all_pass"] = bool(ok)
    checks["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(checks, indent=1))
    print("A4_PRECHECK =", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)
    (repo.parent / "A4_run_manifest.json").write_text(
        json.dumps(checks, indent=1) + "\n", encoding="utf-8")


def cell_work(args):
    obj_idx, obj_name, level, repo_str, data_root, data_meta = args
    t0 = time.perf_counter()
    repo = Path(repo_str)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    import numpy as np
    from calibinfo.allocation.corruption import (
        apply_scaled_corruption, raw_innovations)
    from calibinfo.allocation.policies import (
        SelectionState, budget_scales, select_ordering,
        select_ordering_mode_aware)
    from calibinfo.datasets.openillumination import load_object
    from calibinfo.models.corruption import CorruptionGenerator
    from experiments.openillumination_validation import NominalScene

    cfg = yaml.safe_load(
        (repo / "configs/openillumination_allocation.yaml").read_text(encoding="utf-8"))
    K = cfg["K_lights"]
    all_units = DET_POLICIES + [f"random_{p_i}" for p_i in range(N_RANDOM_PERMS)]
    obj = load_object(data_root, obj_name, data_meta=data_meta)
    scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]))

    u_act = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
    M0_act = np.einsum("kpi,kpj->kij", scen.B_phi, scen.w[:, :, None] * scen.B_phi)
    finf = scen.Finf_diag

    gen = CorruptionGenerator("joint", level)
    sig_logI, sig_rad = gen.sig_logI, np.radians(gen.sig_deg)
    base_lam0 = np.linalg.inv(gen.sigma_phi_diag())

    u142 = np.zeros((K, finf.shape[0], 3))
    M0142 = np.zeros((K, 3, 3))
    lam0142 = np.stack([np.eye(3)] * K)
    u142[scen.sel] = u_act
    M0142[scen.sel] = M0_act
    lam0142[scen.sel] = base_lam0                 # audited F1 wiring
    active142 = np.zeros(K, bool)
    active142[scen.sel] = True
    state = SelectionState(u=u142, M0=M0142, lam0=lam0142, finf=finf,
                           active=active142)
    assert state.u.shape[0] == K
    assert np.allclose(state.lam0[scen.sel], base_lam0)   # F1 wiring gate
    _deg, V_level = scen.predicted_degradation(gen.sigma_phi_diag())

    E_sum = {regime: {unit: {b: 0.0 for b in BUDGETS} for unit in all_units}
             for regime in REGIMES}
    orderings = {}
    per_run = []

    for r_i, regime in enumerate(REGIMES):
        ords = {}
        for pol in DET_POLICIES:
            if pol == "mode_aware":
                ords[pol] = select_ordering_mode_aware(state, regime)[0]
            else:
                ords[pol] = select_ordering(state, pol, regime)[0]
        for p_i in range(N_RANDOM_PERMS):
            rperm = np.random.default_rng([20260911, obj_idx, r_i, p_i])
            ords[f"random_{p_i}"] = select_ordering(state, "random", rng=rperm)[0]
        for unit, ordr in ords.items():
            orderings[f"{obj_name}|{level}|{regime}|{unit}"] = ordr

        for s_i in range(SEEDS_PER_LEVEL):
            z_rng = np.random.default_rng([20260910, 7777, obj_idx, lv_i, s_i])
            raw = raw_innovations(z_rng, len(scen.sel))
            for unit, ordr in ords.items():
                for b, k in zip(BUDGETS, BUDGET_COUNTS):
                    slot_scales = budget_scales(ordr, k, regime)
                    assert (slot_scales < 1.0).sum() == k
                    scales = slot_scales[scen.sel]
                    d2, g = apply_scaled_corruption(
                        scen.dirs, sig_logI, sig_rad, scales, raw)
                    rho_t = scen.estimate_albedo(d2, g)
                    e = rho_t - scen.rho
                    s_g = (e * scen.rho).sum() / (scen.rho ** 2).sum()
                    e = e - s_g * scen.rho             # gauge alignment (CI04)
                    E_run = float(np.sum((e @ V_level) ** 2))
                    E_sum[regime][unit][b] += E_run
                    per_run.append(dict(
                        object=obj_name, level=level, seed=s_i, regime=regime,
                        policy=unit, budget=b,
                        n_improved=int((slot_scales[scen.sel] < 1.0).sum()),
                        E_run=E_run))

    out = {"cell": f"{obj_name}|{level}", "U": {}, "orderings": orderings}
    for regime in REGIMES:
        for unit in all_units:
            out["U"][f"{obj_name}|{regime}|{unit}"] = {
                b: E_sum[regime][unit][b] / SEEDS_PER_LEVEL for b in BUDGETS}
    return out, per_run, time.perf_counter() - t0


def run_all(repo_str, workers):
    repo = Path(repo_str)
    _setup_repo(repo)
    import numpy as np
    cells = [(i, name, level)
             for i, name in enumerate(COHORT) for level in LEVELS]
    cells_dir = Path.cwd() / "cells"
    cells_dir.mkdir(exist_ok=True)
    print(f"[a4] {len(cells)} cells x 2 regimes x (9 units x 5 budgets x "
          f"{SEEDS_PER_LEVEL} seeds) = "
          f"{len(cells)*2*9*5*SEEDS_PER_LEVEL} reconstructions", flush=True)
    todo = [c for c in cells
            if not (cells_dir / f"cell_{c[0]:02d}_{c[2]}" / "U.json").exists()]
    skipped = len(cells) - len(todo)
    if skipped:
        print(f"[a4] resuming: {skipped} cells already done, {len(todo)} to go",
              flush=True)
    done = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(cell_work, c): c for c in todo}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                out, per_run, dt = fut.result()
                cd = cells_dir / f"cell_{c[0]:02d}_{c[2]}"
                cd.mkdir(exist_ok=True)
                (cd / "U.json").write_text(json.dumps(out["U"], indent=1),
                                           encoding="utf-8")
                (cd / "orderings.json").write_text(
                    json.dumps(out["orderings"], indent=1), encoding="utf-8")
                (cd / "per_run_errors.json").write_text(
                    json.dumps(per_run, indent=1), encoding="utf-8")
                done += 1
                el = time.perf_counter() - t0
                eta = (len(todo) - done) / max(done, 1) * el
                print(f"[a4] done {c[1]}|{c[2]} in {dt/60:.1f} min "
                      f"({done}/{len(todo)}; ETA {eta/60:.0f} min)", flush=True)
            except Exception as exc:
                flog = Path.cwd() / "failure.log"
                with open(flog, "a", encoding="utf-8") as f:
                    f.write(f"CELL {c}: {exc!r}\n{traceback.format_exc()}\n")
                print(f"[a4] CELL FAILED {c}: {exc!r} (logged)", flush=True)
    print(f"[a4] cells completed: {done}/{len(todo)} "
          f"(+{skipped} previously done); total wall "
          f"{(time.perf_counter()-t0)/60:.0f} min", flush=True)
    if done != len(todo):
        print("[a4] INCOMPLETE — re-run this script to resume (completed cells "
              "are skipped)", flush=True)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO_DEFAULT))
    ap.add_argument("--workers", type=int,
                    default=max(1, min(33, (os.cpu_count() or 4) - 1)))
    ap.add_argument("--precheck", action="store_true")
    args = ap.parse_args()
    if args.precheck:
        precheck(Path(args.repo).resolve())
        return
    run_all(args.repo, args.workers)


if __name__ == "__main__":
    main()
