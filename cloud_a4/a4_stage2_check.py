"""A4 stage-2 integrity checks (mechanical only; final task book section 4).

Assembles the five official deliverables from per-cell artifacts and runs the
frozen checklist. No statistical judgement happens here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

BUDGETS = [0.1, 0.2, 0.4, 0.6, 0.8]
BUDGET_COUNTS = [14, 28, 57, 85, 114]
REGIMES = [10, 100]
LEVELS = [0.2, 0.5, 1.0]
SEEDS_PER_LEVEL = 10
POLICIES = ["mode_aware", "e_opt", "a_opt", "d_opt", "random"]
DET_POLICIES = POLICIES[:-1]
N_RANDOM_PERMS = 5
ALL_UNITS = DET_POLICIES + [f"random_{p_i}" for p_i in range(N_RANDOM_PERMS)]
COHORT = ["obj_03_pumpkin", "obj_04_dolphin", "obj_07_pumpkin2", "obj_09_ball",
          "obj_10_pumpkin3", "obj_11_pine", "obj_13_mushroom",
          "obj_16_friends_cup", "obj_17_pumpkin5", "obj_18_fabric_hat",
          "obj_19_cylinder"]

fail = []


def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not cond:
        fail.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cells", required=True)
    args = ap.parse_args()
    repo, cells_dir = Path(args.repo), Path(args.cells)
    out = repo / "results/openillumination/allocation"
    import numpy as np

    expected_cells = {(o, lv) for o in COHORT for lv in LEVELS}
    got_cells = set()
    U, orderings, per_run = {}, {}, []
    for cd in sorted(cells_dir.glob("cell_*")):
        if not (cd / "U.json").exists():
            continue
        uj = json.loads((cd / "U.json").read_text(encoding="utf-8"))
        obj_name, level = uj["cell"].rsplit("|", 1)
        got_cells.add((obj_name, float(level)))
        for key, budgets in uj["U"].items():
            U.setdefault(key, {b: 0.0 for b in BUDGETS})
            for b in BUDGETS:
                U[key][b] += budgets[b]           # accumulate across levels
        orderings.update(uj["orderings"])
        per_run.extend(json.loads((cd / "per_run_errors.json").read_text(encoding="utf-8")))

    print("== stage 2: integrity checks (mechanical only) ==")
    check("11 objects x 3 levels = 33 cells", got_cells == expected_cells,
          f"got {len(got_cells)}")
    n_expected = (len(COHORT) * len(LEVELS) * SEEDS_PER_LEVEL * len(REGIMES)
                  * len(ALL_UNITS) * len(BUDGETS))
    check(f"per-run rows == {n_expected}", len(per_run) == n_expected,
          f"got {len(per_run)}")
    combos = Counter((r["object"], r["level"], r["seed"], r["regime"],
                      r["policy"], r["budget"]) for r in per_run)
    check("no duplicate runs", all(v == 1 for v in combos.values()),
          f"{sum(1 for v in combos.values() if v > 1)} duplicated keys")
    bad = [r for r in per_run if not math.isfinite(r["E_run"])]
    check("no NaN/Inf in E_run", not bad, f"{len(bad)} bad")
    flog = cells_dir / "failure.log"
    check("failure log empty", not flog.exists() or not flog.read_text().strip())
    bycell = {}
    for r in per_run:
        bycell.setdefault((r["object"], r["level"], r["seed"], r["regime"],
                           r["budget"]), set()).add(r["n_improved"])
    check("every policy uses the same calibration budget per cell",
          all(len(v) == 1 for v in bycell.values()))
    check("random realized as 5 permutation units",
          {r["policy"] for r in per_run if r["policy"].startswith("random")}
          == {f"random_{i}" for i in range(N_RANDOM_PERMS)})
    check("selection orders: 33 cells x 9 units", len(orderings) == 33 * 9,
          f"got {len(orderings)}")
    check("every ordering is a full 142-light permutation",
          all(len(v) == 142 and sorted(v) == list(range(142))
              for v in orderings.values()))
    check("selection trace carries no reconstruction-side labels",
          not [k for k in orderings if "rho" in k or "residual" in k])

    # leakage-surface re-check on the audited code state (five fields)
    import inspect
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    from calibinfo.allocation.policies import SelectionState
    import dataclasses
    names = {f.name for f in dataclasses.fields(SelectionState)}
    check("SelectionState field set == prediction-side five", names ==
          {"u", "M0", "lam0", "finf", "active"}, str(sorted(names)))

    # assemble deliverables
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_run_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["object", "level", "seed", "regime", "policy", "budget",
                    "n_improved", "E_run"])
        for r in per_run:
            w.writerow([r["object"], r["level"], r["seed"], r["regime"],
                        r["policy"], r["budget"], r["n_improved"],
                        repr(r["E_run"])])
    (out / "selection_orders.json").write_text(
        json.dumps(orderings, indent=1), encoding="utf-8")
    n_lev = len(LEVELS)
    uos = {}
    for key, budgets in U.items():
        obj_name, regime, unit = key.rsplit("|", 2)
        uos[(obj_name, int(regime), unit)] = {b: budgets[b] / n_lev for b in BUDGETS}
    with open(out / "uos_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["object", "regime", "policy", "budget", "E_osb"])
        for (o, regime, unit) in sorted(uos, key=str):
            for b in BUDGETS:
                w.writerow([o, regime, unit, b,
                            repr(uos[(o, regime, unit)][b])])
    print("assembled: per_run_errors.csv / uos_table.csv / selection_orders.json")
    print("STAGE2 =", "FAIL: " + ", ".join(fail) if fail else "PASS")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
