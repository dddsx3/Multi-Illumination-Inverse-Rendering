"""Results packager for the cloud (called by run_all.sh; see package_results.sh)."""
import argparse
import zipfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True)
ap.add_argument("--results", required=True)
ap.add_argument("--logs", required=True)
ap.add_argument("--a4man", default=None)
ap.add_argument("--prerun", default=None)
ap.add_argument("--runlog", default=None)
args = ap.parse_args()

st = Path(args.zip).with_suffix("").name
stage = Path("_package_stage")
stage.mkdir(exist_ok=True)
files = {
    "per_run_errors.csv": Path(args.results) / "per_run_errors.csv",
    "uos_table.csv": Path(args.results) / "uos_table.csv",
    "selection_orders.json": Path(args.results) / "selection_orders.json",
    "allocation_summary.json": Path(args.results) / "allocation_summary.json",
    "allocation_deltas.csv": Path(args.results) / "allocation_deltas.csv",
    "allocation_object_pairs.csv": Path(args.results) / "allocation_object_pairs.csv",
    "prerun_manifest.sha256": Path(args.prerun) if args.prerun else None,
    "A4_run_manifest.json": Path(args.a4man) if args.a4man else None,
}
if args.runlog:
    files["log_a4_run.txt"] = Path(args.runlog)
logs = Path(args.logs)
if logs.exists():
    for l in sorted(logs.glob("*.log")):
        files[f"log_{l.name}"] = l
fl = Path("_cells_failure.log")
if fl.exists():
    files["failure.log"] = fl

n = 0
with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
    for name, src in files.items():
        if src and Path(src).exists():
            z.write(src, name)
            n += 1
        else:
            print(f"[package] SKIP (missing): {name}")
print(f"[package] {n} files -> {args.zip}")
