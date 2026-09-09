#!/usr/bin/env bash
# =============================================================================
# ONE-KEY RESULTS PACKAGER — zips every artifact the author needs for review:
#   5 official deliverables + summary + per-object pairs + orders + logs
#   + A4_run_manifest.json + prerun manifest + failure log (if any)
# Output: a4_results_<timestamp>.zip  (deliver this file to the author)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
RESULTS="${RESULTS:-$(pwd)/repo-new/results/openillumination/allocation}"
LOGS="${LOGS:-$PWD/logs}"
CELLS="${CELLS:-$PWD/cells}"
TS=$(date -u +%Y%m%d_%H%M%S)
ZIP="a4_results_${TS}.zip"
STAGE="$PWD/_package_stage"
rm -rf "$STAGE"; mkdir -p "$STAGE"

# 5 official deliverables
for f in per_run_errors.csv uos_table.csv selection_orders.json \
         allocation_summary.json allocation_deltas.csv \
         allocation_object_pairs.csv; do
  [ -f "$RESULTS/$f" ] && cp "$RESULTS/$f" "$STAGE/$f"
done
# provenance
[ -f "$RESULTS/prerun_manifest.sha256" ] && cp "$RESULTS/prerun_manifest.sha256" "$STAGE/"
[ -f "${A4MAN:-$PWD/A4_run_manifest.json}" ] && cp "${A4MAN:-$PWD/A4_run_manifest.json}" "$STAGE/" 2>/dev/null || \
  [ -f "$LOGS/../A4_run_manifest.json" ] && cp "$LOGS/../A4_run_manifest.json" "$STAGE/" 2>/dev/null
# logs
for l in "$LOGS"/*.log; do [ -f "$l" ] && cp "$l" "$STAGE/log_$(basename "$l")"; done
[ -f "$CELLS/failure.log" ] && cp "$CELLS/failure.log" "$STAGE/failure.log"
# git state proof
git -C repo-new rev-parse HEAD > "$STAGE/run_commit.txt" 2>/dev/null || true
git -C repo-new tag --points-at HEAD >> "$STAGE/run_commit.txt" 2>/dev/null || true

zip -q -r "$ZIP" "$STAGE"
rm -rf "$STAGE"
echo ""
echo "=================================================================="
echo " PACKAGE READY: $ZIP"
echo " contents:"
unzip -l "$ZIP" | tail -n +4 | head -20
echo "=================================================================="
echo " Deliver this zip to the author (download via scp / cloud console)."
