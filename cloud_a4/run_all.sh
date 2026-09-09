#!/usr/bin/env bash
# =============================================================================
# A4 CLOUD RUN · one-key operator script (frozen grid, no editable science)
# =============================================================================
# WHAT THIS DOES (in order):
#   STEP 1  Environment preflight (python3, pip, git, dataset probe)
#   STEP 2  Clone the audited code (mode-resolved-calibration @ ddd44b5)
#   STEP 3  Python venv + dependencies
#   STEP 4  Pre-Run Seal (7 mechanical checks; F5 manifest gate)
#   STEP 5  A4 formal run (29,700 reconstructions, parallel/resumable)
#   STEP 6  Stage-2 integrity checks (mechanical)
#   STEP 7  A5 frozen statistics + grades
#   STEP 8  Package all artifacts into a single zip
#
# USAGE (from the cloud_a4/ directory):
#   bash run_all.sh                # interactive (confirms each step)
#   bash run_all.sh --auto         # unattended (no prompts)
#
# ALL STATE IS RESUMABLE: completed cells are skipped on re-run.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

AUTO=0
[ "${1:-}" = "--auto" ] && AUTO=1

WORK_ROOT="$PWD"                       # everything lives under the script dir
NEW_REPO="$WORK_ROOT/repo-new"
CELLS="$WORK_ROOT/cells"
LOGS="$WORK_ROOT/logs"
RESULTS="$NEW_REPO/results/openillumination/allocation"
DATA_ROOT="${MRC_DATA_ROOT:-$WORK_ROOT/data/OpenIllumination}"
DATA_META="${MRC_DATA_META:-$WORK_ROOT/data/OpenIllumination_meta}"
AUDITED_SHA="ddd44b5332de03337720ad9b1095b4a9d7ecdf6f"
NEW_REPO_URL="https://github.com/dddsx3/mode-resolved-calibration.git"
export MRC_DATA_ROOT="$DATA_ROOT"
export MRC_DATA_META="$DATA_META"

mkdir -p "$LOGS"

banner() { echo ""; echo "=================================================================="; echo " $1"; echo "=================================================================="; }
confirm() {
  if [ "$AUTO" = "1" ]; then echo "  [auto] skipping prompt"; return 0; fi
  read -r -p "  >> Continue to $1? [y/N] " reply
  case "$reply" in y|Y|yes|YES) return 0 ;; *) echo "ABORTED at $1"; exit 1 ;; esac
}

banner "A4 CLOUD RUN · preregistered allocation experiment"
echo "  audited code : $AUDITED_SHA (tag allocation-prerun-audited)"
echo "  data root    : $DATA_ROOT"
echo "  work root    : $WORK_ROOT"
echo "  workers      : ${MRC_WORKERS:-auto}"
confirm "STEP 1 (environment preflight)"

# ------------------------------------------------ STEP 1: preflight
banner "STEP 1 · Environment preflight"
command -v python3 >/dev/null || { echo "FATAL: python3 not found"; exit 1; }
PYV=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo "  python3: $PYV"
python3 -c "import sys; assert sys.version_info >= (3,10)" || {
  echo "FATAL: python >= 3.10 required"; exit 1; }
command -v git >/dev/null || { echo "FATAL: git not found"; exit 1; }
echo "  git: $(git --version)"

# dataset probe
found_data=0
for d in "$DATA_ROOT" "$WORK_ROOT/data/OpenIllumination" \
         "$HOME/data/OpenIllumination" "/data/OpenIllumination" \
         "$WORK_ROOT/../data/OpenIllumination"; do
  if [ -d "$d/OLAT/obj_03_pumpkin/Lights/000/com_masked_thumbnail" ]; then
    DATA_ROOT="$d"; export MRC_DATA_ROOT="$d"
    DM="$d/../OpenIllumination_meta"
    [ -f "$DM/light_pos.npy" ] && DATA_META="$DM" && export MRC_DATA_META="$DM"
    found_data=1; echo "  dataset found: $DATA_ROOT"; break
  fi
done
if [ "$found_data" = "0" ]; then
  echo "  !! DATASET NOT FOUND — operator action required:"
  echo "  !! upload openillumination_data_bundle.zip (17 MB) and extract so that:"
  echo "  !!   $DATA_ROOT/OLAT/<obj>/Lights/NNN/com_masked_thumbnail/A1.png"
  echo "  !!   $DATA_META/light_pos.npy"
  echo "  !! then re-run this script."
  exit 1
fi
N_LIGHTS_OK=1
for o in obj_03_pumpkin obj_04_dolphin obj_07_pumpkin2 obj_09_ball obj_10_pumpkin3 \
         obj_11_pine obj_13_mushroom obj_16_friends_cup obj_17_pumpkin5 \
         obj_18_fabric_hat obj_19_cylinder; do
  CNT=$(find "$DATA_ROOT/OLAT/$o/Lights" -name "A1.png" -path "*com_masked_thumbnail*" 2>/dev/null | wc -l)
  [ "$CNT" = "142" ] || { echo "  !! $o has $CNT/142 thumbnails — INCOMPLETE DATA"; exit 1; }
done
echo "  dataset: 11 objects x 142 thumbnails verified"

# ------------------------------------------------ STEP 2: audited code
banner "STEP 2 · Audited code (mode-resolved-calibration @ ddd44b5)"
if [ -d "$NEW_REPO/.git" ]; then
  echo "  repo-new already cloned"
  git -C "$NEW_REPO" fetch origin main 2>/dev/null || true
else
  git clone "$NEW_REPO_URL" "$NEW_REPO" || {
    echo "FATAL: clone failed (private repo? add credentials or upload a zip of"
    echo "       mode-resolved-calibration @ ddd44b5 to the cloud and extract as repo-new/)"
    exit 1; }
fi
git -C "$NEW_REPO" checkout -q "$AUDITED_SHA" 2>/dev/null || {
  git -C "$NEW_REPO" checkout -q ddd44b5 2>/dev/null || {
    echo "FATAL: cannot checkout $AUDITED_SHA"; exit 1; }; }
GOT=$(git -C "$NEW_REPO" rev-parse HEAD)
[ "$GOT" = "$AUDITED_SHA" ] || { echo "FATAL: HEAD=$GOT != $AUDITED_SHA"; exit 1; }
echo "  repo-new @ $GOT = allocation-prerun-audited ✓"

# ------------------------------------------------ STEP 3: venv + deps
banner "STEP 3 · Python venv + dependencies"
if [ ! -d "$WORK_ROOT/.venv" ]; then
  python3 -m venv "$WORK_ROOT/.venv"
fi
source "$WORK_ROOT/.venv/bin/activate"
pip install -q --upgrade pip 2>/dev/null || true
pip install -q numpy scipy pyyaml Pillow pytest
pip install -q -e "$NEW_REPO"
python -c "import calibinfo, numpy, scipy, yaml, PIL; print('  imports OK')"
PYEXE=$(command -v python)

# ------------------------------------------------ STEP 4: pre-run seal
banner "STEP 4 · Pre-Run Seal (7 mechanical checks)"
cd "$NEW_REPO"
python "$WORK_ROOT/a4_cloud_driver.py" --precheck 2>&1 | tee "$LOGS/precheck.log"
grep -q "A4_PRECHECK = PASS" "$LOGS/precheck.log" || {
  echo "FATAL: pre-run seal FAILED — see $LOGS/precheck.log"; exit 1; }
confirm "STEP 5 (A4 formal run, ~30-60 min on 32-64 cores)"

# ------------------------------------------------ STEP 5: A4 formal run
banner "STEP 5 · A4 formal run (29,700 reconstructions)"
NPROC=$(nproc 2>/dev/null || echo 8)
WORKERS="${MRC_WORKERS:-$(( NPROC > 2 ? NPROC - 2 : 1 ))}"
[ "$WORKERS" -gt 33 ] && WORKERS=33      # 33 cells is the hard parallel ceiling
echo "  workers: $WORKERS (nproc=$NPROC; capped at 33 cells)"
cd "$WORK_ROOT"
python a4_cloud_driver.py --repo "$NEW_REPO" --workers "$WORKERS" \
    2>&1 | tee "$LOGS/a4_run.log"
grep -q "cells completed: 33/33" "$LOGS/a4_run.log" || {
  echo "FATAL: not all cells completed — see $LOGS/a4_run.log (resumable: re-run)"; exit 1; }
confirm "STEP 6 (integrity checks)"

# ------------------------------------------------ STEP 6: integrity
banner "STEP 6 · Stage-2 integrity checks (mechanical)"
python a4_stage2_check.py --repo "$NEW_REPO" --cells "$CELLS" \
    2>&1 | tee "$LOGS/stage2.log"
grep -q "STAGE2 = PASS" "$LOGS/stage2.log" || {
  echo "FATAL: integrity checks FAILED — see $LOGS/stage2.log"; exit 1; }
confirm "STEP 7 (A5 frozen statistics)"

# ------------------------------------------------ STEP 7: A5 analysis
banner "STEP 7 · A5 frozen statistics (preregistered, zero design)"
python a5_analyze.py --repo "$NEW_REPO" --results "$RESULTS" \
    2>&1 | tee "$LOGS/a5.log"
grep -q "A5 = DONE" "$LOGS/a5.log" || {
  echo "FATAL: A5 failed — see $LOGS/a5.log"; exit 1; }
confirm "STEP 8 (package results)"

# ------------------------------------------------ STEP 8: package
banner "STEP 8 · Package all artifacts"
TS=$(date -u +%Y%m%d_%H%M%S)
ZIP="$WORK_ROOT/a4_results_$TS.zip"
A4MAN="$WORK_ROOT/A4_run_manifest.json"
python package_results.py --zip "$ZIP" --results "$RESULTS" \
    --logs "$LOGS" --a4man "$A4MAN" --prerun "$RESULTS/prerun_manifest.sha256" \
    --runlog "$LOGS/a4_run.log" 2>&1 | tail -2
echo ""
echo "=================================================================="
echo " A4 COMPLETE"
echo " results zip: $ZIP"
echo " grade(s)   : see allocation_summary.json / allocation_deltas.csv"
echo " next       : download the zip and deliver to the author for A5 review"
echo "=================================================================="
