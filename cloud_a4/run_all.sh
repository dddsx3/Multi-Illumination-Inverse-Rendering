#!/usr/bin/env bash
# =============================================================================
# A4 CLOUD RUN · one-key operator script (frozen grid, audited code, OFFLINE)
# =============================================================================
# Everything ships inside A4_CLOUD_PACKAGE.zip. NO GitHub access is needed.
#
#   unzip A4_CLOUD_PACKAGE.zip && cd A4_CLOUD_PACKAGE && bash run_all.sh
#
# Interactive confirms between steps; use `bash run_all.sh --auto` for unattended.
# Resumable: completed cells are skipped on re-run.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

AUTO=0
[ "${1:-}" = "--auto" ] && AUTO=1

WORK_ROOT="$PWD"
NEW_REPO="$WORK_ROOT/repo-new"
CELLS="$WORK_ROOT/cells"
LOGS="$WORK_ROOT/logs"
RESULTS="$NEW_REPO/results/openillumination/allocation"
DATA_ROOT="${MRC_DATA_ROOT:-$WORK_ROOT/data/OpenIllumination}"
DATA_META="${MRC_DATA_META:-$WORK_ROOT/data/OpenIllumination_meta}"
AUDITED_SHA="ddd44b5332de03337720ad9b1095b4a9d7ecdf6f"
BUNDLE="$WORK_ROOT/repo-new.bundle"
export MRC_DATA_ROOT="$DATA_ROOT"
export MRC_DATA_META="$DATA_META"

mkdir -p "$LOGS"

banner() { echo ""; echo "=================================================================="; echo " $1"; echo "=================================================================="; }
confirm() {
  if [ "$AUTO" = "1" ]; then echo "  [auto] skipping prompt"; return 0; fi
  read -r -p "  >> Continue to $1? [y/N] " reply
  case "$reply" in y|Y|yes|YES) return 0 ;; *) echo "ABORTED at $1"; exit 1 ;; esac
}

banner "A4 CLOUD RUN · preregistered allocation experiment (offline package)"
echo "  audited code : $AUDITED_SHA (allocation-prerun-audited)"
echo "  work root    : $WORK_ROOT"
confirm "STEP 1 (environment preflight)"

# ------------------------------------------------ STEP 1: preflight
banner "STEP 1 · Environment preflight"
command -v python3 >/dev/null || {
  echo "FATAL: python3 not found. Ask your admin:"
  echo "  sudo apt update && sudo apt install -y python3 python3-venv git zip"
  exit 1; }
PYV=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
python3 -c "import sys; assert sys.version_info >= (3,10)" || {
  echo "FATAL: python >= 3.10 required (found $PYV)"; exit 1; }
echo "  python3: $PYV"
command -v git >/dev/null || { echo "FATAL: git not found (apt install git)"; exit 1; }
command -v zip >/dev/null || { echo "FATAL: zip not found (apt install zip)"; exit 1; }
echo "  git: $(git --version)"

# dataset probe (exact loader contract: 11 objects x 142 thumbnails)
found_data=0
for d in "$DATA_ROOT" "$WORK_ROOT/data/OpenIllumination" "$HOME/data/OpenIllumination" "/data/OpenIllumination"; do
  if [ -f "$d/OLAT/obj_03_pumpkin/Lights/000/com_masked_thumbnail/A1.png" ]; then
    DATA_ROOT="$d"; export MRC_DATA_ROOT="$d"
    DM="$(dirname "$d")/OpenIllumination_meta"
    [ -f "$DM/light_pos.npy" ] && DATA_META="$DM" && export MRC_DATA_META="$DM"
    found_data=1; echo "  dataset found: $DATA_ROOT"; break
  fi
done
[ "$found_data" = "1" ] || {
  echo "FATAL: dataset not found. Expected $DATA_ROOT with:"
  echo "  OLAT/obj_03_pumpkin/Lights/000/com_masked_thumbnail/A1.png  (x142 per object)"
  echo "The package ships data/ inside — extract was incomplete?"; exit 1; }
for o in obj_03_pumpkin obj_04_dolphin obj_07_pumpkin2 obj_09_ball obj_10_pumpkin3 \
         obj_11_pine obj_13_mushroom obj_16_friends_cup obj_17_pumpkin5 \
         obj_18_fabric_hat obj_19_cylinder; do
  CNT=$(find "$DATA_ROOT/OLAT/$o/Lights" -name "A1.png" -path "*com_masked_thumbnail*" 2>/dev/null | wc -l)
  [ "$CNT" = "142" ] || { echo "FATAL: $o has $CNT/142 thumbnails — incomplete data"; exit 1; }
done
[ -f "$DATA_META/light_pos.npy" ] || { echo "FATAL: light_pos.npy missing at $DATA_META"; exit 1; }
echo "  dataset: 11 objects x 142 thumbnails + light_pos.npy verified"

# ------------------------------------------------ STEP 2: audited code (offline)
banner "STEP 2 · Audited code from the offline git bundle"
if [ -f "$NEW_REPO/.git/HEAD" ]; then
  echo "  repo-new already restored"
else
  # robust bundle restore: init + fetch (works with or without HEAD in bundle)
  git init -q "$NEW_REPO"
  git -C "$NEW_REPO" fetch -q "$BUNDLE" "$AUDITED_SHA" "refs/tags/allocation-prerun-audited:refs/tags/allocation-prerun-audited" || {
    echo "FATAL: bundle fetch failed — is repo-new.bundle present/intact?"; exit 1; }
  git -C "$NEW_REPO" checkout -q -B main FETCH_HEAD
fi
git -C "$NEW_REPO" checkout -q "$AUDITED_SHA"
GOT=$(git -C "$NEW_REPO" rev-parse HEAD)
[ "$GOT" = "$AUDITED_SHA" ] || { echo "FATAL: HEAD=$GOT != $AUDITED_SHA"; exit 1; }
echo "  repo-new @ $GOT = allocation-prerun-audited  ✓"

# ------------------------------------------------ STEP 3: venv + deps
banner "STEP 3 · Python venv + dependencies"
if [ ! -d "$WORK_ROOT/.venv" ]; then
  python3 -m venv "$WORK_ROOT/.venv"
fi
. "$WORK_ROOT/.venv/bin/activate"
pip install -q --upgrade pip 2>/dev/null || true
pip install -q numpy scipy pyyaml Pillow pytest || { echo "FATAL: pip install failed"; exit 1; }
pip install -q -e "$NEW_REPO" || { echo "FATAL: editable install failed"; exit 1; }
python -c "import calibinfo, numpy, scipy, yaml, PIL; print('  imports OK')"
PYEXE=$(command -v python)

# ------------------------------------------------ STEP 4: pre-run seal
banner "STEP 4 · Pre-Run Seal (7 mechanical checks incl. F5 manifest gate)"
cd "$NEW_REPO"
"$PYEXE" "$WORK_ROOT/a4_cloud_driver.py" --precheck --repo "$NEW_REPO" 2>&1 | tee "$LOGS/precheck.log"
grep -q "A4_PRECHECK = PASS" "$LOGS/precheck.log" || {
  echo "FATAL: pre-run seal FAILED — see $LOGS/precheck.log (send it to the author)"; exit 1; }
confirm "STEP 5 (A4 formal run)"

# ------------------------------------------------ STEP 5: A4 formal run
banner "STEP 5 · A4 formal run (29,700 reconstructions)"
NPROC=$(nproc 2>/dev/null || echo 4)
WORKERS="${MRC_WORKERS:-$(( NPROC > 2 ? NPROC - 2 : 1 ))}"
[ "$WORKERS" -gt 33 ] && WORKERS=33
echo "  workers: $WORKERS (nproc=$NPROC; hard ceiling 33 cells)"
cd "$WORK_ROOT"
"$PYEXE" a4_cloud_driver.py --repo "$NEW_REPO" --workers "$WORKERS" \
    2>&1 | tee "$LOGS/a4_run.log"
grep -q "cells completed: 33/33" "$LOGS/a4_run.log" || {
  echo "FATAL: run incomplete — RE-RUN this script to resume (done cells skip)"; exit 1; }
confirm "STEP 6 (integrity checks)"

# ------------------------------------------------ STEP 6: integrity
banner "STEP 6 · Stage-2 integrity checks (mechanical)"
"$PYEXE" a4_stage2_check.py --repo "$NEW_REPO" --cells "$CELLS" \
    2>&1 | tee "$LOGS/stage2.log"
grep -q "STAGE2 = PASS" "$LOGS/stage2.log" || {
  echo "FATAL: integrity FAILED — see $LOGS/stage2.log (send it to the author)"; exit 1; }
confirm "STEP 7 (A5 frozen statistics)"

# ------------------------------------------------ STEP 7: A5 analysis
banner "STEP 7 · A5 frozen statistics (preregistered, zero design)"
"$PYEXE" a5_analyze.py --repo "$NEW_REPO" --results "$RESULTS" \
    2>&1 | tee "$LOGS/a5.log"
grep -q "A5 = DONE" "$LOGS/a5.log" || { echo "FATAL: A5 failed"; exit 1; }
confirm "STEP 8 (package results)"

# ------------------------------------------------ STEP 8: package
banner "STEP 8 · Package all artifacts"
TS=$(date -u +%Y%m%d_%H%M%S)
ZIP="$WORK_ROOT/a4_results_${TS}.zip"
"$PYEXE" package_results.py --zip "$ZIP" --results "$RESULTS" --logs "$LOGS" \
    --a4man "$WORK_ROOT/A4_run_manifest.json" \
    --prerun "$RESULTS/prerun_manifest.sha256" --runlog "$LOGS/a4_run.log" \
    --cells "$CELLS" 2>&1 | tail -3
echo ""
echo "=================================================================="
echo " A4 COMPLETE"
echo " deliverable : $ZIP"
echo " next        : download this zip and send it to the author"
echo "=================================================================="
