#!/usr/bin/env bash
# =============================================================================
# LIVE STATUS MONITOR — run this in a second terminal while A4 is running:
#   bash status.sh
# Shows: completion progress, running workers, elapsed/ETA, latest log lines.
# =============================================================================
cd "$(dirname "$0")"
CELLS_DIR="${CELLS_DIR:-$PWD/cells}"
LOG="${LOG:-$PWD/logs/a4_run.log}"
TOTAL=33

echo "=================================================================="
echo " A4 RUN STATUS · $(date '+%H:%M:%S')"
echo "=================================================================="
DONE=$(ls "$CELLS_DIR" 2>/dev/null | grep -c "^cell_.*_0\.2$")
DONE=$(find "$CELLS_DIR" -maxdepth 1 -name "cell_*" -type d 2>/dev/null | wc -l)
echo " cells completed : $DONE / $TOTAL"
if [ "$DONE" -gt 0 ] && [ -f "$LOG" ]; then
  FIRST=$(grep -m1 "done" "$LOG" | head -1)
  LAST=$(grep "done" "$LOG" | tail -1)
  echo " latest          : $LAST"
fi
NW=$(pgrep -fc "a4_cloud_driver" 2>/dev/null || echo 0)
echo " worker processes: $NW"
if [ -f "$LOG" ]; then
  echo "------------------------------------------------------------------"
  tail -6 "$LOG"
fi
echo "------------------------------------------------------------------"
if [ "$DONE" = "$TOTAL" ]; then
  echo " STATUS: ALL CELLS DONE — proceed to stage 2 (run_all.sh handles it)"
else
  REM=$(( (TOTAL - DONE) * 21 / (NW > 0 ? NW : 1) ))
  echo " STATUS: RUNNING — approx $(( REM / 60 ))h $(( REM % 60 ))min remaining"
fi
echo ""
echo " (refresh: watch -n 30 bash status.sh   |   stop A4: see README_CLOUD.md)"
