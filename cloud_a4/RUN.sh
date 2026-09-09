#!/usr/bin/env zsh
# =============================================================================
# RUN.sh — THE one command (operator wraps run_all.sh with zsh-safe defaults)
# Usage (from inside cloud_a4/):  zsh RUN.sh
# =============================================================================
cd "$(dirname "$0")"
# run_all.sh is bash; zsh operators get it transparently
if command -v bash >/dev/null; then
  bash run_all.sh "$@"
else
  echo "bash not found — run_all.sh requires bash (apt install bash)"
  exit 1
fi
