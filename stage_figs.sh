#!/bin/bash
# Usage:       ./stage_figs.sh OUTDIR ROOT [ROOT ...]
#   e.g.       ./stage_figs.sh scp_figs saved/tscurv_* saved/temporalsmooth* saved/temporalprior*
# Description: Collect every .png/.pdf figure found under the given run-dir ROOTs into a
#              single flat OUTDIR for scp. Filenames are prefixed by their experiment
#              family (the first path component under saved/) so figures from different
#              experiments never collide. Then reports any leaf run dir that has a
#              trained model or a dumped uspace_series.csv but NO figure -- i.e. runs
#              whose plot step never produced output and likely need re-plotting.
# Date:        2026-07-27
set -e

# --- Args ---
if [ "$#" -lt 2 ]; then
  echo "usage: $0 OUTDIR ROOT [ROOT ...]" >&2
  exit 1
fi
OUTDIR="$1"; shift
mkdir -p "$OUTDIR"

# --- Step 1: copy all figures, prefixing with the experiment-family dir name ---
echo "[1/2] Staging figures into $OUTDIR/ ..."
n_copied=0
for root in "$@"; do
  [ -e "$root" ] || { echo "  skip (no match): $root"; continue; }
  while IFS= read -r fig; do
    # Experiment family = first path component after 'saved/' (unique per experiment).
    family="$(echo "$fig" | sed -E 's#^saved/([^/]+)/.*#\1#')"
    dest="$OUTDIR/${family}__$(basename "$fig")"
    cp -f "$fig" "$dest"
    n_copied=$((n_copied + 1))
  done < <(find "$root" -type f \( -name '*.png' -o -name '*.pdf' \))
done
echo "  Copied $n_copied file(s)."

# --- Step 2: warn about run dirs that finished/dumped but have no figure ---
echo "[2/2] Checking for runs missing figures (trained/dumped but not plotted)..."
missing=0
for root in "$@"; do
  [ -e "$root" ] || continue
  # A leaf run dir is one containing either a model checkpoint or a dumped series.
  while IFS= read -r marker; do
    rundir="$(dirname "$marker")"
    # walk up to the run root (dir holding the uspace_series.csv, if any)
    if ! find "$rundir" -maxdepth 2 -name '*.png' | grep -q .; then
      echo "  MISSING FIGURE: $rundir"
      missing=$((missing + 1))
    fi
  done < <(find "$root" -type f -name 'uspace_series.csv')
done
[ "$missing" -eq 0 ] && echo "  None. All dumped runs have figures."
echo "Done. Staged in $OUTDIR/ ; rm when scp'd."
