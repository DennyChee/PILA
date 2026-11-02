#!/bin/bash
# Usage:       bash stage_k1_snr_figures.sh        (from the project root, after the K=1 SNR jobs finish)
# Description: Stages figures for the single-source SNR sweep (generate_k1_snr_jobs.py):
#              1. summary.csv (summarize_k1_snr_runs.py) + k1_vs_k2_snr_summary.png (plot_k1_snr_summary.py)
#              2. per-run 4-panel PILA-vs-truth report (plot_multisource_report.py, same format as the K=2 study)
#                 into figures/multisource_k1_snr/<album>/seed<S>/
#              plot_multisource_report.py exits non-zero on noisy albums (its truth==data check fails by design),
#              so its exit status is not treated as an error; missing PNGs are reported instead.
# Date:        2026-10-20
set -e

SAVE_ROOT="saved/multisource_k1_snr"
FIG_ROOT="figures/multisource_k1_snr"
NOISY_DIR="synthetic/multisource/pila_stacks_v2/albums"
SNR_DIR="synthetic/multisource/pila_stacks_v2/albums_snr"

echo "[1/3] Scoring runs..."
python summarize_k1_snr_runs.py --root "$SAVE_ROOT" --out-csv "$FIG_ROOT/summary.csv"

echo "[2/3] Summary figure..."
python plot_k1_snr_summary.py --k1-csv "$FIG_ROOT/summary.csv" --out "$FIG_ROOT/k1_vs_k2_snr_summary.png"

echo "[3/3] Per-run report figures..."
for album_root in "$SAVE_ROOT"/k1snr_seed*/*; do
    seed_tag="$(basename "$(dirname "$album_root")")"          # k1snr_seed<S>
    album="$(basename "$album_root")"
    case "$album" in *_n[0-9][0-9][0-9]) truth="$SNR_DIR/${album}_truth.json" ;; *) truth="$NOISY_DIR/${album}_truth.json" ;; esac
    out_dir="$FIG_ROOT/$album/${seed_tag#k1snr_}"
    mkdir -p "$out_dir"
    python plot_multisource_report.py --album-root "$album_root" --truth "$truth" --fit-k 1 --out-dir "$out_dir" \
        > "$out_dir/report.log" 2>&1 || true
    n_png=$(ls "$out_dir"/*.png 2>/dev/null | wc -l)
    [ "$n_png" -lt 4 ] && echo "  WARNING: $out_dir has only $n_png PNGs (see report.log)"
done
echo "Done. Output saved to $FIG_ROOT/"
