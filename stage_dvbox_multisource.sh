#!/bin/bash
# Usage:       bash stage_dvbox_multisource.sh        (from the project root, env pila)
# Description: Score + figures for the data-scaled dV box rerun of the standard-recipe multi-source runs, with the
#              SAME tools used for the original (fixed-box) results, written to figures/dvbox_rerun/ only:
#                std     sequential_select_k.py stats/calibrate + summarize_sequential.py   (orig: figures/multisource_seq/step3_std)
#                clean   plot_multisource_report.py per album + --summarise                  (orig: figures/multisource_clean_pila)
#                nonoise plot_sequential_compare.py per album                                (orig: figures/multisource_seq/nonoise)
#                k1      summarize_k1_snr_runs.py + per-run reports, strong album            (orig: figures/multisource_k1_snr)
#              plot_multisource_report.py exits non-zero on noisy albums by design, so its status is not an error.
# Date:        2026-10-22
set -e

SAVE_ROOT="saved/multisource_seq_dvbox"
FIG_ROOT="figures/dvbox_rerun"
CLEAN_DIR="synthetic/multisource/clean_scenarios"
NONOISE_DIR="synthetic/multisource/pila_stacks_v2/albums_nonoise"
NOISY_DIR="synthetic/multisource/pila_stacks_v2/albums"
SNR_DIR="synthetic/multisource/pila_stacks_v2/albums_snr"
mkdir -p "$FIG_ROOT"/{multisource_std,multisource_clean,multisource_nonoise,k1_strong}

echo "[1/4] std: source-count statistics, 5 % bars, Step-3 summary..."
python sequential_select_k.py stats --runs "$SAVE_ROOT/std/*" --out "$FIG_ROOT/multisource_std/stats.csv"
python sequential_select_k.py calibrate --stats "$FIG_ROOT/multisource_std/stats.csv" --pattern "_cal" \
    --out "$FIG_ROOT/multisource_std/thresholds.json"
python summarize_sequential.py --stats "$FIG_ROOT/multisource_std/stats.csv" \
    --thresholds "$FIG_ROOT/multisource_std/thresholds.json" --out-dir "$FIG_ROOT/multisource_std"

echo "[2/4] clean: per-album reports + summary matrix..."
for album_root in "$SAVE_ROOT"/clean/*/; do
    album="$(basename "$album_root")"
    out_dir="$FIG_ROOT/multisource_clean/$album"
    mkdir -p "$out_dir"
    python plot_multisource_report.py --album-root "$album_root" --truth "$CLEAN_DIR/${album}_truth.json" \
        --out-dir "$out_dir" > "$out_dir/report.log" 2>&1 || echo "  WARNING: report failed for $album (see $out_dir/report.log)"
done
python plot_multisource_report.py --summarise --summary-root "$FIG_ROOT/multisource_clean"

echo "[3/4] nonoise: truth-vs-reconstruction per album..."
for album_root in "$SAVE_ROOT"/nonoise/*/; do
    album="$(basename "$album_root")"
    python plot_sequential_compare.py --album-root "$album_root" --truth "$NONOISE_DIR/${album}_truth.json" \
        --out "$FIG_ROOT/multisource_nonoise/${album}_compare.png"
done

echo "[4/4] K=1 strong album, data-scaled box..."
python summarize_k1_snr_runs.py --root saved/multisource_k1_dvbox/rule_strong --out-csv "$FIG_ROOT/k1_strong/rule_strong_summary.csv"
for album_root in saved/multisource_k1_dvbox/rule_strong/k1snr_seed*/*; do
    seed_tag="$(basename "$(dirname "$album_root")")"
    album="$(basename "$album_root")"
    case "$album" in *_n[0-9][0-9][0-9]) truth="$SNR_DIR/${album}_truth.json" ;; *) truth="$NOISY_DIR/${album}_truth.json" ;; esac
    out_dir="$FIG_ROOT/k1_strong/$album/${seed_tag#k1snr_}"
    mkdir -p "$out_dir"
    python plot_multisource_report.py --album-root "$album_root" --truth "$truth" --fit-k 1 --out-dir "$out_dir" \
        > "$out_dir/report.log" 2>&1 || true
done
echo "Done. Output saved to $FIG_ROOT/"
