#!/bin/bash
# Usage:       bash stage_k2_keepbest_figures.sh        (from the project root, env pila, after compare_k2_keepbest.py)
# Description: Stages figures for the noisy two-source test (generate_k2_keepbest_jobs.py) into figures/multisource_k2_keepbest/:
#              1. per-seed summary + one epoch trace (plot_k2_keepbest_summary.py)
#              2. 4-panel PILA-vs-truth reports (plot_multisource_report.py) for selected runs, at the LOWEST-LOSS epoch
#                 (via a mirror album folder in a temp dir whose fit_K2.npz links to models/lowest_loss/fit_K2.npz; the
#                 saved runs are not touched) and, for contrast, at the final epoch.
#              plot_multisource_report.py exits non-zero on noisy albums by design (truth != data check).
# Date:        2026-10-22
set -e

FIG_ROOT="figures/multisource_k2_keepbest"
SAVE_ROOT="saved/multisource_k2_tune"
SNR_DIR="synthetic/multisource/pila_stacks_v2/albums_snr"
MIRROR_ROOT="$(mktemp -d)"

# label | noise tag | range | seed | epoch choice
RUNS=(
  "01_SNR12_std_seed15_PICK|n025|std|15|lowest"
  "02_SNR12_box60_seed4_PICK|n025|box60|4|lowest"
  "03_SNR12_std_seed12_lowestloss_rescued|n025|std|12|lowest"
  "04_SNR12_std_seed12_final_trapped|n025|std|12|final"
  "05_SNR6_std_seed14_PICK_fails|n050|std|14|lowest"
)

echo "[1/2] Summary figures..."
python plot_k2_keepbest_summary.py --fig-root "$FIG_ROOT" --seed-trace 12

echo "[2/2] Per-run reports..."
for entry in "${RUNS[@]}"; do
    IFS='|' read -r label noise range seed choice <<< "$entry"
    album="E1_S3_K2_separated_$noise"
    run_root="$SAVE_ROOT/k2tune_snrE1${noise}kb_${range}_wr100_seed${seed}/$album"
    if [ "$choice" = "lowest" ]; then
        models_dir="$(ls -d "$run_root"/K2/*/*/models)"
        mirror_models="$MIRROR_ROOT/$label/$album/K2/run/lowest/models"
        mkdir -p "$mirror_models"
        ln -s "$PWD/$models_dir/lowest_loss/fit_K2.npz" "$mirror_models/fit_K2.npz"
        album_root="$MIRROR_ROOT/$label/$album"
    else
        album_root="$run_root"
    fi
    out_dir="$FIG_ROOT/$label"
    mkdir -p "$out_dir"
    python plot_multisource_report.py --album-root "$album_root" --truth "$SNR_DIR/${album}_truth.json" --fit-k 2 \
        --out-dir "$out_dir" > "$out_dir/report.log" 2>&1 || true
    n_png=$(ls "$out_dir"/*.png 2>/dev/null | wc -l)
    echo "  $label: $n_png PNGs"
done
echo "Done. Output saved to $FIG_ROOT/ (temp mirror $MIRROR_ROOT holds only symlinks)"
