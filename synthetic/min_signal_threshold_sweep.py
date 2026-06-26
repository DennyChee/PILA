#!/usr/bin/env python
# Usage:       python -m synthetic.min_signal_threshold_sweep \
#                  --base-spec synthetic/specs/marapi_mogi_buildup_combined.json \
#                  --base-name marapi_mogi_buildup_combined --out synthetic/snr_sweep_ml1
# Description: Tests how sensitive the 50%-recovery SNR (and recovered fraction) is to the
#              min_signal_mm cutoff that decides which epochs count as "Mogi present".
#              Collects every epoch-inversion ONCE (min_signal_mm=0 -> all epochs, inference
#              run once), then re-applies a range of thresholds in memory and refits the
#              logistic recovery curve at each. Confirms whether including/excluding the weak
#              low-signal tail moves the breakdown SNR.
# Date:        2026-06-23

import argparse

import numpy as np

from synthetic.snr_failure_analysis import collect, logistic_fit, auc


def main():
    ap = argparse.ArgumentParser(description="Sensitivity of SNR50 to the min_signal_mm cutoff.")
    ap.add_argument('--base-spec', default='synthetic/specs/marapi_mogi_buildup_combined.json')
    ap.add_argument('--base-name', default='marapi_mogi_buildup_combined')
    ap.add_argument('--out', default='synthetic/snr_sweep_ml1')
    ap.add_argument('--multilook', type=int, default=1)
    args = ap.parse_args()

    # --- Collect every epoch once (no signal filter) so inference runs only once ---
    print("[1/2] Collecting ALL epoch-inversions (min_signal_mm=0, inference runs once) ...")
    rows, _ = collect(args.base_spec, args.base_name, args.out, min_signal_mm=0.0)
    sig = np.array([r['signal_rms_mm'] for r in rows])
    scene = np.array([r['scene_snr'] for r in rows])
    local = np.array([r['local_snr'] for r in rows])
    loc = np.array([r['loc_err_m'] for r in rows])
    cell_m = (40000.0 / 128) * (args.multilook or 1)
    rec = (loc < cell_m).astype(int)
    print(f"  {len(rows)} total epoch-inversions; signal_rms range "
          f"[{sig.min():.3f}, {sig.max():.1f}] mm; breakdown cell = {cell_m:.0f} m")

    # --- Re-apply each threshold in memory and refit ---
    print("[2/2] Refitting logistic recovery curve at each min_signal_mm threshold\n")
    thresholds_mm = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    hdr = f"{'min_sig_mm':>10} {'n_kept':>7} {'n_dropped':>9} {'rec_frac':>9} " \
          f"{'SNR50_scene':>12} {'SNR50_local':>12} {'AUC_local':>10}"
    print(hdr)
    print("-" * len(hdr))
    for thr in thresholds_mm:
        keep = sig >= thr
        n_kept = int(keep.sum())
        n_drop = int((~keep).sum())
        if n_kept < 5 or rec[keep].sum() in (0, n_kept):
            print(f"{thr:10.1f} {n_kept:7d} {n_drop:9d}   (degenerate: all recovered/failed)")
            continue
        rec_frac = float(rec[keep].mean())
        mu_scene, _ = logistic_fit(scene[keep], rec[keep])
        mu_local, _ = logistic_fit(local[keep], rec[keep])
        a_local = auc(local[keep], rec[keep])
        print(f"{thr:10.1f} {n_kept:7d} {n_drop:9d} {rec_frac:9.3f} "
              f"{mu_scene:12.2f} {mu_local:12.2f} {a_local:10.3f}")

    print("\nNote: SNR50 = SNR (dB) at which P(recover)=0.5. "
          "Stable SNR50 across thresholds => the breakdown curve is insensitive to the cutoff.")


if __name__ == '__main__':
    main()
