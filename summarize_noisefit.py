#!/usr/bin/env python3
# Usage:       python summarize_noisefit.py \
#                  [--out figures/presentation_noisefit/noisefit_summary.csv]
# Description: Quantify the "jitter = noise-fitting, not identifiability" argument by
#              comparing the PILA baseline (no prior) on two synthetic Mogi cubes that
#              share the SAME rising-dV source trajectory and differ ONLY in whether
#              APS noise was added: marapi_mogi_rising_clean (noise: none) vs
#              marapi_mogi_rising_combined (noisy). For each regime it reports, over the
#              QUIESCENT window (epochs whose TRUE dV is ~0), the recovered-dV behaviour
#              and the across-epoch scatter ("jitter") of the recovered source geometry.
# Date:        2026-08-11
#
# Scientific notes:
#   * Quiescent window = epochs with |truth_dV| < quiescent_frac * max|truth_dV|
#     (quiescent_frac=0.1, matching plot_per_epoch_recon.py's default). In these epochs
#     the true surface deformation is ~0, so a well-behaved inversion should return
#     dV ~ 0 and a stable (non-jittering) source location/depth.
#   * A "spurious spike" is a quiescent epoch whose recovered |dV| exceeds
#     spike_frac * max|truth_dV| (spike_frac=0.2): the encoder invented a source where
#     there is no real deformation.
#   * dV in m^3; xcen/ycen/depth in m. LOS convention positive = toward satellite.
#   * Key expectation: the CLEAN (noise-free) baseline recovers dV~0 with ~no spikes and
#     tiny geometry scatter -> the zero-deformation state is IDENTIFIABLE. The NOISY
#     baseline, same trajectory, spikes and jitters -> the jitter is driven by fitting
#     APS noise, not by non-identifiability.

import argparse
import csv
import json
import math
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Physical dV values are ~1e6-1e8 m^3; report in 1e6 m^3 for readability.
DV_SCALE = 1e-6
DV_UNIT = "x10^6 m^3"

# The two runs being compared. Both are the PILA baseline (no prior); the ONLY
# experimental difference is APS noise on/off (identical source trajectory).
REGIMES = [
    {
        "label": "no noise (baseline)",
        "csv": "saved/dvl1u_marapi_mogi_rising_clean/lam0/uspace_series.csv",
        "truth": "synthetic/cubes/marapi_mogi_rising_clean/marapi_mogi_rising_clean_truth.json",
    },
    {
        "label": "with noise (baseline)",
        "csv": "saved/baseline_marapi_mogi_rising_combined/run/uspace_series.csv",
        "truth": "synthetic/cubes/marapi_mogi_rising_combined/marapi_mogi_rising_combined_truth.json",
    },
]


def load_recovered_series(csv_path):
    """
    Load the dumped per-epoch recovered physical parameters.

    Input:  csv_path : uspace_series.csv (cols include xcen, ycen, d, dV in physical units).
    Output: dict of lists {epoch_idx, xcen_m, ycen_m, depth_m, dV_m3}, ordered by epoch.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"recovered CSV not found: {csv_path}")
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    rows.sort(key=lambda r: int(r["epoch_idx"]))
    return {
        "xcen_m": [float(r["xcen"]) for r in rows],
        "ycen_m": [float(r["ycen"]) for r in rows],
        "depth_m": [float(r["d"]) for r in rows],
        "dV_m3": [float(r["dV"]) for r in rows],
    }


def load_truth_dV(truth_path):
    """
    Load per-epoch TRUE dV (m^3) from a synthetic cube's *_truth.json.

    Input:  truth_path : path to <name>_truth.json.
    Output: list of truth dV (m^3), one per epoch (chronological order).
    """
    if not os.path.exists(truth_path):
        raise FileNotFoundError(f"truth JSON not found: {truth_path}")
    with open(truth_path, "r") as f:
        truth = json.load(f)
    dv_idx = truth["param_names"].index("dV")
    return [float(p[dv_idx]) for p in truth["params_per_epoch"]]


def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def _std(vals):
    """Population std (ddof=0); the across-epoch scatter = jitter magnitude."""
    if len(vals) < 2:
        return 0.0
    mu = _mean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))


def _rms(vals):
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else float("nan")


def summarize_regime(regime, quiescent_frac=0.1, spike_frac=0.2):
    """
    Compute the quiescent-window jitter/noise-fitting diagnostics for one regime.

    Inputs:
        regime         : dict with 'label', 'csv', 'truth'.
        quiescent_frac : epoch is quiescent if |truth_dV| < quiescent_frac * max|truth_dV|.
        spike_frac     : quiescent epoch is a spurious spike if |rec_dV| > spike_frac * max|truth_dV|.
    Output:
        dict of scalar metrics (units in the key names).
    """
    rec = load_recovered_series(os.path.join(CURRENT_DIR, regime["csv"]))
    truth_dV_m3 = load_truth_dV(os.path.join(CURRENT_DIR, regime["truth"]))

    n_rec = len(rec["dV_m3"])
    if n_rec != len(truth_dV_m3):
        raise ValueError(
            f"{regime['label']}: recovered has {n_rec} epochs but truth has "
            f"{len(truth_dV_m3)} — refusing to align mismatched series.")

    max_abs_truth = max(abs(v) for v in truth_dV_m3)
    quiescent_thresh_m3 = quiescent_frac * max_abs_truth
    spike_thresh_m3 = spike_frac * max_abs_truth

    # Indices of quiescent (near-zero true deformation) epochs.
    q_idx = [i for i, tv in enumerate(truth_dV_m3) if abs(tv) < quiescent_thresh_m3]

    q_rec_dV = [rec["dV_m3"][i] for i in q_idx]
    q_truth_dV = [truth_dV_m3[i] for i in q_idx]
    q_dV_error = [r - t for r, t in zip(q_rec_dV, q_truth_dV)]

    n_spikes = sum(1 for v in q_rec_dV if abs(v) > spike_thresh_m3)

    return {
        "label": regime["label"],
        "n_epochs": n_rec,
        "n_quiescent_epochs": len(q_idx),
        "max_abs_truth_dV_m3": max_abs_truth,
        "quiescent_mean_abs_dV_m3": _mean([abs(v) for v in q_rec_dV]),
        "quiescent_dV_rmse_m3": _rms(q_dV_error),
        "quiescent_dV_rmse_pct_of_peak": 100.0 * _rms(q_dV_error) / max_abs_truth,
        "n_spurious_spikes": n_spikes,
        "spike_frac_of_quiescent": (n_spikes / len(q_idx)) if q_idx else float("nan"),
        # Geometry jitter = across-epoch std of the recovered source position/depth in
        # the quiescent window (truth geometry is ~constant, so this IS the jitter).
        "quiescent_xcen_std_m": _std([rec["xcen_m"][i] for i in q_idx]),
        "quiescent_ycen_std_m": _std([rec["ycen_m"][i] for i in q_idx]),
        "quiescent_depth_std_m": _std([rec["depth_m"][i] for i in q_idx]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Quantify noise-fitting vs identifiability from the clean/noisy baselines.")
    parser.add_argument("--out", type=str,
                        default="figures/presentation_noisefit/noisefit_summary.csv",
                        help="output CSV path for the comparison table")
    parser.add_argument("--quiescent_frac", type=float, default=0.1,
                        help="epoch quiescent if |truth_dV| < frac*max|truth_dV| (default 0.1)")
    parser.add_argument("--spike_frac", type=float, default=0.2,
                        help="quiescent epoch is a spurious spike if |rec_dV| > frac*max|truth_dV| (default 0.2)")
    args = parser.parse_args()

    print("[1/3] Computing per-regime quiescent-window diagnostics...")
    results = [summarize_regime(r, args.quiescent_frac, args.spike_frac) for r in REGIMES]

    print("[2/3] Comparison table (quiescent window = epochs with true dV ~ 0):\n")
    hdr = (f"  {'regime':22s} {'quiet ep':>8s} {'mean|dV|':>12s} {'dV RMSE':>12s} "
           f"{'RMSE %pk':>9s} {'spikes':>10s} {'xcen std':>10s} {'ycen std':>10s} {'depth std':>10s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        print(f"  {r['label']:22s} {r['n_quiescent_epochs']:8d} "
              f"{r['quiescent_mean_abs_dV_m3']*DV_SCALE:11.2f}{'':1s} "
              f"{r['quiescent_dV_rmse_m3']*DV_SCALE:11.2f}{'':1s} "
              f"{r['quiescent_dV_rmse_pct_of_peak']:8.1f}% "
              f"{r['n_spurious_spikes']:3d}/{r['n_quiescent_epochs']:<3d}{'':3s} "
              f"{r['quiescent_xcen_std_m']:9.0f}m "
              f"{r['quiescent_ycen_std_m']:9.0f}m "
              f"{r['quiescent_depth_std_m']:9.0f}m")
    print(f"\n  (dV columns in {DV_UNIT}; std columns = across-epoch scatter of the "
          f"recovered source in the quiescent window = jitter)")

    print("\n[3/3] Writing table CSV...")
    out_path = os.path.join(CURRENT_DIR, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Wrote {len(results)} regimes x {len(fieldnames)} metrics -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
