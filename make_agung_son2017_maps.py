#!/usr/bin/env python
# Usage:       python make_agung_son2017_maps.py
# Description: Agung Sept-Oct-Nov 2017 LOS maps from the retrained PILA Okada model:
#              columns = Observed LOS | Physics-only (x_P) | PILA residual (x_PB - x_P).
#              The third column is the LEARNED z_aux augmentation (what PILA adds on top of
#              the analytical source), NOT the data-minus-model residual. Gridded to the
#              coherence raster + DEM hillshade (house style). One row per SON-2017 epoch.
# Date:        2026-06-22

import os
import json

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.preprocessing.insar_mintpy import load_insar_mintpy
from plot_insar_results import build_model, points_to_grid, load_hillshade, _r2_rmse
from plot_insar_timeseries import run_inference_all_epochs

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(CURRENT_DIR, "saved/agung_okada_insar_A/Agung_Okada_InSAR_A/0622_173754")
OUT = os.path.join(CURRENT_DIR, "comparison_out", "agung_okada_SON2017_maps")


def main():
    cfg = json.load(open(os.path.join(RUN, "models", "config.json")))
    ia = cfg["arch"]["args"]["insar"]
    d = load_insar_mintpy(ia["timeseries"], ia["geometry"], ia.get("mask"),
                          ia["lat0"], ia["lon0"], multilook=int(ia.get("multilook", 20)),
                          coh_valid_frac=ia.get("coh_valid_frac", 0.5),
                          bbox=ia.get("bbox"), verbose=False)
    dev = torch.device("cpu")
    model = build_model(cfg, os.path.join(RUN, "models", "model_best.pth"), dev)
    actual, full, phys, _ = run_inference_all_epochs(model, d, cfg, dev)
    resid_pila = full - phys                      # the learned z_aux augmentation

    # SON 2017 epochs
    idxs = [i for i, dt in enumerate(d.dates)
            if dt.startswith("2017-") and dt.split("-")[1] in ("09", "10", "11")]
    dates = [d.dates[i] for i in idxs]
    print(f"SON-2017 epochs: {dates}")

    disp_extent = list(d.extent_lonlat)
    hs, hs_extent = load_hillshade(ia["geometry"])

    # Shared scale: observed+physics share one symmetric scale; residual gets its own.
    vmax_d = float(np.nanpercentile(np.abs(actual[idxs]), 98))
    vmax_r = float(np.nanpercentile(np.abs(resid_pila[idxs]), 98))

    nrow = len(idxs)
    fig, axes = plt.subplots(nrow, 3, figsize=(13, 3.4 * nrow),
                             constrained_layout=True, sharex=True, sharey=True)
    col_titles = ["Observed LOS", "Physics only (x_P)", "PILA residual (x_PB − x_P)"]
    im_d = im_r = None
    for r, i in enumerate(idxs):
        r2p, rmp = _r2_rmse(phys[i], actual[i])
        fields = [(actual[i], vmax_d), (phys[i], vmax_d), (resid_pila[i], vmax_r)]
        for c, (vals, vmax) in enumerate(fields):
            ax = axes[r, c] if nrow > 1 else axes[c]
            ax.imshow(hs, extent=hs_extent, cmap="gray", origin="upper")
            grid = points_to_grid(vals, d.mask_d)
            im = ax.imshow(np.ma.masked_invalid(grid), extent=disp_extent, origin="upper",
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.95)
            ax.set_xlim(disp_extent[0], disp_extent[1]); ax.set_ylim(disp_extent[2], disp_extent[3])
            if c < 2:
                im_d = im
            else:
                im_r = im
            if r == 0:
                ax.set_title(col_titles[c], fontsize=12)
        ad = axes[r, 0] if nrow > 1 else axes[0]
        ad.set_ylabel(f"{dates[r]}\nLat (°)", fontsize=9)
        ap = axes[r, 1] if nrow > 1 else axes[1]
        ap.text(0.02, 0.04, f"phys R²={r2p:.3f}\nRMSE={rmp:.1f} mm", transform=ap.transAxes,
                fontsize=8, va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="white", alpha=0.7, lw=0))
    for c in range(3):
        (axes[nrow - 1, c] if nrow > 1 else axes[c]).set_xlabel("Longitude (°)")

    cb_d = fig.colorbar(im_d, ax=axes[:, :2] if nrow > 1 else axes[:2], shrink=0.6, location="right")
    cb_d.set_label("LOS / physics (mm)  [+ = toward satellite]")
    cb_r = fig.colorbar(im_r, ax=axes[:, 2] if nrow > 1 else axes[2], shrink=0.6, location="right")
    cb_r.set_label("PILA residual (mm)")
    fig.suptitle("Agung — Okada — Sept–Oct–Nov 2017  (cumulative LOS, descending TD32)",
                 fontsize=14, fontweight="bold")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".png", dpi=150)
    fig.savefig(OUT + ".pdf")
    plt.close(fig)
    print(f"Done. Saved {OUT}.png / .pdf")


if __name__ == "__main__":
    main()
