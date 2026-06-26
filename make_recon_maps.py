#!/usr/bin/env python
# Usage:       python make_recon_maps.py [--config Etna_Mogi_TA44_A ...]
# Description: For each benchmark config, plot the observed LOS field and the physics-only
#              reconstructions from MCMC (Bayesian posterior median) and PILA, gridded to the
#              coarse coherence raster and overlaid on a DEM hillshade (house style, matching
#              plot_insar_results.py), on a shared diverging colour scale. Same epoch, same
#              points for both inversions. One 3-panel figure per config.
# Date:        2026-06-22

import os
import json
import glob
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless HPC node
import matplotlib.pyplot as plt
from scipy.io import loadmat

import compare_pila_vs_classical as C            # forward model + param parsing
from plot_insar_results import points_to_grid, load_hillshade
from datasets.preprocessing.insar_mintpy import load_insar_mintpy


def r2(pred_mm, obs_mm):
    ss_res = np.sum((obs_mm - pred_mm) ** 2)
    ss_tot = np.sum((obs_mm - obs_mm.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=None, help="subset of config names (default: all)")
    ap.add_argument("--out", default=os.path.join(C.CURRENT_DIR, "comparison_out", "recon_maps"))
    ap.add_argument("--hs_azimuth", type=float, default=315.0)
    ap.add_argument("--hs_altitude", type=float, default=45.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    exported = sorted(glob.glob(os.path.join(C.CURRENT_DIR, "exported", "*.mat")))
    saved_root = os.path.join(C.CURRENT_DIR, "saved")
    classic_dir = os.path.join(C.CURRENT_DIR, "classic_out")

    written = []
    for i, exp in enumerate(exported, 1):
        cfg = os.path.splitext(os.path.basename(exp))[0]
        if args.config and cfg not in args.config:
            continue
        E = loadmat(exp)
        model = str(E["model"][0])
        epoch = str(E["epoch_iso"][0]) if "epoch_iso" in E else str(E["epoch_yyyymmdd"][0])

        # --- locate the PILA run used by the benchmark (latest params.txt) + its config.json ---
        ptxt = C.find_pila_params_txt(cfg, saved_root)
        if not ptxt:
            print(f"[{i}] {cfg}: no PILA params.txt — skipping"); continue
        cfg_json = os.path.join(os.path.dirname(os.path.dirname(ptxt)), "models", "config.json")
        with open(cfg_json) as fh:
            ia = json.load(fh)["arch"]["args"]["insar"]
        lat0, lon0 = float(ia["lat0"]), float(ia["lon0"])

        # --- load the multilooked LOS field (same points the model used) ---
        d = load_insar_mintpy(ia["timeseries"], ia["geometry"], ia.get("mask"),
                              ia["lat0"], ia["lon0"], multilook=int(ia.get("multilook", 20)),
                              coh_valid_frac=ia.get("coh_valid_frac", 0.5),
                              bbox=ia.get("bbox"), verbose=False)
        print(f"[{i}] {cfg}  model={model}  epoch={epoch}  N={d.n_points}  grid={d.mask_d.shape}")

        # Observed = final epoch (the benchmark epoch); point coords in metres for the forward model.
        actual_mm = d.los_points_mm[-1].astype(np.float64)
        x_m, y_m = d.xE_pts * 1000.0, d.yN_pts * 1000.0
        lE, lN, lU = d.losE_pts, d.losN_pts, d.losU_pts

        # --- MCMC (Bayesian post_med) and PILA physics-only reconstructions on the SAME points ---
        Cc = C.load_classical(classic_dir, cfg, model)
        mcmc_mm = (C.evaluate_los_mm(model, C.vec_to_si(model, Cc["post_med"], Cc["param_names"]),
                                     x_m, y_m, lE, lN, lU) if Cc is not None else None)
        p_pila, _, _ = C.parse_pila_params(ptxt, model, lat0, lon0)
        pila_mm = (C.evaluate_los_mm(model, p_pila, x_m, y_m, lE, lN, lU)
                   if p_pila is not None else None)

        # --- grids + hillshade + shared symmetric scale ---
        disp_extent = list(d.extent_lonlat)
        hs, hs_extent = load_hillshade(ia["geometry"], args.hs_azimuth, args.hs_altitude)
        vmax = float(np.nanpercentile(np.abs(actual_mm), 98))

        panels = [("Observed LOS", actual_mm)]
        if mcmc_mm is not None:
            panels.append((f"MCMC reconstruction\n(R²={r2(mcmc_mm, actual_mm):.3f})", mcmc_mm))
        if pila_mm is not None:
            panels.append((f"PILA reconstruction\n(R²={r2(pila_mm, actual_mm):.3f})", pila_mm))

        fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.8),
                                 constrained_layout=True, sharex=True, sharey=True)
        if len(panels) == 1:
            axes = [axes]
        im = None
        for ax, (title, vals) in zip(axes, panels):
            ax.imshow(hs, extent=hs_extent, cmap="gray", origin="upper")
            grid = points_to_grid(vals, d.mask_d)
            im = ax.imshow(np.ma.masked_invalid(grid), extent=disp_extent, origin="upper",
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.95)
            ax.set_xlim(disp_extent[0], disp_extent[1])
            ax.set_ylim(disp_extent[2], disp_extent[3])
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Longitude (°)")
        axes[0].set_ylabel("Latitude (°)")
        cb = fig.colorbar(im, ax=axes, shrink=0.85, location="right")
        cb.set_label("LOS displacement (mm)  [+ = toward satellite]")
        fig.suptitle(f"{cfg}  —  {model.capitalize()}  —  epoch {epoch}", fontsize=13, fontweight="bold")

        out = os.path.join(args.out, f"{cfg}_recon_maps.png")
        fig.savefig(out, dpi=150)
        fig.savefig(out.replace(".png", ".pdf"))
        plt.close(fig)
        written.append(out)

    print(f"\nDone. Wrote {len(written)} figures to {args.out}/")


if __name__ == "__main__":
    main()
