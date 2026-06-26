#!/usr/bin/env python
# Usage:       python -m synthetic.perepoch_snr_diagnostic \
#                  --out synthetic/snr_sweep_ml1 --base-name marapi_mogi_buildup_combined \
#                  --base-spec synthetic/specs/marapi_mogi_buildup_combined.json --multilook 1
# Description: Dense per-epoch breakdown. PILA inverts each epoch independently, so every
#              epoch of every run is one inversion at its OWN instantaneous SNR:
#                SNR(run, k) = 20*log10( signal_rms(k) / noise_rms(k) )
#              with signal_rms(k) from each run's truth json (dV-scaled) and noise_rms(k)
#              the per-epoch RMS of the (fixed) noise cube. Pools all (run x epoch) points
#              into a scatter of per-epoch source-location error vs per-epoch SNR, plus a
#              binned-median curve — many more points than the per-run (peak-epoch) sweep.
# Date:        2026-06-22

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from synthetic.evaluate import run_inference, PHYSICS_ATTRS, LOC_KEYS, los_field_r2
from synthetic.noise import load_marapi_noise, _valid_rms
from synthetic.generate_timeseries import load_full_res_geometry
from synthetic.forward_numpy import run_model
from synthetic.snr_failure_analysis import logistic_fit, detection_limit_lines
from synthetic import plots


def per_epoch_noise_rms(components=('combined',), reference_epoch0=True, scale=1.0, mask=None):
    """Per-epoch RMS (mm) of the fixed noise cube (same for every run)."""
    cube = load_marapi_noise(components)                  # [n_epoch, Ny, Nx], metres
    if reference_epoch0:
        cube = cube - cube[0]
    cube = cube * scale * 1000.0                          # -> mm
    return np.array([_valid_rms(cube[i], mask) for i in range(cube.shape[0])])


def collect(out_dir, base_name, base_spec_path, min_signal_mm=1.0):
    """For every run with a checkpoint, gather per-epoch records: SNR (dB), linear
    signal/noise ratio, noise RMS (mm), peak |LOS| deformation (mm), location error,
    and dV %error. Peak deformation is recomputed on the fly from each truth json's
    params + forward model (no regeneration needed)."""
    noise_rms = per_epoch_noise_rms()                     # [n_epoch] mm, run-independent

    # Geometry to recompute the clean LOS field per epoch -> peak |LOS| deformation.
    with open(base_spec_path) as f:
        base = json.load(f)['base_scene']
    xE, yN, losE, losN, losU, _ = load_full_res_geometry(base['geometry'],
                                                         base['lat0'], base['lon0'])
    xflat, yflat = xE.ravel(), yN.ravel()
    eflat, nflat, uflat = losE.ravel(), losN.ravel(), losU.ravel()

    pts = []   # list of dicts
    # Enumerate tags from THIS out-dir's eval/ so variants (e.g. _ff) stay separated.
    eval_dirs = sorted(glob.glob(os.path.join(out_dir, 'eval', f'{base_name}_*')))
    for ed in eval_dirs:
        tag = os.path.basename(ed)
        cfg_path = os.path.join('configs', 'synth', f'{tag}.json')
        if not os.path.exists(cfg_path):
            continue
        truth_p = os.path.join('synthetic', 'cubes', tag, f'{tag}_truth.json')
        ckpts = glob.glob(os.path.join('saved', 'synth', tag, '*', '*', 'models', 'model_best.pth'))
        if not (os.path.exists(truth_p) and ckpts):
            continue
        ckpt = sorted(ckpts, key=os.path.getmtime)[-1]
        with open(cfg_path) as f:
            config = json.load(f)
        with open(truth_p) as f:
            truth = json.load(f)
        physics = config['arch']['args']['physics']
        attrs = PHYSICS_ATTRS[physics]
        kx, ky = LOC_KEYS[physics]

        sig_rms = np.asarray(truth['signal_rms_mm_per_epoch'])       # per-epoch clean signal RMS
        model_name = truth['forward_model']
        tnames = truth['param_names']; tparams = np.asarray(truth['params_per_epoch'])
        ix, iy = tnames.index(kx), tnames.index(ky)
        idV_true = tnames.index('dV') if 'dV' in tnames else None    # true source dV (m^3)

        _, params_inf, _, _, _, _ = run_inference(config, ckpt)      # [n_epoch, n_param]
        ax = attrs.index(kx); ay = attrs.index(ky)
        adV = attrs.index('dV') if 'dV' in attrs else None
        # Column map attrs-order -> truth/forward-model order, so recovered params can be
        # fed straight into run_model (whose arg order matches truth['param_names']).
        col = [attrs.index(p) for p in tnames]

        n = min(len(sig_rms), params_inf.shape[0], len(noise_rms))
        for k in range(n):
            if sig_rms[k] < min_signal_mm or noise_rms[k] <= 0:
                continue                                             # near-zero signal: SNR undefined
            ratio = sig_rms[k] / noise_rms[k]
            snr = 20.0 * np.log10(ratio)
            # Truth clean |LOS| field on the surface at this epoch (mm) from the forward model.
            uE_mm, uN_mm, uU_mm = run_model(model_name, xflat, yflat, tparams[k])
            sig_los_mm = eflat * uE_mm + nflat * uN_mm + uflat * uU_mm
            peak_def_mm = float(np.max(np.abs(sig_los_mm)))
            # Reconstructed clean LOS field from PILA's RECOVERED params (same grid, no
            # noise) -> R^2 of recovered-vs-truth deformation pattern.
            rE_mm, rN_mm, rU_mm = run_model(model_name, xflat, yflat, params_inf[k, col])
            recon_los_mm = eflat * rE_mm + nflat * rN_mm + uflat * rU_mm
            recon_r2 = los_field_r2(recon_los_mm, sig_los_mm)
            loc_err_m = float(np.sqrt((params_inf[k, ax] - tparams[k, ix]) ** 2
                                      + (params_inf[k, ay] - tparams[k, iy]) ** 2))
            dv_true = float(tparams[k, idV_true]) if idV_true is not None else np.nan
            dv_pct = (abs(params_inf[k, adV] - tparams[k, tnames.index('dV')])
                      / (abs(tparams[k, tnames.index('dV')]) + 1e-30) * 100) if adV is not None else np.nan
            pts.append({'snr_db': snr, 'snr_ratio': ratio, 'noise_rms_mm': noise_rms[k],
                        'peak_def_mm': peak_def_mm, 'loc_err_m': loc_err_m, 'recon_r2': recon_r2,
                        'dV_true_m3': dv_true, 'dV_pcterr': dv_pct, 'run_tag': tag})
    return pts, noise_rms


def main():
    ap = argparse.ArgumentParser(description="Per-epoch SNR breakdown diagnostic (pooled over runs).")
    ap.add_argument('--out', default='synthetic/snr_sweep_ml1')
    ap.add_argument('--base-name', default='marapi_mogi_buildup_combined')
    ap.add_argument('--base-spec', default='synthetic/specs/marapi_mogi_buildup_combined.json')
    ap.add_argument('--multilook', type=int, default=1)
    ap.add_argument('--min-signal-mm', type=float, default=1.0)
    args = ap.parse_args()

    print("[1/2] Collecting per-epoch points across all runs ...")
    pts, _ = collect(args.out, args.base_name, args.base_spec, args.min_signal_mm)
    if not pts:
        print("No points collected (need configs/synth/<tag>.json + checkpoints + truth)."); return
    snr = np.array([p['snr_db'] for p in pts]); ratio = np.array([p['snr_ratio'] for p in pts])
    noise = np.array([p['noise_rms_mm'] for p in pts]); peak = np.array([p['peak_def_mm'] for p in pts])
    loc = np.array([p['loc_err_m'] for p in pts])
    r2 = np.array([p['recon_r2'] for p in pts])                    # recon-vs-truth LOS field R^2
    dv_Mm3 = np.array([p['dV_true_m3'] for p in pts]) * 1e-6        # true source dV -> Mm^3
    print(f"  {len(pts)} per-epoch points from {len(set(p['run_tag'] for p in pts))} runs; "
          f"SNR range [{snr.min():.1f}, {snr.max():.1f}] dB")

    cell_m = (40000.0 / 128) * (args.multilook or 1)
    recovered = loc < cell_m
    base = os.path.join(args.out, f'{args.base_name}_perepoch')

    print("[2/2] Plotting")

    def plot_vs(xvals, xlabel, stem, bin_w):
        """Scatter (loc err vs predictor) + binned median/IQR + recovery-fraction plot."""
        edges = np.arange(np.floor(xvals.min()), np.ceil(xvals.max()) + bin_w, bin_w)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.scatter(xvals[recovered], loc[recovered], s=10, c='C0', alpha=0.4,
                   label='recovered (<1 cell)')
        ax.scatter(xvals[~recovered], loc[~recovered], s=10, c='C3', alpha=0.4, label='failed')
        cx, cy, clo, chi = plots.binned_median(xvals, loc, edges)
        if len(cx):
            ax.plot(cx, cy, 'k-', lw=2, label='median')
            ax.fill_between(cx, clo, chi, color='0.6', alpha=0.3, label='IQR')
        ax.axhline(cell_m, ls=':', color='0.4', label=f'breakdown ({cell_m:.0f} m)')
        ax.set_yscale('log'); ax.set_xlabel(f'per-epoch {xlabel}')
        ax.set_ylabel('per-epoch source-location error (m)')
        ax.set_title(f'{args.base_name}: per-epoch breakdown ({len(pts)} epoch-inversions pooled)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
        fig.tight_layout()
        fig.savefig(f'{base}_{stem}.png', dpi=150); fig.savefig(f'{base}_{stem}.pdf'); plt.close(fig)
        print(f"  wrote {base}_{stem}.png")

        # Recovery fraction vs predictor (the empirical breakdown probability)
        fr_x, fr_y = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (xvals >= a) & (xvals < b)
            if m.sum() >= 3:
                fr_x.append(0.5 * (a + b)); fr_y.append(float(np.mean(recovered[m])))
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2.plot(fr_x, fr_y, 'o-', color='C2')
        ax2.axhline(0.5, ls=':', color='0.5'); ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlabel(f'per-epoch {xlabel}'); ax2.set_ylabel('fraction of epochs recovered')
        ax2.set_title(f'{args.base_name}: recovery probability vs per-epoch {xlabel}')
        ax2.grid(alpha=0.3); fig2.tight_layout()
        fig2.savefig(f'{base}_{stem}_recovery_fraction.png', dpi=150); plt.close(fig2)
        print(f"  wrote {base}_{stem}_recovery_fraction.png")

    # Breakdown vs the two meaningful predictors: dB and linear ratio. (Noise RMS alone
    # is NOT a useful recovery predictor — at a given noise level the outcome still
    # depends on signal strength — so it is intentionally omitted here; noise enters
    # only via the detectability plot below, where signal is the y-axis.)
    plot_vs(snr, 'instantaneous SNR (dB)', 'snr', 0.5)
    plot_vs(ratio, 'signal/noise ratio', 'ratio', max((ratio.max() - ratio.min()) / 30, 1e-6))

    # DEFINITIVE diagnostic #1 — reconstruction r² (recovered vs truth LOS pattern) vs the
    # LINEAR signal/noise ratio (not dB). Continuous accuracy axis (cf. the demoted
    # loc-error / recovery-fraction plots above).
    plots.plot_recon_r2_vs_snr(args.out, f'{args.base_name}_perepoch', ratio, r2, recovered,
                               xlabel='per-epoch signal/noise ratio (linear)')
    print(f"  wrote {base}_recon_r2_vs_snr.png")

    # Detectability: peak surface deformation (y) vs noise RMS (x), with the SNR50 and
    # SNR90 recovery boundaries (peak = k * noise) + bootstrap 95% CI overlaid. sig_rms is
    # recovered as ratio*noise (ratio = sig_rms/noise by construction).
    sig_rms_perepoch = ratio * noise
    det_lines, det_summary = detection_limit_lines(ratio, recovered, peak, sig_rms_perepoch)
    plots.plot_deformation_vs_noise(args.out, f'{args.base_name}_perepoch', noise, peak,
                                    recovered, det_lines=det_lines)
    print(f"  wrote {os.path.join(args.out, args.base_name)}_perepoch_deformation_vs_noise.png "
          f"(SNR50 k={det_summary['k50_peak_per_noise']:.1f}, "
          f"SNR90 k={det_summary['k90_peak_per_noise']:.1f})")
    # Companion logistic recovery-probability curve in deformation/noise terms.
    plots.plot_deformation_recovery_curve(
        args.out, f'{args.base_name}_perepoch', peak, noise, recovered,
        det_summary['k50_peak_per_noise'], det_summary['k90_peak_per_noise'],
        k50_ci=det_summary['k50_ci95'], k90_ci=det_summary['k90_ci95'])
    print(f"  wrote {base}_deformation_recovery_curve.png")

    # Per-epoch signal/noise ratio vs true source dV: ties physical source volume to the
    # SNR it produces (and whether PILA recovered it). LINEAR ratio on x (to match the r²
    # plot), dV (Mm^3, log) on y; ratio50 marks the 50%-recovery threshold. Only
    # meaningful when the model has a dV (Mogi/Sun69).
    good_dv = np.isfinite(dv_Mm3) & (dv_Mm3 > 0)
    if good_dv.any():
        ratio50, _ = logistic_fit(ratio, recovered.astype(int))   # fit in linear ratio units
        figd, axd = plt.subplots(figsize=(9, 5.5))
        g_rec = good_dv & recovered; g_fail = good_dv & ~recovered
        axd.scatter(ratio[g_rec], dv_Mm3[g_rec], s=10, c='C0', alpha=0.4, label='recovered (<1 cell)')
        axd.scatter(ratio[g_fail], dv_Mm3[g_fail], s=10, c='C3', alpha=0.4, label='failed')
        axd.axvline(ratio50, ls='--', color='0.3', label=f'ratio50 = {ratio50:.2f}')
        axd.set_yscale('log'); axd.set_xlabel('per-epoch signal/noise ratio (linear)')
        axd.set_ylabel('true source volume change dV (Mm³)')
        axd.set_title(f'{plots._pretty(args.base_name)}: signal/noise ratio vs source dV')
        axd.legend(); axd.grid(alpha=0.3, which='both')
        figd.tight_layout()
        figd.savefig(f'{base}_snr_vs_dV.png', dpi=150); figd.savefig(f'{base}_snr_vs_dV.pdf')
        plt.close(figd)
        print(f"  wrote {base}_snr_vs_dV.png")

    # CSV of all points
    import csv
    cols = ['snr_db', 'snr_ratio', 'noise_rms_mm', 'peak_def_mm', 'loc_err_m', 'recon_r2',
            'dV_true_m3', 'dV_pcterr', 'run_tag']
    with open(base + '_snr.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(pts)
    print(f"  wrote {base}_snr.csv")


if __name__ == '__main__':
    main()
