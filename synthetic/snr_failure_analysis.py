#!/usr/bin/env python
# Usage:       python -m synthetic.snr_failure_analysis \
#                  --base-spec synthetic/specs/marapi_mogi_buildup_combined.json \
#                  --base-name marapi_mogi_buildup_combined --out synthetic/snr_sweep_ml1 --multilook 1
# Description: Per-epoch PILA failure-mode analysis. For every epoch of every run computes
#              TWO signal-to-noise definitions and whether PILA recovered the source:
#                * scene SNR  = 20log10( rms(signal) / rms(noise) )         [whole scene]
#                * local SNR  = 20log10( rms(signal) / wrms(noise; w=signal^2) )
#                               i.e. noise weighted into the deformation footprint
#              Then (1) fits a logistic recovery-probability curve to each (50%-recovery SNR
#              + bootstrap CI + which predictor is better via AUC), and (2) converts the
#              scene 50% threshold into a physical minimum-detectable dV / LOS, with a formula
#              usable for any real scene's noise level.
# Why local SNR: the deformation is compact; recovery should track the noise CO-LOCATED with
#              the signal, not the scene average. If the source sits in a locally-quiet patch
#              scene SNR is pessimistic; if under a noisy patch it is optimistic.
# Date:        2026-06-22

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from synthetic.generate_timeseries import load_full_res_geometry
from synthetic.forward_numpy import run_model, FORWARD_MODELS
from synthetic.noise import load_marapi_noise, _valid_rms
from synthetic.evaluate import run_inference, PHYSICS_ATTRS, LOC_KEYS
from synthetic import plots


def _wrms(field, weights):
    """Weighted RMS of `field` with non-negative weights."""
    w = weights / (weights.sum() + 1e-30)
    return float(np.sqrt(np.sum(w * field ** 2)))


def logistic_fit(snr, recovered):
    """MLE logistic p(recover)=sigmoid((snr-mu)/s). Returns mu, s, and per-point p."""
    x = np.asarray(snr, float); y = np.asarray(recovered, float)

    def nll(theta):
        mu, logs = theta; s = np.exp(logs)
        z = (x - mu) / s
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize(nll, x0=[np.median(x), np.log(2.0)], method='Nelder-Mead')
    mu, s = res.x[0], np.exp(res.x[1])
    return mu, s


def boot_ci(snr, recovered, n_boot=500):
    """Bootstrap 95% CI for the 50%-recovery SNR (mu)."""
    snr = np.asarray(snr); recovered = np.asarray(recovered)
    n = len(snr); mus = []
    rng = np.random.RandomState(0)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            mu, _ = logistic_fit(snr[idx], recovered[idx]); mus.append(mu)
        except Exception:
            pass
    return (np.percentile(mus, 2.5), np.percentile(mus, 97.5)) if mus else (np.nan, np.nan)


def detection_limit_lines(ratio, recovered, peak_def_mm, sig_rms_mm, n_boot=500, seed=0):
    """Detection-limit slopes (deformation = k * noise_rms) at 50% and 90% recovery.

    The logistic recovery model gives the signal/noise RATIO at which PILA recovers the
    source with probability P: ratio50 = mu, ratio90 = mu + s*ln(9) (s>0). Peak |LOS|
    deformation is ~linear in source amplitude, so peak ~ (peak/sig_rms) * sig_rms, and
    sig_rms = ratioP * noise_rms by definition of the ratio. The P% detection boundary on
    a deformation-vs-noise plot is therefore the straight line through the origin
        peak_def = k_P * noise_rms ,   k_P = median(peak/sig_rms) * ratioP .
    Bootstrap (resample epochs, refit logistic, recompute median peak/sig_rms) gives the
    95% CI on each slope.

    Parameters
    ----------
    ratio        : per-epoch linear signal/noise ratio (sig_rms/noise_rms).
    recovered    : per-epoch 0/1 (loc-error < 1 multilook cell).
    peak_def_mm  : per-epoch peak |LOS| deformation (mm).
    sig_rms_mm   : per-epoch signal RMS (mm).

    Returns
    -------
    det_lines : list of dicts {label, k, k_lo, k_hi, color} ready for
                plots.plot_deformation_vs_noise(..., det_lines=...).
    summary   : dict of the scalar slopes/ratios + CIs (for the metrics JSON).
    """
    ratio = np.asarray(ratio, float); rec = np.asarray(recovered).astype(int)
    peak = np.asarray(peak_def_mm, float); sig = np.asarray(sig_rms_mm, float)
    ln9 = np.log(9.0)
    pk_over_rms_per = peak / np.where(sig > 0, sig, np.nan)     # per-epoch peak/sig_rms

    def slopes(mu, s, pk):
        return pk * mu, pk * (mu + s * ln9)                    # (k50, k90)

    mu, s = logistic_fit(ratio, rec)
    pk = float(np.nanmedian(pk_over_rms_per))
    k50, k90 = slopes(mu, s, pk)

    rng = np.random.RandomState(seed); n = len(ratio); b50, b90 = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            m, ss = logistic_fit(ratio[idx], rec[idx])
        except Exception:
            continue
        a, b = slopes(m, ss, float(np.nanmedian(pk_over_rms_per[idx])))
        b50.append(a); b90.append(b)

    def ci(v):
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)
    c50, c90 = ci(b50), ci(b90)
    ratio50, ratio90 = float(mu), float(mu + s * ln9)
    det_lines = [
        {'label': f'SNR50 (50% recovery, ratio={ratio50:.2f}): peak={k50:.1f}×noise',
         'k': k50, 'k_lo': c50[0], 'k_hi': c50[1], 'color': 'k'},
        {'label': f'SNR90 (90% recovery, ratio={ratio90:.2f}): peak={k90:.1f}×noise',
         'k': k90, 'k_lo': c90[0], 'k_hi': c90[1], 'color': 'C2'},
    ]
    summary = {'ratio50': ratio50, 'ratio90': ratio90,
               'k50_peak_per_noise': k50, 'k50_ci95': list(c50),
               'k90_peak_per_noise': k90, 'k90_ci95': list(c90),
               'peak_over_sig_rms': pk}
    return det_lines, summary


def auc(score, label):
    """Area under ROC (probability a recovered epoch has higher SNR than a failed one)."""
    score = np.asarray(score); label = np.asarray(label).astype(bool)
    pos = score[label]; neg = score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(score); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(score) + 1)
    return (ranks[label].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def collect(base_spec_path, base_name, out_dir, min_signal_mm=1.0):
    with open(base_spec_path) as f:
        base = json.load(f)['base_scene']
    xE, yN, losE, losN, losU, (L, W) = load_full_res_geometry(base['geometry'], base['lat0'], base['lon0'])
    xflat, yflat = xE.ravel(), yN.ravel()
    eflat, nflat, uflat = losE.ravel(), losN.ravel(), losU.ravel()

    # Fixed noise cube (mm), already epoch0-referenced + north-up in the loader.
    noise_cube = load_marapi_noise(['combined'])
    noise_cube = (noise_cube - noise_cube[0]) * 1000.0          # ref epoch0, -> mm

    rows = []
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
        config = json.load(open(cfg_path)); truth = json.load(open(truth_p))
        model_name = truth['forward_model']
        attrs = PHYSICS_ATTRS[config['arch']['args']['physics']]
        kx, ky = LOC_KEYS[config['arch']['args']['physics']]
        tnames = truth['param_names']; tparams = np.asarray(truth['params_per_epoch'])
        ix, iy = tnames.index(kx), tnames.index(ky)
        idV = tnames.index('dV') if 'dV' in tnames else None

        _, params_inf, _, _, _, _ = run_inference(config, ckpt)
        ax, ay = attrs.index(kx), attrs.index(ky)

        n = min(len(tparams), params_inf.shape[0], noise_cube.shape[0])
        for k in range(n):
            uE, uN, uU = run_model(model_name, xflat, yflat, tparams[k])
            sig = (eflat * uE + nflat * uN + uflat * uU)         # truth clean LOS mm, [L*W]
            sig_rms = np.sqrt(np.mean(sig ** 2))
            if sig_rms < min_signal_mm:
                continue
            noise = noise_cube[k].ravel()
            noise_rms_mm = float(np.sqrt(np.mean(noise ** 2)))
            peak_def_mm = float(np.max(np.abs(sig)))            # peak |LOS| on the surface
            scene_ratio = sig_rms / (noise_rms_mm + 1e-30)      # linear signal/noise ratio
            scene_snr = 20 * np.log10(scene_ratio + 1e-30)
            wnoise = _wrms(noise, sig ** 2)                      # noise in the signal footprint
            local_snr = 20 * np.log10(sig_rms / (wnoise + 1e-30))
            loc_err = float(np.sqrt((params_inf[k, ax] - tparams[k, ix]) ** 2
                                    + (params_inf[k, ay] - tparams[k, iy]) ** 2))
            dV_true = tparams[k, idV] if idV is not None else np.nan
            rows.append({'tag': tag, 'epoch': k, 'scene_snr': scene_snr, 'local_snr': local_snr,
                         'scene_ratio': scene_ratio, 'noise_rms_mm': noise_rms_mm,
                         'loc_err_m': loc_err, 'signal_rms_mm': sig_rms,
                         'peak_def_mm': peak_def_mm, 'dV_true_m3': dV_true})
    return rows, base


def main():
    ap = argparse.ArgumentParser(description="PILA SNR failure-mode analysis (scene vs local SNR).")
    ap.add_argument('--base-spec', default='synthetic/specs/marapi_mogi_buildup_combined.json')
    ap.add_argument('--base-name', default='marapi_mogi_buildup_combined')
    ap.add_argument('--out', default='synthetic/snr_sweep_ml1')
    ap.add_argument('--multilook', type=int, default=1)
    args = ap.parse_args()

    print("[1/4] Collecting per-epoch scene & local SNR + recovery ...")
    rows, base = collect(args.base_spec, args.base_name, args.out)
    cell_m = (40000.0 / 128) * (args.multilook or 1)
    scene = np.array([r['scene_snr'] for r in rows])
    local = np.array([r['local_snr'] for r in rows])
    ratio = np.array([r['scene_ratio'] for r in rows])      # linear signal/noise ratio
    noise = np.array([r['noise_rms_mm'] for r in rows])      # absolute noise level (mm)
    peak = np.array([r['peak_def_mm'] for r in rows])        # peak |LOS| deformation (mm)
    loc = np.array([r['loc_err_m'] for r in rows])
    rec = (loc < cell_m).astype(int)
    print(f"  {len(rows)} epoch-inversions; recovered {rec.sum()} ({100*rec.mean():.0f}%)")

    print("[2/4] Logistic fits (#1)")
    out = {}
    # dB and linear-ratio are monotonic transforms -> identical AUC; the ratio fit is
    # for readability. (scene/local are increasing predictors; all fit cleanly.)
    for name, x, key in [('scene', scene, 'db'), ('local', local, 'db'),
                         ('scene_ratio', ratio, 'ratio')]:
        mu, s = logistic_fit(x, rec)
        lo, hi = boot_ci(x, rec)
        a = auc(x, rec)
        out[name] = {f'snr50_{key}': mu, f'width_s_{key}': s, f'ci95_{key}': [lo, hi], 'auc': a}
        print(f"  {name:11s} SNR50 = {mu:7.3g} {key:5s} (95% CI [{lo:.3g},{hi:.3g}])  "
              f"width s = {s:.3g}  AUC = {a:.3f}")
    better = 'local' if out['local']['auc'] > out['scene']['auc'] else 'scene'
    print(f"  -> {better.upper()} SNR is the better predictor of recovery (higher AUC).")

    print("[3/4] Physical detectability (#2)")
    # Calibrate dV <-> signal_rms (linear through origin) from the data.
    sig_rms = np.array([r['signal_rms_mm'] for r in rows]); dV = np.array([r['dV_true_m3'] for r in rows])
    good = np.isfinite(dV) & (sig_rms > 0)
    k_dV_per_mm = float(np.sum(dV[good] * sig_rms[good]) / np.sum(sig_rms[good] ** 2))  # dV per mm signal_rms
    noise_rms_scene = float(np.median([10 ** (-r['scene_snr'] / 20) * r['signal_rms_mm'] for r in rows]))
    mu_scene = out['scene']['snr50_db']
    sig_rms_50 = noise_rms_scene * 10 ** (mu_scene / 20)
    dV_50 = k_dV_per_mm * sig_rms_50
    # Peak |LOS| is also linear in source amplitude, so peak/signal_rms is ~constant;
    # express the SNR50 threshold as a minimum-detectable PEAK deformation and as the
    # slope of the detectability line peak = k_det * noise_rms (used in the plot below).
    pk_over_rms = float(np.median(peak / np.where(sig_rms > 0, sig_rms, np.nan)))
    peak_def_50 = pk_over_rms * sig_rms_50
    k_det = peak_def_50 / noise_rms_scene if noise_rms_scene > 0 else float('nan')
    out['detectability'] = {
        'noise_rms_scene_mm': noise_rms_scene, 'k_dV_per_mm_signal_rms': k_dV_per_mm,
        'snr50_db': mu_scene, 'min_detectable_signal_rms_mm': sig_rms_50,
        'min_detectable_dV_m3': dV_50,
        'peak_over_rms': pk_over_rms, 'min_detectable_peak_def_mm': peak_def_50,
        'k_det_peak_per_noise': k_det,
        'formula': 'min_dV(m3) = k_dV_per_mm * noise_rms_mm * 10^(SNR50/20); '
                   'min_peak_def(mm) = k_det * noise_rms_mm'}
    print(f"  scene noise RMS ~ {noise_rms_scene:.1f} mm; dV per mm signal_rms = {k_dV_per_mm:.3e} m3/mm")
    print(f"  50% detectable: signal_rms >= {sig_rms_50:.1f} mm  ->  dV >= {dV_50:.3e} m3")
    print(f"  50% detectable: peak |LOS| >= {peak_def_50:.1f} mm  (k_det = {k_det:.2f} x noise RMS)")
    print(f"  GENERAL: min_dV = {k_dV_per_mm:.3e} * noise_rms_mm * 10^({mu_scene:.2f}/20)  m3")

    print("[4/4] Plots")
    edges = np.arange(np.floor(min(scene.min(), local.min())), np.ceil(max(scene.max(), local.max())) + 1, 1.0)

    def frac(x):
        fx, fy = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (x >= a) & (x < b)
            if m.sum() >= 3:
                fx.append(0.5 * (a + b)); fy.append(rec[m].mean())
        return np.array(fx), np.array(fy)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    grid = np.linspace(edges[0], edges[-1], 200)
    for name, x, c in [('scene', scene, 'C0'), ('local', local, 'C3')]:
        fx, fy = frac(x)
        ax.plot(fx, fy, 'o', color=c, alpha=0.6, label=f'{name} SNR (AUC {out[name]["auc"]:.2f})')
        mu, s = out[name]['snr50_db'], out[name]['width_s_db']
        ax.plot(grid, 1 / (1 + np.exp(-(grid - mu) / s)), '-', color=c,
                label=f'{name} logistic, SNR50={mu:.1f} dB')
        ax.axvline(mu, ls=':', color=c, alpha=0.6)
    ax.axhline(0.5, ls=':', color='0.6')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('fraction of epochs recovered')
    ax.set_title(f'{args.base_name}: recovery probability — scene vs local SNR')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(args.out, f'{args.base_name}_scene_vs_local_snr.png')
    fig.savefig(p, dpi=150); fig.savefig(p.replace('.png', '.pdf')); plt.close(fig)
    print(f"  wrote {p}")

    # Same scene recovery curve in the linear signal/noise ratio (request: drop 20log10).
    redges = np.linspace(ratio.min(), ratio.max(), 25)
    rfx, rfy = [], []
    for a, b in zip(redges[:-1], redges[1:]):
        m = (ratio >= a) & (ratio < b)
        if m.sum() >= 3:
            rfx.append(0.5 * (a + b)); rfy.append(rec[m].mean())
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(rfx, rfy, 'o', color='C0', alpha=0.7,
            label=f'scene ratio (AUC {out["scene_ratio"]["auc"]:.2f})')
    mu_r, s_r = out['scene_ratio']['snr50_ratio'], out['scene_ratio']['width_s_ratio']
    rgrid = np.linspace(ratio.min(), ratio.max(), 200)
    ax.plot(rgrid, 1 / (1 + np.exp(-(rgrid - mu_r) / s_r)), '-', color='C0',
            label=f'logistic, ratio50={mu_r:.2f}')
    ax.axvline(mu_r, ls=':', color='C0', alpha=0.6); ax.axhline(0.5, ls=':', color='0.6')
    ax.set_xlabel('signal/noise ratio (linear)'); ax.set_ylabel('fraction of epochs recovered')
    ax.set_title(f'{args.base_name}: recovery probability vs signal/noise ratio')
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    pr = os.path.join(args.out, f'{args.base_name}_recovery_vs_ratio.png')
    fig.savefig(pr, dpi=150); fig.savefig(pr.replace('.png', '.pdf')); plt.close(fig)
    print(f"  wrote {pr}")

    # Detectability: peak surface deformation (y) vs noise RMS (x), with the SNR50
    # boundary peak = k_det * noise overlaid (read off detectable deformation at a noise
    # level). RMS deformation drawn faintly for the peak-vs-RMS cross-check.
    # SNR50 + SNR90 detection slopes (peak = k*noise) with bootstrap 95% CI -> recorded in
    # the metrics JSON. The deformation-vs-noise and deformation-recovery-curve FIGURES are
    # produced by perepoch_snr_diagnostic.py (identical points), so not duplicated here.
    _, det_summary = detection_limit_lines(ratio, rec, peak, sig_rms)
    out['detectability'].update(det_summary)
    print(f"  detectability: SNR50 k={det_summary['k50_peak_per_noise']:.1f} "
          f"CI{det_summary['k50_ci95']}, SNR90 k={det_summary['k90_peak_per_noise']:.1f} "
          f"CI{det_summary['k90_ci95']} (figures via perepoch_snr_diagnostic.py)")
    # NOTE: the reconstruction-r² vs SNR figure is produced by perepoch_snr_diagnostic.py
    # (identical per-epoch pooled points + SNR), so it is intentionally not duplicated here.

    with open(os.path.join(args.out, f'{args.base_name}_failure_analysis.json'), 'w') as f:
        json.dump(out, f, indent=2)
    import csv
    with open(os.path.join(args.out, f'{args.base_name}_scene_vs_local_snr.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  wrote failure_analysis.json + scene_vs_local_snr.csv")


if __name__ == '__main__':
    main()
