#!/usr/bin/env python
# Usage:       python -m synthetic.make_presentation_figures \
#                  --out synthetic/snr_sweep_ml1 --base-name marapi_mogi_buildup_combined \
#                  --multilook 1
# Description: Rebuilds the two SNR-sweep figures in presentation-ready form, straight from
#              the CSVs already on disk (no inference re-run):
#                (1) per-epoch source-location error vs per-epoch SNR scatter, cropped to the
#                    informative SNR band with large fonts (the "breakdown is real & bimodal"
#                    slide); from <base>_perepoch_snr.csv
#                (2) clean logistic recovery-probability curve with the fitted SNR50 marker and
#                    its 95% bootstrap CI band, replacing the noisy raw recovery-fraction plot
#                    (the "operating envelope" slide); from <base>_scene_vs_local_snr.csv
#              recovered = source localized within 1 grid pixel (40 km / 128 * multilook).
# Date:        2026-06-23

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from synthetic.snr_failure_analysis import logistic_fit, boot_ci, auc


# --- Presentation-grade defaults (large, legible from the back of a room) ---
plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 17,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'lines.linewidth': 2.2,
    'axes.linewidth': 1.1,
    'figure.dpi': 150,
})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def save(fig, base):
    """Save both PNG (dpi 150) and PDF, per plotting standards."""
    fig.savefig(base + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(base + '.pdf', bbox_inches='tight')
    print(f"  wrote {base}.png / .pdf")


def binned_curve(snr, val, edges, reducer, min_count=5):
    """Bin snr and apply reducer(val_in_bin); skip sparse bins to kill low-count artifacts."""
    cx, cy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (snr >= a) & (snr < b)
        if m.sum() >= min_count:
            cx.append(0.5 * (a + b))
            cy.append(reducer(val[m]))
    return np.array(cx), np.array(cy)


# ----------------------------------------------------------------------------
def figure_scatter(out_dir, base_name, cell_m):
    """(1) Per-epoch location error vs SNR scatter, cropped + enlarged."""
    csv_path = os.path.join(out_dir, f'{base_name}_perepoch_snr.csv')
    rows = read_csv(csv_path)
    snr = np.array([float(r['snr_db']) for r in rows])
    loc = np.array([float(r['loc_err_m']) for r in rows])
    recovered = loc < cell_m

    # Crop to the informative band: a little below the breakdown up to the top of the data.
    x_lo, x_hi = -10.0, np.ceil(snr.max()) + 1
    edges = np.arange(np.floor(snr.min()), np.ceil(snr.max()) + 0.5, 1.0)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.scatter(snr[recovered], loc[recovered], s=22, c='#1f6fb4', alpha=0.45,
               edgecolors='none', label=f'recovered (< 1 pixel, {cell_m:.0f} m)')
    ax.scatter(snr[~recovered], loc[~recovered], s=22, c='#c0392b', alpha=0.45,
               edgecolors='none', label='failed')

    # Median + IQR location error vs SNR (robust trend through the cloud).
    mx, my = binned_curve(snr, loc, edges, np.median)
    _, mlo = binned_curve(snr, loc, edges, lambda v: np.percentile(v, 25))
    _, mhi = binned_curve(snr, loc, edges, lambda v: np.percentile(v, 75))
    if len(mx):
        ax.fill_between(mx, mlo, mhi, color='0.5', alpha=0.25, label='IQR')
        ax.plot(mx, my, 'k-', lw=2.6, label='median (1 dB bins)')

    ax.axhline(cell_m, ls='--', color='0.3', lw=1.8,
               label=f'breakdown threshold ({cell_m:.0f} m = 1 pixel)')
    ax.set_yscale('log')
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel('Per-epoch instantaneous SNR (dB)')
    ax.set_ylabel('Source-location error (m)')
    ax.set_title('PILA Mogi recovery: resolved vs unresolved source')
    ax.grid(alpha=0.3, which='both')
    ax.legend(loc='upper right', framealpha=0.9)
    # Honesty note: ~21% of epochs sit in the 1-10 km transition; the log y-axis + binary
    # recovered/failed coloring make the gap look emptier than it is (not strictly bimodal).
    frac_mid = float(((loc >= 1000) & (loc < 10000)).mean()) * 100
    fig.text(0.5, -0.02,
             f'Note: not strictly bimodal — {frac_mid:.0f}% of epochs fall in the 1-10 km '
             'transition; the log axis exaggerates the gap.',
             ha='center', va='top', fontsize=11, style='italic', color='0.3')
    fig.tight_layout()
    save(fig, os.path.join(out_dir, f'{base_name}_perepoch_snr_pres'))
    plt.close(fig)
    n_shown = int(((snr >= x_lo) & (snr <= x_hi)).sum())
    print(f"    {len(rows)} epochs total; {n_shown} in cropped [{x_lo:.0f}, {x_hi:.0f}] dB window")


# ----------------------------------------------------------------------------
def figure_logistic(out_dir, base_name, cell_m):
    """(2) Clean logistic recovery-probability curve with SNR50 marker + 95% CI band."""
    csv_path = os.path.join(out_dir, f'{base_name}_scene_vs_local_snr.csv')
    rows = read_csv(csv_path)
    scene = np.array([float(r['scene_snr']) for r in rows])
    local = np.array([float(r['local_snr']) for r in rows])
    loc = np.array([float(r['loc_err_m']) for r in rows])
    rec = (loc < cell_m).astype(int)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    xgrid = np.linspace(min(scene.min(), local.min()) - 1,
                        max(scene.max(), local.max()) + 1, 400)

    summary = {}
    # Scene SNR is the primary predictor (better AUC); plot it bold with CI. Local as context.
    styles = {
        'scene': dict(color='#1f6fb4', label='scene-wide SNR', primary=True),
        'local': dict(color='#27ae60', label='local SNR', primary=False),
    }
    for name, x in [('scene', scene), ('local', local)]:
        st = styles[name]
        mu, s = logistic_fit(x, rec)
        lo, hi = boot_ci(x, rec)
        a = auc(x, rec)
        summary[name] = (mu, s, lo, hi, a)
        p = 1.0 / (1.0 + np.exp(-(xgrid - mu) / s))
        lw = 3.2 if st['primary'] else 2.2
        ls = '-' if st['primary'] else '--'
        ax.plot(xgrid, p, ls, color=st['color'], lw=lw,
                label=f"{st['label']}: SNR50={mu:.1f} dB (AUC {a:.2f})")
        # Binned empirical recovery fraction (markers only, sparse bins dropped).
        edges = np.arange(np.floor(x.min()), np.ceil(x.max()) + 2, 2.0)
        bx, by = binned_curve(x, rec.astype(float), edges, np.mean, min_count=8)
        ax.plot(bx, by, 'o', color=st['color'], ms=7, alpha=0.55, mec='none')
        if st['primary']:
            # SNR50 marker + 95% bootstrap CI band for the primary (scene) predictor.
            ax.axvspan(lo, hi, color=st['color'], alpha=0.12,
                       label=f'95% CI [{lo:.1f}, {hi:.1f}] dB')
            ax.axvline(mu, color=st['color'], ls=':', lw=1.8)
            ax.plot([mu], [0.5], marker='*', ms=18, color=st['color'],
                    mec='k', mew=0.8, zorder=5)
            # 90%-recovery SNR: x90 = mu + s*ln(0.9/0.1) — the conservative "reliable" threshold.
            x90 = mu + s * np.log(9.0)
            ax.axvline(x90, color='#c0392b', ls=':', lw=1.8)
            ax.plot([x90], [0.9], marker='D', ms=11, color='#c0392b',
                    mec='k', mew=0.8, zorder=5)

    mu_s, s_s = summary['scene'][0], summary['scene'][1]
    x90 = mu_s + s_s * np.log(9.0)
    ax.axhline(0.5, ls=':', color='0.5', lw=1.4)
    ax.axhline(0.9, ls=':', color='0.5', lw=1.0)

    # --- In-figure explanation (so the slide is self-contained) ---
    # SNR50 callout (the curve midpoint / characteristic threshold).
    ax.annotate(f'SNR50 = {mu_s:.1f} dB\n50% recovery\n(characteristic threshold)',
                xy=(mu_s, 0.5), xytext=(mu_s - 27, 0.58),
                fontsize=12, color='#1f6fb4', va='center',
                arrowprops=dict(arrowstyle='->', color='#1f6fb4', lw=1.6))
    # 90% callout (the conservative / reliable threshold).
    ax.annotate(f'90% recovery = {x90:.1f} dB\n(reliable detection)',
                xy=(x90, 0.9), xytext=(x90 + 3, 0.66),
                fontsize=12, color='#c0392b', va='center',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.6))
    # Regime labels along the bottom: unresolved -> transition -> resolved.
    ax.text(mu_s - 20, 0.04, 'source\nUNRESOLVED', ha='center', va='bottom',
            fontsize=11, color='0.35')
    ax.text(x90 + 9, 0.04, 'source reliably\nRESOLVED', ha='center', va='bottom',
            fontsize=11, color='0.35')

    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(xgrid.min(), xgrid.max())
    ax.set_xlabel('Per-epoch SNR (dB)')
    ax.set_ylabel('Probability of source recovery')
    ax.set_title('PILA Mogi operating envelope: recovery probability vs SNR')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, f'{base_name}_recovery_logistic_pres'))
    plt.close(fig)
    mu, s, lo, hi, a = summary['scene']
    print(f"    scene SNR50 = {mu:.2f} dB (95% CI [{lo:.2f}, {hi:.2f}]), width {s:.2f} dB, "
          f"AUC {a:.3f}; n={len(rows)} epochs")


def figure_loc_vs_dv(out_dir, base_name, cell_m, snr50=3.71):
    """(4) Location vs dV accuracy vs SNR: median + IQR for BOTH.

    Honest framing (NOT 'step vs slide' — both medians degrade continuously). The real
    distinction is the SPREAD: below the detection threshold both metrics are unreliable, but
    LOCATION is non-identifiable -> bimodal, domain-scale IQR (resolved or railed), whereas dV
    stays unimodal with a tight IQR. Both fail together below SNR50.
    """
    rows = read_csv(os.path.join(out_dir, f'{base_name}_perepoch_snr.csv'))
    snr = np.array([float(r['snr_db']) for r in rows])
    loc = np.array([float(r['loc_err_m']) for r in rows])
    dv = np.array([float(r['dV_pcterr']) for r in rows])
    x_lo, x_hi = -10.0, np.ceil(snr.max()) + 1
    edges = np.arange(np.floor(snr.min()), np.ceil(snr.max()) + 0.5, 1.0)

    fig, (axL, axD) = plt.subplots(2, 1, figsize=(10, 8.4), sharex=True)

    def panel(ax, val, color, ylabel, median_label, spread_note):
        ax.scatter(snr, val, s=12, c='0.7', alpha=0.30, edgecolors='none')
        mx, my = binned_curve(snr, val, edges, np.median)
        _, mlo = binned_curve(snr, val, edges, lambda v: np.percentile(v, 25))
        _, mhi = binned_curve(snr, val, edges, lambda v: np.percentile(v, 75))
        if len(mx):
            ax.fill_between(mx, mlo, mhi, color=color, alpha=0.25, label='IQR (25-75%)')
            ax.plot(mx, my, '-', color=color, lw=3.0, label=median_label)
        ax.axvline(snr50, color='0.25', ls=':', lw=1.8)            # shared detection threshold
        ax.set_yscale('log'); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which='both'); ax.legend(loc='upper right', framealpha=0.9)
        ax.text(0.015, 0.07, spread_note, transform=ax.transAxes, fontsize=12,
                color=color, va='bottom')

    # --- Top: location error ---
    panel(axL, loc, '#1f6fb4', 'Source-location\nerror (m)', 'median location error',
          'non-identifiable below threshold:\nbimodal, domain-scale spread')
    axL.axhline(cell_m, ls='--', color='0.3', lw=1.6, label=f'1 pixel ({cell_m:.0f} m)')
    axL.legend(loc='upper right', framealpha=0.9)
    axL.set_title('Below SNR50, BOTH location and volume are unreliable\n'
                  '(medians degrade together; location has the heavier tails)')

    # --- Bottom: dV percentage error ---
    panel(axD, dv, '#c0392b', 'Volume change |dV|\n% error', 'median |dV| %error',
          'unimodal, tighter spread —\nbut equally degraded below threshold')
    axD.set_ylim(1, None); axD.set_xlim(x_lo, x_hi)
    axD.set_xlabel('Per-epoch instantaneous SNR (dB)')

    # Shared SNR50 callout near the top panel's threshold line.
    axL.annotate(f'SNR50 = {snr50:.1f} dB', xy=(snr50, axL.get_ylim()[1] * 0.4),
                 xytext=(snr50 + 2.5, axL.get_ylim()[1] * 0.5), fontsize=12, color='0.25',
                 arrowprops=dict(arrowstyle='->', color='0.25', lw=1.4))

    fig.tight_layout()
    save(fig, os.path.join(out_dir, f'{base_name}_loc_vs_dV_pres'))
    plt.close(fig)
    print(f"    both metrics: median + IQR vs SNR (co-degradation; location heavier-tailed)")


def figure_detectability(out_dir, base_name, cell_m):
    """(3) Minimum detectable dV vs scene noise RMS — translates SNR50/SNR90 into a physical limit.

    From the failure-analysis calibration:
        min_dV(m3) = k_dV_per_mm * noise_rms_mm * 10^(SNR/20)
    i.e. for any real track, take its post-correction residual noise RMS (mm) and read off
    the smallest source volume change PILA can localize. 95% CIs (bootstrap over epochs) are
    shown for BOTH the 50% detection limit and the 90% reliable-detection threshold; the SNR90
    band is wider because it inherits uncertainty from both the curve centre AND its width.
    """
    import json
    js_path = os.path.join(out_dir, f'{base_name}_failure_analysis.json')
    with open(js_path) as f:
        det = json.load(f)['detectability']
    k_dV = det['k_dV_per_mm_signal_rms']          # m3 per mm signal RMS
    snr50 = det['snr50_db']
    # CI + width from the scene logistic fit (same JSON).
    with open(js_path) as f:
        scene = json.load(f)['scene']
    ci = scene['ci95_db']
    snr90 = snr50 + scene['width_s_db'] * np.log(9.0)   # 90% (reliable) detection threshold
    noise_example = det['noise_rms_scene_mm']      # the synthetic Marapi scene noise

    # --- Bootstrap a 95% CI for SNR90 from the raw scene SNR + recovery flags ---
    srows = read_csv(os.path.join(out_dir, f'{base_name}_scene_vs_local_snr.csv'))
    bscene = np.array([float(r['scene_snr']) for r in srows])
    brec = (np.array([float(r['loc_err_m']) for r in srows]) < cell_m).astype(int)
    rng = np.random.RandomState(0); s90s = []
    for _ in range(500):
        idx = rng.randint(0, len(bscene), len(bscene))
        try:
            mu_b, s_b = logistic_fit(bscene[idx], brec[idx])
            s90s.append(mu_b + s_b * np.log(9.0))
        except Exception:
            pass
    ci90 = (float(np.percentile(s90s, 2.5)), float(np.percentile(s90s, 97.5)))

    # min_dV is linear in noise RMS; gain factor depends on the chosen recovery SNR.
    noise_mm = np.linspace(5.0, 100.0, 200)
    to_Mm3 = 1e-6
    def min_dV_Mm3(snr_db):
        return k_dV * noise_mm * 10 ** (snr_db / 20.0) * to_Mm3

    central = min_dV_Mm3(snr50)
    central90 = min_dV_Mm3(snr90)
    lo = min_dV_Mm3(ci[0]); hi = min_dV_Mm3(ci[1])
    lo90 = min_dV_Mm3(ci90[0]); hi90 = min_dV_Mm3(ci90[1])

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.fill_between(noise_mm, lo, hi, color='#1f6fb4', alpha=0.18,
                    label=f'95% CI (SNR50 [{ci[0]:.1f}, {ci[1]:.1f}] dB)')
    ax.fill_between(noise_mm, lo90, hi90, color='#c0392b', alpha=0.12,
                    label=f'95% CI (SNR90 [{ci90[0]:.1f}, {ci90[1]:.1f}] dB)')
    # SNR90 line (conservative / reliable) sits above the SNR50 line.
    ax.plot(noise_mm, central90, color='#c0392b', lw=3.0, ls='--',
            label=f'reliable detection (SNR90 = {snr90:.1f} dB)')
    ax.plot(noise_mm, central, color='#1f6fb4', lw=3.2,
            label=f'detection limit (SNR50 = {snr50:.1f} dB)')

    # Worked example: the synthetic Marapi scene, on BOTH thresholds.
    ex_dV = k_dV * noise_example * 10 ** (snr50 / 20.0) * to_Mm3
    ex_dV90 = k_dV * noise_example * 10 ** (snr90 / 20.0) * to_Mm3
    ax.plot([noise_example], [ex_dV], marker='*', ms=20, color='#1f6fb4', mec='k',
            mew=0.8, zorder=5,
            label=f'Marapi @ {noise_example:.0f} mm: {ex_dV:.0f} Mm³ (50%) / {ex_dV90:.0f} Mm³ (90%)')
    ax.plot([noise_example], [ex_dV90], marker='D', ms=11, color='#c0392b', mec='k',
            mew=0.8, zorder=5)

    ax.set_xlim(5, 100); ax.set_ylim(0, hi90.max())
    ax.set_xlabel('Scene noise level, LOS RMS (mm)')
    ax.set_ylabel('Minimum detectable volume change |dV| (Mm³)')
    ax.set_title('PILA Mogi detectability: from SNR limit to a physical volume')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, f'{base_name}_detectability_pres'))
    plt.close(fig)
    print(f"    k = {k_dV:.3e} m3/mm; Marapi {noise_example:.1f} mm -> {ex_dV:.1f} Mm3 (50%) / "
          f"{ex_dV90:.1f} Mm3 (90%); SNR90 = {snr90:.2f} dB")


def figure_detectability_deformation(out_dir, base_name, cell_m):
    """(5) Minimum detectable SURFACE DEFORMATION vs scene noise RMS.

    The volume-change twin of figure_detectability, but in the directly-observable LOS
    deformation that the noise competes with — answers "given a scene's residual noise,
    how much surface deformation must be present before PILA can localize the source?".

        min_signal_rms(mm) = noise_rms_mm * 10^(SNR/20)                  (scene RMS LOS)
        min_peak_def(mm)   = (peak/rms) * noise_rms_mm * 10^(SNR/20)     (peak |LOS|)

    Peak |LOS| is the headline (largest deformation actually on the surface; bold, with
    CI bands at the 50% and 90% thresholds); scene RMS LOS is drawn faintly as the
    diluted-over-the-scene reference. Both are linear through the origin in noise RMS.
    """
    import json
    js_path = os.path.join(out_dir, f'{base_name}_failure_analysis.json')
    with open(js_path) as f:
        full = json.load(f)
    det = full['detectability']; scene = full['scene']
    peak_over_rms = det['peak_over_rms']           # peak |LOS| / scene RMS (~constant)
    snr50 = det['snr50_db']
    ci = scene['ci95_db']
    snr90 = snr50 + scene['width_s_db'] * np.log(9.0)
    noise_example = det['noise_rms_scene_mm']

    # Bootstrap a 95% CI for SNR90 (same procedure as the dV figure).
    srows = read_csv(os.path.join(out_dir, f'{base_name}_scene_vs_local_snr.csv'))
    bscene = np.array([float(r['scene_snr']) for r in srows])
    brec = (np.array([float(r['loc_err_m']) for r in srows]) < cell_m).astype(int)
    rng = np.random.RandomState(0); s90s = []
    for _ in range(500):
        idx = rng.randint(0, len(bscene), len(bscene))
        try:
            mu_b, s_b = logistic_fit(bscene[idx], brec[idx])
            s90s.append(mu_b + s_b * np.log(9.0))
        except Exception:
            pass
    ci90 = (float(np.percentile(s90s, 2.5)), float(np.percentile(s90s, 97.5)))

    noise_mm = np.linspace(5.0, 100.0, 200)
    def min_peak_mm(snr_db):
        return peak_over_rms * noise_mm * 10 ** (snr_db / 20.0)
    def min_rms_mm(snr_db):
        return noise_mm * 10 ** (snr_db / 20.0)

    central = min_peak_mm(snr50); central90 = min_peak_mm(snr90)
    lo = min_peak_mm(ci[0]); hi = min_peak_mm(ci[1])
    lo90 = min_peak_mm(ci90[0]); hi90 = min_peak_mm(ci90[1])

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.fill_between(noise_mm, lo, hi, color='#1f6fb4', alpha=0.18,
                    label=f'95% CI (SNR50 [{ci[0]:.1f}, {ci[1]:.1f}] dB)')
    ax.fill_between(noise_mm, lo90, hi90, color='#c0392b', alpha=0.12,
                    label=f'95% CI (SNR90 [{ci90[0]:.1f}, {ci90[1]:.1f}] dB)')
    ax.plot(noise_mm, central90, color='#c0392b', lw=3.0, ls='--',
            label=f'reliable detection (SNR90 = {snr90:.1f} dB)')
    ax.plot(noise_mm, central, color='#1f6fb4', lw=3.2,
            label=f'detection limit (SNR50 = {snr50:.1f} dB)')
    # Scene RMS LOS reference (diluted over the whole scene; thinner, grey).
    ax.plot(noise_mm, min_rms_mm(snr50), color='0.45', lw=2.0, ls=':',
            label='scene RMS LOS @ SNR50 (reference)')

    # Worked example: the synthetic Marapi scene, peak threshold on BOTH SNR levels.
    ex_peak = peak_over_rms * noise_example * 10 ** (snr50 / 20.0)
    ex_peak90 = peak_over_rms * noise_example * 10 ** (snr90 / 20.0)
    ax.plot([noise_example], [ex_peak], marker='*', ms=20, color='#1f6fb4', mec='k',
            mew=0.8, zorder=5,
            label=f'Marapi @ {noise_example:.0f} mm: {ex_peak:.0f} mm (50%) / {ex_peak90:.0f} mm (90%)')
    ax.plot([noise_example], [ex_peak90], marker='D', ms=11, color='#c0392b', mec='k',
            mew=0.8, zorder=5)

    ax.set_xlim(5, 100); ax.set_ylim(0, hi90.max())
    ax.set_xlabel('Scene noise level, LOS RMS (mm)')
    ax.set_ylabel('Minimum detectable surface deformation,\npeak |LOS| (mm)')
    ax.set_title('PILA Mogi detectability: minimum surface deformation vs scene noise')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, f'{base_name}_detectability_deformation_pres'))
    plt.close(fig)
    print(f"    peak/rms = {peak_over_rms:.1f}; Marapi {noise_example:.1f} mm -> peak |LOS| "
          f">= {ex_peak:.0f} mm (50%) / {ex_peak90:.0f} mm (90%); SNR90 = {snr90:.2f} dB")


def figure_logistic_ratio(out_dir, base_name, cell_m):
    """(2b) Logistic recovery-probability curve in LINEAR signal/noise ratio (no 20log10).

    Companion to figure_logistic (which uses dB). Recovery is naturally sigmoidal in dB
    (= log-ratio), so a logistic in linear ratio is right-skewed and a slightly poorer
    functional fit; the AUC is IDENTICAL to the dB fit (rank-based, and dB<->ratio is
    monotonic). Shown for readers who think in plain signal/noise ratio. Scene ratio is
    primary (with bootstrap CI); local ratio is drawn for context.
    """
    csv_path = os.path.join(out_dir, f'{base_name}_scene_vs_local_snr.csv')
    rows = read_csv(csv_path)
    loc = np.array([float(r['loc_err_m']) for r in rows])
    rec = (loc < cell_m).astype(int)
    # scene_ratio is in the CSV; local has no ratio column -> derive from local_snr (dB).
    scene_r = np.array([float(r['scene_ratio']) for r in rows])
    local_r = 10 ** (np.array([float(r['local_snr']) for r in rows]) / 20.0)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    xgrid = np.linspace(0, max(scene_r.max(), local_r.max()) * 1.02, 400)
    styles = {'scene': dict(color='#1f6fb4', label='scene-wide ratio', primary=True),
              'local': dict(color='#27ae60', label='local ratio', primary=False)}
    summary = {}
    for name, x in [('scene', scene_r), ('local', local_r)]:
        st = styles[name]
        mu, s = logistic_fit(x, rec)
        lo, hi = boot_ci(x, rec)
        a = auc(x, rec)
        summary[name] = (mu, s, lo, hi, a)
        p = 1.0 / (1.0 + np.exp(-(xgrid - mu) / s))
        ax.plot(xgrid, p, '-' if st['primary'] else '--', color=st['color'],
                lw=3.2 if st['primary'] else 2.2,
                label=f"{st['label']}: ratio50={mu:.2f} (AUC {a:.2f})")
        edges = np.linspace(x.min(), x.max(), 18)
        bx, by = binned_curve(x, rec.astype(float), edges, np.mean, min_count=8)
        ax.plot(bx, by, 'o', color=st['color'], ms=7, alpha=0.55, mec='none')
        if st['primary']:
            ax.axvspan(lo, hi, color=st['color'], alpha=0.12,
                       label=f'95% CI [{lo:.2f}, {hi:.2f}]')
            ax.axvline(mu, color=st['color'], ls=':', lw=1.8)
            ax.plot([mu], [0.5], marker='*', ms=18, color=st['color'], mec='k', mew=0.8, zorder=5)
            # ratio90: the 90%-recovery (reliable) threshold, mirroring the dB figure's SNR90.
            r90 = mu + s * np.log(9.0)
            ax.axvline(r90, color='#c0392b', ls=':', lw=1.8,
                       label=f'ratio90 = {r90:.2f} (reliable)')
            ax.plot([r90], [0.9], marker='D', ms=11, color='#c0392b', mec='k', mew=0.8, zorder=5)

    ax.axhline(0.5, ls=':', color='0.5', lw=1.4)
    ax.axhline(0.9, ls=':', color='0.5', lw=1.0)
    # ratio50 / ratio90 callouts (50% characteristic, 90% reliable).
    mu_r = summary['scene'][0]; r90 = mu_r + summary['scene'][1] * np.log(9.0)
    ax.annotate(f'ratio50 = {mu_r:.2f}\n50% recovery', xy=(mu_r, 0.5),
                xytext=(mu_r + 1.2, 0.40), fontsize=11, color='#1f6fb4', va='center',
                arrowprops=dict(arrowstyle='->', color='#1f6fb4', lw=1.4))
    ax.annotate(f'ratio90 = {r90:.2f}\n(reliable)', xy=(r90, 0.9),
                xytext=(r90 + 1.4, 0.80), fontsize=11, color='#c0392b', va='center',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.4))
    # Cap x so the transition is legible; beyond ~10 every epoch recovers (p=1).
    x_hi = min(10.0, xgrid.max())
    ax.set_ylim(-0.03, 1.05); ax.set_xlim(0, x_hi)
    ax.set_xlabel('Signal/noise ratio (linear, = 10^(SNR_dB/20))')
    ax.set_ylabel('Probability of source recovery')
    ax.set_title('PILA Mogi operating envelope: recovery probability vs signal/noise ratio')
    ax.grid(alpha=0.3); ax.legend(loc='lower right', framealpha=0.9)
    fig.text(0.5, -0.02, 'Same data & AUC as the dB curve; the linear-ratio logistic is '
             'right-skewed (recovery is naturally sigmoidal in dB = log-ratio) — shown for readability.',
             ha='center', va='top', fontsize=10.5, style='italic', color='0.3')
    fig.tight_layout()
    save(fig, os.path.join(out_dir, f'{base_name}_recovery_logistic_ratio_pres'))
    plt.close(fig)
    mu, s, lo, hi, a = summary['scene']
    print(f"    scene ratio50 = {mu:.2f} (95% CI [{lo:.2f}, {hi:.2f}]), "
          f"ratio90 = {mu + s*np.log(9.0):.2f}, width {s:.2f}, AUC {a:.3f}")


def figure_snr_vs_dv(out_dir, base_name, cell_m):
    """(6) Per-epoch SNR vs true source dV — ties physical source volume to the SNR it
    produces and whether PILA recovered it.

    SNR (dB) on x, true dV (Mm³, log) on y; points coloured recovered/failed, with the
    fitted SNR50 (50% recovery) and SNR90 (reliable) thresholds marked. Reads from
    <base>_perepoch_snr.csv (needs the dV_true_m3 column). Skipped for models without dV.
    """
    csv_path = os.path.join(out_dir, f'{base_name}_perepoch_snr.csv')
    rows = read_csv(csv_path)
    if 'dV_true_m3' not in rows[0]:
        print("    no dV_true_m3 column (non-dV model?); skipping SNR-vs-dV figure")
        return
    snr = np.array([float(r['snr_db']) for r in rows])
    loc = np.array([float(r['loc_err_m']) for r in rows])
    dv_Mm3 = np.array([float(r['dV_true_m3']) for r in rows]) * 1e-6
    rec = loc < cell_m
    good = np.isfinite(dv_Mm3) & (dv_Mm3 > 0)
    if not good.any():
        print("    dV column present but empty/NaN; skipping SNR-vs-dV figure")
        return

    mu, s = logistic_fit(snr, rec.astype(int))          # SNR50 + width
    snr90 = mu + s * np.log(9.0)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    g_rec = good & rec; g_fail = good & ~rec
    ax.scatter(snr[g_rec], dv_Mm3[g_rec], s=22, c='#1f6fb4', alpha=0.45, edgecolors='none',
               label=f'recovered (< 1 pixel, {cell_m:.0f} m)')
    ax.scatter(snr[g_fail], dv_Mm3[g_fail], s=22, c='#c0392b', alpha=0.45, edgecolors='none',
               label='failed')

    # Median dV-at-recovery trend (robust line through the cloud).
    edges = np.arange(np.floor(snr.min()), np.ceil(snr.max()) + 1, 2.0)
    mx, my = binned_curve(snr[good], dv_Mm3[good], edges, np.median, min_count=8)
    if len(mx):
        ax.plot(mx, my, 'k-', lw=2.4, label='median dV (2 dB bins)')

    ax.axvline(mu, ls=':', color='#1f6fb4', lw=2.0)
    ax.plot([mu], [np.median(dv_Mm3[good])], marker='*', ms=18, color='#1f6fb4',
            mec='k', mew=0.8, zorder=5)
    ax.axvline(snr90, ls=':', color='#c0392b', lw=2.0)
    # Rotated labels along each threshold line, near the bottom, to avoid overlap.
    ax.text(mu, 0.02, f'SNR50 = {mu:.1f} dB  ', color='#1f6fb4', rotation=90,
            va='bottom', ha='right', fontsize=12, transform=ax.get_xaxis_transform())
    ax.text(snr90, 0.02, f'SNR90 = {snr90:.1f} dB  ', color='#c0392b', rotation=90,
            va='bottom', ha='right', fontsize=12, transform=ax.get_xaxis_transform())

    ax.set_yscale('log')
    ax.set_xlabel('Per-epoch instantaneous SNR (dB)')
    ax.set_ylabel('True source volume change dV (Mm³)')
    ax.set_title('PILA Mogi: source volume vs the SNR it produces (and whether it is recovered)')
    ax.grid(alpha=0.3, which='both')
    ax.legend(loc='lower right', framealpha=0.9)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, f'{base_name}_snr_vs_dV_pres'))
    plt.close(fig)
    print(f"    {good.sum()} epochs with dV; SNR50 = {mu:.2f} dB, SNR90 = {snr90:.2f} dB")


def main():
    ap = argparse.ArgumentParser(description="Presentation-ready SNR-sweep figures from CSVs.")
    ap.add_argument('--out', default='synthetic/snr_sweep_ml1')
    ap.add_argument('--base-name', default='marapi_mogi_buildup_combined')
    ap.add_argument('--multilook', type=int, default=1)
    args = ap.parse_args()

    cell_m = (40000.0 / 128) * (args.multilook or 1)   # 1 grid pixel = recovery threshold
    print(f"[1/4] Scatter figure (breakdown, cropped) — recovery bar = {cell_m:.0f} m (1 pixel)")
    figure_scatter(args.out, args.base_name, cell_m)
    print("[2/4] Logistic recovery-probability figure (operating envelope, dB)")
    figure_logistic(args.out, args.base_name, cell_m)
    print("[2b] Logistic recovery-probability figure (linear signal/noise ratio)")
    figure_logistic_ratio(args.out, args.base_name, cell_m)
    print("[3/4] Location vs dV accuracy figure (median+IQR; co-degradation, location heavier-tailed)")
    figure_loc_vs_dv(args.out, args.base_name, cell_m)
    print("[4/5] Detectability figure (min dV vs scene noise RMS)")
    figure_detectability(args.out, args.base_name, cell_m)
    print("[5/6] Detectability figure (min surface deformation vs scene noise RMS)")
    figure_detectability_deformation(args.out, args.base_name, cell_m)
    print("[6/6] Per-epoch SNR vs source dV figure")
    figure_snr_vs_dv(args.out, args.base_name, cell_m)
    print("Done. Presentation figures saved alongside the source CSVs.")


if __name__ == '__main__':
    main()
