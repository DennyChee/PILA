#!/usr/bin/env python
# Usage:       python -m synthetic.snr_sweep \
#                  --base-spec synthetic/specs/marapi_sun69_buildup_combined.json \
#                  --template configs/phys_smpl/SierraNegra_Sun69_InSAR_A.json \
#                  --amp-list 5e6,1e7,2e7,5e7,1e8,1.5e8 --multilook 4 --epochs 120
# Description: Sweep the source amplitude (dV for Mogi/Sun69, opening for Okada) at
#              FIXED noise to find the SNR at which PILA's recovery breaks down.
#              For each amplitude: regenerate the synthetic cube, template a config,
#              train PILA (subprocess), evaluate parameter recovery, and record
#              {realized SNR, LOS RMSE, source-location error, key per-param errors}.
#              Produces a CSV and a breakdown-SNR curve.
# Scientific note: SNR is reported at the peak epoch; PILA only ever sees the
#              post-multilook field, so the realized SNR recorded here is computed on
#              the full-resolution cube (MATLAB convention) AND should be read with
#              the multilook caveat in mind (multilook suppresses white > correlated).
# Date:        2026-06-22

import argparse
import copy
import csv
import json
import os
import subprocess
import sys

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from synthetic.generate_timeseries import generate
from synthetic.make_synth_config import make_config
from synthetic.evaluate import evaluate
from synthetic.validate_all import validate_all


def _newest_ckpt(save_dir):
    """Return the newest model_best.pth under a save_dir tree."""
    hits = []
    for dp, _, fns in os.walk(save_dir):
        if 'model_best.pth' in fns:
            p = os.path.join(dp, 'model_best.pth')
            hits.append((os.path.getmtime(p), p))
    if not hits:
        raise FileNotFoundError(f"no model_best.pth under {save_dir}")
    return sorted(hits)[-1][1]


def run_sweep(base_spec_path, template, amp_list, multilook, epochs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(base_spec_path, 'r') as f:
        base_spec = json.load(f)
    amp_key = base_spec['trajectory'].get('amplitude_key', 'dV')
    base_name = base_spec['name']

    rows = []
    for amp in amp_list:
        tag = f"{base_name}_{amp_key}{amp:.2e}".replace('+', '').replace('.', 'p')
        spec = copy.deepcopy(base_spec)
        spec['name'] = tag
        spec['trajectory']['amp_end'] = float(amp)

        print(f"\n########## SWEEP {amp_key}={amp:.3e}  ({tag}) ##########")
        eval_dir = os.path.join(out_dir, 'eval', tag)
        metrics_json = os.path.join(eval_dir, f'{tag}_metrics.json')
        truth_json = os.path.join('synthetic', 'cubes', tag, f'{tag}_truth.json')

        # Resumable: if this point was already evaluated, reuse it (so a re-run only
        # trains the iterations that are missing, e.g. one killed by SIGTERM).
        if os.path.exists(metrics_json) and os.path.exists(truth_json):
            print("  already evaluated; reusing metrics.")
            m = json.load(open(metrics_json))
            n_pts = m.get('n_points') or json.load(open(
                os.path.join('configs', 'synth', f'{tag}.json')))['arch']['args']['input_dim']
        else:
            out_h5, truth_json = generate(spec)
            cfg_path = os.path.join('configs', 'synth', f'{tag}.json')
            _, n_pts = make_config(template, out_h5, cfg_path, name=tag,
                                   paras=spec['paras'], epochs=epochs, multilook=multilook,
                                   base_scene=spec['base_scene'])
            # Train (subprocess; MKL_THREADING_LAYER=GNU avoids the mkl-service/libgomp
            # conflict when launching a numpy/torch python from an initialized one).
            log = os.path.join(out_dir, f'{tag}_train.log')
            env = dict(os.environ); env['MKL_THREADING_LAYER'] = 'GNU'
            with open(log, 'w') as lf:
                r = subprocess.run([sys.executable, 'train_pila.py', '-c', cfg_path],
                                   stdout=lf, stderr=subprocess.STDOUT, env=env)
            if r.returncode != 0:
                print(f"  TRAIN FAILED (rc={r.returncode}); see {log}")
                continue
            ckpt = _newest_ckpt(os.path.join('saved', 'synth', tag))
            m = evaluate(truth_json, ckpt, out_dir=eval_dir)
            m['n_points'] = n_pts
            json.dump(m, open(metrics_json, 'w'), indent=2)

            # Full validation plot set — only for FRESHLY trained runs (cached runs
            # already have their plots, so a denser re-sweep doesn't re-plot them).
            try:
                validate_all(spec, multilook or 1, ckpt=ckpt)
            except Exception as e:
                print(f"  validate_all failed for {tag}: {e}")

        with open(truth_json) as f:
            truth = json.load(f)
        noise_info = truth['noise']
        snr = noise_info.get('realized_snr_db', float('nan'))
        # Linear ratio (= 10^(SNR_dB/20)) and absolute noise level, both more
        # physically readable than dB. signal/noise RMS are already in the truth sidecar.
        signal_rms_mm = noise_info.get('signal_rms_mm', float('nan'))
        noise_rms_mm = noise_info.get('noise_rms_mm', float('nan'))
        snr_ratio = (signal_rms_mm / noise_rms_mm
                     if noise_rms_mm and np.isfinite(noise_rms_mm) and noise_rms_mm > 0
                     else float('nan'))
        pp = m['per_param']
        row = {
            'amp': amp, 'amp_key': amp_key, 'realized_snr_db': snr,
            'snr_ratio': snr_ratio, 'signal_rms_mm': signal_rms_mm,
            'noise_rms_mm': noise_rms_mm,
            'n_points': n_pts,
            'los_rmse_mm_peak': m['los_rmse_mm_peak'],
            'loc_err_km_peak': m['source_loc_err_km_peak'],
            'loc_err_km_strong': m['source_loc_err_km_strong_mean'],
        }
        for p in pp:
            row[f'{p}_pcterr_peak'] = pp[p]['pct_err_peak']
        rows.append(row)
        print(f"  -> SNR {snr:.2f} dB | loc_err {row['loc_err_km_peak']*1000:.0f} m | "
              f"LOS RMSE {row['los_rmse_mm_peak']:.2f} mm")

    # --- CSV ---
    csv_path = os.path.join(out_dir, f'{base_name}_snr_sweep.csv')
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"\nWrote {csv_path}")

        # --- Breakdown curve ---
        from synthetic import plots
        snr = [r['realized_snr_db'] for r in rows]
        loc_err_m = [r['loc_err_km_peak'] * 1000 for r in rows]
        amp_param = 'dV_pcterr_peak' if 'dV_pcterr_peak' in rows[0] else \
            ('opening_pcterr_peak' if 'opening_pcterr_peak' in rows[0] else None)
        secondary = {'LOS RMSE (mm)': [r['los_rmse_mm_peak'] for r in rows]}
        if amp_param:
            secondary[f'{amp_param.replace("_pcterr_peak","")} %err'] = \
                [r[amp_param] for r in rows]
        # Breakdown threshold ~ one multilook coarse cell (px_size * multilook).
        cell_m = (40000.0 / 128) * (multilook or 1)
        plots.plot_breakdown_curve(out_dir, base_name, snr, loc_err_m,
                                   xlabel='SNR (dB)', stem='breakdown_snr',
                                   secondary=secondary, breakdown_threshold_m=cell_m)
        # Also in linear ratio (more readable than dB). Noise RMS is fixed across this
        # sweep AND is not a standalone recovery predictor, so no noise-RMS breakdown.
        ratio = [r['snr_ratio'] for r in rows]
        plots.plot_breakdown_curve(out_dir, base_name, ratio, loc_err_m,
                                   xlabel='signal/noise ratio', stem='breakdown_ratio',
                                   secondary=secondary, breakdown_threshold_m=cell_m)
    return rows


def main():
    ap = argparse.ArgumentParser(description="PILA SNR-breakdown sweep.")
    ap.add_argument('--base-spec', required=True)
    ap.add_argument('--template', required=True)
    ap.add_argument('--amp-list', required=True,
                    help='Comma-separated amplitudes (dV in m^3 or opening in m).')
    ap.add_argument('--multilook', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--out', default=os.path.join('synthetic', 'snr_sweep'))
    args = ap.parse_args()
    amp_list = [float(x) for x in args.amp_list.split(',')]
    run_sweep(args.base_spec, args.template, amp_list, args.multilook, args.epochs, args.out)


if __name__ == '__main__':
    main()
