#!/usr/bin/env python
# Usage:       python -m synthetic.run_one_snr_point --base-spec <spec> --template <cfg> \
#                  --dv 2.83e7 --multilook 1 --epochs 120 --out synthetic/snr_sweep_ml1
# Description: Run ONE SNR-sweep point end-to-end (generate cube -> config -> train ->
#              evaluate -> full validation plot set) for a single source amplitude.
#              Designed to be the body of one PBS job so the sweep can fan out across
#              the cluster. Resumable: if the point's metrics already exist, it exits.
#              Aggregation into the CSV + breakdown curve is done separately by
#              synthetic/aggregate_sweep.py once all jobs finish.
# Date:        2026-06-22

import argparse
import copy
import json
import os
import subprocess
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from synthetic.generate_timeseries import generate
from synthetic.make_synth_config import make_config
from synthetic.evaluate import evaluate
from synthetic.validate_all import validate_all
from synthetic.snr_sweep import _newest_ckpt


def _far_field_from_spec(spec, radius_km):
    """Far-field exclusion disk centred on the synthetic source (km), for standardization."""
    tr = spec['trajectory']
    geom = tr.get('geometry') or tr.get('start') or {}
    sx = float(geom.get('xcen', geom.get('xoff', 0.0))) / 1000.0   # m -> km
    sy = float(geom.get('ycen', geom.get('yoff', 0.0))) / 1000.0
    return {'source_xE_km': sx, 'source_yN_km': sy, 'radius_km': float(radius_km)}


def run_point(base_spec_path, template, dv, multilook, epochs, out_dir,
              std_mode='global', ff_radius_km=8.0, tag_suffix=''):
    with open(base_spec_path) as f:
        base_spec = json.load(f)
    amp_key = base_spec['trajectory'].get('amplitude_key', 'dV')
    base_name = base_spec['name']
    tag = f"{base_name}_{amp_key}{dv:.2e}".replace('+', '').replace('.', 'p')
    if tag_suffix:
        tag = f"{tag}_{tag_suffix}"

    eval_dir = os.path.join(out_dir, 'eval', tag)
    metrics_json = os.path.join(eval_dir, f'{tag}_metrics.json')
    if os.path.exists(metrics_json):
        print(f"[{tag}] already done; skipping.")
        return

    spec = copy.deepcopy(base_spec)
    spec['name'] = tag
    spec['trajectory']['amp_end'] = float(dv)

    print(f"[{tag}] generate (dV={dv:.3e})")
    out_h5, truth_json = generate(spec)

    far_field = _far_field_from_spec(spec, ff_radius_km) if std_mode == 'far_field_robust' else None
    cfg_path = os.path.join('configs', 'synth', f'{tag}.json')
    _, n_pts = make_config(template, out_h5, cfg_path, name=tag, paras=spec['paras'],
                           epochs=epochs, multilook=multilook, base_scene=spec['base_scene'],
                           std_mode=std_mode, far_field=far_field)

    print(f"[{tag}] train ({n_pts} px, {epochs} epochs)")
    env = dict(os.environ); env['MKL_THREADING_LAYER'] = 'GNU'
    log = os.path.join(out_dir, f'{tag}_train.log'); os.makedirs(out_dir, exist_ok=True)
    with open(log, 'w') as lf:
        r = subprocess.run([sys.executable, 'train_pila.py', '-c', cfg_path],
                           stdout=lf, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"train failed (rc={r.returncode}); see {log}")

    ckpt = _newest_ckpt(os.path.join('saved', 'synth', tag))
    m = evaluate(truth_json, ckpt, out_dir=eval_dir)
    m['n_points'] = n_pts
    os.makedirs(eval_dir, exist_ok=True)
    json.dump(m, open(metrics_json, 'w'), indent=2)

    print(f"[{tag}] full validation plot set")
    validate_all(spec, multilook or 1, ckpt=ckpt)
    print(f"[{tag}] DONE.")


def main():
    ap = argparse.ArgumentParser(description="Run one SNR-sweep point (for a PBS job).")
    ap.add_argument('--base-spec', required=True)
    ap.add_argument('--template', required=True)
    ap.add_argument('--dv', type=float, required=True, help='Source amplitude (dV m^3 / opening m).')
    ap.add_argument('--multilook', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--out', default=os.path.join('synthetic', 'snr_sweep_ml1'))
    ap.add_argument('--std-mode', default='global', choices=['global', 'far_field_robust'])
    ap.add_argument('--ff-radius', type=float, default=8.0, help='Far-field exclusion radius (km).')
    ap.add_argument('--tag-suffix', default='', help='Suffix to isolate runs (e.g. "ff").')
    args = ap.parse_args()
    run_point(args.base_spec, args.template, args.dv, args.multilook, args.epochs, args.out,
              std_mode=args.std_mode, ff_radius_km=args.ff_radius, tag_suffix=args.tag_suffix)


if __name__ == '__main__':
    main()
