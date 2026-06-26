#!/usr/bin/env python
# Usage:       python -m synthetic.run_moving_tests \
#                  --specs synthetic/specs/marapi_mogi_rising_combined.json \
#                          synthetic/specs/marapi_sun69_sill_combined.json \
#                          synthetic/specs/marapi_okada_dike_combined.json \
#                  --epochs 120
# Description: Train PILA on each ALREADY-GENERATED synthetic moving-source cube and run
#              inference, producing per-parameter recovery line plots (truth vs inferred)
#              and a metrics json per case. Reuses the same machinery as the SNR sweep
#              (make_synth_config -> train_pila.py -> synthetic.evaluate) but on existing
#              cubes (no regeneration / no dV sweep).
# Scientific note: PILA inversion is per-epoch / amortized. Training is self-supervised
#              (reconstruction of the LOS field through the physics decoder); the truth
#              params are used ONLY for scoring, never for training.
# Date:        2026-06-23

import argparse
import json
import os
import subprocess
import sys
import time

from synthetic.make_synth_config import make_config
from synthetic.evaluate import evaluate
from synthetic.snr_sweep import _newest_ckpt

# Physics -> SierraNegra template config (parameter bounds get overridden by spec['paras']).
TEMPLATE = {
    'mogi':  'configs/phys_smpl/SierraNegra_Mogi_InSAR_A.json',
    'sun69': 'configs/phys_smpl/SierraNegra_Sun69_InSAR_A.json',
    'okada': 'configs/phys_smpl/SierraNegra_Okada_InSAR_A.json',
}


def run_one(spec_path, epochs, multilook, std_mode):
    """Train + evaluate one synthetic cube described by spec_path. Returns metrics dict."""
    with open(spec_path) as f:
        spec = json.load(f)
    name = spec['name']
    model = spec['model']
    cube_h5 = os.path.join('synthetic', 'cubes', name, f'{name}.h5')
    truth_json = os.path.join('synthetic', 'cubes', name, f'{name}_truth.json')
    for p in (cube_h5, truth_json):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} (generate the cube first)")

    template = TEMPLATE[model]
    cfg_path = os.path.join('configs', 'synth', f'{name}.json')

    print(f"\n=== [{name}] make config (template {os.path.basename(template)}) ===")
    _, n_pts = make_config(template, cube_h5, cfg_path, name=name, paras=spec['paras'],
                           epochs=epochs, multilook=multilook,
                           base_scene=spec['base_scene'], std_mode=std_mode)

    print(f"=== [{name}] train ({n_pts} px, {epochs} epochs) ===")
    t0 = time.time()
    env = dict(os.environ); env['MKL_THREADING_LAYER'] = 'GNU'
    log = os.path.join('saved', 'synth', f'{name}_train.log')
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, 'w') as lf:
        r = subprocess.run([sys.executable, 'train_pila.py', '-c', cfg_path],
                           stdout=lf, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"[{name}] train failed (rc={r.returncode}); see {log}")
    print(f"  trained in {time.time()-t0:.1f}s; log {log}")

    ckpt = _newest_ckpt(os.path.join('saved', 'synth', name))
    eval_dir = os.path.join('synthetic', 'eval', name)
    print(f"=== [{name}] evaluate -> {eval_dir} (ckpt {ckpt}) ===")
    m = evaluate(truth_json, ckpt, out_dir=eval_dir)
    m['n_points'] = n_pts
    json.dump(m, open(os.path.join(eval_dir, f'{name}_metrics.json'), 'w'), indent=2)
    print(f"=== [{name}] DONE. metrics in {eval_dir} ===")
    return m


def main():
    ap = argparse.ArgumentParser(description="Train+infer PILA on existing synthetic cubes.")
    ap.add_argument('--specs', nargs='+', required=True, help='Scenario spec JSON path(s).')
    ap.add_argument('--epochs', type=int, default=120, help='Training epochs (default 120).')
    ap.add_argument('--multilook', type=int, default=1, help='Multilook factor (default 1).')
    ap.add_argument('--std-mode', default=None, help='Standardization mode (default: template).')
    args = ap.parse_args()

    t0 = time.time()
    for spec_path in args.specs:
        run_one(spec_path, args.epochs, args.multilook, args.std_mode)
    print(f"\nAll done in {time.time()-t0:.1f}s.")


if __name__ == '__main__':
    main()
