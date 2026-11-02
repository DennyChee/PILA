#!/usr/bin/env python
# Usage:       python tests/test_explicit_init.py [--album synthetic/multisource/clean_scenarios/K2_separated_static.npz]
# Description: Sanity checks for warm_start_explicit_sources (fresh K=2 network started at explicit sources).
#              1 requested (x, y, d, dV) -> z -> sigmoid(u) round trip (units km / m^3, source-major order)
#              2 encoder output at epoch 0 is EXACTLY constant over inputs (head weights zeroed); lnvar head starts tight;
#                the returned info has the keys train_pila.log_compute_usage reads
#              3 rendered per-source LOS fields == independent mogi_los_mm_torch of the requested sources (<0.5 mm)
#              4 summed field == noise-free truth scene (when the start is the truth) -> loss at init ~ 0
#              5 dV = 0 start is rejected; K != len(init_sources) is rejected
#              6 the default (no init_sources) path is untouched: make_sequential_config(K=2) without init_from still raises
# Date:        2026-10-08

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from model.model_phys_smpl import PHYS_VAE_SMPL                                    # noqa: E402
from sequential_configs import make_sequential_config                              # noqa: E402
from synthetic.multisource.make_pila_stack import load_ta69_geometry               # noqa: E402
from synthetic.multisource.torch_generator import mogi_los_mm_torch                # noqa: E402
from utils.warm_start import warm_start_explicit_sources                           # noqa: E402
from data_loader import data_loaders as module_data                                # noqa: E402

RESULTS = []


def check(name, ok, detail):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--album', default='synthetic/multisource/clean_scenarios/K2_separated_static.npz')
    args = parser.parse_args()
    album = os.path.abspath(args.album)
    truth = json.load(open(album[:-4] + '_truth.json'))
    sources = [[s['xcen_km'], s['ycen_km'], s['depth_km'], s['dV_m3']] for s in truth['sources']]
    logger = logging.getLogger('test')

    print("[1/2] Building a fresh K=2 model at the TRUTH...")
    cfg = make_sequential_config(album, 2, 'unused', init_sources=sources, epochs=3)
    loader = getattr(module_data, cfg['data_loader']['type'])(**cfg['data_loader']['args'])   # standard recipe = Seq loader
    model = PHYS_VAE_SMPL(json.loads(json.dumps(cfg)))
    info = warm_start_explicit_sources(model, cfg, logger)
    series = torch.tensor(np.asarray(loader.dataset.series[:6]), dtype=torch.float32)
    print(f"  Loaded: series shape={tuple(series.shape)}, dtype={series.dtype}")

    print("[2/2] Checks...")
    ranges = model.physics_model.z_phy_ranges
    model.eval()
    with torch.no_grad():
        stat, _ = model.encode(series, None)
        z = torch.sigmoid(stat['mean'])
        fields_mm = model.physics_model.per_source_los_mm(z)                      # [B, K, N]
        rescaled = model.physics_model.rescale(z.reshape(-1, 4))
    # 1: z -> physical round trip (rescale returns dict-like with km*1000 for x, y, d)
    mean_z = z.mean(dim=0).reshape(2, 4).numpy()
    expected_z = np.array(info['z'])
    check('1 z round trip', np.allclose(mean_z, expected_z, atol=2e-3), f"max |z - requested z| {np.abs(mean_z - expected_z).max():.1e}")
    # 2: constant outputs
    spread = float(stat['mean'].std(dim=0).max())
    check('2 outputs exactly constant at epoch 0', spread < 1e-6, f"max std of u over inputs {spread:.1e}")
    lnvar_mean = float(stat['lnvar'].mean()); lnvar_spread = float(stat['lnvar'].std(dim=0).max())
    check('2b lnvar head starts tight and constant', abs(lnvar_mean - info['lnvar_init']) < 1e-4 and lnvar_spread < 1e-6,
          f"lnvar mean {lnvar_mean:.3f} (target {info['lnvar_init']}), spread {lnvar_spread:.1e}")
    needed = ['warm_start_sec', 'init_from', 'new_source_start']      # keys train_pila.log_compute_usage reads
    check('2c compute_usage keys present', all(k in info for k in needed), f"keys {sorted(info)}")
    # 3: independent render
    geom = load_ta69_geometry(); grid = geom['grid']
    xE = torch.tensor(grid['xE_m']); yN = torch.tensor(grid['yN_m'])
    los = torch.tensor(np.stack([grid['losE'], grid['losN'], grid['losU']]))
    worst_mm = 0.0
    total_mm = np.zeros(grid['xE_m'].shape)
    for k, (x_km, y_km, d_km, dV_m3) in enumerate(sources):
        ref = mogi_los_mm_torch(xE, yN, los, torch.tensor([[x_km * 1e3]], dtype=torch.float64), torch.tensor([[y_km * 1e3]], dtype=torch.float64),
                                torch.tensor([[d_km * 1e3]], dtype=torch.float64), torch.tensor([[dV_m3]], dtype=torch.float64))[0, 0].numpy()
        total_mm += ref
        worst_mm = max(worst_mm, float(np.abs(fields_mm[:, k].mean(dim=0).numpy() - ref).max()))
    check('3 per-source field == independent Mogi render', worst_mm < 0.5, f"worst |diff| {worst_mm:.3f} mm (peak {np.abs(total_mm).max():.1f} mm)")
    # 4: summed field vs the scene (truth start => loss ~ 0)
    scene_mm = np.load(album)['los_asc_m'][0].astype(np.float64).ravel() * 1e3
    summed = fields_mm.sum(dim=1).mean(dim=0).numpy()
    rms = float(np.sqrt(np.mean((summed - scene_mm) ** 2)))
    check('4 truth start reproduces the scene', rms < 0.5, f"RMS(sum of sources - scene) {rms:.3f} mm")
    # 5: invalid inputs
    bad = json.loads(json.dumps(cfg)); bad['arch']['phys_vae']['init_sources'][0][3] = 0.0
    try:
        warm_start_explicit_sources(PHYS_VAE_SMPL(json.loads(json.dumps(bad))), bad, logger); ok = False
    except ValueError:
        ok = True
    check('5a dV = 0 rejected', ok, 'ValueError raised' if ok else 'no error')
    try:
        make_sequential_config(album, 3, 'unused', init_sources=sources); ok = False
    except ValueError:
        ok = True
    check('5b K != len(init_sources) rejected', ok, 'ValueError raised' if ok else 'no error')
    for kwargs, label in (({'init_from': 'x'}, 'init_from'), ({'seed_init': {'depth_km': 3.0}}, 'seed_init')):
        try:
            make_sequential_config(album, 2, 'unused', init_sources=sources, **kwargs); ok = False
        except ValueError:
            ok = True
        check(f'5c init_sources + {label} rejected', ok, 'ValueError raised' if ok else 'no error')
    cfg_off = make_sequential_config(album, 2, 'unused', init_sources=sources,
                                     trainer_overrides={'use_kl_term_z_phy': False, 'dv_l1_weight': 0.0, 'temporal_smoothness_weight': 0.0})
    tp = cfg_off['trainer']['phys_vae']
    check('5d penalties-off override applied', tp['use_kl_term_z_phy'] is False and tp['dv_l1_weight'] == 0.0 and tp['temporal_smoothness_weight'] == 0.0,
          f"kl={tp['use_kl_term_z_phy']}, dv_l1={tp['dv_l1_weight']}, curv={tp['temporal_smoothness_weight']}")
    # 6: default path unchanged
    try:
        make_sequential_config(album, 2, 'unused'); ok = False
    except ValueError:
        ok = True
    check('6 K=2 without init_from/init_sources still raises', ok, 'ValueError raised' if ok else 'no error')
    n_fail = RESULTS.count(False)
    print(f"Done. {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed.")
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
