#!/usr/bin/env python
# Usage:       python tests/test_multimogi.py [--album synthetic/multisource/pila_stacks_v2/albums/E1_S3_K2_separated.npz]
#                  [--workdir <scratch dir>] [--skip-train]
# Description: Step-1 sanity checks for the K-source Mogi decoder (MultiMogi_LOS), the trainer
#              generalisation, the K-1 -> K warm start, the timing instrumentation and the per-source
#              dump. Prints PASS/FAIL per check and exits non-zero on any failure. Plain script (the
#              pila env has no pytest).
# Checks:
#   1  K=1 MultiMogi output == Physics_Mogi_LOS output (bit-for-bit) for random z.
#   2  K=2 with source 2 at dV = 0 == K=1 with source 1.
#   3  K=2 summed output == sum of the two single-source fields (linearity), per_source consistent.
#   4  MultiMogi per-source LOS == independent mogi_los_mm_torch for the same physical params.
#   5  dV penalty: new multi-index form == old single-index form for K=1 (exact); K=2 = sum.
#   6  Trainer dims for K=2 (read from the K=2 training log): dV [3,7], smoothed [0,1,2,4,5,6].
#   7  Short end-to-end training (3 epochs): K=1, then K=2 warm-started from it:
#      7a warm start copies source-1 rows exactly (encoder outputs of source 1 unchanged)
#      7b new source starts near the residual peak (inside the grid, nonzero dV)
#      7c timing_epochs.csv written; per-phase timed total within 5% of measured train_sec
#      7d fit_K2.npz written by the auto-dump; per-source sum == decoder x_P (checked in dump)
# Date:        2026-10-06

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from model.model_phys_smpl import PHYS_VAE_SMPL, Physics_Mogi_LOS, Physics_MultiMogi_LOS   # noqa: E402
from sequential_configs import make_sequential_config                                       # noqa: E402
from synthetic.multisource.torch_generator import mogi_los_mm_torch                         # noqa: E402

RESULTS = []


def check(name, ok, detail):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def physics_pair(album, k):
    """MultiMogi decoder with K = k plus the config used to build it."""
    cfg = make_sequential_config(album, k, save_dir='unused', init_from='unused' if k > 1 else None)
    return Physics_MultiMogi_LOS(cfg), cfg


def unit_checks(album):
    torch.manual_seed(0)
    print("[1/2] Unit checks...")
    multi1, cfg1 = physics_pair(album, 1)
    cfg_single = json.loads(json.dumps(cfg1)); cfg_single['arch']['args']['physics'] = 'Mogi_LOS'
    single = Physics_Mogi_LOS(cfg_single)
    z1 = torch.rand(16, 4)
    diff = (multi1(z1) - single(z1)).abs().max().item()
    check('1 K=1 MultiMogi == Mogi_LOS', diff == 0.0, f"max |diff| {diff:.1e} (standardised units)")

    multi2, _ = physics_pair(album, 2)
    z2 = torch.rand(16, 8)
    z2_off = z2.clone(); z2_off[:, 7] = 0.5                       # symdV: z = 0.5 -> dV = 0 m^3
    diff = (multi2(z2_off) - multi1(z2_off[:, :4])).abs().max().item()
    check('2 K=2 with dV_2 = 0 == K=1', diff < 1e-6, f"max |diff| {diff:.1e}")

    per_src = multi2.per_source_los_mm(z2)                        # [B, 2, N] mm
    fields_k1 = [multi1.per_source_los_mm(z2[:, 4 * k:4 * k + 4])[:, 0] for k in range(2)]
    lin = (per_src.sum(1) - (fields_k1[0] + fields_k1[1])).abs().max().item()
    # Relative: random z gives dV up to 2e8 m^3 (fields of ~1e3-1e4 mm), so float32
    # standardise/un-standardise round-off is ~1e-7 relative, not a fixed mm amount.
    field_max = per_src.sum(1).abs().max().item()
    std_sum = (multi2(z2) * multi2.x_scale + multi2.x_mean - per_src.sum(1)).abs().max().item() / field_max
    check('3 linearity / per-source consistency', lin < 1e-4 and std_sum < 1e-6,
          f"max |sum - K1 fields| {lin:.1e} mm, max |forward - per-source sum| {std_sum:.1e} relative "
          f"(field max {field_max:.0f} mm)")

    params = multi2.rescale(z2.reshape(-1, 4))
    ref = mogi_los_mm_torch(multi2.model.x.double(), multi2.model.y.double(),
                            torch.cat([multi2.los_E, multi2.los_N, multi2.los_U]).double().cpu(),
                            params['xcen'].double().view(16, 2), params['ycen'].double().view(16, 2),
                            params['d'].double().view(16, 2), params['dV'].double().view(16, 2))
    rel = ((per_src.double() - ref).abs().max() / ref.abs().max()).item()
    check('4 vs independent mogi_los_mm_torch', rel < 1e-5, f"max relative diff {rel:.1e}")

    u = torch.randn(32, 4); u0 = 0.0
    old = (u[..., 3] - u0).abs().mean(); new = (u[..., [3]] - u0).abs().sum(dim=-1).mean()
    u2 = torch.randn(32, 8)
    new2 = (u2[..., [3, 7]] - u0).abs().sum(dim=-1).mean()
    sep = (u2[..., 3] - u0).abs().mean() + (u2[..., 7] - u0).abs().mean()
    check('5 dV penalty form', old.item() == new.item() and abs(new2.item() - sep.item()) < 1e-6,
          f"K=1 old {old.item():.6f} new {new.item():.6f}; K=2 {new2.item():.6f} = sum {sep.item():.6f}")
    return multi1


def find_run(save_dir):
    runs = sorted(glob.glob(os.path.join(save_dir, '*', '*', 'models', 'model_best.pth')))
    if not runs:
        raise FileNotFoundError(f"no model_best.pth under {save_dir}")
    return runs[-1]


def run_train(cfg, workdir, tag):
    cfg_path = os.path.join(workdir, f'{tag}.json')
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    log_path = os.path.join(workdir, f'{tag}.log')
    t0 = time.time()
    with open(log_path, 'w') as log:
        # Same env as the PBS jobs: GNU MKL threading layer (avoids the mkl/libgomp clash).
        env = {**os.environ, 'MKL_THREADING_LAYER': 'GNU', 'PYTHONUNBUFFERED': '1'}
        proc = subprocess.run([sys.executable, 'train_pila.py', '-c', cfg_path], stdout=log,
                              stderr=subprocess.STDOUT, cwd=PROJECT_ROOT, env=env)
    print(f"  {tag}: exit {proc.returncode} in {time.time() - t0:.0f}s (log {log_path})")
    return proc.returncode


def integration_checks(album, workdir):
    print("[2/2] End-to-end: K=1 (3 epochs) -> K=2 warm start (3 epochs)...")
    os.makedirs(workdir, exist_ok=True)
    save1, save2 = os.path.join(workdir, 'K1'), os.path.join(workdir, 'K2')
    for path in (save1, save2):
        if os.path.exists(path):
            raise FileExistsError(f"{path} exists; use a fresh --workdir")
    rc = run_train(make_sequential_config(album, 1, save1, epochs=3), workdir, 'K1')
    check('7 K=1 training ran', rc == 0, f"exit code {rc}")
    if rc:
        return
    ckpt1 = find_run(save1)

    # 7a: warm start in-process, compare encoder outputs of source 1
    cfg2 = make_sequential_config(album, 2, save2, init_from=ckpt1, epochs=3)
    from data_loader import data_loaders as module_data
    from utils.warm_start import warm_start_multimogi
    import logging
    logger = logging.getLogger('test')
    loader = module_data.InSARh5DataLoader(**cfg2['data_loader']['args'])
    model1 = PHYS_VAE_SMPL(json.loads(json.dumps(torch.load(ckpt1, map_location='cpu', weights_only=False)['config'].config)))
    model1.load_state_dict(torch.load(ckpt1, map_location='cpu', weights_only=False)['state_dict'])
    model2 = PHYS_VAE_SMPL(json.loads(json.dumps(cfg2)))
    info = warm_start_multimogi(model2, cfg2, loader, logger)
    x = torch.tensor(np.asarray(loader.dataset.series[:5]), dtype=torch.float32)
    model1.eval(); model2.eval()
    with torch.no_grad():
        s1, _ = model1.encode(x, None); s2, _ = model2.encode(x, None)
    diff = max((s2['mean'][:, :4] - s1['mean']).abs().max().item(), (s2['lnvar'][:, :4] - s1['lnvar']).abs().max().item())
    check('7a warm start keeps source 1', diff < 1e-6, f"max |u diff| {diff:.1e}")
    start = info['new_source_start']
    check('7b new source start', abs(start['xcen_km']) <= 24.5 and abs(start['ycen_km']) <= 21.5 and start['dV_m3'] != 0,
          f"x={start['xcen_km']:.1f} km, y={start['ycen_km']:.1f} km, dV={start['dV_m3']:.2e} m^3, "
          f"residual peak {info['residual_peak_mm']:.1f} mm")

    rc = run_train(cfg2, workdir, 'K2')
    check('7 K=2 training ran', rc == 0, f"exit code {rc}")
    if rc:
        return
    k2_log = open(os.path.join(workdir, 'K2.log')).read()
    expected = "n_sources=2: dV dims [3, 7], temporal-smoothness dims [0, 1, 2, 4, 5, 6]"
    check('6 trainer K=2 dV / smoothed dims', expected in k2_log, expected if expected in k2_log else 'line not found in K2.log')
    run_dir = os.path.dirname(find_run(save2))
    usage = json.load(open(os.path.join(run_dir, 'compute_usage.json')))
    timed = usage['timing']['timed_total_sec']
    rows = open(os.path.join(run_dir, 'timing_epochs.csv')).read().strip().splitlines()
    frac = timed / usage['train_sec']
    check('7c timing', len(rows) == 4 and 0.95 <= frac <= 1.05 and usage.get('n_sources') == 2
          and 'warm_start_sec' in usage,
          f"{len(rows) - 1} epoch rows; timed {timed:.2f}s vs train_sec {usage['train_sec']:.2f}s ({100 * frac:.1f}%); "
          f"physics_decode {usage['timing'].get('physics_decode_ms_per_sample', 'NA')} ms/sample")
    fit = os.path.join(run_dir, 'fit_K2.npz')
    ok = os.path.exists(fit)
    detail = 'missing'
    if ok:
        with np.load(fit) as f:
            detail = f"params_phys {f['params_phys'].shape}, xP_src_mm {f['xP_src_mm'].shape}"
            ok = f['params_phys'].shape == (30, 2, 4) and f['xP_src_mm'].shape == (30, 2, 128, 128)
    check('7d auto-dump fit_K2.npz', ok, detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--album', default='synthetic/multisource/pila_stacks_v2/albums/E1_S3_K2_separated.npz')
    parser.add_argument('--workdir', default=None, help='scratch dir for the short training runs')
    parser.add_argument('--skip-train', action='store_true')
    args = parser.parse_args()
    album = os.path.abspath(args.album)
    unit_checks(album)

    if not args.skip_train:
        if args.workdir is None:
            raise ValueError("--workdir is required for the training checks")
        integration_checks(album, os.path.abspath(args.workdir))
    n_fail = RESULTS.count(False)
    print(f"Done. {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed.")
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
