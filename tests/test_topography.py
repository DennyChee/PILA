#!/usr/bin/env python
# Usage:       python tests/test_topography.py [--workdir <scratch dir>]
# Description: Sanity checks for the Williams & Wadge (1998) varying-depth topography correction
#              (arch.args.topography = {"mode": "varying_depth", "h_ref_m": ...}) in the Mogi,
#              OkadaDike and Sun69 forward models and the InSAR LOS decoders. Prints PASS/FAIL per
#              check and exits non-zero on any failure. Plain script (the pila env has no pytest).
# Checks:
#   1  dz = 0 equals the flat model (dz = None) for Mogi, OkadaDike, Sun69: bit-identical, or
#      within float round-off (a [B,N] vs [B,1] depth can change torch's vectorised kernel, e.g.
#      Sun69 differs by ~1 ULP). mode 'none' never builds dz, so existing configs are untouched.
#   2  constant dz = c equals the flat model evaluated at depth d + c.
#   3  random per-pixel dz matches the independent NumPy models (synthetic/forward_numpy.py)
#      called pixel-by-pixel with depth d + dz_i (float64).
#   4  gradients w.r.t. every source parameter are finite with topography on.
#   5  decoder level, zero-height scene: varying_depth decoder == flat decoder (Mogi/Okada/Sun69,
#      point + grid, and MultiMogi K=1 == Mogi_LOS).
#   6  decoder level, COMSOL cone scene: loader height == revolved profile at the coherent cells;
#      grid dz has no NaN; grid and point decoders agree on coherent cells; Mogi_LOS output ==
#      NumPy Mogi LOS with per-pixel depth d + h_i (h_ref = 0).
#   7  positivity guard raises when d_min + min(h - h_ref) <= 0 (cone, h_ref = 'max').
#   8  config errors: unknown mode, and varying_depth on a scene without elevation.
# Date:        2026-10-12

import argparse
import copy
import json
import os
import sys
import tempfile

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from physics.mogi.mogi import Mogi                                                # noqa: E402
from physics.okada.okada_dike import OkadaDike                                    # noqa: E402
from physics.sun69.sun69 import Sun69                                             # noqa: E402
from synthetic.forward_numpy import mogi as mogi_np, okada_dike as okada_np, sun69 as sun69_np  # noqa: E402
from model.model_phys_smpl import (Physics_Mogi_LOS, Physics_Mogi_LOS_Grid,       # noqa: E402
                                   Physics_Okada_LOS, Physics_Okada_LOS_Grid,
                                   Physics_Sun69_LOS, Physics_Sun69_LOS_Grid)
try:   # multi-source decoder lives on the multi-source work; skip its check if absent
    from model.model_phys_smpl import Physics_MultiMogi_LOS                     # noqa: E402
except ImportError:
    Physics_MultiMogi_LOS = None

# --- Inputs ---
TEMPLATE_CONFIG = 'configs/comsol/per_chamber/comsol_wk6_cid1000_cone_desc_mogi.json'
CONE_SCENE = 'data/comsol/wk6_per_chamber_topo/wk6_cid1000_cone.npz'      # has height_m
FLAT_SCENE_NO_HEIGHT = 'data/comsol/wk6_per_chamber/wk6_cid1000_flat.npz'  # no height_m
OKADA_PARAS = 'configs/okada_dike_paras.json'
SUN69_PARAS = 'configs/sierranegra_sun69_paras.json'

RESULTS = []


def report(name, passed, detail=''):
    """Record and print one PASS/FAIL line."""
    RESULTS.append(bool(passed))
    print(f"  {'PASS' if passed else 'FAIL'}: {name}" + (f"  ({detail})" if detail else ''))


def max_relative_difference(reference, candidate):
    """Max |candidate - reference| relative to max |reference| (0.0 when bit-identical)."""
    return float((candidate - reference).abs().max() / reference.abs().max())


# --- Shared random sources (SI units: m, m^3, deg) ---
def random_sources(n_batch, generator, dtype):
    """Random batched source parameters for the three models, as torch tensors."""
    def uniform(low, high):
        return low + (high - low) * torch.rand(n_batch, generator=generator, dtype=dtype)
    mogi = dict(xcen=uniform(-2e3, 2e3), ycen=uniform(-2e3, 2e3), d=uniform(1.5e3, 6e3),
                dV=uniform(-5e6, 5e6))
    okada = dict(xoff=uniform(-2e3, 2e3), yoff=uniform(-2e3, 2e3), depth=uniform(2e3, 6e3),
                 strike=uniform(0, 360), dip=uniform(30, 89), length=uniform(500, 3e3),
                 width=uniform(500, 2e3), opening=uniform(0.1, 2.0))
    sun69 = dict(xcen=uniform(-2e3, 2e3), ycen=uniform(-2e3, 2e3), depth=uniform(3e3, 6e3),
                 radius=uniform(300, 1e3), dV=uniform(-5e6, 5e6))
    return {'mogi': (Mogi, 'd', mogi), 'okada': (OkadaDike, 'depth', okada),
            'sun69': (Sun69, 'depth', sun69)}


def check_forward_models():
    """Checks 1-4 on the raw torch forward models."""
    generator = torch.Generator().manual_seed(0)
    n_points, n_batch = 300, 4
    x_m = (torch.rand(n_points, generator=generator, dtype=torch.float64) - 0.5) * 20e3
    y_m = (torch.rand(n_points, generator=generator, dtype=torch.float64) - 0.5) * 20e3
    dz_random_m = torch.rand(n_points, generator=generator, dtype=torch.float64) * 1000.0  # 0-1 km relief
    sources = random_sources(n_batch, generator, torch.float64)

    print("[1/4] dz = 0 == flat (bit-identical or fp64 round-off)")
    for name, (model_class, depth_key, params) in sources.items():
        flat_out = model_class(x_m, y_m).run(**params)
        zero_out = model_class(x_m, y_m, dz_topo_m=torch.zeros(n_points, dtype=torch.float64)).run(**params)
        max_rel = max_relative_difference(flat_out, zero_out)
        report(f"{name}: dz=0 == flat", max_rel < 1e-12,
               'bit-identical' if max_rel == 0 else f"max rel diff = {max_rel:.1e}")

    print("[2/4] constant dz = c == flat at depth d + c")
    offset_m = 750.0
    for name, (model_class, depth_key, params) in sources.items():
        shifted = dict(params)
        shifted[depth_key] = params[depth_key] + offset_m
        flat_out = model_class(x_m, y_m).run(**shifted)
        topo_out = model_class(x_m, y_m, dz_topo_m=torch.full((n_points,), offset_m,
                                                               dtype=torch.float64)).run(**params)
        max_diff = float((flat_out - topo_out).abs().max())
        report(f"{name}: const dz", torch.allclose(flat_out, topo_out, rtol=1e-10, atol=1e-10),
               f"max |diff| = {max_diff:.2e} mm")

    print("[3/4] per-pixel dz vs NumPy models (pixel-by-pixel depth d + dz_i)")
    numpy_models = {'mogi': mogi_np, 'okada': okada_np, 'sun69': sun69_np}
    order = {'mogi': ['xcen', 'ycen', 'd', 'dV'],
             'okada': ['xoff', 'yoff', 'depth', 'strike', 'dip', 'length', 'width', 'opening'],
             'sun69': ['xcen', 'ycen', 'depth', 'radius', 'dV']}
    x_np, y_np, dz_np = x_m.numpy(), y_m.numpy(), dz_random_m.numpy()
    for name, (model_class, depth_key, params) in sources.items():
        torch_out = model_class(x_m, y_m, dz_topo_m=dz_random_m).run(**params).numpy()  # [B, 3N] mm
        max_rel = 0.0
        for batch_index in range(n_batch):
            values = {key: float(params[key][batch_index]) for key in order[name]}
            reference = np.zeros(3 * n_points)
            # Evaluate the NumPy model one pixel at a time with that pixel's own depth, so the
            # reference never relies on array-valued depth support in forward_numpy.
            for point_index in range(n_points):
                pixel_values = dict(values)
                pixel_values[depth_key] = values[depth_key] + dz_np[point_index]
                east, north, up = numpy_models[name](
                    x_np[point_index:point_index + 1], y_np[point_index:point_index + 1],
                    *[pixel_values[key] for key in order[name]])
                reference[[point_index, n_points + point_index, 2 * n_points + point_index]] = \
                    [east[0], north[0], up[0]]
            scale = np.abs(reference).max()
            max_rel = max(max_rel, np.abs(torch_out[batch_index] - reference).max() / scale)
        report(f"{name}: torch vs numpy", max_rel < 1e-8, f"max rel err = {max_rel:.1e}")

    print("[4/4] gradients finite with topography")
    for name, (model_class, depth_key, params) in sources.items():
        leaf_params = {key: value.clone().requires_grad_(True) for key, value in params.items()}
        output = model_class(x_m, y_m, dz_topo_m=dz_random_m).run(**leaf_params)
        output.pow(2).sum().backward()
        finite = all(torch.isfinite(value.grad).all() for value in leaf_params.values())
        nonzero_depth = bool(leaf_params[depth_key].grad.abs().sum() > 0)
        report(f"{name}: grads finite, d(depth) != 0", finite and nonzero_depth)


# --- Decoder-level helpers ---
def make_config(scene_path, topography=None, physics='Mogi_LOS', paras_key='mogi_paras',
                paras_file=None, dim_z_phy=4, n_sources=None):
    """Clone the COMSOL template config onto a scene, with an optional topography block."""
    with open(TEMPLATE_CONFIG) as f:
        config = json.load(f)
    args = config['arch']['args']
    args['insar']['timeseries'] = scene_path
    args['physics'] = physics
    if paras_file is not None:
        args.pop('mogi_paras', None)
        args[paras_key] = paras_file
    if topography is not None:
        args['topography'] = topography
    if n_sources is not None:
        args['n_sources'] = n_sources
    config['arch']['phys_vae']['dim_z_phy'] = dim_z_phy
    return config


DECODERS = [  # (label, point class, grid class, paras key, paras file, dim_z_phy)
    ('Mogi', Physics_Mogi_LOS, Physics_Mogi_LOS_Grid, 'mogi_paras', None, 4),
    ('Okada', Physics_Okada_LOS, Physics_Okada_LOS_Grid, 'okada_paras', OKADA_PARAS, 8),
    ('Sun69', Physics_Sun69_LOS, Physics_Sun69_LOS_Grid, 'sun69_paras', SUN69_PARAS, 5),
]


def check_decoders(workdir):
    """Checks 5-8 on the InSAR LOS decoders."""
    generator = torch.Generator().manual_seed(1)
    topo_on = {'mode': 'varying_depth', 'h_ref_m': 0.0}

    # Zero-height copy of the cone scene (same grid, height_m = 0) in the scratch workdir.
    zero_scene = os.path.join(workdir, 'wk6_cid1000_zeroheight.npz')
    with np.load(CONE_SCENE) as scene:
        arrays = {key: scene[key] for key in scene.files}
    arrays['height_m'] = np.zeros_like(arrays['height_m'])
    np.savez(zero_scene, **arrays)

    print("[5/8] zero-height scene: varying_depth decoder == flat decoder (fp32 round-off)")
    for label, point_class, grid_class, paras_key, paras_file, dim_z_phy in DECODERS:
        z_phy = torch.rand(3, dim_z_phy, generator=generator)
        for path_name, decoder_class in (('point', point_class), ('grid', grid_class)):
            flat_decoder = decoder_class(make_config(zero_scene, None, paras_key=paras_key,
                                                     paras_file=paras_file, dim_z_phy=dim_z_phy))
            topo_decoder = decoder_class(make_config(zero_scene, topo_on, paras_key=paras_key,
                                                     paras_file=paras_file, dim_z_phy=dim_z_phy))
            max_rel = max_relative_difference(flat_decoder(z_phy), topo_decoder(z_phy))
            report(f"{label} {path_name}: zero height == flat", max_rel < 1e-6,
                   'bit-identical' if max_rel == 0 else f"max rel diff = {max_rel:.1e} (float32)")

    print("[6/8] COMSOL cone scene")
    cone_config = make_config(CONE_SCENE, topo_on)
    mogi_point = Physics_Mogi_LOS(cone_config)
    mogi_grid = Physics_Mogi_LOS_Grid(cone_config)
    from model.model_phys_smpl import _load_insar_data
    scene_data = _load_insar_data(cone_config)
    with np.load(CONE_SCENE) as scene:
        x_axis_m, y_axis_m = scene['x_m'], scene['y_m']
    radius_m = np.hypot(scene_data.xE_pts * 1e3, scene_data.yN_pts * 1e3)
    profile = np.loadtxt('data/comsol/wk6_chambers_inversion_20260928/profiles/wk6_cid1000_cone.csv',
                         delimiter=',', skiprows=1)
    expected_height_m = np.interp(radius_m, profile[:, 0], profile[:, 1])
    height_err = float(np.abs(scene_data.height_pts_m - expected_height_m).max())
    report("loader height == revolved profile at coherent cells", height_err < 1e-2,
           f"max |diff| = {height_err:.2e} m; grid {x_axis_m.size}x{y_axis_m.size}")
    report("grid dz has no NaN", bool(torch.isfinite(mogi_grid.model.dz_topo_m).all()))

    # MultiMogi K=1 with topography == single-source Mogi_LOS with topography.
    if Physics_MultiMogi_LOS is None:
        print("  SKIP: MultiMogi K=1 check (Physics_MultiMogi_LOS not in this code base)")
    else:
        multi_decoder = Physics_MultiMogi_LOS(make_config(CONE_SCENE, topo_on, physics='MultiMogi_LOS',
                                                          n_sources=1))
        z_phy = torch.rand(3, 4, generator=generator)
        report("MultiMogi K=1 == Mogi_LOS (topography on)",
               torch.equal(multi_decoder(z_phy), mogi_point(z_phy)))

    # Grid decoder restricted to coherent cells == point decoder (same row-major mask order).
    grid_out = mogi_grid(z_phy)[:, torch.from_numpy(scene_data.mask_d.reshape(-1))]
    point_out = mogi_point(z_phy)
    report("grid == point decoder on coherent cells",
           torch.allclose(grid_out, point_out, rtol=1e-5, atol=1e-5),
           f"max |diff| = {float((grid_out - point_out).abs().max()):.1e} (standardized)")

    # Physical LOS (mm) vs NumPy Mogi with per-pixel depth d + h_i.
    params = mogi_point.rescale(z_phy)
    los_mm = (point_out * mogi_point.x_scale + mogi_point.x_mean).detach().numpy()
    los_vectors = np.stack([scene_data.losE_pts, scene_data.losN_pts, scene_data.losU_pts])
    max_rel = 0.0
    for batch_index in range(z_phy.shape[0]):
        east, north, up = mogi_np(scene_data.xE_pts.astype(np.float64) * 1e3,
                                  scene_data.yN_pts.astype(np.float64) * 1e3,
                                  float(params['xcen'][batch_index]), float(params['ycen'][batch_index]),
                                  float(params['d'][batch_index]) + scene_data.height_pts_m.astype(np.float64),
                                  float(params['dV'][batch_index]))
        reference = los_vectors[0] * east + los_vectors[1] * north + los_vectors[2] * up
        max_rel = max(max_rel, np.abs(los_mm[batch_index] - reference).max() / np.abs(reference).max())
    report("Mogi_LOS on cone == NumPy Mogi LOS with depth d + h_i", max_rel < 1e-4,
           f"max rel err = {max_rel:.1e} (float32 decoder)")

    print("[7/8] positivity guard")
    try:
        Physics_Mogi_LOS(make_config(CONE_SCENE, {'mode': 'varying_depth', 'h_ref_m': 'max'}))
        report("raises when d_min + min(h - h_ref) <= 0", False, "no error raised")
    except ValueError as error:
        report("raises when d_min + min(h - h_ref) <= 0", 'effective depth' in str(error))

    print("[8/8] config errors")
    try:
        Physics_Mogi_LOS(make_config(CONE_SCENE, {'mode': 'bogus'}))
        report("unknown mode raises", False)
    except ValueError:
        report("unknown mode raises", True)
    try:
        Physics_Mogi_LOS(make_config(FLAT_SCENE_NO_HEIGHT, topo_on))
        report("varying_depth without elevation raises", False)
    except ValueError as error:
        report("varying_depth without elevation raises", 'no elevation' in str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', default=None, help='scratch dir for the zero-height scene')
    args = parser.parse_args()
    for required in (TEMPLATE_CONFIG, CONE_SCENE, FLAT_SCENE_NO_HEIGHT):
        if not os.path.exists(required):
            raise FileNotFoundError(required)
    workdir = args.workdir or tempfile.mkdtemp(prefix='test_topography_')
    os.makedirs(workdir, exist_ok=True)

    check_forward_models()
    check_decoders(workdir)

    n_failed = RESULTS.count(False)
    print(f"Done. {len(RESULTS) - n_failed}/{len(RESULTS)} checks passed.")
    sys.exit(1 if n_failed else 0)


if __name__ == '__main__':
    main()
