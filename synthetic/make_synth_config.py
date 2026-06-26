#!/usr/bin/env python
# Usage:       python -m synthetic.make_synth_config \
#                  --template configs/phys_smpl/SierraNegra_Sun69_InSAR_A.json \
#                  --synth-h5 synthetic/cubes/<name>/<name>.h5 \
#                  --out configs/synth/<name>.json
# Description: Template an existing PILA InSAR config to point at a SYNTHETIC
#              timeseries cube while keeping the real geometry/mask/origin. Only
#              data paths, the experiment name, the save_dir, and input_dim change;
#              the model architecture and parameter bounds are untouched. input_dim
#              is recomputed from the synthetic cube (its coherent-cell count can
#              differ from the real scene's, because the synthetic cube is finite
#              everywhere whereas the real cube may have NaNs inside the mask), and
#              Physics_*_LOS asserts input_dim == N.
# Date:        2026-06-22

import argparse
import copy
import json
import os

from datasets.preprocessing.insar_mintpy import load_insar_mintpy


# Map the physics type to the *_paras config key the arch.args carries.
PARAS_KEY = {'Mogi_LOS': 'mogi_paras', 'Okada_LOS': 'okada_paras',
             'Sun69_LOS': 'sun69_paras'}


def _set_insar_timeseries(insar_block, synth_h5, base_scene=None):
    """
    Repoint the timeseries at the synthetic cube. If base_scene is given, also set
    geometry/mask/lat0/lon0 from it (needed for Mode B idealized grids whose geometry
    differs from the template's real scene; harmless for real-geometry scenarios where
    base_scene matches the template).
    """
    out = copy.deepcopy(insar_block)
    out['timeseries'] = synth_h5
    if base_scene is not None:
        out['geometry'] = base_scene['geometry']
        if base_scene.get('mask') is not None:
            out['mask'] = base_scene['mask']
        out['lat0'] = base_scene['lat0']
        out['lon0'] = base_scene['lon0']
    return out


def _set_multilook(insar_block, multilook):
    insar_block['multilook'] = int(multilook)


def _set_standardization(insar_block, std_mode, far_field):
    if std_mode is not None:
        insar_block['std_mode'] = std_mode
    if far_field is not None:
        insar_block['far_field'] = far_field


def make_config(template_path, synth_h5, out_path, name=None, paras=None,
                save_dir=None, epochs=None, multilook=None, base_scene=None,
                std_mode=None, far_field=None):
    """
    Build a synthetic-data config from a template and write it to out_path.

    Returns the config dict and the computed input_dim (N coherent cells).
    """
    with open(template_path, 'r') as f:
        cfg = json.load(f)

    synth_h5 = os.path.abspath(synth_h5)
    if not os.path.exists(synth_h5):
        raise FileNotFoundError(synth_h5)

    name = name or os.path.splitext(os.path.basename(out_path))[0]
    cfg['name'] = name

    args = cfg['arch']['args']

    # --- Repoint every insar block at the synthetic timeseries (+ base geometry) ---
    args['insar'] = _set_insar_timeseries(args['insar'], synth_h5, base_scene)
    dl = cfg['data_loader']
    dl['args']['insar'] = _set_insar_timeseries(dl['args']['insar'], synth_h5, base_scene)
    for key in ('data_dir_valid', 'data_dir_test'):
        if key in dl:
            dl[key] = _set_insar_timeseries(dl[key], synth_h5, base_scene)

    # --- Optionally override multilook + standardization mode on all insar blocks ---
    for blk in (args['insar'], dl['args']['insar'],
                dl.get('data_dir_valid'), dl.get('data_dir_test')):
        if blk is None:
            continue
        if multilook is not None:
            _set_multilook(blk, multilook)
        _set_standardization(blk, std_mode, far_field)

    # --- Optionally swap the parameter-bounds file (e.g. widened synth bounds) ---
    if paras is not None:
        paras_key = PARAS_KEY.get(args['physics'])
        if paras_key is None:
            raise ValueError(f"Unsupported physics '{args['physics']}'")
        args[paras_key] = paras

    # --- Recompute input_dim from the synthetic cube ---
    ins = args['insar']
    data = load_insar_mintpy(ins['timeseries'], ins['geometry'], ins.get('mask'),
                             ins['lat0'], ins['lon0'],
                             multilook=ins.get('multilook', 20),
                             coh_valid_frac=ins.get('coh_valid_frac', 0.5),
                             verbose=False)
    n_points = data.n_points
    args['input_dim'] = n_points

    # --- save_dir + epochs ---
    cfg['trainer']['save_dir'] = save_dir or os.path.join('saved', 'synth', name)
    if epochs is not None:
        cfg['trainer']['epochs'] = int(epochs)
        if 'lr_scheduler' in cfg and 'args' in cfg['lr_scheduler'] \
                and 'T_max' in cfg['lr_scheduler']['args']:
            cfg['lr_scheduler']['args']['T_max'] = int(epochs)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(cfg, f, indent=4)
    print(f"Wrote {out_path}")
    print(f"  physics={args['physics']}  input_dim(N coherent cells)={n_points}  "
          f"save_dir={cfg['trainer']['save_dir']}")
    return cfg, n_points


def main():
    ap = argparse.ArgumentParser(description="Template a PILA config onto a synthetic cube.")
    ap.add_argument('--template', required=True, help='Existing configs/phys_smpl/*.json to clone.')
    ap.add_argument('--synth-h5', required=True, help='Synthetic timeseries h5 to point at.')
    ap.add_argument('--out', required=True, help='Output config path (configs/synth/*.json).')
    ap.add_argument('--name', default=None, help='Experiment name (default: out basename).')
    ap.add_argument('--paras', default=None, help='Override *_paras.json bounds file.')
    ap.add_argument('--save-dir', default=None, help='Override trainer.save_dir.')
    ap.add_argument('--epochs', type=int, default=None, help='Override trainer.epochs.')
    ap.add_argument('--multilook', type=int, default=None, help='Override insar multilook factor.')
    ap.add_argument('--spec', default=None,
                    help='Scenario spec JSON; supplies base_scene (geometry/mask/lat0/lon0) — '
                         'REQUIRED when the synthetic grid differs from the template scene '
                         "(e.g. Mode-B Marapi grid), and the paras if --paras not given.")
    ap.add_argument('--std-mode', default='global', choices=['global', 'far_field_robust'],
                    help="Standardization: 'global' (default) or 'far_field_robust'.")
    ap.add_argument('--ff-radius', type=float, default=8.0,
                    help='Far-field exclusion radius (km); used with --std-mode far_field_robust.')
    args = ap.parse_args()
    base_scene = None
    paras = args.paras
    spec = None
    if args.spec:
        with open(args.spec) as f:
            spec = json.load(f)
        base_scene = spec.get('base_scene')
        paras = paras or spec.get('paras')

    # Build the far-field exclusion disk from the spec's source location (km).
    far_field = None
    if args.std_mode == 'far_field_robust':
        if spec is None:
            raise SystemExit("--std-mode far_field_robust needs --spec (for the source location).")
        geom = spec['trajectory'].get('geometry') or spec['trajectory'].get('start') or {}
        far_field = {'source_xE_km': float(geom.get('xcen', geom.get('xoff', 0.0))) / 1000.0,
                     'source_yN_km': float(geom.get('ycen', geom.get('yoff', 0.0))) / 1000.0,
                     'radius_km': float(args.ff_radius)}

    make_config(args.template, args.synth_h5, args.out, name=args.name,
                paras=paras, save_dir=args.save_dir, epochs=args.epochs,
                multilook=args.multilook, base_scene=base_scene,
                std_mode=args.std_mode, far_field=far_field)


if __name__ == '__main__':
    main()
