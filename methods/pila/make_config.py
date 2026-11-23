#!/usr/bin/env python
# Usage:       python methods/pila/make_config.py --scenario-id <id> [--manifest scenarios/manifest.csv] \
#                  [--model mogi] [--n-sources K] [--overlay multisource_k2] [--seed 1] [--epochs N] --out cfg.json
# Description: Build a PILA run config for one manifest scenario from the ONE standard template
#              (configs/templates/pila_standard.json) plus optional named overlays
#              (configs/templates/overlay_<name>.json). Fills the physics-specific keys (physics, prior file,
#              dim_z_phy, n_sources) and the insar block (synthetic npz: path + track token; real: MintPy files and
#              the manifest's preparation settings, identical for every method). Every run is seeded explicitly.
# Notes:       Which recipe/overlay and which prior box are THE PILA setting of the comparison is a Phase 4
#              decision; the defaults here are placeholders that reproduce the legacy standard cell.
# Date:        2026-11-23
import argparse
import copy
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from scenarios.manifest import DEFAULT_MANIFEST, load_manifest, resolve_row  # noqa: E402

TEMPLATE = os.path.join(REPO_ROOT, 'configs', 'templates', 'pila_standard.json')
# model -> (PILA physics for K=1, physics for K>1, prior-file config key, default prior file, params per source)
MODELS = {
    'mogi':  ('MultiMogi_LOS', 'MultiMogi_LOS', 'mogi_paras', 'configs/mogi_paras_symdV.json', 4),
    'sun69': ('Sun69_LOS', None, 'sun69_paras', 'configs/sierranegra_sun69_paras_symdV.json', 5),
    'okada': ('Okada_LOS', None, 'okada_paras', 'configs/okada_dike_paras.json', 8),
}
REPLACE_WHOLE = ('lr_scheduler', 'optimizer')        # overlay replaces these blocks entirely (no stale args)


def deep_merge(base, overlay):
    """Recursively merge overlay into a copy of base (dicts merged, other values replaced)."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if key.startswith('_'):
            continue                                   # documentation keys
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key not in REPLACE_WHOLE:
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def insar_block(row):
    """The insar dict PILA's loader reads, from a resolved manifest row."""
    if row['kind'] == 'synthetic':
        return dict(timeseries=row['scenario_path'], geometry=row['geometry'] or row['tracks'].split(';')[0],
                    lat0=0.0, lon0=0.0, multilook=1, coh_valid_frac=0.5, verbose=False)
    block = dict(timeseries=row['timeseries'], geometry=row['geometry'], mask=row['mask'] or None,
                 lat0=float(row['lat0']), lon0=float(row['lon0']), multilook=int(row['multilook'] or 1),
                 coh_valid_frac=float(row['coh_valid_frac'] or 0.5), verbose=False)
    if row['ref_date']:
        block['ref_date'] = row['ref_date']
    if row['bbox']:
        block['bbox'] = [float(v) for v in row['bbox'].split(';')]
    return block


def make_config(row, model, n_sources, save_dir, overlays=(), seed=1, epochs=None, prior_file=None):
    """
    Build the config dict for one scenario row.

    Inputs:  row (manifest row); model ('mogi'|'sun69'|'okada'); n_sources K (oracle K from the manifest by default);
             save_dir; overlays (names); seed; epochs override; prior_file override (repo-relative).
    """
    physics_k1, physics_multi, prior_key, default_prior, n_per_source = MODELS[model]
    if n_sources > 1 and physics_multi is None:
        raise ValueError(f"PILA has no multi-source {model} physics (only Mogi supports K > 1)")
    with open(TEMPLATE) as handle:
        config = json.load(handle)
    config.pop('_template_note', None)
    for name in overlays:
        with open(os.path.join(REPO_ROOT, 'configs', 'templates', f'overlay_{name}.json')) as handle:
            config = deep_merge(config, json.load(handle))
    resolved = resolve_row(row)
    block = insar_block(resolved)
    config['name'] = row['scenario_id']
    config['seed'] = int(seed)
    config['arch']['args'].update(physics=physics_multi if n_sources > 1 or physics_k1 == 'MultiMogi_LOS' else physics_k1,
                                  **{prior_key: prior_file or default_prior})
    if config['arch']['args']['physics'] == 'MultiMogi_LOS':
        config['arch']['args']['n_sources'] = int(n_sources)
    config['arch']['phys_vae']['dim_z_phy'] = int(n_per_source * n_sources)
    config['arch']['args']['insar'] = copy.deepcopy(block)
    config['data_loader']['args']['insar'] = copy.deepcopy(block)
    config['data_loader']['data_dir_valid'] = copy.deepcopy(block)
    config['data_loader']['data_dir_test'] = copy.deepcopy(block)
    config['trainer']['save_dir'] = save_dir
    if epochs is not None:
        config['trainer']['epochs'] = config['trainer']['early_stop'] = config['trainer']['save_period'] = int(epochs)
        basin = config['trainer']['phys_vae'].get('basin_hop')
        if basin:
            basin['total_epochs'] = int(epochs)
    basin = config['trainer']['phys_vae'].get('basin_hop')
    if basin is not None:
        basin['seed'] = int(seed)
    leftover = [k for k, v in _walk(config) if isinstance(v, str) and v.startswith('__') and v.endswith('__')]
    if leftover:
        raise ValueError(f"unfilled template placeholders: {leftover}")
    return config


def _walk(obj, prefix=''):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _walk(value, f'{prefix}{key}.')
    else:
        yield prefix.rstrip('.'), obj


def main():
    parser = argparse.ArgumentParser(description='Build a PILA config for a manifest scenario')
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    parser.add_argument('--model', default=None, help='default: first model in the manifest row')
    parser.add_argument('--n-sources', type=int, default=None, help='default: manifest n_sources (oracle K)')
    parser.add_argument('--overlay', action='append', default=[])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--prior-file', default=None)
    parser.add_argument('--save-dir', default=None, help='default: runs/pila/<scenario_id>')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    rows = {row['scenario_id']: row for row in load_manifest(args.manifest)}
    if args.scenario_id not in rows:
        raise KeyError(f"{args.scenario_id} not in {args.manifest}")
    row = rows[args.scenario_id]
    model = args.model or row['models'].split(';')[0]
    n_sources = args.n_sources if args.n_sources is not None else int(row['n_sources'])
    save_dir = args.save_dir or os.path.join('runs', 'pila', args.scenario_id)
    config = make_config(row, model, n_sources, save_dir, args.overlay, args.seed, args.epochs, args.prior_file)
    if os.path.exists(args.out):
        raise FileExistsError(f"{args.out} exists; not overwriting")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(config, handle, indent=2)
    print(f"Done. Output saved to {args.out} (model={model}, K={n_sources}, overlays={args.overlay}, seed={args.seed})")


if __name__ == '__main__':
    main()
