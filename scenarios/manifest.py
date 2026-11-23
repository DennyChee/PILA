#!/usr/bin/env python
# Usage:       from scenarios.manifest import load_manifest, add_scenario, resolve_row
#              python -m scenarios.manifest --check [--manifest scenarios/manifest.csv]
# Description: The scenario manifest: one CSV row per scenario, the single list every method adapter and the
#              scorer iterate over. Synthetic rows point to a scenario npz + truth json (scenarios/schema.py);
#              real rows point to the MintPy files and record the preparation settings that EVERY method applies
#              identically (each method reads MintPy itself -- user decision 2026-11-23).
#              Paths are repo-relative or use ${NAME} tokens from paths.yaml (pila/paths.py).
# Columns:     scenario_id, kind (synthetic|real), family, models ('mogi;mogi'), n_sources, tracks ('asc' |
#              'asc;desc'), split (test|train|val), scenario_path, truth_path, sha256 (of the npz),
#              timeseries, geometry, mask, lat0, lon0, multilook, ref_date, bbox, coh_valid_frac, notes
# Date:        2026-11-23
import argparse
import csv
import hashlib
import os

from pila.paths import REPO_ROOT, resolve_tokens

DEFAULT_MANIFEST = os.path.join(REPO_ROOT, 'scenarios', 'manifest.csv')
COLUMNS = ['scenario_id', 'kind', 'family', 'models', 'n_sources', 'tracks', 'split', 'scenario_path', 'truth_path',
           'sha256', 'timeseries', 'geometry', 'mask', 'lat0', 'lon0', 'multilook', 'ref_date', 'bbox',
           'coh_valid_frac', 'notes']
KINDS = ('synthetic', 'real')
SPLITS = ('test', 'train', 'val')       # train/val reserved for the (on-hold) supervised method


def sha256_file(path, chunk_bytes=1 << 20):
    """SHA-256 of a file (read in 1 MB chunks), to detect a scenario silently changing after registration."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value):
    """Expand ${NAME} tokens and make repo-relative paths absolute ('' stays '')."""
    if not value:
        return ''
    value = resolve_tokens(value)
    return value if os.path.isabs(value) else os.path.join(REPO_ROOT, value)


def load_manifest(path=DEFAULT_MANIFEST):
    """Read the manifest into a list of row dicts (empty list if the file does not exist yet)."""
    if not os.path.exists(path):
        return []
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if list(row) != COLUMNS:
            raise ValueError(f"{path}: columns {list(row)} != {COLUMNS}")
        break
    return rows


def resolve_row(row):
    """Row with every path column resolved to an absolute path (tokens expanded)."""
    out = dict(row)
    for key in ('scenario_path', 'truth_path', 'timeseries', 'geometry', 'mask'):
        if key == 'geometry' and row.get('kind') == 'synthetic':
            continue                                   # synthetic npz: geometry column holds the track token
        out[key] = resolve_path(row.get(key, ''))
    return out


def add_scenario(row, path=DEFAULT_MANIFEST):
    """
    Append one scenario row (refuses a duplicate scenario_id). For synthetic rows the npz sha256 is computed here.

    Input:  row dict with any subset of COLUMNS (missing columns -> '').
    Output: the stored row.
    """
    rows = load_manifest(path)
    full = {key: str(row.get(key, '')) for key in COLUMNS}
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown manifest columns {sorted(unknown)}")
    if any(existing['scenario_id'] == full['scenario_id'] for existing in rows):
        raise ValueError(f"scenario_id {full['scenario_id']!r} already in {path}")
    if full['kind'] not in KINDS or full['split'] not in SPLITS:
        raise ValueError(f"kind must be in {KINDS} and split in {SPLITS}: {full['kind']!r}, {full['split']!r}")
    if full['kind'] == 'synthetic' and not full['sha256']:
        full['sha256'] = sha256_file(resolve_path(full['scenario_path']))
    new_file = not os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(full)
    return full


def check_manifest(path=DEFAULT_MANIFEST):
    """Return a list of problems: missing files, changed sha256, invalid scenario/truth files."""
    from scenarios.schema import read_truth, validate_scenario
    problems = []
    for row in load_manifest(path):
        resolved, sid = resolve_row(row), row['scenario_id']
        if row['kind'] == 'synthetic':
            if not os.path.exists(resolved['scenario_path']):
                problems.append(f"{sid}: missing {resolved['scenario_path']}")
                continue
            if sha256_file(resolved['scenario_path']) != row['sha256']:
                problems.append(f"{sid}: sha256 changed since registration")
            problems += [f"{sid}: {p}" for p in validate_scenario(resolved['scenario_path'])]
            if row['truth_path']:
                try:
                    read_truth(resolved['truth_path'])
                except (OSError, ValueError) as exc:
                    problems.append(f"{sid}: truth: {exc}")
        else:
            for key in ('timeseries', 'geometry'):
                if not os.path.exists(resolved[key]):
                    problems.append(f"{sid}: missing {key} {resolved[key]}")
    return problems


def main():
    parser = argparse.ArgumentParser(description='Check the scenario manifest')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    problems = check_manifest(args.manifest)
    print(f"{len(rows)} scenarios in {args.manifest}")
    print("OK" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
    raise SystemExit(0 if not problems else 1)


if __name__ == '__main__':
    main()
