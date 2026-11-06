#!/usr/bin/env python
# Usage:       python generate_seq_dvbox_jobs.py [--dry-run]
# Description: Clone the standard-recipe sequential multi-source PBS jobs (std 19, clean 21, nonoise 5) into
#              jobs/multisource_seq_dvbox/, each album now run with --amp-box (data-scaled dV prior box) and saved under
#              saved/multisource_seq_dvbox/<tag>/ so nothing is overwritten. Everything else (K = 1..3, recipe, seeds,
#              epochs, resources) is copied verbatim. Prints each album's LOS std and resulting box for review.
# Date:        2026-10-22

import argparse
import glob
import os
import re

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_SETS = {                          # tag -> source PBS dir
    'std': 'jobs/multisource_seq_std',
    'clean': 'jobs/multisource_seq_clean',
    'nonoise': 'jobs/multisource_seq_nonoise',
}
OUT_DIR = 'jobs/multisource_seq_dvbox'
SAVE_ROOT = 'saved/multisource_seq_dvbox'
STANDARD_DV_MAX_M3, ANCHOR_STD_MM = 2.0e8, 12.0      # same constants as utils/amp_box.py


def album_std_mm(album_npz):
    """Global LOS std (mm) of a dense album (== the loader's x_scale_global; checked against the loader on weak_n012)."""
    with np.load(album_npz) as f:
        los_mm = f['los_asc_m'].astype(np.float64) * 1e3
    std_mm = float(np.nanstd(los_mm[np.isfinite(los_mm)]))
    return std_mm if std_mm >= 1e-6 else 1.0          # loader falls back to 1.0 for an all-zero scene


def rewrite_line(line, tag):
    """Point one run_sequential_album.py call at the dvbox save root/tag and add --amp-box."""
    line = re.sub(r'\s--save-root\s+\S+', '', line)
    line = re.sub(r'--tag\s+\S+', f'--tag {tag}', line)
    if '--tag' not in line:
        line += f' --tag {tag}'
    return line + f' --save-root {SAVE_ROOT} --amp-box'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print(f"[1/2] Cloning {list(SOURCE_SETS)} into {OUT_DIR}/ ...")
    os.makedirs(os.path.join(PROJECT_ROOT, OUT_DIR), exist_ok=True)
    qsub_lines, rows = [], []
    for tag, src_dir in SOURCE_SETS.items():
        for src in sorted(glob.glob(os.path.join(PROJECT_ROOT, src_dir, '*.pbs'))):
            name = 'dvbox_' + os.path.basename(src)[:-4]
            out_rel = os.path.join(OUT_DIR, name + '.pbs')
            lines = []
            for line in open(src).read().splitlines():
                old_name = os.path.basename(src)[:-4]
                if 'run_sequential_album.py' in line:
                    line = rewrite_line(line, tag)
                    album = re.search(r'--album\s+(\S+)', line).group(1)
                    std_mm = album_std_mm(os.path.join(PROJECT_ROOT, album))
                    rows.append((tag, name, os.path.basename(album)[:-4], std_mm, STANDARD_DV_MAX_M3 * std_mm / ANCHOR_STD_MM))
                elif line.startswith('# Usage:'):
                    line = f'# Usage:       qsub {out_rel}'
                elif line.startswith('# Date:'):
                    line = '# Date:        2026-10-22 (clone of ' + os.path.relpath(src, PROJECT_ROOT) + ' + --amp-box)'
                else:
                    line = line.replace(old_name, name)        # #PBS -N / -o log / echo tags
                lines.append(line)
            if not args.dry_run:
                out_abs = os.path.join(PROJECT_ROOT, out_rel)
                if os.path.exists(out_abs):
                    raise FileExistsError(out_abs)
                with open(out_abs, 'w') as handle:
                    handle.write('\n'.join(lines) + '\n')
            qsub_lines.append(f'qsub {out_rel}')

    print(f"[2/2] {len(qsub_lines)} PBS, {len(rows)} albums:")
    print(f"  {'tag':8s} {'job':22s} {'album':34s} {'std_mm':>8s} {'box_m3':>10s}")
    for tag, name, album, std_mm, box in rows:
        print(f"  {tag:8s} {name:22s} {album:34s} {std_mm:8.3f} {box:10.3e}")
    if not args.dry_run:
        with open(os.path.join(PROJECT_ROOT, OUT_DIR, 'submit_all.sh'), 'w') as handle:
            handle.write('#!/bin/bash\nset -e\ncd /eos-rs/INSAR_processing/denny/PILA\n' + '\n'.join(qsub_lines) + '\n')
    print(f"Done. {'(dry run) ' if args.dry_run else ''}Submit with: bash {OUT_DIR}/submit_all.sh")


if __name__ == '__main__':
    main()
