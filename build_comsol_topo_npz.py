#!/usr/bin/env python
# Usage:       python build_comsol_topo_npz.py [--surface cone] [--out-dir data/comsol/wk6_per_chamber_topo]
#                  [--fig figures/comsol_wk6_topo/cone_height_map.png]
# Description: Adds a surface-elevation grid (key 'height_m', metres) to the COMSOL wk6
#              per-chamber scenes, so PILA's topography correction (arch.args.topography =
#              varying_depth) can be tested on the cone runs. The elevation is the COMSOL cone
#              profile z_m(r_m) revolved about the chamber axis onto the map grid. Writes NEW
#              npz files (all original keys copied unchanged + height_m); never overwrites.
# Date:        2026-10-12
#
# Assumptions (READ):
#   * The cone surface z(r) is identical for all 40 chambers (verified: max |dz| = 0 across
#     profiles/wk6_cid*_cone.csv); it is checked again here and the script fails otherwise.
#   * Elevation datum = the flat base of the COMSOL model (z = 0), the same datum as the
#     truth column z_centroid_m, so PILA depth with h_ref_m = 0 compares directly to
#     -z_centroid_m. Summit = 1000 m at r = 0, base reached at r ~ 2747 m (~20 deg slope).
#   * The displacement maps themselves are untouched: COMSOL placed cone displacements by
#     radial coordinate only, consistent with PILA keeping x/y unchanged under topography.
#   * Flat-surface scenes would get height_m = 0 everywhere, which the unit tests already
#     prove is bit-identical to the flat model, so only the cone is built by default.

import argparse
import glob
import os
import time

import numpy as np

# --- Project-root-relative paths (run from the repo root) ---
CATALOGUE_DIR = 'data/comsol/wk6_chambers_inversion_20260928'
PROFILE_GLOB = os.path.join(CATALOGUE_DIR, 'profiles', 'wk6_cid*_{surface}.csv')
PER_CHAMBER_DIR = 'data/comsol/wk6_per_chamber'          # existing single-chamber scenes (read only)


def load_surface_profile(surface):
    """
    Read the COMSOL surface elevation profile z(r), checking it is shared by every chamber.

    Inputs : surface str ('cone' or 'flat')
    Output : (radius_m [M], elevation_m [M]) float64, radius increasing.
    """
    profile_files = sorted(glob.glob(PROFILE_GLOB.format(surface=surface)))
    if not profile_files:
        raise FileNotFoundError(PROFILE_GLOB.format(surface=surface))
    reference = np.loadtxt(profile_files[0], delimiter=',', skiprows=1)   # r_m, z_m, w_m, u_m
    for profile_file in profile_files[1:]:
        table = np.loadtxt(profile_file, delimiter=',', skiprows=1)
        # The revolved height grid is only valid if every chamber has the same surface.
        if table.shape != reference.shape or np.max(np.abs(table[:, :2] - reference[:, :2])) > 1e-6:
            raise ValueError(f"surface profile of {profile_file} differs from {profile_files[0]}")
    radius_m, elevation_m = reference[:, 0], reference[:, 1]
    if np.any(np.diff(radius_m) <= 0):
        raise ValueError("profile radius must be strictly increasing for np.interp")
    print(f"  {len(profile_files)} {surface} profiles share one surface: "
          f"summit {elevation_m.max():.1f} m, r_max {radius_m.max():.0f} m")
    return radius_m, elevation_m


def revolve_profile(x_m, y_m, radius_m, elevation_m):
    """
    Revolve an axisymmetric elevation profile about (0, 0) onto a map grid.

    Inputs : x_m [W], y_m [H] map axes (m); radius_m, elevation_m profile (m).
    Output : height_m [H, W] float32, metres; layout map[iy, ix] like the displacement maps.
    """
    x_grid_m, y_grid_m = np.meshgrid(x_m, y_m)                 # [H, W], row index = y
    radial_distance_m = np.hypot(x_grid_m, y_grid_m)
    if radial_distance_m.max() > radius_m.max():
        raise ValueError("map extends beyond the last profile radius")
    return np.interp(radial_distance_m, radius_m, elevation_m).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--surface', default='cone', choices=['cone', 'flat'])
    parser.add_argument('--out-dir', default='data/comsol/wk6_per_chamber_topo')
    parser.add_argument('--fig', default='figures/comsol_wk6_topo/cone_height_map.png')
    args = parser.parse_args()
    t0 = time.time()

    # --- [1/3] Surface profile ---
    print(f"[1/3] Reading {args.surface} surface profiles ...")
    radius_m, elevation_m = load_surface_profile(args.surface)

    # --- [2/3] Write per-chamber scenes with height_m ---
    scene_files = sorted(glob.glob(os.path.join(PER_CHAMBER_DIR, f'wk6_cid*_{args.surface}.npz')))
    if not scene_files:
        raise FileNotFoundError(f"no per-chamber scenes in {PER_CHAMBER_DIR} "
                                f"(run generate_comsol_jobs.py first)")
    print(f"[2/3] Adding height_m to {len(scene_files)} scenes -> {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    n_written, n_skipped = 0, 0
    height_m = None
    for scene_file in scene_files:
        out_file = os.path.join(args.out_dir, os.path.basename(scene_file))
        if os.path.exists(out_file):
            # Never overwrite: a stale file must be moved aside by hand.
            print(f"  exists, skipped: {out_file}")
            n_skipped += 1
            continue
        with np.load(scene_file) as scene:
            arrays = {key: scene[key] for key in scene.files}
        if 'height_m' in arrays:
            raise ValueError(f"{scene_file} already has height_m")
        height_m = revolve_profile(arrays['x_m'], arrays['y_m'], radius_m, elevation_m)
        arrays['height_m'] = height_m
        np.savez_compressed(out_file, **arrays)
        n_written += 1
    print(f"  wrote {n_written}, skipped {n_skipped} existing")

    # --- [3/3] Sanity checks + height map figure ---
    print("[3/3] Sanity checks ...")
    check_file = os.path.join(args.out_dir, os.path.basename(scene_files[0]))
    with np.load(scene_files[0]) as source, np.load(check_file) as built:
        x_m, y_m, height_m = built['x_m'], built['y_m'], built['height_m']
        print(f"  Loaded: height_m shape={height_m.shape}, dtype={height_m.dtype}")
        # Displacement arrays must be byte-identical to the source scene.
        for key in source.files:
            assert np.array_equal(source[key], built[key]), f"{key} changed"
    radial_distance_m = np.hypot(*np.meshgrid(x_m, y_m))
    centre = np.unravel_index(np.argmin(radial_distance_m), radial_distance_m.shape)
    checks = {
        'summit at (0,0) == profile max': np.isclose(height_m[centre], elevation_m.max(), atol=1.0),
        'height 0 beyond cone base': np.all(height_m[radial_distance_m > 2800.0] == 0.0),
        'radially symmetric (flip x/y)': (np.allclose(height_m, height_m[::-1, :])
                                          and np.allclose(height_m, height_m[:, ::-1])),
        'no NaN': np.all(np.isfinite(height_m)),
    }
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    if not all(checks.values()):
        raise SystemExit("sanity checks failed")

    if args.fig:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(args.fig), exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        image = ax.imshow(height_m, origin='lower', cmap='terrain',
                          extent=[x_m[0] / 1e3, x_m[-1] / 1e3, y_m[0] / 1e3, y_m[-1] / 1e3])
        fig.colorbar(image, ax=ax, label='Surface elevation (m)')
        ax.set_xlabel('East (km)')
        ax.set_ylabel('North (km)')
        ax.set_title(f'COMSOL wk6 {args.surface} surface (revolved profile)')
        plt.tight_layout()
        fig.savefig(args.fig, dpi=150)
        plt.close(fig)
        print(f"  figure: {args.fig}")

    print(f"Done in {time.time() - t0:.1f}s. Output saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
