#!/usr/bin/env python
# Usage:       python -m synthetic.multisource.noise_covariance [--noise-dir synthetic/multisource/pila_stacks_v2/noise_sets]
#                  [--multilook 4] [--out synthetic/multisource/pila_stacks_v2/noise_covariance.npz]
#              from synthetic.multisource.noise_covariance import NoiseCovariance
#              cov = NoiseCovariance.load(path); chi2 = cov.chi2(residual_maps_mm)   # [T] values
# Description: Step 2 of the sequential multi-source plan. Spatial noise covariance for the whitened
#              "keep this source?" misfit, estimated from the CALIBRATION noise sets (other volcanoes'
#              ERA5 + turbulence + white + ramp) after the same per-acquisition offset + planar ramp
#              removal that PILA's loss uses, on a 4x multilooked grid (128 -> 32 px, ~1.5 km pixels):
#                  C = (1 - alpha) S + alpha T + floor I
#              S = sample covariance of the 1170 multilooked maps (captures the non-stationarity that
#              ramp removal introduces), T = stationary model built from the binned empirical
#              covariance vs distance (eigenvalue-clipped to PSD), floor = nugget for fine-scale noise of
#              unseen tracks. alpha and floor are chosen by held-out Gaussian log-likelihood (sets
#              C01-C30 fit, C31-C39 = 3 unseen tracks held out). chi2(r) = r^T C^-1 r.
#              (A single exponential model was tried first: it overshoots the zero-lag variance and
#              misses the negative lobe from ramp removal; whitened noise variance 0.30 instead of ~1.)
# Scientific notes:
#   * Long-wavelength APS is spatially correlated; treating it as white makes every extra source look
#     significant. Whitening down-weights broad patterns that the noise routinely produces.
#   * Acquisitions of one album share the reference-day APS, so they are NOT independent; chi2 summed
#     over acquisitions is therefore not chi2-distributed. The keep/stop bar is calibrated empirically
#     on noise-only albums (sequential_select_k.py), so this only needs to be a sensible weighting.
#   * Offset + ramp removal makes C singular in 3 directions; a small ridge (1e-6 sigma^2) keeps the
#     Cholesky stable, and residuals are always ramp-removed first so those directions carry no data.
# Date:        2026-10-06

import argparse
import glob
import json
import os
import time

import numpy as np



def remove_ramp_maps(maps, x_km, y_km):
    """Remove each map's least-squares offset + planar ramp. maps [T, H, W]; x_km/y_km [H, W]."""
    G = np.stack([np.ones(x_km.size), x_km.ravel(), y_km.ravel()], axis=1)
    flat = maps.reshape(maps.shape[0], -1)
    return (flat - (flat @ np.linalg.pinv(G).T) @ G.T).reshape(maps.shape)


def multilook(maps, factor):
    """Block-mean multilook [T, H, W] -> [T, H/f, W/f]."""
    n, h, w = maps.shape
    return maps[:, :h - h % factor, :w - w % factor].reshape(n, h // factor, factor, w // factor, factor).mean(axis=(2, 4))


class NoiseCovariance:
    """Covariance matrix on the multilooked grid with a Cholesky factor for whitening."""

    def __init__(self, C, x_km, y_km, multilook_factor, meta=None):
        self.C = np.asarray(C, dtype=np.float64)
        self.multilook_factor = int(multilook_factor)
        self.x_km_full, self.y_km_full = x_km, y_km                     # full-resolution grid (km)
        self.meta = meta or {}
        ridge = 1e-6 * float(np.mean(np.diag(self.C)))                   # see notes (ramp null space)
        self.chol = np.linalg.cholesky(self.C + ridge * np.eye(len(self.C)))

    def project(self, maps_full_mm):
        """[T, H, W] full-res maps (mm) -> ramp-removed, multilooked vectors [T, N_ml]."""
        r = remove_ramp_maps(maps_full_mm, self.x_km_full, self.y_km_full)
        return multilook(r, self.multilook_factor).reshape(r.shape[0], -1)

    def whiten(self, maps_full_mm):
        """[T, H, W] full-resolution residuals (mm) -> whitened vectors [T, N_ml]."""
        from scipy.linalg import solve_triangular
        return solve_triangular(self.chol, self.project(maps_full_mm).T, lower=True).T

    def chi2(self, maps_full_mm):
        """Per-acquisition r^T C^-1 r of ramp-removed, multilooked residuals: [T]."""
        return np.sum(self.whiten(maps_full_mm) ** 2, axis=1)

    def save(self, path):
        np.savez(path, C=self.C, x_km=self.x_km_full, y_km=self.y_km_full,
                 multilook_factor=self.multilook_factor, meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        f = np.load(path)
        return cls(f['C'], f['x_km'], f['y_km'], int(f['multilook_factor']), json.loads(str(f['meta'])))


def stationary_matrix(x_ml, y_ml, centres, cov_emp, var0):
    """Stationary covariance matrix from the binned empirical curve (linear interp), clipped to PSD."""
    pts = np.stack([x_ml.ravel(), y_ml.ravel()], axis=1)
    r = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    T = np.interp(r, np.concatenate([[0.0], centres]), np.concatenate([[var0], cov_emp]))
    vals, vecs = np.linalg.eigh(T)
    return (vecs * np.clip(vals, 1e-6 * var0, None)) @ vecs.T


def gaussian_loglik(C, X):
    """Mean per-map Gaussian log-likelihood of rows of X [n, d] under N(0, C) (up to a constant)."""
    L = np.linalg.cholesky(C + 1e-6 * np.mean(np.diag(C)) * np.eye(len(C)))
    from scipy.linalg import solve_triangular
    W = solve_triangular(L, X.T, lower=True)
    return float(-np.sum(np.log(np.diag(L))) - 0.5 * np.mean(np.sum(W ** 2, axis=0)))


def empirical_covariance(maps, x_km, y_km, bin_km=1.5, max_km=30.0, n_pairs=400000, rng=None):
    """Binned isotropic empirical covariance from random pixel pairs pooled over all maps."""
    rng = rng or np.random.default_rng(0)
    n, h, w = maps.shape
    flat = maps.reshape(n, -1)
    pts = np.stack([x_km.ravel(), y_km.ravel()], axis=1)
    edges = np.arange(0, max_km + bin_km, bin_km)
    sums = np.zeros(len(edges) - 1); counts = np.zeros(len(edges) - 1)
    m = rng.integers(0, n, n_pairs); i = rng.integers(0, h * w, n_pairs); j = rng.integers(0, h * w, n_pairs)
    r = np.linalg.norm(pts[i] - pts[j], axis=1)
    prod = flat[m, i] * flat[m, j]
    idx = np.digitize(r, edges) - 1
    ok = (idx >= 0) & (idx < len(sums))
    np.add.at(sums, idx[ok], prod[ok]); np.add.at(counts, idx[ok], 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    var0 = float(np.mean(flat ** 2))                                    # zero-lag (incl. white part)
    return centres, sums / np.maximum(counts, 1), counts, var0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--noise-dir', default='synthetic/multisource/pila_stacks_v2/noise_sets')
    parser.add_argument('--album', default='synthetic/multisource/pila_stacks_v2/albums/C01_cal.npz',
                        help='any album, for the grid coordinates')
    parser.add_argument('--multilook', type=int, default=4)
    parser.add_argument('--out', default='synthetic/multisource/pila_stacks_v2/noise_covariance_shrunk.npz')
    parser.add_argument('--fig', default='figures/multisource_seq/step2/noise_covariance_shrunk.png')
    args = parser.parse_args()
    if os.path.exists(args.out):
        raise FileExistsError(f"{args.out} exists; not overwriting")
    t0 = time.time()

    print("[1/4] Loading calibration noise sets (C*) ...")
    with np.load(args.album) as f:
        x_km_1d, y_km_1d = f['x_m'] / 1e3, f['y_m'] / 1e3
    x_km, y_km = np.meshgrid(x_km_1d, y_km_1d)
    paths = sorted(glob.glob(os.path.join(args.noise_dir, 'C*.npz')))
    maps = np.concatenate([np.load(p)['noise_mm'] for p in paths]).astype(np.float64)
    print(f"  Loaded: shape={maps.shape}, dtype={maps.dtype} from {len(paths)} sets")

    print("[2/4] Offset + ramp removal, multilook ...")
    maps = remove_ramp_maps(maps, x_km, y_km)
    maps_ml = multilook(maps, args.multilook)
    x_ml = multilook(x_km[None], args.multilook)[0]; y_ml = multilook(y_km[None], args.multilook)[0]
    print(f"  Multilooked: shape={maps_ml.shape}, RMS {np.sqrt(np.mean(maps_ml ** 2)):.2f} mm")

    print("[3/4] Shrinkage covariance: alpha by held-out log-likelihood (fit C01-C30, test C31-C39) ...")
    n_per_set = maps_ml.shape[0] // len(paths)
    split = 30 * n_per_set
    X = maps_ml.reshape(maps_ml.shape[0], -1)

    def build(Xfit, alpha, floor_mm2):
        centres, cov_emp, counts, var0 = empirical_covariance(Xfit.reshape(-1, *maps_ml.shape[1:]), x_ml, y_ml)
        T = stationary_matrix(x_ml, y_ml, centres, cov_emp, var0)
        S = Xfit.T @ Xfit / Xfit.shape[0]                                # zero-mean by construction
        # Noise floor (nugget): unseen tracks carry fine-scale structure the calibration maps under-
        # represent; without a floor those directions get huge weights (held-out whitened var ~7).
        C = (1 - alpha) * S + alpha * T + floor_mm2 * np.eye(len(S))
        return C, (centres, cov_emp, var0)

    scores = {}
    for alpha in [0.1, 0.3, 0.5]:
        for floor_mm2 in [0.25, 1.0, 4.0, 9.0, 16.0]:
            C_fit, _ = build(X[:split], alpha, floor_mm2)
            scores[(alpha, floor_mm2)] = gaussian_loglik(C_fit, X[split:])
            print(f"  alpha {alpha:.2f}, floor {floor_mm2:5.2f} mm^2: held-out log-likelihood {scores[(alpha, floor_mm2)]:.1f}")
    alpha, floor_mm2 = max(scores, key=scores.get)
    C, (centres, cov_emp, var0) = build(X, alpha, floor_mm2)
    print(f"  chosen alpha = {alpha}, floor = {floor_mm2} mm^2 (sigma {np.sqrt(floor_mm2):.1f} mm)")

    print("[4/4] Checks + saving ...")
    cov = NoiseCovariance(C, x_km, y_km, args.multilook,
                          meta=dict(alpha=alpha, floor_mm2=floor_mm2, n_sets=len(paths),
                                    heldout_loglik={f'{a}_{f}': v for (a, f), v in scores.items()}))
    cov.save(args.out)
    # Whitened noise should have ~unit variance per element: on held-out calibration sets (fitted with
    # the train-only matrix) and on the EVALUATION noise sets (Marapi track, never used for fitting).
    C_tr, _ = build(X[:split], alpha, floor_mm2)
    cov_tr = NoiseCovariance(C_tr, x_km, y_km, args.multilook)
    held = np.concatenate([np.load(p)['noise_mm'] for p in paths[30:]]).astype(np.float64)
    w_held = cov_tr.whiten(held)
    eval_paths = sorted(glob.glob(os.path.join(args.noise_dir, 'E*.npz')))
    w_eval = cov.whiten(np.concatenate([np.load(p)['noise_mm'] for p in eval_paths]).astype(np.float64))
    print(f"  whitened held-out calibration noise: variance {w_held.var():.3f} (target ~1)")
    print(f"  whitened evaluation noise (Marapi E1-E3): variance {w_eval.var():.3f} (target ~1)")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(centres, cov_emp, 'o-', label='binned empirical (stationary model T)')
    ax.plot([0], [var0], 'k*', ms=10, label='zero lag')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('Separation (km)'); ax.set_ylabel('Covariance (mm²)'); ax.legend(fontsize=8)
    ax.set_title(f'Calibration noise covariance (ramp removed, {args.multilook}x ML)')
    ax = axes[1]
    ax.hist(w_held.ravel(), bins=80, density=True, alpha=0.6, label=f'held-out calibration (var {w_held.var():.2f})')
    ax.hist(w_eval.ravel(), bins=80, density=True, alpha=0.6, label=f'evaluation E1-E3 (var {w_eval.var():.2f})')
    g = np.linspace(-5, 5, 200); ax.plot(g, np.exp(-g ** 2 / 2) / np.sqrt(2 * np.pi), 'k--', label='N(0,1)')
    ax.set_xlabel('Whitened residual (dimensionless)'); ax.set_ylabel('Density'); ax.legend(fontsize=8)
    ax.set_title(f'Whitening check (alpha = {alpha}, floor = {floor_mm2} mm²)')
    plt.tight_layout(); fig.savefig(args.fig, dpi=150); plt.close(fig)
    with open(args.out.replace('.npz', '.json'), 'w') as f:
        json.dump(dict(alpha=alpha, floor_mm2=floor_mm2, heldout_loglik={f'{a}_{f}': v for (a, f), v in scores.items()},
                       multilook=args.multilook,
                       n_sets=len(paths), whitened_var_heldout=float(w_held.var()),
                       whitened_var_eval=float(w_eval.var())), f, indent=1)
    print(f"Done in {time.time() - t0:.1f}s. Output saved to {args.out}")


if __name__ == '__main__':
    main()
