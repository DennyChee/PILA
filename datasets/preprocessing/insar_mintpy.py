#!/usr/bin/env python
# Usage:       imported, not run directly:
#                from datasets.preprocessing.insar_mintpy import load_insar_mintpy
#                data = load_insar_mintpy(ts_h5, geom_h5, mask_h5, lat0, lon0, multilook=20)
# Description: Read a MintPy LOS time-series HDF5 directly (no intermediate CSV) and
#              multilook the coherent field to a coarse grid. Produces everything BOTH
#              PILA InSAR paths need: per-cell local-ENU coordinates, per-cell LOS unit
#              vectors, the per-epoch LOS field (mm), a coherence mask, and per-point /
#              global standardization. The result is memoized so the dataset side and the
#              model side derive the IDENTICAL points from one computation.
# Date:        2026-06-12
#
# Scientific conventions (READ — these affect interpretation):
#   * LOS sign: positive LOS = motion TOWARD the satellite (range decrease), MintPy
#     convention. Per-cell LOS unit vectors are derived through MintPy's enu2los so the
#     model-side projection e_E*uE + e_N*uN + e_U*uU is sign-consistent.
#   * Referencing: KEPT AS MintPy PRODUCED IT (its REF_DATE / REF pixel). No extra
#     re-referencing is applied here.
#   * Local ENU origin (lat0, lon0): the fixed frame peg mapping to (xE, yN) = (0, 0) km.
#     It is NOT the source location — the source (xcen/ycen, xoff/yoff) is inverted and
#     reported relative to this peg. It MUST match the origin used to define the
#     Mogi/Okada source-parameter ranges.
#   * Multilooking: mask-aware block averaging over `multilook` x `multilook` pixels. A
#     block is coherent if its valid (coherent & finite) fraction >= coh_valid_frac.
#     This both reduces 1.4M coherent pixels to a trainable count and suppresses noise.
#   * Units: LOS is converted metres -> mm to match the physics decoders' mm output.

import os
from dataclasses import dataclass

import h5py
import numpy as np

try:
    import pyproj
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("pyproj is required for insar_mintpy: " + str(exc))

from datasets import PARENT_DIR


# --- LOS unit-vector helper -------------------------------------------------
# These helpers are defined here (self-contained, no MintPy / pandas dependency) so the
# loader runs in the training env, which does not have MintPy installed. The projection
# below is the EXACT MintPy (>=1.6) enu2los convention, verified against
# mintpy.utils.utils0.enu2los:
#     v_los = -v_E*sin(inc)*sin(az) + v_N*sin(inc)*cos(az) + v_U*cos(inc)
# (az = MintPy azimuthAngle: LOS ground->satellite, from north, anti-clockwise positive;
#  positive LOS = motion toward the satellite). NOTE: this differs from the documented
# fallback in akutan_insar_prepare.py, which had e_E/e_N swapped and the wrong N sign.
def enu_to_los_unit_vectors(incidence_angle_deg, azimuth_angle_deg):
    """
    Per-point ground->satellite LOS unit-vector components (e_E, e_N, e_U).

    Parameters
    ----------
    incidence_angle_deg : np.ndarray   LOS incidence angle from vertical, deg.
    azimuth_angle_deg   : np.ndarray   MintPy azimuthAngle of the LOS vector, deg.

    Returns
    -------
    los_E, los_N, los_U : np.ndarray   LOS unit-vector components (normalized).
    """
    inc = np.deg2rad(incidence_angle_deg)
    az = np.deg2rad(azimuth_angle_deg)
    los_E = -np.sin(inc) * np.sin(az)   # MintPy enu2los E coefficient
    los_N = np.sin(inc) * np.cos(az)    # MintPy enu2los N coefficient
    los_U = np.cos(inc)                 # MintPy enu2los U coefficient

    # Normalize to unit length (guards tiny numerical drift).
    norm = np.sqrt(los_E ** 2 + los_N ** 2 + los_U ** 2)
    norm = np.where(norm == 0, 1.0, norm)
    return los_E / norm, los_N / norm, los_U / norm


def read_lonlat_grid(geom_file, n_rows, n_cols):
    """
    Per-pixel longitude/latitude arrays [n_rows, n_cols] (degrees).

    PREFERS the geocoding affine (X_FIRST/Y_FIRST/X_STEP/Y_STEP), which is authoritative
    for a geocoded product. The per-pixel 'longitude'/'latitude' datasets are only a
    fallback: on some files they are stale/shifted relative to the affine grid (e.g. the
    Agung geometry is offset by ~0.13 deg), which would mis-georeference both the ENU
    coordinates and the displacement maps.
    """
    required = ('X_FIRST', 'Y_FIRST', 'X_STEP', 'Y_STEP')
    with h5py.File(geom_file, 'r') as f:
        attrs = dict(f.attrs)
        if all(k in attrs for k in required):
            x_first = float(attrs['X_FIRST'])
            y_first = float(attrs['Y_FIRST'])
            x_step = float(attrs['X_STEP'])
            y_step = float(attrs['Y_STEP'])
            cols = x_first + x_step * np.arange(n_cols)
            rows = y_first + y_step * np.arange(n_rows)
            return np.meshgrid(cols, rows)
        if 'longitude' in f and 'latitude' in f:
            lon = np.array(f['longitude'], dtype=np.float64)
            lat = np.array(f['latitude'], dtype=np.float64)
            return lon, lat

    raise ValueError(
        "Geometry file has neither the geocoding attributes "
        f"{required} nor longitude/latitude datasets; cannot build a lon/lat grid.")


def lonlat_to_local_enu_km(lon, lat, lat0, lon0):
    """
    Convert lon/lat (deg) to local ENU East/North in km, centred on (lat0, lon0).

    Azimuthal-equidistant projection about the origin: the origin maps to (0, 0) and
    near-origin distances are accurate. The same origin must define the source ranges.
    """
    aeqd = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    transformer = pyproj.Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)
    east_m, north_m = transformer.transform(lon, lat)
    return east_m / 1000.0, north_m / 1000.0  # m -> km


def date_bytes_to_iso(date_array):
    """MintPy 'date' dataset (bytes 'YYYYMMDD') -> list of 'YYYY-MM-DD' strings."""
    iso = []
    for d in date_array:
        s = d.decode() if isinstance(d, (bytes, bytearray)) else str(d)
        iso.append(f"{s[0:4]}-{s[4:6]}-{s[6:8]}")
    return iso


@dataclass
class InSARData:
    """
    Container for a multilooked MintPy LOS time series, serving both PILA paths.

    Grid arrays are [Hd, Wd] (coarse rows/cols); the cube is [n_epoch, Hd, Wd].
    Point arrays are the coherent cells flattened in row-major order; that same order
    is what both the dataset and the model must agree on.

    Attributes
    ----------
    los_mm        : [n_epoch, Hd, Wd]  LOS displacement, mm (NaN where incoherent).
    mask_d        : [Hd, Wd] bool      coarse coherence mask (True = keep).
    dates         : list[str]          'YYYY-MM-DD' per epoch.
    xE_grid,yN_grid : [Hd, Wd]         local-ENU East/North of each cell centre, km.
    losE_grid,losN_grid,losU_grid : [Hd, Wd]  LOS unit-vector components (0 where masked).
    n_points      : int                number of coherent cells N.
    los_points_mm : [n_epoch, N]       LOS at coherent cells, mm (row-major over mask).
    xE_pts,yN_pts : [N]                coherent-cell coordinates, km.
    losE_pts,losN_pts,losU_pts : [N]   coherent-cell LOS unit vectors.
    x_mean_pts,x_scale_pts : [N]       per-point standardization (point/MLP path).
    x_mean_global,x_scale_global : float  global z-score (CNN/grid path).
    multilook,lat0,lon0 : echo of inputs.
    """
    los_mm: np.ndarray
    mask_d: np.ndarray
    dates: list
    xE_grid: np.ndarray
    yN_grid: np.ndarray
    losE_grid: np.ndarray
    losN_grid: np.ndarray
    losU_grid: np.ndarray
    n_points: int
    los_points_mm: np.ndarray
    xE_pts: np.ndarray
    yN_pts: np.ndarray
    losE_pts: np.ndarray
    losN_pts: np.ndarray
    losU_pts: np.ndarray
    x_mean_pts: np.ndarray
    x_scale_pts: np.ndarray
    x_mean_global: float
    x_scale_global: float
    multilook: int
    lat0: float
    lon0: float
    extent_lonlat: tuple   # [lon_min, lon_max, lat_min, lat_max] of the coarse grid (deg)


# --- block-multilook helpers ------------------------------------------------
def _crop_to_multiple(arr_2d_or_3d, factor):
    """Crop the two trailing (spatial) axes down to a whole multiple of `factor`."""
    n_rows, n_cols = arr_2d_or_3d.shape[-2], arr_2d_or_3d.shape[-1]
    rows_keep = (n_rows // factor) * factor
    cols_keep = (n_cols // factor) * factor
    return arr_2d_or_3d[..., :rows_keep, :cols_keep]


def mask_aware_multilook(cube_3d, mask_2d, factor, coh_valid_frac=0.5):
    """
    Mask-aware block average of a [n_epoch, L, W] cube over `factor` x `factor` blocks.

    Only coherent (mask True) pixels contribute to each block mean. A coarse cell is
    kept (coherent) if its valid fraction >= coh_valid_frac.

    Parameters
    ----------
    cube_3d  : [n_epoch, L, W]  values (may contain NaN at incoherent pixels).
    mask_2d  : [L, W] bool       fine coherence mask.
    factor   : int               multilook factor.
    coh_valid_frac : float       min coherent fraction for a coarse cell to be kept.

    Returns
    -------
    coarse_cube : [n_epoch, Hd, Wd]  block mean over coherent pixels (NaN where dropped).
    coarse_mask : [Hd, Wd] bool      coarse coherence mask.
    """
    # factor == 1 means NO multilooking: identity grid. Short-circuit to avoid allocating
    # a float64 copy + 4-D reshape of the (potentially huge) full-resolution cube. Result
    # matches the general path: coherent pixels keep their value, incoherent -> NaN.
    if factor == 1:
        coarse_mask = mask_2d.astype(bool)
        coarse_cube = np.where(coarse_mask[None, :, :], cube_3d, np.nan).astype(np.float32)
        return coarse_cube, coarse_mask

    cube_c = _crop_to_multiple(cube_3d, factor).astype(np.float64)
    mask_c = _crop_to_multiple(mask_2d, factor).astype(np.float64)
    n_epoch = cube_c.shape[0]
    rows_d = cube_c.shape[1] // factor
    cols_d = cube_c.shape[2] // factor

    # Reshape spatial axes into (block_row, in_block_row, block_col, in_block_col).
    mask_blk = mask_c.reshape(rows_d, factor, cols_d, factor)
    coh_count = mask_blk.sum(axis=(1, 3))                     # [Hd, Wd] coherent px per cell
    coarse_mask = (coh_count / (factor * factor)) >= coh_valid_frac

    # Zero out incoherent pixels (and any NaN) before summing so they don't contribute.
    # NOTE: callers pass a mask that is already intersected with finite-at-every-epoch
    # (see load_insar_mintpy), so coherent pixels carry no per-epoch NaN and the per-cell
    # coherent count (coh_count) is the correct divisor. If a caller ever passes a mask
    # whose True pixels can be NaN at some epoch, the block mean would be biased low.
    values = np.where(mask_2d[None, :, :].astype(bool), cube_3d, np.nan)
    values = _crop_to_multiple(values, factor)
    values = np.nan_to_num(values, nan=0.0)
    values_blk = values.reshape(n_epoch, rows_d, factor, cols_d, factor)
    block_sum = values_blk.sum(axis=(2, 4))                   # [n_epoch, Hd, Wd]

    # Mean over coherent pixels; guard divide-by-zero, then NaN-fill dropped cells.
    safe_count = np.where(coh_count == 0, 1.0, coh_count)[None, :, :]
    coarse_cube = block_sum / safe_count
    coarse_cube[:, ~coarse_mask] = np.nan
    return coarse_cube.astype(np.float32), coarse_mask


def _multilook_2d(arr_2d, mask_2d, factor):
    """Mask-aware block mean of a single [L, W] geometry array -> [Hd, Wd]."""
    coarse, _ = mask_aware_multilook(arr_2d[None, :, :].astype(np.float64), mask_2d, factor)
    return coarse[0]


# --- main loader (memoized) -------------------------------------------------
_CACHE = {}  # key -> InSARData; guarantees dataset & model see identical points


def load_insar_mintpy(timeseries_h5, geometry_h5, mask_h5, lat0, lon0,
                      multilook=1, coh_valid_frac=0.5, bbox=None, verbose=True,
                      standardization='global', far_field=None,
                      stride=1, offset=0,
                      bootstrap_k=None, bootstrap_block=3, bootstrap_seed=0):
    """
    Load + multilook a MintPy LOS time series into an InSARData bundle (memoized).

    Parameters
    ----------
    timeseries_h5 : str    MintPy geo_timeseries_*.h5 (timeseries in metres).
    geometry_h5   : str    MintPy geo_geometryRadar.h5 (incidence/azimuth, lat/lon).
    mask_h5        : str    MintPy mask .h5 (e.g. geo_maskTempCoh.h5); 'mask' dataset.
    lat0, lon0     : float  local-ENU origin (deg); MUST match the source-param frame.
    multilook      : int    block-average factor. DEFAULT 1 = NO multilooking (full
                            resolution); set >1 to block-average to a coarser grid.
    coh_valid_frac : float  min coherent fraction for a coarse cell to be kept.
    stride         : int    UQ strided-decimation factor n. DEFAULT 1 = keep all coherent
                            cells. If n>1, keep only every n-th coherent cell (row-major
                            order) starting at `offset` -- i.e. ensemble member `offset`
                            of n disjoint, jointly-complete subsets. The standardizer is
                            then computed from this subset only (per-subset scaler).
    offset         : int    UQ phase offset m in [0, stride). Selects which of the n
                            strided subsets this call returns. Ignored when stride==1.
    bootstrap_k    : int or None  Alternative to stride: SPATIAL BLOCK-BOOTSTRAP subsampling.
                            If set, keep ~bootstrap_k coherent cells by randomly selecting
                            whole `bootstrap_block`-square tiles (without replacement, member
                            seed = `bootstrap_seed`) until the target is reached. Resampling
                            contiguous tiles (not independent pixels) respects spatially-
                            correlated InSAR/atmospheric noise, and a fixed k DECOUPLES member
                            density from ensemble size. Mutually exclusive with stride>1.
    bootstrap_block: int    tile side in COARSE pixels for the block bootstrap (default 3).
    bootstrap_seed : int    RNG seed selecting this member's random tile set.
    bbox           : [lon_min, lon_max, lat_min, lat_max] or None. If given, crop the scene
                     to this geographic window BEFORE masking/multilooking. Use it to
                     isolate one target (e.g. Bali/Agung) so unrelated deformation elsewhere
                     in the scene (e.g. the 2018 Lombok earthquakes) cannot contaminate the
                     single-source inversion.

    Returns
    -------
    InSARData

    Notes
    -----
    The module-level memo (_CACHE) lives in the calling process. Use DataLoader
    num_workers=0 (all provided configs do): forked workers would each re-read and
    re-multilook the full HDF5, exploding I/O/memory and defeating the dataset<->model
    consistency the cache provides.
    """
    # Resolve to absolute paths so cache keys and h5py see the same thing regardless of cwd.
    ts_path = timeseries_h5 if os.path.isabs(timeseries_h5) else os.path.join(PARENT_DIR, timeseries_h5)
    geom_path = geometry_h5 if os.path.isabs(geometry_h5) else os.path.join(PARENT_DIR, geometry_h5)
    mask_path = mask_h5 if (mask_h5 and os.path.isabs(mask_h5)) else (
        os.path.join(PARENT_DIR, mask_h5) if mask_h5 else None)

    # Validate the UQ decimation params up front (cheap, fail fast before any I/O).
    stride = int(stride)
    offset = int(offset)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    # offset is a strided PHASE only when stride>1; for the block-bootstrap path (stride==1)
    # it is reused purely as the member id (0..B-1) for downstream dedup, so don't bound it.
    if stride > 1 and not (0 <= offset < stride):
        raise ValueError(f"offset must be in [0, stride)={[0, stride]}, got {offset}")
    # Block-bootstrap is an ALTERNATIVE decimation; refuse to combine it with stride>1.
    if bootstrap_k is not None:
        bootstrap_k = int(bootstrap_k)
        bootstrap_block = int(bootstrap_block)
        bootstrap_seed = int(bootstrap_seed)
        if bootstrap_k < 1:
            raise ValueError(f"bootstrap_k must be >= 1, got {bootstrap_k}")
        if bootstrap_block < 1:
            raise ValueError(f"bootstrap_block must be >= 1, got {bootstrap_block}")
        if stride > 1:
            raise ValueError("bootstrap_k and stride>1 are mutually exclusive "
                             f"(got stride={stride}, bootstrap_k={bootstrap_k})")

    bbox_key = tuple(float(b) for b in bbox) if bbox is not None else None
    ff_key = (tuple(sorted((k, float(v)) for k, v in far_field.items()))
              if far_field else None)
    # stride/offset (and the bootstrap triplet) are part of the key so the ensemble members
    # never collide in the memo, AND the dataset side and the model side agree on the SAME
    # subset per member.
    cache_key = (ts_path, geom_path, mask_path, float(lat0), float(lon0),
                 int(multilook), float(coh_valid_frac), bbox_key,
                 str(standardization), ff_key, stride, offset,
                 bootstrap_k, bootstrap_block if bootstrap_k is not None else None,
                 bootstrap_seed if bootstrap_k is not None else None)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    for path in (ts_path, geom_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    # --- [1/5] Load LOS cube + dates ---------------------------------------
    if verbose:
        print(f"[insar_mintpy 1/5] Loading time series from {ts_path} ...")
    with h5py.File(ts_path, 'r') as f:
        ts_cube_m = np.array(f['timeseries'], dtype=np.float32)   # [n_epoch, L, W], metres
        dates = date_bytes_to_iso(np.array(f['date']))
    n_epoch, full_rows, full_cols = ts_cube_m.shape
    if verbose:
        print(f"  Loaded: shape={ts_cube_m.shape}, dtype={ts_cube_m.dtype}, n_epoch={n_epoch}")

    # Full-resolution lon/lat (used for the optional crop and, after multilook, for coords).
    lon_grid, lat_grid = read_lonlat_grid(geom_path, full_rows, full_cols)

    # --- Optional spatial crop to a lon/lat bounding box -------------------
    row0, row1, col0, col1 = 0, full_rows, 0, full_cols
    if bbox is not None:
        lon_min, lon_max, lat_min, lat_max = bbox
        in_box = ((lon_grid >= lon_min) & (lon_grid <= lon_max) &
                  (lat_grid >= lat_min) & (lat_grid <= lat_max))
        if not in_box.any():
            raise ValueError(f"bbox {bbox} selects no pixels in the scene")
        rws, cls = np.where(in_box)
        row0, row1 = int(rws.min()), int(rws.max()) + 1
        col0, col1 = int(cls.min()), int(cls.max()) + 1
        ts_cube_m = ts_cube_m[:, row0:row1, col0:col1]
        lon_grid = lon_grid[row0:row1, col0:col1]
        lat_grid = lat_grid[row0:row1, col0:col1]
        if verbose:
            print(f"  Cropped to bbox {bbox}: rows {row0}:{row1}, cols {col0}:{col1} "
                  f"-> {ts_cube_m.shape[1]}x{ts_cube_m.shape[2]}")
    n_rows, n_cols = ts_cube_m.shape[1], ts_cube_m.shape[2]

    # --- [2/5] Fine coherence mask -----------------------------------------
    finite_mask = np.isfinite(ts_cube_m).all(axis=0)            # finite at every epoch
    if mask_path is not None and os.path.exists(mask_path):
        with h5py.File(mask_path, 'r') as f:
            mask_key = 'mask' if 'mask' in f else list(f.keys())[0]
            keep_mask = np.array(f[mask_key], dtype=bool)[row0:row1, col0:col1]
        if verbose:
            print(f"[insar_mintpy 2/5] Using mask dataset '{mask_key}'")
    else:
        keep_mask = np.ones((n_rows, n_cols), dtype=bool)
        if verbose:
            print("[insar_mintpy 2/5] No mask file; keeping all finite pixels")
    keep_mask = keep_mask & finite_mask
    if verbose:
        print(f"  Fine coherent pixels: {int(keep_mask.sum())} / {n_rows * n_cols}")

    # --- [3/5] Multilook cube + geometry to the coarse grid ----------------
    if verbose:
        print(f"[insar_mintpy 3/5] Multilooking (factor={multilook}) ...")
    coarse_cube_m, mask_d = mask_aware_multilook(ts_cube_m, keep_mask, multilook, coh_valid_frac)
    los_mm = coarse_cube_m * 1000.0                              # m -> mm

    # --- UQ strided decimation -------------------------------------------------
    # Keep only every `stride`-th coherent cell, in the same row-major order that is the
    # shared dataset<->model contract, starting at phase `offset`. We DEMOTE the dropped
    # cells in mask_d itself (rather than subsetting indices later) so that mask_d.sum()
    # stays equal to n_points -- every downstream array (coords, LOS vectors, los_points_mm,
    # the standardizer, and the grid/plotting path) is then derived from this thinned mask
    # and remains mutually consistent. The n subsets (offset=0..stride-1) are disjoint and
    # together tile the full coherent set, so the spread of inverted parameters across the
    # n ensemble members is a systematic spatial-subsampling uncertainty estimate.
    if stride > 1:
        coh_rows_full, coh_cols_full = np.where(mask_d)          # row-major coherent cells
        keep_sel = np.zeros(coh_rows_full.size, dtype=bool)
        keep_sel[offset::stride] = True                          # member `offset` of `stride`
        drop_rows = coh_rows_full[~keep_sel]
        drop_cols = coh_cols_full[~keep_sel]
        mask_d = mask_d.copy()
        mask_d[drop_rows, drop_cols] = False                     # thin the coherence mask
        if verbose:
            print(f"  UQ decimation: stride={stride}, offset={offset} -> "
                  f"{int(mask_d.sum())} / {coh_rows_full.size} coherent cells kept")
        if int(mask_d.sum()) == 0:
            raise ValueError(f"stride={stride}, offset={offset} selects 0 cells "
                             f"(only {coh_rows_full.size} coherent cells available)")

    # --- UQ spatial BLOCK-BOOTSTRAP subsampling --------------------------------
    # Alternative to stride: partition the coherent footprint into bootstrap_block-square
    # tiles and randomly keep WHOLE tiles (without replacement, this member's seed) until
    # ~bootstrap_k coherent cells are accumulated. Resampling contiguous tiles respects the
    # spatial correlation of InSAR/atmospheric noise (independent-pixel resampling treats
    # correlated noise as independent -> over-tight UQ), and a fixed target k decouples
    # member density from ensemble size. We thin mask_d in place, exactly like the stride
    # path, so every downstream array stays mutually consistent.
    elif bootstrap_k is not None:
        coh_rows_full, coh_cols_full = np.where(mask_d)          # row-major coherent cells
        n_coh = coh_rows_full.size
        if bootstrap_k >= n_coh:
            if verbose:
                print(f"  UQ block-bootstrap: k={bootstrap_k} >= {n_coh} coherent cells; "
                      f"keeping all.")
        else:
            b = bootstrap_block
            # Tile id per coherent cell (row-major over the tile grid); group cells by tile.
            n_tile_cols = mask_d.shape[1] // b + 1
            tile_id = (coh_rows_full // b) * n_tile_cols + (coh_cols_full // b)
            uniq_tiles, inv, counts = np.unique(tile_id, return_inverse=True, return_counts=True)
            rng = np.random.default_rng(bootstrap_seed)
            order = rng.permutation(uniq_tiles.size)              # member-specific tile order
            # Take whole tiles until the cumulative cell count first reaches k.
            cum = np.cumsum(counts[order])
            n_take = int(np.searchsorted(cum, bootstrap_k, side='left')) + 1
            keep_tile_pos = order[:n_take]
            keep_sel = np.isin(inv, keep_tile_pos)               # bool over coherent cells
            mask_d = mask_d.copy()
            mask_d[coh_rows_full[~keep_sel], coh_cols_full[~keep_sel]] = False
            if verbose:
                print(f"  UQ block-bootstrap: seed={bootstrap_seed}, block={b}px, "
                      f"target k={bootstrap_k} -> {int(mask_d.sum())} cells from "
                      f"{n_take}/{uniq_tiles.size} tiles ({n_coh} coherent available)")
            if int(mask_d.sum()) == 0:
                raise ValueError(f"block-bootstrap seed={bootstrap_seed} selects 0 cells")

    with h5py.File(geom_path, 'r') as f:
        inc_full = np.array(f['incidenceAngle'], dtype=np.float64)[row0:row1, col0:col1]
        az_full = np.array(f['azimuthAngle'], dtype=np.float64)[row0:row1, col0:col1]
    lon_d = _multilook_2d(lon_grid, keep_mask, multilook)
    lat_d = _multilook_2d(lat_grid, keep_mask, multilook)
    inc_d = _multilook_2d(inc_full, keep_mask, multilook)
    az_d = _multilook_2d(az_full, keep_mask, multilook)
    rows_d, cols_d = mask_d.shape

    # Geographic extent of the coarse grid (the multilook crops to whole blocks), for plots.
    rk, ck = rows_d * multilook, cols_d * multilook
    extent_lonlat = (float(lon_grid[:rk, :ck].min()), float(lon_grid[:rk, :ck].max()),
                     float(lat_grid[:rk, :ck].min()), float(lat_grid[:rk, :ck].max()))
    if verbose:
        print(f"  Coarse grid: {rows_d} x {cols_d}; coherent cells: {int(mask_d.sum())}")

    # --- [4/5] Per-cell ENU coords + LOS unit vectors ----------------------
    # Compute on coherent cells only (avoids NaN propagating through enu2los), scatter back.
    coh_rows, coh_cols = np.where(mask_d)
    east_km, north_km = lonlat_to_local_enu_km(
        lon_d[coh_rows, coh_cols], lat_d[coh_rows, coh_cols], lat0, lon0)
    losE_c, losN_c, losU_c = enu_to_los_unit_vectors(
        inc_d[coh_rows, coh_cols], az_d[coh_rows, coh_cols])

    xE_grid = np.zeros((rows_d, cols_d), np.float64)
    yN_grid = np.zeros((rows_d, cols_d), np.float64)
    losE_grid = np.zeros((rows_d, cols_d), np.float64)
    losN_grid = np.zeros((rows_d, cols_d), np.float64)
    losU_grid = np.zeros((rows_d, cols_d), np.float64)
    xE_grid[coh_rows, coh_cols] = east_km
    yN_grid[coh_rows, coh_cols] = north_km
    losE_grid[coh_rows, coh_cols] = losE_c
    losN_grid[coh_rows, coh_cols] = losN_c
    losU_grid[coh_rows, coh_cols] = losU_c

    mean_eU = float(losU_c.mean())
    if verbose:
        print(f"  Sanity: mean e_U={mean_eU:.3f} (expect ~0.7-0.85, positive)")
    if mean_eU < 0.5:
        print("  WARNING: mean e_U < 0.5 — unexpected for Sentinel-1; check inc/az + enu2los.")

    # --- [5/5] Point series + standardization ------------------------------
    # Coherent-cell LOS, row-major over the mask -> [n_epoch, N]; this fixed ordering is
    # the shared contract between the dataset and the model.
    los_points_mm = los_mm[:, coh_rows, coh_cols]               # [n_epoch, N]
    n_points = los_points_mm.shape[1]

    # Per-point standardization over ALL epochs (so Stage A and Stage B share one scaler).
    # ASSUMPTION: epoch 0 is MintPy's reference date (identically 0). It is INCLUDED here,
    # which slightly shrinks the mean/std (the effect is ~1/n_epoch). This is a benign,
    # consistent scaling choice — the model uses the same scaler — not a correctness issue.
    x_mean_pts = np.nanmean(los_points_mm, axis=0).astype(np.float32)        # [N]
    x_scale_pts = np.nanstd(los_points_mm, axis=0).astype(np.float32)        # [N]
    x_scale_pts = np.where(x_scale_pts < 1e-6, 1.0, x_scale_pts).astype(np.float32)
    n_tiny = int(np.sum(x_scale_pts < 1e-3))
    if verbose and n_tiny > 0:
        print(f"  NOTE: {n_tiny} points have std < 1e-3 mm (near-static).")

    # Global standardization (CNN/grid + h5-LOS path). Two modes:
    #   'global'           : mean / std over ALL valid cells & epochs (default; legacy).
    #   'far_field_robust' : robust median / MAD over FAR-FIELD cells only (those farther
    #                        than far_field['radius_km'] from the source). The far field is
    #                        signal-free, so the scale reflects the NOISE FLOOR rather than
    #                        the scene-wide signal+noise mixture; this decouples PILA's input
    #                        scale from scene-wide noise and from signal self-suppression.
    #                        ALL points are still standardized with this scaler (n_points
    #                        unchanged) — only the statistic's support changes.
    if standardization == 'far_field_robust':
        if not far_field or 'radius_km' not in far_field:
            raise ValueError("standardization='far_field_robust' needs far_field with "
                             "source_xE_km/source_yN_km/radius_km")
        sx = float(far_field.get('source_xE_km', 0.0))
        sy = float(far_field.get('source_yN_km', 0.0))
        R = float(far_field['radius_km'])
        dist_km = np.sqrt((east_km - sx) ** 2 + (north_km - sy) ** 2)   # [N]
        ff_pts = dist_km > R                                            # far-field cells
        if int(ff_pts.sum()) < 50:
            raise ValueError(f"far-field mask keeps only {int(ff_pts.sum())} cells "
                             f"(radius {R} km too large for this scene)")
        ff_vals = los_points_mm[:, ff_pts].ravel()
        ff_vals = ff_vals[np.isfinite(ff_vals)]
        med = float(np.median(ff_vals))
        mad = float(np.median(np.abs(ff_vals - med)))
        x_mean_global = med
        x_scale_global = 1.4826 * mad                                  # MAD -> sigma-equivalent
        if x_scale_global < 1e-6:
            x_scale_global = 1.0
        if verbose:
            print(f"  Standardization: far_field_robust over {int(ff_pts.sum())}/{n_points} "
                  f"cells (>{R} km from source); median={med:.3f} mm, "
                  f"MAD-scale={x_scale_global:.3f} mm")
    else:
        x_mean_global = float(np.nanmean(los_points_mm))
        x_scale_global = float(np.nanstd(los_points_mm))
        if x_scale_global < 1e-6:
            x_scale_global = 1.0

    data = InSARData(
        los_mm=los_mm, mask_d=mask_d, dates=dates,
        xE_grid=xE_grid, yN_grid=yN_grid,
        losE_grid=losE_grid, losN_grid=losN_grid, losU_grid=losU_grid,
        n_points=n_points, los_points_mm=los_points_mm.astype(np.float32),
        xE_pts=east_km.astype(np.float32), yN_pts=north_km.astype(np.float32),
        losE_pts=losE_c.astype(np.float32), losN_pts=losN_c.astype(np.float32),
        losU_pts=losU_c.astype(np.float32),
        x_mean_pts=x_mean_pts, x_scale_pts=x_scale_pts,
        x_mean_global=x_mean_global, x_scale_global=x_scale_global,
        multilook=int(multilook), lat0=float(lat0), lon0=float(lon0),
        extent_lonlat=extent_lonlat,
    )
    _CACHE[cache_key] = data
    if verbose:
        print(f"  Done. N points = {n_points}, grid = {rows_d}x{cols_d}.")
    return data
