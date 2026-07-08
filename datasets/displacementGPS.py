import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import os
import math
import datetime
from datasets import PARENT_DIR


def time_feats(d, time_feat_dim=4):
    """
    Convert date to temporal features (annual and semi-annual cycles).
    
    Args:
        d: datetime.date or 'YYYY-MM-DD' string
        time_feat_dim: dimension of time features (2 for annual only, 4 for annual+semi-annual)
    
    Returns:
        np.array: temporal features [sin(annual), cos(annual), sin(semi-annual), cos(semi-annual)]
    """
    if isinstance(d, str):
        # Handle different date formats: 'YYYY-MM-DD' or 'YYYY.MM.DD'
        if '-' in d:
            y, m, dd = map(int, d.split('-'))
        elif '.' in d:
            y, m, dd = map(int, d.split('.'))
        else:
            raise ValueError(f"Unsupported date format: {d}")
        d = datetime.date(y, m, dd)
    
    doy = d.timetuple().tm_yday
    two_pi = 2 * math.pi
    
    # Use a consistent year length for all calculations
    year_length = 365.25
    
    # Annual cycle (normalized by consistent year length)
    a1 = two_pi * (doy / year_length)
    annual_feats = [math.sin(a1), math.cos(a1)]
    
    if time_feat_dim == 4:
        # Semi-annual cycle
        a2 = two_pi * (2 * doy / year_length)
        semi_annual_feats = [math.sin(a2), math.cos(a2)]
        return np.array(annual_feats + semi_annual_feats, dtype=np.float32)
    elif time_feat_dim == 2:
        return np.array(annual_feats, dtype=np.float32)
    else:
        raise ValueError(f"time_feat_dim must be 2 or 4, got {time_feat_dim}")


class DisplacementGPS(data.Dataset):
    def __init__(self, csv_path):
        super(DisplacementGPS, self).__init__()
        # the dataset is a tabular data of GPS displacement data
        # each row is displacements from 12 stations at the same time point
        self.data_df = pd.read_csv(os.path.join(PARENT_DIR, csv_path))

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]
        data_dict = {}
        data_dict['displacement'] = torch.tensor(
            sample[:36].values.astype('float32')
        ).to(torch.float32)
        data_dict['date'] = sample[-3]
        
        # Always add time features (4-dim: annual + semi-annual)
        # The model will use only the first time_feat_dim elements
        date_str = sample[-3]  # Assuming date is in the last column
        time_features = time_feats(date_str, time_feat_dim=4)  # Always generate 4-dim features
        data_dict['time_feats'] = torch.tensor(time_features).to(torch.float32)

        return data_dict

# dataset by slicing data into sequences with temporal smoothness
class DisplacementGPSSeq(data.Dataset):
    def __init__(self, csv_path, seq_len=7, mode='train'):
        super(DisplacementGPSSeq, self).__init__()
        self.seq_len = seq_len
        self.mode = mode  # 'train' or 'inference'
        self.data_df = pd.read_csv(os.path.join(PARENT_DIR, csv_path))
        self.data_df['date'] = pd.to_datetime(self.data_df['date'])
        self.data_df = self.data_df.sort_values(by='date')

    def __len__(self):
        if self.mode == 'inference':
            # Non-overlapping sequences for inference
            return len(self.data_df) // self.seq_len
        else:
            # Overlapping sequences for training
            return max(1, len(self.data_df) - self.seq_len + 1)

    def __getitem__(self, idx):
        if self.mode == 'inference':
            # Non-overlapping sequences for inference (no repeated dates)
            start_idx = idx * self.seq_len
            end_idx = start_idx + self.seq_len
            if end_idx > len(self.data_df):
                # Handle the last incomplete sequence
                start_idx = max(0, len(self.data_df) - self.seq_len)
                end_idx = len(self.data_df)
            sample = self.data_df.iloc[start_idx:end_idx]
        else:
            # Random overlapping sequences for training
            max_start_idx = len(self.data_df) - self.seq_len
            start_idx = np.random.randint(0, max_start_idx + 1)
            sample = self.data_df.iloc[start_idx:start_idx+self.seq_len]
        
        data_dict = {}
        data_dict['displacement'] = torch.tensor(
            sample.iloc[:, :36].values.astype('float32')
        ).to(torch.float32)
        
        # Always add time features (4-dim: annual + semi-annual)
        # The model will use only the first time_feat_dim elements
        dates = sample['date'].values
        time_features_list = []
        for date in dates:
            # Convert pandas timestamp to string format
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            time_features = time_feats(date_str, time_feat_dim=4)  # Always generate 4-dim features
            time_features_list.append(time_features)
        
        data_dict['time_feats'] = torch.tensor(
            np.array(time_features_list, dtype=np.float32)
        ).to(torch.float32)

        return data_dict


# Columns that are metadata (not LOS observations) in the InSAR CSV.
_INSAR_META_COLS = ('date', 'sin_date', 'cos_date')


class DisplacementInSAR(data.Dataset):
    """
    InSAR line-of-sight (LOS) displacement dataset (per-epoch; Stage A/B).

    Mirrors DisplacementGPS but each row holds N LOS values (one per downsampled
    observation point) at a single epoch, instead of 36 GNSS ENU values. The N
    LOS columns are every column except the metadata columns in _INSAR_META_COLS;
    their file order must match the point order in the points-info JSON.

    Each row of the CSV is the (standardized) LOS field at one epoch, in mm-space
    before standardization. The 'displacement' key is kept so the trainer/test
    loops and config input_key/output_key need no change.
    """
    def __init__(self, csv_path):
        super(DisplacementInSAR, self).__init__()
        self.data_df = pd.read_csv(os.path.join(PARENT_DIR, csv_path))
        # LOS columns: everything that is not metadata, in file order.
        self.los_cols = [c for c in self.data_df.columns
                         if c not in _INSAR_META_COLS]
        self.n_points = len(self.los_cols)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]
        data_dict = {}
        data_dict['displacement'] = torch.tensor(
            sample[self.los_cols].values.astype('float32')
        ).to(torch.float32)
        data_dict['date'] = sample['date']

        # Always add 4-dim time features (annual + semi-annual); the model uses
        # only the first time_feat_dim elements.
        time_features = time_feats(sample['date'], time_feat_dim=4)
        data_dict['time_feats'] = torch.tensor(time_features).to(torch.float32)

        return data_dict


# Columns that are metadata in legacy CSV-based InSAR datasets above; the h5-native
# datasets below read MintPy directly and have no such columns.


def _standardized_los_series_h5(insar_args):
    """
    Load + multilook a MintPy LOS time series and return the per-point STANDARDIZED
    cumulative series plus its dates (h5-native; no CSV).

    Parameters
    ----------
    insar_args : dict   must contain keys: timeseries, geometry, mask (optional),
                        lat0, lon0, multilook (optional). Same block the model reads, so
                        the memoized loader guarantees identical points + scaler.

    Returns
    -------
    std_series : np.ndarray [n_epoch, N]   per-point standardized LOS (mm-space scaler).
    dates      : list[str]                 'YYYY-MM-DD' per epoch.
    n_points   : int                       number of coherent cells N.
    """
    # Imported lazily so importing this module never pulls in h5py/pyproj unnecessarily.
    from datasets.preprocessing.insar_mintpy import load_insar_mintpy
    d = load_insar_mintpy(
        insar_args['timeseries'], insar_args['geometry'], insar_args.get('mask'),
        insar_args['lat0'], insar_args['lon0'],
        multilook=insar_args.get('multilook', 1),   # default 1 = full resolution (no multilook)
        coh_valid_frac=insar_args.get('coh_valid_frac', 0.5),
        bbox=insar_args.get('bbox'),
        verbose=insar_args.get('verbose', True),
        standardization=insar_args.get('std_mode', 'global'),
        far_field=insar_args.get('far_field'),
        stride=insar_args.get('stride', 1),         # UQ strided-decimation factor n
        offset=insar_args.get('offset', 0),         # UQ phase offset (this ensemble member)
        bootstrap_k=insar_args.get('bootstrap_k'),  # UQ block-bootstrap target px (None = off)
        bootstrap_block=insar_args.get('bootstrap_block', 3),
        bootstrap_seed=insar_args.get('bootstrap_seed', 0),
        ref_date=insar_args.get('ref_date'))          # temporal re-referencing (None = MintPy default)
    # GLOBAL (scene-wide) standardization with the SAME scaler the model's physics decoder
    # uses (see _load_insar_inputs). Per-point z-scoring would amplify the ~94% quiescent
    # cells' noise to signal scale and flatten the deformation, collapsing the fit to ~0.
    std_series = (d.los_points_mm - d.x_mean_global) / d.x_scale_global
    return std_series.astype(np.float32), list(d.dates), d.n_points


class DisplacementInSARh5(data.Dataset):
    """
    h5-native InSAR LOS dataset for the point/MLP path (Stage A).

    Mirrors the GNSS DisplacementGPS: serves EVERY epoch of the (referenced) cumulative
    LOS time series as an independent 'displacement' vector [N]. A VAE needs many samples,
    so Stage A trains on the whole time series (each epoch a separate sample) rather than a
    single field; Stage B (DisplacementInSARSeqh5) adds the temporal-smoothness sequences
    on top. Returns the same keys as DisplacementInSAR so the trainer needs no change.

    insar_args : dict  (timeseries, geometry, mask, lat0, lon0, multilook, ...)
    """
    def __init__(self, insar_args):
        super(DisplacementInSARh5, self).__init__()
        std_series, dates, n_points = _standardized_los_series_h5(insar_args)
        self.n_points = n_points
        self.series = std_series                      # [n_epoch, N] (all epochs)
        self.dates = dates

    def __len__(self):
        return self.series.shape[0]

    def __getitem__(self, index):
        data_dict = {}
        data_dict['displacement'] = torch.tensor(self.series[index]).to(torch.float32)
        data_dict['date'] = self.dates[index]
        # 4-dim time features (annual + semi-annual); model uses first time_feat_dim.
        time_features = time_feats(self.dates[index], time_feat_dim=4)
        data_dict['time_feats'] = torch.tensor(time_features).to(torch.float32)
        return data_dict


class DisplacementInSARSeqh5(data.Dataset):
    """
    h5-native InSAR LOS dataset sliced into temporal sequences (Stage B = whole series).

    Mirrors DisplacementInSARSeq but reads MintPy directly. Sequences drive the
    temporal-smoothness regularizer (source location/geometry pinned across consecutive
    epochs while amplitude is free).
    """
    def __init__(self, insar_args, seq_len=7, mode='train'):
        super(DisplacementInSARSeqh5, self).__init__()
        std_series, dates, n_points = _standardized_los_series_h5(insar_args)
        # Clamp seq_len to the stack length so short series can't produce an empty/negative
        # sampling window (np.random.randint would raise).
        self.seq_len = max(1, min(seq_len, len(dates)))
        if self.seq_len != seq_len:
            print(f"  NOTE: seq_len clamped {seq_len} -> {self.seq_len} (only {len(dates)} epochs)")
        self.mode = mode
        self.n_points = n_points
        self.series = std_series                       # [n_epoch, N]
        self.dates = dates

    def __len__(self):
        if self.mode == 'inference':
            return len(self.dates) // self.seq_len
        return max(1, len(self.dates) - self.seq_len + 1)

    def __getitem__(self, idx):
        if self.mode == 'inference':
            start_idx = idx * self.seq_len
            end_idx = start_idx + self.seq_len
            if end_idx > len(self.dates):
                start_idx = max(0, len(self.dates) - self.seq_len)
                end_idx = len(self.dates)
        else:
            max_start_idx = len(self.dates) - self.seq_len
            start_idx = np.random.randint(0, max_start_idx + 1)
            end_idx = start_idx + self.seq_len

        data_dict = {}
        data_dict['displacement'] = torch.tensor(
            self.series[start_idx:end_idx]).to(torch.float32)       # [seq_len, N]
        seq_dates = self.dates[start_idx:end_idx]
        time_features_list = [time_feats(d, time_feat_dim=4) for d in seq_dates]
        data_dict['time_feats'] = torch.tensor(
            np.array(time_features_list, dtype=np.float32)).to(torch.float32)
        return data_dict


class DisplacementInSARImage(data.Dataset):
    """
    h5-native full-field InSAR LOS image dataset for the CNN path.

    Reads a MintPy geo_timeseries_*.h5 via the shared multilook loader and serves EVERY
    epoch's coarse LOS field as a single-channel image [1, Hd, Wd], globally z-scored, with
    incoherent cells zeroed. Also returns a coherence 'mask' [1, Hd, Wd] for the masked
    reconstruction loss. The image flatten order (row-major over Hd x Wd) matches the
    grid-cell order the CNN physics decoder renders, so masked-MSE compares like for like.

    Like the point path, the whole time series is used (each epoch an independent image) so
    the VAE has many samples to train on.
    """
    def __init__(self, insar_args):
        super(DisplacementInSARImage, self).__init__()
        from datasets.preprocessing.insar_mintpy import load_insar_mintpy
        d = load_insar_mintpy(
            insar_args['timeseries'], insar_args['geometry'], insar_args.get('mask'),
            insar_args['lat0'], insar_args['lon0'],
            multilook=insar_args.get('multilook', 20),
            coh_valid_frac=insar_args.get('coh_valid_frac', 0.5),
            bbox=insar_args.get('bbox'),
            verbose=insar_args.get('verbose', True),
            ref_date=insar_args.get('ref_date'))          # temporal re-referencing (None = MintPy default)
        # Global z-score (same scaler the grid physics decoder uses); incoherent -> 0.
        std_grid = (d.los_mm - d.x_mean_global) / d.x_scale_global   # [n_epoch, Hd, Wd]
        std_grid = np.nan_to_num(std_grid, nan=0.0).astype(np.float32)
        self.mask = d.mask_d.astype(np.float32)[None, :, :]          # [1, Hd, Wd]
        self.images = std_grid[:, None, :, :]                        # [n_epoch, 1, Hd, Wd]
        self.dates = list(d.dates)
        self.Hd, self.Wd = d.mask_d.shape

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        data_dict = {}
        data_dict['displacement'] = torch.tensor(self.images[index]).to(torch.float32)  # [1,Hd,Wd]
        data_dict['mask'] = torch.tensor(self.mask).to(torch.float32)                    # [1,Hd,Wd]
        data_dict['date'] = self.dates[index]
        time_features = time_feats(self.dates[index], time_feat_dim=4)
        data_dict['time_feats'] = torch.tensor(time_features).to(torch.float32)
        return data_dict


class DisplacementInSARSeq(data.Dataset):
    """
    InSAR LOS displacement dataset sliced into temporal sequences (Stage B).

    Mirrors DisplacementGPSSeq but with N LOS columns instead of 36 GNSS columns.
    Sequences drive the temporal-smoothness regularizer, which pins the inferred
    source location/geometry across consecutive epochs while amplitude is free.
    """
    def __init__(self, csv_path, seq_len=7, mode='train'):
        super(DisplacementInSARSeq, self).__init__()
        self.seq_len = seq_len
        self.mode = mode  # 'train' or 'inference'
        self.data_df = pd.read_csv(os.path.join(PARENT_DIR, csv_path))
        self.data_df['date'] = pd.to_datetime(self.data_df['date'])
        self.data_df = self.data_df.sort_values(by='date')
        # LOS columns: everything that is not metadata, in file order.
        self.los_cols = [c for c in self.data_df.columns
                         if c not in _INSAR_META_COLS]
        self.n_points = len(self.los_cols)

    def __len__(self):
        if self.mode == 'inference':
            # Non-overlapping sequences for inference
            return len(self.data_df) // self.seq_len
        else:
            # Overlapping sequences for training
            return max(1, len(self.data_df) - self.seq_len + 1)

    def __getitem__(self, idx):
        if self.mode == 'inference':
            # Non-overlapping sequences for inference (no repeated dates)
            start_idx = idx * self.seq_len
            end_idx = start_idx + self.seq_len
            if end_idx > len(self.data_df):
                start_idx = max(0, len(self.data_df) - self.seq_len)
                end_idx = len(self.data_df)
            sample = self.data_df.iloc[start_idx:end_idx]
        else:
            # Random overlapping sequences for training
            max_start_idx = len(self.data_df) - self.seq_len
            start_idx = np.random.randint(0, max_start_idx + 1)
            sample = self.data_df.iloc[start_idx:start_idx + self.seq_len]

        data_dict = {}
        data_dict['displacement'] = torch.tensor(
            sample[self.los_cols].values.astype('float32')
        ).to(torch.float32)

        dates = sample['date'].values
        time_features_list = []
        for date in dates:
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            time_features = time_feats(date_str, time_feat_dim=4)
            time_features_list.append(time_features)

        data_dict['time_feats'] = torch.tensor(
            np.array(time_features_list, dtype=np.float32)
        ).to(torch.float32)

        return data_dict
