from torchvision import datasets, transforms
from base import BaseDataLoader
from datasets.spectrumS2 import SpectrumS2, SyntheticS2
from datasets.displacementGPS import (DisplacementGPS, DisplacementGPSSeq,
                                       DisplacementInSAR, DisplacementInSARSeq,
                                       DisplacementInSARh5, DisplacementInSARSeqh5,
                                       DisplacementInSARImage)
from datasets.timeseriesEVI import TimeSeriesEVI
import numpy as np
import torch

class SpectrumS2DataLoader(BaseDataLoader):
    """
    SpectrumS2 data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, with_const=False):
        self.data_dir = data_dir
        self.dataset = SpectrumS2(self.data_dir, with_const=with_const)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class SyntheticS2DataLoader(BaseDataLoader):
    """
    SyntheticS2 data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, with_const=False):
        self.data_dir = data_dir
        self.dataset = SyntheticS2(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class GPSDataLoader(BaseDataLoader):
    """
    GPS data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, with_const=False):
        self.data_dir = data_dir
        self.dataset = DisplacementGPS(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class GPSSeqDataLoader(BaseDataLoader):
    """
    GPS data loading demo using BaseDataLoader with temporal smoothness support
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, with_const=False, 
                 seq_len=5, mode='train'):
        self.data_dir = data_dir
        self.dataset = DisplacementGPSSeq(self.data_dir, seq_len=seq_len, mode=mode)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class InSARDataLoader(BaseDataLoader):
    """
    InSAR per-epoch LOS data loading using BaseDataLoader (Stage A).
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, with_const=False):
        self.data_dir = data_dir
        self.dataset = DisplacementInSAR(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class InSARSeqDataLoader(BaseDataLoader):
    """
    InSAR LOS sequence data loading using BaseDataLoader with temporal smoothness support (Stage B).
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, with_const=False,
                 seq_len=5, mode='train'):
        self.data_dir = data_dir
        self.dataset = DisplacementInSARSeq(self.data_dir, seq_len=seq_len, mode=mode)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class InSARh5DataLoader(BaseDataLoader):
    """
    h5-native InSAR per-epoch LOS loader for the point/MLP path (Stage A = whole series).

    Serves EVERY epoch of the cumulative LOS time series as an independent sample
    (mirrors the GNSS DisplacementGPS), NOT just the final/cumulative epoch. Reads a
    MintPy geo_timeseries_*.h5 directly via the shared multilook loader. The `insar`
    arg is the same dict the model's physics decoder reads (timeseries, geometry,
    mask, lat0, lon0, multilook), so both sides derive identical points.
    """

    def __init__(self, insar, batch_size, shuffle=True, validation_split=0.0, num_workers=1,
                 with_const=False):
        self.insar = insar
        self.dataset = DisplacementInSARh5(insar)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class InSARSeqh5DataLoader(BaseDataLoader):
    """
    h5-native InSAR LOS sequence loader for the point/MLP path (Stage B = whole series).
    """

    def __init__(self, insar, batch_size, shuffle=True, validation_split=0.0, num_workers=1,
                 with_const=False, seq_len=5, mode='train'):
        self.insar = insar
        self.dataset = DisplacementInSARSeqh5(insar, seq_len=seq_len, mode=mode)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class InSARImageDataLoader(BaseDataLoader):
    """
    h5-native full-field InSAR LOS image loader for the CNN path.

    Serves every epoch as an independent single-channel image (whole time series). The
    `insar` arg matches the model's arch.args.insar block.
    """

    def __init__(self, insar, batch_size, shuffle=True, validation_split=0.0, num_workers=1,
                 with_const=False):
        self.insar = insar
        self.dataset = DisplacementInSARImage(insar)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class EVIDataLoader(BaseDataLoader):
    """
    GPS data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, with_const=False):
        self.data_dir = data_dir
        self.dataset = TimeSeriesEVI(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)