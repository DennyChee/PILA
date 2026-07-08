"""
PILA model (physics-informed low-rank augmentation).

This implementation provides the PILA architecture used in the paper.

To extend PILA with a new physics model, implement it under `physics/` and
create matching configs under `configs/phys_smpl/`.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import warnings
from base import BaseModel
from physics.rtm.rtm import RTM
from physics.mogi.mogi import Mogi
from physics.okada.okada_dike import OkadaDike
from physics.sun69.sun69 import Sun69
from model import SCRIPT_DIR, PARENT_DIR
from utils import MLP, draw_normal

# CUDA for PyTorch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# RTM spectral bands to be used
S2_FULL_BANDS = ['B01', 'B02_BLUE', 'B03_GREEN', 'B04_RED','B05_RE1', 
                 'B06_RE2', 'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B10', 
                 'B11_SWI1', 'B12_SWI2']
SD = 500.0 # Stem Density (SD), assumed to be 500 trees/ha

class Encoders(nn.Module):
    def __init__(self, config:dict):
        super(Encoders, self).__init__()

        in_channels = config['arch']['args']['input_dim'] #11 for RTM, 36 for Mogi
        no_phy = config['arch']['phys_vae']['no_phy']
        dim_z_aux = config['arch']['phys_vae']['dim_z_aux']#2
        dim_z_phy = config['arch']['phys_vae']['dim_z_phy']#7 for RTM, 4 for Mogi
        activation = config['arch']['phys_vae']['activation'] # e.g., 'elu'
        num_units_feat = config['arch']['phys_vae']['num_units_feat']#64
        
        self.func_feat = FeatureExtractor(config)

        if dim_z_aux > 0:
            hidlayers_z_aux = config['arch']['phys_vae']['hidlayers_z_aux']
            # z_aux encoding from feature only (as before)
            self.func_z_aux_mean = MLP([num_units_feat,]+hidlayers_z_aux+[dim_z_aux,], activation)
            self.func_z_aux_lnvar = MLP([num_units_feat,]+hidlayers_z_aux+[dim_z_aux,], activation)

        if not no_phy:
            hidlayers_z_phy = config['arch']['phys_vae']['hidlayers_z_phy']
            # CHANGED: remove Softplus on mean; final layer is linear (u-space)
            # ORIGINAL: self.func_z_phy_mean = nn.Sequential(MLP([...],[dim_z_phy]), nn.Softplus())
            self.func_z_phy_mean = MLP([num_units_feat,]+hidlayers_z_phy+[dim_z_phy,], activation)  # NEW (u-mean)
            self.func_z_phy_lnvar = MLP([num_units_feat,]+hidlayers_z_phy+[dim_z_phy,], activation) # u-lnvar

            # REMOVED: unmixing path; keep ablation-ready by simply not creating it now.
            # ORIGINAL: self.func_unmixer_coeff = nn.Sequential(MLP([...,[in_channels]]), nn.Tanh())

class Decoders(nn.Module):
    def __init__(self, config:dict):
        super(Decoders, self).__init__()

        in_channels = config['arch']['args']['input_dim'] #11 for RTM, 36 for Mogi
        dim_z_aux = config['arch']['phys_vae']['dim_z_aux'] #2
        dim_z_phy = config['arch']['phys_vae']['dim_z_phy'] #7 for RTM, 4 for Mogi
        activation = config['arch']['phys_vae']['activation'] #elu 
        no_phy = config['arch']['phys_vae']['no_phy']
        
        # Get time dimension and usage flags from config
        time_feat_dim = config['arch']['args'].get('time_feat_dim', 0)
        self.time_feat_dim = config['arch']['args'].get('time_feat_dim', 0)
        
        # Option to use time features in residual coefficient computation
        self.use_time_in_residual = config['arch']['args'].get('use_time_in_residual', False)

        if not no_phy:
            if dim_z_aux > 0:
                # IMPROVED: [z_aux, x_P_det] -> Linear -> c -> tanh(c/tau) -> delta = (c*s)@B.T
                residual_rank = config['arch']['phys_vae'].get('residual_rank', dim_z_aux)  # Default to dim_z_aux
                
                # Linear coefficient transformation: [z_aux, x_P_det, time_feats] -> residual_rank
                # Only add time_feat_dim if use_time_in_residual is True
                coeff_input_dim = dim_z_aux + in_channels + (time_feat_dim if self.use_time_in_residual else 0)
                self.coeff = nn.Linear(coeff_input_dim, residual_rank, bias=True)
                
                # Per-direction scale parameters
                self.s = nn.Parameter(torch.ones(residual_rank))  # per-direction scale
                
                # Basis matrix B: R^D x k (D = in_channels, k = residual_rank)
                self.B = nn.Parameter(torch.randn(in_channels, residual_rank))
                nn.init.orthogonal_(self.B)
                
                # Temperature annealing for coefficient computation
                self.tau_init = config['arch']['phys_vae'].get('tau_init', 3.0)  # Initial temperature
                self.tau_final = config['arch']['phys_vae'].get('tau_final', 1.0)  # Final temperature
                self.tau_warmup_epochs = config['arch']['phys_vae'].get('tau_warmup_epochs', 20)
                
                # Global residual scale warmup
                self.r_init = config['arch']['phys_vae'].get('r_init', 0.0)
                self.r_final = config['arch']['phys_vae'].get('r_final', 1.0)
                self.r_warmup_epochs = config['arch']['phys_vae'].get('r_warmup_epochs', 20)
                
                # Orthogonality penalty weight
                self.ortho_penalty_weight = config['arch']['phys_vae'].get('ortho_penalty_weight', 0.1)
            else:
                # When dim_z_aux = 0, set default values for tau/r methods to avoid errors
                self.tau_init = 1.0
                self.tau_final = 1.0
                self.tau_warmup_epochs = 0
                self.r_init = 0.0
                self.r_final = 0.0
                self.r_warmup_epochs = 0
        else:
            # no phy
            if dim_z_aux > 0:
                self.func_aux_dec = MLP([dim_z_aux, 16, 32, 64, in_channels], activation)
            else:
                # If dim_z_aux = 0, create a simple identity mapping
                self.func_aux_dec = nn.Identity()

    def get_tau(self, epoch, epochs_pretrain):
        """Get current temperature for annealing (starts after pretraining)"""
        if epoch < epochs_pretrain:
            return self.tau_init  # Keep initial temperature during pretraining
        elif epoch < epochs_pretrain + self.tau_warmup_epochs:
            progress = (epoch - epochs_pretrain) / self.tau_warmup_epochs
            return self.tau_init + progress * (self.tau_final - self.tau_init)
        return self.tau_final
    
    def get_r(self, epoch, epochs_pretrain):
        """Get current global residual scale (starts after pretraining)"""
        if epoch < epochs_pretrain:
            return self.r_init  # Keep initial scale during pretraining
        elif epoch < epochs_pretrain + self.r_warmup_epochs:
            progress = (epoch - epochs_pretrain) / self.r_warmup_epochs
            return self.r_init + progress * (self.r_final - self.r_init)
        return self.r_final
    
    def get_current_tau_r(self, epoch, epochs_pretrain):
        """Get current tau and r values for saving in checkpoint"""
        return {
            'tau': self.get_tau(epoch, epochs_pretrain),
            'r': self.get_r(epoch, epochs_pretrain),
            'epoch': epoch,
            'epochs_pretrain': epochs_pretrain
        }
    
    def set_tau_r_from_checkpoint(self, tau_r_dict):
        """Set tau and r values from checkpoint for inference"""
        if tau_r_dict is not None:
            self._inference_tau = tau_r_dict.get('tau', self.tau_final)
            self._inference_r = tau_r_dict.get('r', self.r_final)
        else:
            self._inference_tau = self.tau_final
            self._inference_r = self.r_final
    
    def get_tau_for_inference(self):
        """Get tau value for inference (uses saved value if available)"""
        return getattr(self, '_inference_tau', self.tau_final)
    
    def get_r_for_inference(self):
        """Get r value for inference (uses saved value if available)"""
        return getattr(self, '_inference_r', self.r_final)
    
    def compute_coefficient(self, z_aux, x_P_det, epoch, epochs_pretrain, use_inference_values=False, time_feats=None):
        """
        IMPROVED: Compute coefficient from [z_aux, x_P_det, time_feats] with temperature annealing
        c_raw = Linear([z_aux, x_P_det, time_feats])
        c = tanh(c_raw / tau)
        """
        # Concatenate z_aux, physics context, and time features
        if time_feats is not None and self.time_feat_dim > 0 and self.use_time_in_residual:
            # Use only the first time_feat_dim elements of the 4-dim time features
            time_feats_sliced = time_feats[..., :self.time_feat_dim]
            coeff_input = torch.cat([z_aux, x_P_det, time_feats_sliced], dim=1)
        else:
            coeff_input = torch.cat([z_aux, x_P_det], dim=1)
        c_raw = self.coeff(coeff_input)  # (batch_size, residual_rank)
        
        # Apply temperature annealing (starts after pretraining)
        if use_inference_values:
            tau = self.get_tau_for_inference()
        else:
            tau = self.get_tau(epoch, epochs_pretrain)
        c = torch.tanh(c_raw / tau)  # (batch_size, residual_rank)
        return c
    
    def orthogonality_penalty(self):
        """Compute orthogonality penalty for basis matrix B"""
        BtB = torch.matmul(self.B.T, self.B)
        I = torch.eye(self.B.shape[1], device=self.B.device)
        return torch.norm(BtB - I, p='fro') ** 2

class FeatureExtractor(nn.Module):
    """
    Feature extractor mapping observations -> a num_units_feat vector.

    encoder_type == 'mlp' (default): MLP over the [B, input_dim] point/LOS vector.
    encoder_type == 'cnn': convolutional encoder over the [B, C, H, W] full-field LOS
        image (full-field InSAR path). Conv blocks are ported from the user's Conv-VAE
        (Conv2d/BatchNorm/LeakyReLU); an adaptive average pool makes the output
        independent of the input grid size, then a Linear projects to num_units_feat.
    """
    def __init__(self, config:dict):
        super(FeatureExtractor, self).__init__()

        args = config['arch']['args']
        num_units_feat = config['arch']['phys_vae']['num_units_feat']
        activation = config['arch']['phys_vae']['activation']
        self.encoder_type = args.get('encoder_type', 'mlp')

        # --- Iterative refinement: optionally feed a current u-space estimate alongside x ---
        # When enabled, the feature extractor input is widened by dim_z_phy so the encoder
        # can refine its estimate over multiple passes (see PHYS_VAE_SMPL.encode_iterative).
        iter_cfg = config['arch']['phys_vae'].get('iterative_refinement', {})
        self.use_iterative = iter_cfg.get('enabled', False)
        self.dim_z_phy = config['arch']['phys_vae']['dim_z_phy']
        # Extra input width contributed by the concatenated u_phy_est vector (0 when disabled).
        extra_in = self.dim_z_phy if self.use_iterative else 0

        if self.encoder_type == 'cnn':
            cnn_in_channels = args.get('cnn_in_channels', 1)
            cnn_hidden_dims = args.get('cnn_hidden_dims', [32, 64, 128, 256])
            self.cnn_pool = args.get('cnn_pool', 2)  # adaptive output H=W=cnn_pool
            conv_blocks = []
            current_channels = cnn_in_channels
            for h_dim in cnn_hidden_dims:
                conv_blocks.append(nn.Sequential(
                    nn.Conv2d(current_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()))
                current_channels = h_dim
            self.conv = nn.Sequential(*conv_blocks)
            # Adaptive pool -> fixed bottleneck, so any grid size maps to num_units_feat.
            self.adaptive_pool = nn.AdaptiveAvgPool2d((self.cnn_pool, self.cnn_pool))
            flattened = cnn_hidden_dims[-1] * self.cnn_pool * self.cnn_pool
            # z_phy is a global (non-spatial) vector -> concatenate after the conv flatten,
            # so the projection MLP carries the extra dim_z_phy inputs.
            self.func_feat = MLP([flattened + extra_in, ] + [num_units_feat, ], activation)
        else:
            in_channels = args['input_dim']
            hidlayers_feat = config['arch']['phys_vae']['hidlayers_feat']
            self.func_feat = MLP([in_channels + extra_in, ] + hidlayers_feat + [num_units_feat, ], activation)

    def forward(self, x:torch.Tensor, t:torch.Tensor=None, u_phy_est:torch.Tensor=None):
        # Time features are not used in the feature extractor input.
        if self.encoder_type == 'cnn':
            h = self.conv(x)                       # [B, C', h, w]
            h = self.adaptive_pool(h)              # [B, C', cnn_pool, cnn_pool]
            h = torch.flatten(h, start_dim=1)      # [B, C'*pool*pool]
            if self.use_iterative:
                # Default a missing estimate to zeros (= centre of param range in u-space),
                # so callers that don't thread an estimate still match the widened input dim.
                if u_phy_est is None:
                    u_phy_est = torch.zeros(h.shape[0], self.dim_z_phy, device=h.device, dtype=h.dtype)
                h = torch.cat([h, u_phy_est], dim=1)
            return self.func_feat(h)               # [B, num_units_feat]
        if self.use_iterative:
            if u_phy_est is None:
                u_phy_est = torch.zeros(x.shape[0], self.dim_z_phy, device=x.device, dtype=x.dtype)
            x = torch.cat([x, u_phy_est], dim=1)   # [B, input_dim + dim_z_phy]
        return self.func_feat(x)                   # [B, num_units_feat]


class Physics_RTM(nn.Module):
    def __init__(self, config:dict):
        super(Physics_RTM, self).__init__()
        self.model = RTM()
        self.z_phy_ranges = json.load(open(os.path.join(PARENT_DIR, config['arch']['args']['rtm_paras']), 'r'))
        self.bands_index = [i for i in range(
            len(S2_FULL_BANDS)) if S2_FULL_BANDS[i] not in ['B01', 'B10']]
        # Mean and scale for standardization
        self.x_mean = torch.tensor(
            np.load(os.path.join(PARENT_DIR,config['arch']['args']['standardization']['x_mean']))
            ).float().unsqueeze(0).to(DEVICE)
        self.x_scale = torch.tensor(
            np.load(os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_scale']))
            ).float().unsqueeze(0).to(DEVICE)
    
    def rescale(self, z_phy:torch.Tensor):
        """
        Rescale z in (0,1) to physical parameters in original scales.
        """
        z_phy_rescaled = {}
        for i, para_name in enumerate(self.z_phy_ranges.keys()):
            z_phy_rescaled[para_name] = z_phy[:, i] * (
                self.z_phy_ranges[para_name]['max'] - self.z_phy_ranges[para_name]['min']
                ) + self.z_phy_ranges[para_name]['min']
        
        z_phy_rescaled['cd'] = torch.sqrt(
            (z_phy_rescaled['fc']*10000)/(torch.pi*SD))*2
        z_phy_rescaled['h'] = torch.exp(
            2.117 + 0.507*torch.log(z_phy_rescaled['cd'])) 
        
        return z_phy_rescaled
    
    def forward(self, z_phy:torch.Tensor, const:dict=None):
        z_phy_rescaled = self.rescale(z_phy)
        if const is not None:
            z_phy_rescaled.update(const)
        output = self.model.run(**z_phy_rescaled)[:, self.bands_index]
        return (output - self.x_mean) / self.x_scale 

class Physics_Mogi(nn.Module):
    def __init__(self, config:dict):
        super(Physics_Mogi, self).__init__()

        self.z_phy_ranges = json.load(open(os.path.join(PARENT_DIR, config['arch']['args']['mogi_paras']), 'r'))
        self.station_info = json.load(open(os.path.join(PARENT_DIR, config['arch']['args']['station_info']), 'r'))
        
        x = torch.tensor([self.station_info[k]['xE']
                         for k in self.station_info.keys()])*1000  # m
        y = torch.tensor([self.station_info[k]['yN']
                         for k in self.station_info.keys()])*1000  # m
        self.model = Mogi(x,y)
        
        # Mean and scale for standardization
        self.x_mean = torch.tensor(
            np.load(os.path.join(PARENT_DIR,config['arch']['args']['standardization']['x_mean']))
            ).float().unsqueeze(0).to(DEVICE)
        self.x_scale = torch.tensor(
            np.load(os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_scale']))
            ).float().unsqueeze(0).to(DEVICE)
    
    def rescale(self, z_phy:torch.Tensor):
        """
        Rescale z in (0,1) to physical parameters in the original scale.
        """
        z_phy_rescaled = {}
        for i, para_name in enumerate(self.z_phy_ranges.keys()):
            minv = self.z_phy_ranges[para_name]['min']
            maxv = self.z_phy_ranges[para_name]['max']
            if len(z_phy.shape) == 3:
                z_phy_rescaled[para_name] = z_phy[:, :, i]*(maxv-minv)+minv
            else:
                z_phy_rescaled[para_name] = z_phy[:, i]*(maxv-minv)+minv

            if para_name in ['xcen', 'ycen', 'd']:
                z_phy_rescaled[para_name] = z_phy_rescaled[para_name]*1000

        # dV magnitude transform: physical dV = (rescaled dV) * dV_scale + dV_shift.
        # Defaults (dV_scale=1e5, dV_shift=-1e7) reproduce the legacy hardcoded map, so
        # existing paras files and checkpoints are unchanged. A paras file may set
        # 'scale'/'shift' on the dV entry to define a different physical range -- e.g.
        # configs/mogi_paras_symdV.json uses min=-2e8, max=2e8, scale=1, shift=0 to make
        # a symmetric-about-0 range where z=0.5 (u=0, the prior mean) maps to dV=0 m^3.
        dV_scale = float(self.z_phy_ranges['dV'].get('scale', 1e5))
        dV_shift = float(self.z_phy_ranges['dV'].get('shift', -1e7))
        z_phy_rescaled['dV'] = z_phy_rescaled['dV'] * dV_scale + dV_shift

        return z_phy_rescaled
    
    def forward(self, z_phy:torch.Tensor, const:dict=None):
        z_phy_rescaled = self.rescale(z_phy)
        output = self.model.run(**z_phy_rescaled)
        return (output - self.x_mean) / self.x_scale

class Physics_Sun69(nn.Module):
    """
    Sun (1969) penny-shaped (horizontal circular) crack decoder, mirroring
    Physics_Mogi.

    z_phy (5-dim, in (0,1)) -> physical parameters -> Sun69.run -> ENU (mm),
    then standardized to the input space. Parameter order/names must match the
    keys in sun69_paras.json and Sun69.run's signature:
      xcen, ycen, depth, radius, dV
    (xcen,ycen = crack-centre offset; depth = H; radius = A; dV = injected volume).
    """
    def __init__(self, config:dict):
        super(Physics_Sun69, self).__init__()

        with open(os.path.join(PARENT_DIR, config['arch']['args']['sun69_paras']), 'r') as f:
            self.z_phy_ranges = json.load(f)
        with open(os.path.join(PARENT_DIR, config['arch']['args']['station_info']), 'r') as f:
            self.station_info = json.load(f)

        # rescale() maps z_phy[:, i] to a parameter by ENUMERATION order, so the
        # JSON key order must match Sun69.run's signature.
        expected_order = ['xcen', 'ycen', 'depth', 'radius', 'dV']
        assert list(self.z_phy_ranges.keys()) == expected_order, (
            f"sun69_paras.json keys must be in order {expected_order} to match "
            f"Sun69.run; got {list(self.z_phy_ranges.keys())}")

        x = torch.tensor([self.station_info[k]['xE']
                         for k in self.station_info.keys()])*1000  # km -> m
        y = torch.tensor([self.station_info[k]['yN']
                         for k in self.station_info.keys()])*1000  # km -> m
        self.model = Sun69(x, y)

        # km -> m parameters (dV handled separately by the same transform as Mogi)
        self.km_params = ['xcen', 'ycen', 'depth', 'radius']

        # Mean and scale for standardization
        self.x_mean = torch.tensor(
            np.load(os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_mean']))
            ).float().unsqueeze(0).to(DEVICE)
        self.x_scale = torch.tensor(
            np.load(os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_scale']))
            ).float().unsqueeze(0).to(DEVICE)

    def rescale(self, z_phy:torch.Tensor):
        """
        Rescale z in (0,1) to physical parameters in the original scale.
        """
        z_phy_rescaled = {}
        for i, para_name in enumerate(self.z_phy_ranges.keys()):
            minv = self.z_phy_ranges[para_name]['min']
            maxv = self.z_phy_ranges[para_name]['max']
            if len(z_phy.shape) == 3:
                val = z_phy[:, :, i]*(maxv-minv)+minv
            else:
                val = z_phy[:, i]*(maxv-minv)+minv

            if para_name in self.km_params:
                val = val*1000  # km -> m

            z_phy_rescaled[para_name] = val

        # Volume change uses the same data-driven magnitude transform as Mogi's dV so
        # the two sources share parameterization conventions:
        #   physical dV = (rescaled dV) * dV_scale + dV_shift.
        # Defaults (dV_scale=1e5, dV_shift=-1e7) reproduce the legacy hardcoded map; a
        # paras file may set 'scale'/'shift' on the dV entry for a symmetric-about-0
        # range (see Physics_Mogi.rescale and configs/mogi_paras_symdV.json).
        dV_scale = float(self.z_phy_ranges['dV'].get('scale', 1e5))
        dV_shift = float(self.z_phy_ranges['dV'].get('shift', -1e7))
        z_phy_rescaled['dV'] = z_phy_rescaled['dV'] * dV_scale + dV_shift

        return z_phy_rescaled

    def forward(self, z_phy:torch.Tensor, const:dict=None):
        z_phy_rescaled = self.rescale(z_phy)
        output = self.model.run(**z_phy_rescaled)
        return (output - self.x_mean) / self.x_scale

class Physics_Okada(nn.Module):
    """
    Opening-only Okada (1985) dike/sill decoder, mirroring Physics_Mogi.

    z_phy (8-dim, in (0,1)) -> physical parameters -> OkadaDike.run -> ENU (mm),
    then standardized to the input space. Parameter order/names must match the
    keys in okada_paras.json and OkadaDike.run's signature:
      xoff, yoff, depth, strike, dip, length, width, opening
    (xoff,yoff,depth = fault CENTROID; strike/dip in degrees; opening in m).
    """
    def __init__(self, config:dict):
        super(Physics_Okada, self).__init__()

        with open(os.path.join(PARENT_DIR, config['arch']['args']['okada_paras']), 'r') as f:
            self.z_phy_ranges = json.load(f)
        with open(os.path.join(PARENT_DIR, config['arch']['args']['station_info']), 'r') as f:
            self.station_info = json.load(f)

        # The rescale() below maps z_phy[:, i] to a parameter by ENUMERATION
        # order, so the JSON key order must match OkadaDike.run's signature.
        expected_order = ['xoff', 'yoff', 'depth', 'strike', 'dip',
                          'length', 'width', 'opening']
        assert list(self.z_phy_ranges.keys()) == expected_order, (
            f"okada_paras.json keys must be in order {expected_order} to match "
            f"OkadaDike.run; got {list(self.z_phy_ranges.keys())}")

        x = torch.tensor([self.station_info[k]['xE']
                         for k in self.station_info.keys()])*1000  # km -> m
        y = torch.tensor([self.station_info[k]['yN']
                         for k in self.station_info.keys()])*1000  # km -> m
        self.model = OkadaDike(x, y)

        # km -> m parameters (others kept in their native units: deg, deg, m)
        self.km_params = ['xoff', 'yoff', 'depth', 'length', 'width']

        # Mean and scale for standardization
        x_mean_path = config['arch']['args']['standardization']['x_mean']
        if 'mogi' in x_mean_path.lower():
            warnings.warn(
                f"Physics_Okada: standardization points to Mogi files "
                f"('{x_mean_path}'). These are placeholders — compute "
                f"Okada-specific x_mean/x_scale before training, or inferred "
                f"parameters will be biased.", UserWarning)
        self.x_mean = torch.tensor(
            np.load(os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_mean']))
            ).float().unsqueeze(0).to(DEVICE)
        self.x_scale = torch.tensor(
            np.load(os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_scale']))
            ).float().unsqueeze(0).to(DEVICE)

    def rescale(self, z_phy:torch.Tensor):
        """
        Rescale z in (0,1) to physical parameters in the original scale.
        """
        z_phy_rescaled = {}
        for i, para_name in enumerate(self.z_phy_ranges.keys()):
            minv = self.z_phy_ranges[para_name]['min']
            maxv = self.z_phy_ranges[para_name]['max']
            if len(z_phy.shape) == 3:
                val = z_phy[:, :, i]*(maxv-minv)+minv
            else:
                val = z_phy[:, i]*(maxv-minv)+minv

            if para_name in self.km_params:
                val = val*1000  # km -> m

            z_phy_rescaled[para_name] = val

        return z_phy_rescaled

    def forward(self, z_phy:torch.Tensor, const:dict=None):
        z_phy_rescaled = self.rescale(z_phy)
        output = self.model.run(**z_phy_rescaled)
        return (output - self.x_mean) / self.x_scale

# ---------------------------------------------------------------------------
# InSAR line-of-sight (LOS) physics decoders
# ---------------------------------------------------------------------------
# InSAR measures a single LOS scalar per pixel, not the full 3-D ENU vector that
# GNSS records. The LOS decoders below reuse the existing Mogi / Okada ENU
# forward models (and their rescale()) unchanged, then PROJECT the modelled
# [East | North | Up] displacement onto each observation point's LOS unit
# vector. Output dimension is therefore N (number of points), not 3N, so
# arch.args.input_dim must equal N for these decoders.

def _load_points_los(config:dict):
    """
    Load InSAR observation points: local ENU coordinates + per-point LOS vectors.

    Reads the points-info JSON (the InSAR analog of station_info.json). Each
    entry must provide:
        xE, yN     : local ENU coordinates of the point, in km (same origin as
                     the Mogi/Okada source-parameter ranges)
        los_E, los_N, los_U : ground->satellite LOS unit-vector components
                     (convention: positive LOS = motion toward the satellite)

    Returns
    -------
    points_info : dict             the raw, order-preserving JSON dict
    x_east_m    : Tensor [N]       East coordinates in metres (CPU, mirrors parent)
    y_north_m   : Tensor [N]       North coordinates in metres (CPU, mirrors parent)
    los_E, los_N, los_U : Tensor [1, N]  LOS unit-vector components on DEVICE
    """
    points_path = os.path.join(PARENT_DIR, config['arch']['args']['points_info'])
    with open(points_path, 'r') as f:
        points_info = json.load(f)
    point_keys = list(points_info.keys())

    # Coordinates: km -> m. Created as plain CPU tensors exactly like the parent
    # Physics_Mogi/Physics_Okada classes (Mogi/OkadaDike are not nn.Modules, so
    # these are not moved by model.to(device); the forward() below aligns devices).
    x_east_m = torch.tensor([points_info[k]['xE'] for k in point_keys]) * 1000.0   # km -> m
    y_north_m = torch.tensor([points_info[k]['yN'] for k in point_keys]) * 1000.0  # km -> m

    # LOS unit vectors as [1, N] row tensors for broadcasting over the batch.
    los_E = torch.tensor([[points_info[k]['los_E'] for k in point_keys]]).float().to(DEVICE)
    los_N = torch.tensor([[points_info[k]['los_N'] for k in point_keys]]).float().to(DEVICE)
    los_U = torch.tensor([[points_info[k]['los_U'] for k in point_keys]]).float().to(DEVICE)
    return points_info, x_east_m, y_north_m, los_E, los_N, los_U


def _load_insar_inputs(config:dict):
    """
    Observation inputs for the LOS decoders, from EITHER an h5 'insar' block (MintPy
    direct, multilooked) OR the legacy points-info JSON + standardization .npy files.

    Returns
    -------
    x_east_m, y_north_m : Tensor [N]   coordinates in metres (CPU, mirrors parent classes)
    los_E, los_N, los_U : Tensor [1,N] LOS unit-vector components (DEVICE)
    x_mean, x_scale     : Tensor [1,N] per-point standardization (DEVICE)
    n_points            : int
    """
    args = config['arch']['args']
    if 'insar' in args:
        # h5-native: derive points + per-point scaler from the SAME memoized loader the
        # dataset uses, so the dataset and the decoder agree on identical points exactly.
        from datasets.preprocessing.insar_mintpy import load_insar_mintpy
        ins = args['insar']
        d = load_insar_mintpy(
            ins['timeseries'], ins['geometry'], ins.get('mask'),
            ins['lat0'], ins['lon0'], multilook=ins.get('multilook', 1),   # default 1 = no multilook
            coh_valid_frac=ins.get('coh_valid_frac', 0.5), bbox=ins.get('bbox'), verbose=False,
            standardization=ins.get('std_mode', 'global'), far_field=ins.get('far_field'),
            stride=ins.get('stride', 1), offset=ins.get('offset', 0),   # UQ decimation (must match dataset)
            bootstrap_k=ins.get('bootstrap_k'), bootstrap_block=ins.get('bootstrap_block', 3),
            bootstrap_seed=ins.get('bootstrap_seed', 0),
            ref_date=ins.get('ref_date'))          # temporal re-referencing (must match dataset)
        x_east_m = torch.tensor(d.xE_pts.astype(np.float32)) * 1000.0   # km -> m
        y_north_m = torch.tensor(d.yN_pts.astype(np.float32)) * 1000.0
        los_E = torch.tensor(d.losE_pts[None, :]).float().to(DEVICE)
        los_N = torch.tensor(d.losN_pts[None, :]).float().to(DEVICE)
        los_U = torch.tensor(d.losU_pts[None, :]).float().to(DEVICE)
        # GLOBAL (scene-wide) standardization, not per-point. An InSAR scene is ~94%
        # quiescent; per-point z-scoring divides each quiet cell by its own ~noise std,
        # inflating noise to signal scale and flattening the real deformation (the caldera
        # drops from ~24 sigma to ~1.7 sigma), so MSE collapses to "predict 0 everywhere".
        # A single global scale keeps the deformation dominant. [1,1] broadcasts over N.
        x_mean = torch.tensor([[d.x_mean_global]]).float().to(DEVICE)
        x_scale = torch.tensor([[d.x_scale_global]]).float().to(DEVICE)
        return x_east_m, y_north_m, los_E, los_N, los_U, x_mean, x_scale, d.n_points

    # legacy JSON + npy path
    points_info, x_east_m, y_north_m, los_E, los_N, los_U = _load_points_los(config)
    x_mean = torch.tensor(
        np.load(os.path.join(PARENT_DIR, args['standardization']['x_mean']))
        ).float().unsqueeze(0).to(DEVICE)
    x_scale = torch.tensor(
        np.load(os.path.join(PARENT_DIR, args['standardization']['x_scale']))
        ).float().unsqueeze(0).to(DEVICE)
    return x_east_m, y_north_m, los_E, los_N, los_U, x_mean, x_scale, len(points_info)


def _load_insar_grid_inputs(config:dict):
    """
    Full-field grid inputs for the CNN LOS decoders (renders the whole scene image).

    Uses ALL coarse grid cells (row-major over Hd x Wd, matching the dataset image's
    flatten order), not just coherent ones — incoherent cells have zero LOS vectors so
    they project to ~0 and are excluded by the masked loss.

    Returns
    -------
    x_east_m, y_north_m : Tensor [Ncells]  coordinates in metres (CPU)
    los_E, los_N, los_U : Tensor [1,Ncells] LOS unit-vector components (DEVICE)
    x_mean, x_scale     : Tensor [1,1]      global z-score (DEVICE)
    Hd, Wd, n_cells     : ints
    """
    from datasets.preprocessing.insar_mintpy import load_insar_mintpy
    ins = config['arch']['args']['insar']
    d = load_insar_mintpy(
        ins['timeseries'], ins['geometry'], ins.get('mask'),
        ins['lat0'], ins['lon0'], multilook=ins.get('multilook', 1),   # default 1 = no multilook
        coh_valid_frac=ins.get('coh_valid_frac', 0.5), bbox=ins.get('bbox'), verbose=False,
        standardization=ins.get('std_mode', 'global'), far_field=ins.get('far_field'),
        stride=ins.get('stride', 1), offset=ins.get('offset', 0),   # UQ decimation (grid path)
        bootstrap_k=ins.get('bootstrap_k'), bootstrap_block=ins.get('bootstrap_block', 3),
        bootstrap_seed=ins.get('bootstrap_seed', 0),
        ref_date=ins.get('ref_date'))          # temporal re-referencing (must match dataset)
    Hd, Wd = d.mask_d.shape
    # CRITICAL: cell order here MUST match how the trainer flattens the input image
    # (torch `data.reshape(B, -1)`, i.e. C-order/row-major over Hd x Wd). flat() forces
    # C-order so physics cell i corresponds to image pixel i; a mismatch would silently
    # regress each cell against the wrong pixel.
    def flat(a):
        return np.ascontiguousarray(a, dtype=np.float32).reshape(-1)   # row-major, C-order
    x_east_m = torch.tensor(flat(d.xE_grid)) * 1000.0   # km -> m
    y_north_m = torch.tensor(flat(d.yN_grid)) * 1000.0
    los_E = torch.tensor(flat(d.losE_grid)[None, :]).to(DEVICE)
    los_N = torch.tensor(flat(d.losN_grid)[None, :]).to(DEVICE)
    los_U = torch.tensor(flat(d.losU_grid)[None, :]).to(DEVICE)
    x_mean = torch.tensor([[d.x_mean_global]]).float().to(DEVICE)
    x_scale = torch.tensor([[d.x_scale_global]]).float().to(DEVICE)
    return x_east_m, y_north_m, los_E, los_N, los_U, x_mean, x_scale, Hd, Wd, Hd * Wd


def project_enu_to_los(enu:torch.Tensor, los_E:torch.Tensor, los_N:torch.Tensor,
                       los_U:torch.Tensor, n_points:int):
    """
    Project concatenated ENU surface displacement onto per-point LOS unit vectors.

    Parameters
    ----------
    enu : Tensor [batch, 3*N]      [East | North | Up] displacement in mm, the
                                   layout returned by Mogi.run / OkadaDike.run.
    los_E, los_N, los_U : Tensor [1, N]  ground->satellite LOS unit-vector components.
    n_points : int                 number of observation points N.

    Returns
    -------
    d_los : Tensor [batch, N]      LOS displacement in mm
            (positive = motion toward the satellite, i.e. range decrease).
    """
    assert enu.shape[1] == 3 * n_points, (
        f"ENU tensor has {enu.shape[1]} columns but expected 3*n_points="
        f"{3 * n_points} ([East|North|Up] concat)")
    u_east = enu[:, :n_points]                    # [batch, N] East,  mm
    u_north = enu[:, n_points:2 * n_points]       # [batch, N] North, mm
    u_up = enu[:, 2 * n_points:]                  # [batch, N] Up,    mm
    # Align LOS vectors to the displacement's device (CPU/GPU robust).
    return (los_E.to(enu.device) * u_east
            + los_N.to(enu.device) * u_north
            + los_U.to(enu.device) * u_up)


class Physics_Mogi_LOS(Physics_Mogi):
    """
    LOS-projected Mogi decoder for InSAR.

    Identical physics to Physics_Mogi (same 4-D z_phy -> ENU forward and the same
    rescale(), inherited), but observation points carry per-point LOS unit
    vectors and the output is the scalar LOS projection d_LOS = e . u (mm) at
    each point, then standardized. input_dim therefore equals N (number of
    points), not 3N.
    """
    def __init__(self, config:dict):
        # Skip Physics_Mogi.__init__ (it loads station_info + 3N standardization);
        # set up the LOS-specific attributes here while reusing rescale() and Mogi.
        nn.Module.__init__(self)

        with open(os.path.join(PARENT_DIR, config['arch']['args']['mogi_paras']), 'r') as f:
            self.z_phy_ranges = json.load(f)

        # Points + per-point standardization from h5 (insar block) or legacy JSON/npy.
        (x_east_m, y_north_m, self.los_E, self.los_N, self.los_U,
         self.x_mean, self.x_scale, self.n_points) = _load_insar_inputs(config)
        assert self.n_points == config['arch']['args']['input_dim'], (
            f"InSAR points N={self.n_points} != arch.args.input_dim="
            f"{config['arch']['args']['input_dim']}; they must match")
        self.model = Mogi(x_east_m, y_north_m)

    def forward(self, z_phy:torch.Tensor, const:dict=None):
        z_phy_rescaled = self.rescale(z_phy)              # inherited from Physics_Mogi
        enu = self.model.run(**z_phy_rescaled)           # [batch, 3N] in mm
        d_los = project_enu_to_los(enu, self.los_E, self.los_N, self.los_U, self.n_points)
        return (d_los - self.x_mean.to(enu.device)) / self.x_scale.to(enu.device)


class Physics_Okada_LOS(Physics_Okada):
    """
    LOS-projected opening-only Okada (1985) dike/sill decoder for InSAR.

    Identical physics to Physics_Okada (same 8-D z_phy -> ENU forward and the
    same rescale()/km_params, inherited), but returns the LOS projection at each
    observation point instead of full ENU. input_dim equals N (number of points).
    """
    def __init__(self, config:dict):
        # Skip Physics_Okada.__init__ (station_info + 3N standardization + Mogi
        # warning); set up LOS-specific attributes while reusing rescale().
        nn.Module.__init__(self)

        with open(os.path.join(PARENT_DIR, config['arch']['args']['okada_paras']), 'r') as f:
            self.z_phy_ranges = json.load(f)

        # Same enumeration-order requirement as Physics_Okada: rescale() maps
        # z_phy[:, i] to a parameter by JSON key order, which must match run().
        expected_order = ['xoff', 'yoff', 'depth', 'strike', 'dip',
                          'length', 'width', 'opening']
        assert list(self.z_phy_ranges.keys()) == expected_order, (
            f"okada_paras.json keys must be in order {expected_order} to match "
            f"OkadaDike.run; got {list(self.z_phy_ranges.keys())}")

        # Points + per-point standardization from h5 (insar block) or legacy JSON/npy.
        # (Standardization in LOS-mm space is physics-agnostic — it is derived from the
        # observations, so Mogi and Okada share the same scaler for the same scene.)
        (x_east_m, y_north_m, self.los_E, self.los_N, self.los_U,
         self.x_mean, self.x_scale, self.n_points) = _load_insar_inputs(config)
        assert self.n_points == config['arch']['args']['input_dim'], (
            f"InSAR points N={self.n_points} != arch.args.input_dim="
            f"{config['arch']['args']['input_dim']}; they must match")
        self.model = OkadaDike(x_east_m, y_north_m)

        # km -> m parameters (inherited rescale() reads this attribute).
        self.km_params = ['xoff', 'yoff', 'depth', 'length', 'width']

    def forward(self, z_phy:torch.Tensor, const:dict=None):
        z_phy_rescaled = self.rescale(z_phy)             # inherited from Physics_Okada
        enu = self.model.run(**z_phy_rescaled)          # [batch, 3N] in mm
        d_los = project_enu_to_los(enu, self.los_E, self.los_N, self.los_U, self.n_points)
        return (d_los - self.x_mean.to(enu.device)) / self.x_scale.to(enu.device)


class Physics_Mogi_LOS_Grid(Physics_Mogi):
    """
    Full-field (CNN-path) Mogi LOS decoder: renders LOS over the WHOLE coarse grid.

    Same 4-D z_phy -> ENU forward + rescale() as Physics_Mogi, but observation points are
    ALL Hd x Wd grid cells (row-major), the output is the global-standardized LOS at every
    cell flattened to [batch, Hd*Wd], and standardization is a single global z-score. The
    trainer reshapes/masks this against the input image. input_dim therefore equals Hd*Wd.
    """
    def __init__(self, config:dict):
        nn.Module.__init__(self)
        with open(os.path.join(PARENT_DIR, config['arch']['args']['mogi_paras']), 'r') as f:
            self.z_phy_ranges = json.load(f)
        (x_east_m, y_north_m, self.los_E, self.los_N, self.los_U,
         self.x_mean, self.x_scale, self.Hd, self.Wd, self.n_cells) = _load_insar_grid_inputs(config)
        self.model = Mogi(x_east_m, y_north_m)

    def forward(self, z_phy:torch.Tensor, const:dict=None):
        z_phy_rescaled = self.rescale(z_phy)              # inherited from Physics_Mogi
        enu = self.model.run(**z_phy_rescaled)            # [batch, 3*Ncells] mm
        d_los = project_enu_to_los(enu, self.los_E, self.los_N, self.los_U, self.n_cells)
        return (d_los - self.x_mean.to(enu.device)) / self.x_scale.to(enu.device)  # [batch, Ncells]


class Physics_Okada_LOS_Grid(Physics_Okada):
    """
    Full-field (CNN-path) opening-only Okada LOS decoder over the WHOLE coarse grid.
    Mirrors Physics_Mogi_LOS_Grid; same 8-D z_phy -> ENU forward + rescale() as Physics_Okada.
    """
    def __init__(self, config:dict):
        nn.Module.__init__(self)
        with open(os.path.join(PARENT_DIR, config['arch']['args']['okada_paras']), 'r') as f:
            self.z_phy_ranges = json.load(f)
        expected_order = ['xoff', 'yoff', 'depth', 'strike', 'dip',
                          'length', 'width', 'opening']
        assert list(self.z_phy_ranges.keys()) == expected_order, (
            f"okada_paras.json keys must be in order {expected_order} to match "
            f"OkadaDike.run; got {list(self.z_phy_ranges.keys())}")
        (x_east_m, y_north_m, self.los_E, self.los_N, self.los_U,
         self.x_mean, self.x_scale, self.Hd, self.Wd, self.n_cells) = _load_insar_grid_inputs(config)
        self.model = OkadaDike(x_east_m, y_north_m)
        self.km_params = ['xoff', 'yoff', 'depth', 'length', 'width']

    def forward(self, z_phy:torch.Tensor, const:dict=None):
        z_phy_rescaled = self.rescale(z_phy)              # inherited from Physics_Okada
        enu = self.model.run(**z_phy_rescaled)            # [batch, 3*Ncells] mm
        d_los = project_enu_to_los(enu, self.los_E, self.los_N, self.los_U, self.n_cells)
        return (d_los - self.x_mean.to(enu.device)) / self.x_scale.to(enu.device)  # [batch, Ncells]


class Physics_Sun69_LOS(Physics_Sun69):
    """
    LOS-projected Sun (1969) penny-shaped crack decoder for InSAR.

    Identical physics to Physics_Sun69 (same 5-D z_phy -> ENU forward and the
    same rescale()/km_params, inherited), but returns the LOS projection at each
    observation point instead of full ENU. input_dim equals N (number of points).
    """
    def __init__(self, config:dict):
        # Skip Physics_Sun69.__init__ (station_info + 3N standardization); set up
        # LOS-specific attributes here while reusing rescale() and Sun69.
        nn.Module.__init__(self)

        with open(os.path.join(PARENT_DIR, config['arch']['args']['sun69_paras']), 'r') as f:
            self.z_phy_ranges = json.load(f)

        # Same enumeration-order requirement as Physics_Sun69: rescale() maps
        # z_phy[:, i] to a parameter by JSON key order, which must match run().
        expected_order = ['xcen', 'ycen', 'depth', 'radius', 'dV']
        assert list(self.z_phy_ranges.keys()) == expected_order, (
            f"sun69_paras.json keys must be in order {expected_order} to match "
            f"Sun69.run; got {list(self.z_phy_ranges.keys())}")

        # Points + per-point standardization from h5 (insar block) or legacy JSON/npy.
        (x_east_m, y_north_m, self.los_E, self.los_N, self.los_U,
         self.x_mean, self.x_scale, self.n_points) = _load_insar_inputs(config)
        assert self.n_points == config['arch']['args']['input_dim'], (
            f"InSAR points N={self.n_points} != arch.args.input_dim="
            f"{config['arch']['args']['input_dim']}; they must match")
        self.model = Sun69(x_east_m, y_north_m)

        # km -> m parameters (inherited rescale() reads this attribute).
        self.km_params = ['xcen', 'ycen', 'depth', 'radius']

    def forward(self, z_phy:torch.Tensor, const:dict=None):
        z_phy_rescaled = self.rescale(z_phy)             # inherited from Physics_Sun69
        enu = self.model.run(**z_phy_rescaled)           # [batch, 3N] in mm
        d_los = project_enu_to_los(enu, self.los_E, self.los_N, self.los_U, self.n_points)
        return (d_los - self.x_mean.to(enu.device)) / self.x_scale.to(enu.device)


class Physics_Sun69_LOS_Grid(Physics_Sun69):
    """
    Full-field (CNN-path) Sun (1969) penny-shaped crack LOS decoder over the
    WHOLE coarse grid. Mirrors Physics_Mogi_LOS_Grid; same 5-D z_phy -> ENU
    forward + rescale() as Physics_Sun69.
    """
    def __init__(self, config:dict):
        nn.Module.__init__(self)
        with open(os.path.join(PARENT_DIR, config['arch']['args']['sun69_paras']), 'r') as f:
            self.z_phy_ranges = json.load(f)
        expected_order = ['xcen', 'ycen', 'depth', 'radius', 'dV']
        assert list(self.z_phy_ranges.keys()) == expected_order, (
            f"sun69_paras.json keys must be in order {expected_order} to match "
            f"Sun69.run; got {list(self.z_phy_ranges.keys())}")
        (x_east_m, y_north_m, self.los_E, self.los_N, self.los_U,
         self.x_mean, self.x_scale, self.Hd, self.Wd, self.n_cells) = _load_insar_grid_inputs(config)
        self.model = Sun69(x_east_m, y_north_m)
        self.km_params = ['xcen', 'ycen', 'depth', 'radius']

    def forward(self, z_phy:torch.Tensor, const:dict=None):
        z_phy_rescaled = self.rescale(z_phy)              # inherited from Physics_Sun69
        enu = self.model.run(**z_phy_rescaled)            # [batch, 3*Ncells] mm
        d_los = project_enu_to_los(enu, self.los_E, self.los_N, self.los_U, self.n_cells)
        return (d_los - self.x_mean.to(enu.device)) / self.x_scale.to(enu.device)  # [batch, Ncells]


def _maybe_set_insar_input_dim(config:dict):
    """
    For the h5-native point/MLP LOS path, set arch.args.input_dim from the multilooked
    coherent-cell count BEFORE the encoder/decoder are built, so configs don't have to
    hardcode N (which depends on the multilook factor and mask). No-op otherwise.
    """
    args = config['arch']['args']
    if args.get('physics') not in ('Mogi_LOS', 'Okada_LOS', 'Sun69_LOS') or 'insar' not in args:
        return
    from datasets.preprocessing.insar_mintpy import load_insar_mintpy
    ins = args['insar']
    d = load_insar_mintpy(
        ins['timeseries'], ins['geometry'], ins.get('mask'),
        ins['lat0'], ins['lon0'], multilook=ins.get('multilook', 1),   # default 1 = no multilook
        coh_valid_frac=ins.get('coh_valid_frac', 0.5), bbox=ins.get('bbox'), verbose=False,
        standardization=ins.get('std_mode', 'global'), far_field=ins.get('far_field'),
        stride=ins.get('stride', 1), offset=ins.get('offset', 0),   # UQ decimation -> input_dim = N/stride
        bootstrap_k=ins.get('bootstrap_k'), bootstrap_block=ins.get('bootstrap_block', 3),
        bootstrap_seed=ins.get('bootstrap_seed', 0),
        ref_date=ins.get('ref_date'))          # temporal re-referencing (must match dataset)
    # CNN path renders the whole Hd x Wd grid -> input_dim = Hd*Wd (flattened image);
    # point/MLP path uses the N coherent cells -> input_dim = N.
    if args.get('encoder_type', 'mlp') == 'cnn':
        new_dim = int(d.mask_d.shape[0] * d.mask_d.shape[1])
        label = f"{d.mask_d.shape[0]}x{d.mask_d.shape[1]} grid"
    else:
        new_dim = d.n_points
        label = "MintPy multilook"
    if args.get('input_dim') != new_dim:
        print(f"[PHYS_VAE_SMPL] input_dim set to {new_dim} from {label} "
              f"(was {args.get('input_dim')})")
        args['input_dim'] = new_dim


class PHYS_VAE_SMPL(nn.Module):
    def __init__(self, config:dict):
        super(PHYS_VAE_SMPL, self).__init__()

        # h5-native LOS: derive input_dim (= N coherent cells) before building encoders.
        _maybe_set_insar_input_dim(config)

        self.no_phy = config['arch']['phys_vae']['no_phy']
        self.dim_z_aux = config['arch']['phys_vae']['dim_z_aux']
        self.dim_z_phy = config['arch']['phys_vae']['dim_z_phy']
        self.activation = config['arch']['phys_vae']['activation']
        self.in_channels = config['arch']['args']['input_dim']
        self.detach_x_P_for_bias = config['arch']['phys_vae'].get('detach_x_P_for_bias', True)

        # --- Iterative refinement encoder configuration (default: disabled) ---
        # When enabled, encode() accepts a current u-space estimate and the encoder is run
        # for multiple refinement passes (encode_iterative / encode_iterative_infer).
        iter_cfg = config['arch']['phys_vae'].get('iterative_refinement', {})
        self.use_iterative = iter_cfg.get('enabled', False)
        self.num_passes_train = iter_cfg.get('num_passes_train', 3)   # refinement passes in training
        self.num_passes_infer = iter_cfg.get('num_passes_infer', 5)   # max passes at inference
        self.convergence_tol = iter_cfg.get('convergence_tol', 1e-4)  # early-stop tol on z_phy change
        self.initial_u_noise_std = iter_cfg.get('initial_u_noise_std', 0.0)  # init-estimate jitter (train only)
        self._last_num_passes = 0  # bookkeeping: passes actually taken at last inference

        # EMA prior configuration
        self.use_ema_prior = config['trainer']['phys_vae'].get('use_ema_prior', False)
        self.ema_momentum = config['trainer']['phys_vae'].get('ema_momentum', 0.99)
        
        # EMA variance bounds (hardcoded)
        self.ema_min_var = 1e-3  # variance floor
        self.ema_max_var = 50.0  # variance ceiling

        # Encoding part
        self.enc = Encoders(config)

        # Decoding part
        self.dec = Decoders(config)

        # Physics
        self.physics_model = self.physics_init(config)
        
        # EMA buffers for u_phy statistics (only if EMA prior is enabled)
        if self.use_ema_prior and not self.no_phy:
            self.register_buffer('ema_mean', torch.zeros(self.dim_z_phy))  # E[U]
            self.register_buffer('ema_m2', torch.ones(self.dim_z_phy))     # E[U^2]
            self.register_buffer('ema_var', torch.ones(self.dim_z_phy))    # convenience buffer
        
        # Store time features for decoder use
        self.time_feats = None
    
    def physics_init(self, config:dict):
        if config['arch']['args']['physics'] == 'RTM':
            return Physics_RTM(config)
        elif config['arch']['args']['physics'] == 'Mogi':
            return Physics_Mogi(config)
        elif config['arch']['args']['physics'] == 'Okada':
            return Physics_Okada(config)
        elif config['arch']['args']['physics'] == 'Mogi_LOS':
            if config['arch']['args'].get('encoder_type', 'mlp') == 'cnn':
                return Physics_Mogi_LOS_Grid(config)
            return Physics_Mogi_LOS(config)
        elif config['arch']['args']['physics'] == 'Okada_LOS':
            if config['arch']['args'].get('encoder_type', 'mlp') == 'cnn':
                return Physics_Okada_LOS_Grid(config)
            return Physics_Okada_LOS(config)
        elif config['arch']['args']['physics'] == 'Sun69':
            return Physics_Sun69(config)
        elif config['arch']['args']['physics'] == 'Sun69_LOS':
            if config['arch']['args'].get('encoder_type', 'mlp') == 'cnn':
                return Physics_Sun69_LOS_Grid(config)
            return Physics_Sun69_LOS(config)
        else:
            raise ValueError("Unknown model type")
        
    def generate_physonly(self, z_phy:torch.Tensor, const:dict=None):
        # here z_phy is in (0,1)
        y = self.physics_model(z_phy, const=const) # (n, in_channels) 
        return y

    def update_ema_prior(self, u_phy_mean: torch.Tensor, u_phy_lnvar: torch.Tensor):
        """
        Update EMA statistics for u_phy using exponential moving average.
        
        Args:
            u_phy_mean: Current batch posterior means (shape: [batch_size, dim_z_phy])
            u_phy_lnvar: Current batch posterior log-variances (shape: [batch_size, dim_z_phy])
        """
        if not self.use_ema_prior or self.no_phy:
            return
            
        with torch.no_grad():
            decay = self.ema_momentum  # e.g., 0.999
            mu_q = u_phy_mean.detach()                 # (B, D)
            var_q = torch.exp(u_phy_lnvar.detach())    # (B, D)

            # First moment E[U]
            batch_mu = mu_q.mean(dim=0)                # (D,)

            # Second moment E[U^2] = E[var + mu^2]
            batch_m2 = (var_q + mu_q**2).mean(dim=0)   # (D,)

            # EMA updates
            self.ema_mean.mul_(decay).add_(batch_mu, alpha=1 - decay)
            self.ema_m2.mul_(decay).add_(batch_m2, alpha=1 - decay)

            # Convert moments to variance and clamp
            ema_var = (self.ema_m2 - self.ema_mean**2).clamp(self.ema_min_var, self.ema_max_var)
            self.ema_var.copy_(ema_var)

    def priors(self, n:int, device:torch.device):
        """
        CHANGED: priors now in u-space for physics (standard normal or EMA Gaussian),
        auxiliaries remain standard normal as before.
        """
        if self.use_ema_prior and not self.no_phy:
            # Use EMA Gaussian prior for u_phy
            ema_var = (self.ema_m2 - self.ema_mean**2).clamp(self.ema_min_var, self.ema_max_var)
            prior_u_phy_stat = {
                'mean': self.ema_mean.unsqueeze(0).expand(n, -1),
                'lnvar': ema_var.log().unsqueeze(0).expand(n, -1)
            }
        else:
            # Use standard normal prior for u_phy
            prior_u_phy_stat = {'mean': torch.zeros(n, self.dim_z_phy, device=device),
                                'lnvar': torch.zeros(n, self.dim_z_phy, device=device)}
        
        prior_z_aux_stat = {'mean': torch.zeros(n, max(0,self.dim_z_aux), device=device),
                            'lnvar': torch.zeros(n, max(0,self.dim_z_aux), device=device)}
        return prior_u_phy_stat, prior_z_aux_stat

    def encode(self, x:torch.Tensor, t:torch.Tensor=None, u_phy_est:torch.Tensor=None):
        """
        CHANGED: z_aux encoding from feature only (as before).
        Now supports optional time features.

        u_phy_est: optional current u-space estimate for iterative refinement. Ignored unless
        iterative refinement is enabled (the FeatureExtractor concatenates it to its input).
        """
        x_ = x
        n = x_.shape[0]
        device = x_.device

        # Store time features for decoder use
        self.time_feats = t

        feature = self.enc.func_feat(x_, t, u_phy_est=u_phy_est)

        # infer z_aux from feature only
        if self.dim_z_aux > 0:
            z_aux_stat = {'mean': self.enc.func_z_aux_mean(feature),
                          'lnvar': self.enc.func_z_aux_lnvar(feature)}
        else:
            z_aux_stat = {'mean': torch.empty(n, 0, device=device),
                          'lnvar': torch.empty(n, 0, device=device)}

        # infer u_phy stats (stored in z_phy_stat for backward-compat)
        if not self.no_phy:
            z_phy_stat = {'mean': self.enc.func_z_phy_mean(feature),   # u-mean
                          'lnvar': self.enc.func_z_phy_lnvar(feature)} # u-lnvar
        else:
            z_phy_stat = {'mean': torch.empty(n, 0, device=device),
                          'lnvar': torch.empty(n, 0, device=device)}

        return z_phy_stat, z_aux_stat

    def draw(self, z_phy_stat:dict, z_aux_stat:dict, hard_z_phy:bool=False, hard_z_aux:bool=False):
        """
        Sample in u-space, then squash to z in (0,1).
        z_aux remains in u-space (unbounded) for coefficient computation.
        
        Args:
            hard_z_phy: If True, use deterministic sampling for z_phy (mean)
            hard_z_aux: If True, use deterministic sampling for z_aux (mean)
        """
        # Sample z_phy based on its individual setting
        if not hard_z_phy:
            u_phy = draw_normal(z_phy_stat['mean'], z_phy_stat['lnvar'])
        else:
            u_phy = z_phy_stat['mean'].clone()
            
        # Sample z_aux based on its individual setting
        if not hard_z_aux:
            z_aux = draw_normal(z_aux_stat['mean'], z_aux_stat['lnvar'])  # Keep in u-space
        else:
            z_aux = z_aux_stat['mean'].clone()

        if not self.no_phy:
            z_phy = torch.sigmoid(u_phy)  # CHANGED: no clamping
        else:
            z_phy = torch.zeros(u_phy.shape[0], self.in_channels, device=u_phy.device)

        return z_phy, z_aux  # Return z_aux (unbounded) instead of z_aux

    def decode(self, z_phy:torch.Tensor, z_aux:torch.Tensor, epoch:int=0, epochs_pretrain:int=20, full:bool=False, const:dict=None, use_inference_values:bool=False, detach_x_P_for_bias:bool=True):
        """
        CORRECTED: z_aux (unbounded) -> c = tanh(z_aux/tau) -> delta = c@B.T
        x_PB = x_P + r(t) * delta
        """
        if not self.no_phy:
            y = self.physics_model(z_phy, const=const) # (n, in_channels)
            x_P = y
            if self.dim_z_aux > 0:
                # Compute coefficient from z_aux with temperature annealing
                x_P_input = x_P.detach() if detach_x_P_for_bias else x_P
                # Pass time features if available
                c = self.dec.compute_coefficient(z_aux, x_P_input, epoch, epochs_pretrain, use_inference_values, self.time_feats)
                
                # Compute low-rank residual: delta = (c * s) @ B.T
                delta = torch.matmul(c * self.dec.s, self.dec.B.T)
                
                # Apply global residual scale warmup (starts after pretraining)
                if use_inference_values:
                    r = self.dec.get_r_for_inference()
                else:
                    r = self.dec.get_r(epoch, epochs_pretrain)
                x_PB = x_P + r * delta
            else:
                x_PB = x_P.clone()
                delta = torch.zeros_like(x_P)
                c = torch.zeros(x_P.shape[0], 0, device=x_P.device)
        else:
            y = torch.zeros(z_phy.shape[0], self.in_channels, device=z_phy.device)
            if self.dim_z_aux > 0:
                x_PB = self.dec.func_aux_dec(z_aux) 
            else:
               x_PB = torch.zeros(z_phy.shape[0], self.in_channels, device=z_phy.device)
            x_P = x_PB.clone()
            delta = torch.zeros_like(x_PB)
            c = torch.zeros(x_P.shape[0], 0, device=x_P.device)

        if full:
            return x_PB, x_P, y, delta, c
        else:
            return x_PB

    def encode_iterative(self, x:torch.Tensor, t:torch.Tensor=None, num_passes:int=None):
        """
        Run K refinement passes of the encoder for TRAINING.

        Each pass re-encodes the data together with the previous pass's u-space estimate.
        The estimate is detached between passes, so every pass is a self-contained learning
        problem (no gradient flow across passes, no memory growth).

        Returns:
            all_z_phy_stats: list of K z_phy_stat dicts (one per pass, in order)
            z_aux_stat: the auxiliary-latent stats from the final pass
        """
        n = x.shape[0]
        device = x.device
        K = num_passes if num_passes is not None else self.num_passes_train

        # Initial estimate: zeros in u-space (= centre of the bounded param range after sigmoid).
        u_phy_est = torch.zeros(n, self.dim_z_phy, device=device, dtype=x.dtype)
        # Small jitter during training only -> stops the encoder memorising one init path.
        if self.training and self.initial_u_noise_std > 0:
            u_phy_est = u_phy_est + torch.randn_like(u_phy_est) * self.initial_u_noise_std

        all_z_phy_stats = []
        z_aux_stat = None
        for _pass in range(K):
            z_phy_stat, z_aux_stat = self.encode(x, t, u_phy_est=u_phy_est)
            all_z_phy_stats.append(z_phy_stat)
            # Detach between passes (each pass is a self-contained learning problem) and clamp
            # the unbounded u-space estimate: sigmoid saturates past ~+/-8, so +/-10 loses no
            # information while stopping a runaway early pass from destabilising later ones.
            u_phy_est = z_phy_stat['mean'].detach().clamp(-10.0, 10.0)

        return all_z_phy_stats, z_aux_stat

    def encode_iterative_infer(self, x:torch.Tensor, t:torch.Tensor=None, hard_z_phy:bool=True, hard_z_aux:bool=True):
        """
        Run up to num_passes_infer refinement passes for INFERENCE, stopping early once the
        bounded z_phy estimate stops changing (max abs change < convergence_tol).

        Returns the same objects as a single encode()+draw():
            z_phy_stat, z_aux_stat (final-pass stats), z_phy, z_aux (final-pass draws)
        """
        n = x.shape[0]
        device = x.device

        u_phy_est = torch.zeros(n, self.dim_z_phy, device=device, dtype=x.dtype)
        z_phy_prev = None
        z_phy_stat = z_aux_stat = z_phy = z_aux = None
        k = 0
        for k in range(self.num_passes_infer):
            z_phy_stat, z_aux_stat = self.encode(x, t, u_phy_est=u_phy_est)
            z_phy, z_aux = self.draw(z_phy_stat, z_aux_stat, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)
            # Convergence is measured on the BOUNDED z_phy (the physically meaningful (0,1)
            # parameters), not on the unbounded u-space estimate: we stop once the decoded
            # source parameters stabilise. Near the (0,1) edges the sigmoid compresses large
            # u-changes, so a given tol triggers earlier there -- intended (output has settled).
            if z_phy_prev is not None and (z_phy - z_phy_prev).abs().max().item() < self.convergence_tol:
                break
            z_phy_prev = z_phy.detach()
            u_phy_est = z_phy_stat['mean'].detach().clamp(-10.0, 10.0)  # see encode_iterative
        self._last_num_passes = k + 1  # passes actually taken (for logging/diagnostics)

        return z_phy_stat, z_aux_stat, z_phy, z_aux

    def forward(self, x:torch.Tensor, t:torch.Tensor=None, reconstruct:bool=True, hard_z_phy:bool=False, hard_z_aux:bool=False,
                inference:bool=False, const:dict=None, epoch:int=0, epochs_pretrain:int=20):
        # --- Iterative refinement path (only when enabled and physics is active) ---
        if self.use_iterative and not self.no_phy:
            if not reconstruct:
                # Refine, then return only the final-pass stats (mirrors the non-iterative case).
                all_z_phy_stats, z_aux_stat = self.encode_iterative(x, t)
                return all_z_phy_stats[-1], z_aux_stat

            if not inference:
                # Training: run K passes; decode ONLY the final pass for x_mean. The per-pass
                # reconstruction loss is computed in the trainer from all_z_phy_stats.
                all_z_phy_stats, z_aux_stat = self.encode_iterative(x, t)
                z_phy_stat = all_z_phy_stats[-1]
                x_mean = self.decode(*self.draw(z_phy_stat, z_aux_stat, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux),
                                   epoch=epoch, epochs_pretrain=epochs_pretrain, full=False, const=const, use_inference_values=False, detach_x_P_for_bias=self.detach_x_P_for_bias)
                # First element is the LIST of per-pass stats (trainer expects this when iterative).
                return all_z_phy_stats, z_aux_stat, x_mean
            else:
                # Inference: iterate to convergence, then decode once. Same 4-tuple as before.
                z_phy_stat, z_aux_stat, z_phy, z_aux = self.encode_iterative_infer(x, t, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)
                x_PB, x_P, _y, _delta, _c = self.decode(z_phy, z_aux, epoch=epoch, epochs_pretrain=epochs_pretrain, full=True, const=const, use_inference_values=True, detach_x_P_for_bias=self.detach_x_P_for_bias)
                return z_phy, z_aux, x_PB, x_P  # keep 4-tuple for existing test code

        # --- Original single-pass path (unchanged when iterative refinement is disabled) ---
        z_phy_stat, z_aux_stat = self.encode(x, t)

        if not reconstruct:
            return z_phy_stat, z_aux_stat

        if not inference:
            x_mean = self.decode(*self.draw(z_phy_stat, z_aux_stat, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux),
                               epoch=epoch, epochs_pretrain=epochs_pretrain, full=False, const=const, use_inference_values=False, detach_x_P_for_bias=self.detach_x_P_for_bias)
            return z_phy_stat, z_aux_stat, x_mean
        else:
            z_phy, z_aux = self.draw(z_phy_stat, z_aux_stat, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)
            x_PB, x_P, _y, _delta, _c = self.decode(z_phy, z_aux, epoch=epoch, epochs_pretrain=epochs_pretrain, full=True, const=const, use_inference_values=True, detach_x_P_for_bias=self.detach_x_P_for_bias)
            return z_phy, z_aux, x_PB, x_P  # keep 4-tuple for existing test code
