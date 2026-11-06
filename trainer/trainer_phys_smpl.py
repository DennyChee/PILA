"""
Simplified version of PhysVAE for ablation study.
TODO add wandb logging to check the balance between different terms.

NEW METRICS ADDED:
- residual_loss: L2 difference between raw physics output (x_P) and corrected output (x_PB)
- residual_rel_diff: Relative difference as percentage of raw output magnitude
  This helps monitor how much the correction layer is modifying the physics output.
  Lower values indicate the correction is making smaller changes.
  The residual_loss is computed as torch.sum((x_PB - x_P).pow(2), dim=1).mean().
"""
# Adapted from the training script of Phys-VAE
import numpy as np
import torch
import csv
import time
from utils.timing import PhaseTimer
import os
from torchvision.utils import make_grid
from base import BaseTrainer, PARENT_DIR
from utils import inf_loop, MetricTracker, kldiv_normal_normal
import wandb
from model.loss import mse_loss, mse_loss_per_channel
from IPython import embed

class PhysVAETrainerSMPL(BaseTrainer):
    """
    Trainer for Phys-VAE, with options for physics-based regularization and reconstruction.
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.no_phy = config['arch']['phys_vae']['no_phy']
        self.epochs_pretrain = config['trainer']['phys_vae'].get('epochs_pretrain', 0)
        self.dim_z_phy = config['arch']['phys_vae']['dim_z_phy']
        self.dim_z_aux = config['arch']['phys_vae']['dim_z_aux']

        # CHANGED: separate KL controls for z_phy and z_aux
        self.beta_warmup = config['trainer']['phys_vae'].get('kl_warmup_epochs', 50)  # NEW
        self.gate_loss_weight = config['trainer']['phys_vae'].get('balance_gate', 1e-3)  # NEW
        
        # Separate KL controls for z_phy
        self.use_kl_term_z_phy = config['trainer']['phys_vae'].get('use_kl_term_z_phy', False)
        self.beta_max_z_phy = config['trainer']['phys_vae'].get('beta_max_z_phy', 1.0)
        
        # Separate KL controls for z_aux
        self.use_kl_term_z_aux = config['trainer']['phys_vae'].get('use_kl_term_z_aux', False)
        self.beta_max_z_aux = config['trainer']['phys_vae'].get('beta_max_z_aux', 1.0)
        
        # ========================================================================
        # CAPACITY CONTROL MODULE (Optional add-on to original implementation)
        # ========================================================================
        self.use_capacity_control = config['trainer']['phys_vae'].get('use_capacity_control', False)
        self.C_max = config['trainer']['phys_vae'].get('C_max', float(self.dim_z_phy))  # e.g., 7.0 for RTM
        self.C_gamma = config['trainer']['phys_vae'].get('C_gamma', 10.0)  # Capacity penalty weight
        self.beta_aux = config['trainer']['phys_vae'].get('beta_aux', 1.0)  # Auxiliary KL weight

        # for Stage A bootstrap in u-space
        self.synthetic_data_loss_weight = config['trainer']['phys_vae'].get('balance_data_aug', 1.0)
        
        # NEW: Loss weights for improved residual architecture
        self.ortho_penalty_weight = config['trainer']['phys_vae'].get('ortho_penalty_weight', 0.1)
        self.coeff_penalty_weight = config['trainer']['phys_vae'].get('coeff_penalty_weight', 1e-4)
        self.delta_penalty_weight = config['trainer']['phys_vae'].get('delta_penalty_weight', 1e-4)
        
        # NEW: Edge penalty for z_phy to avoid extreme values (0, 1)
        self.edge_penalty_weight = config['trainer']['phys_vae'].get('edge_penalty_weight', 0.0)
        self.edge_penalty_power = config['trainer']['phys_vae'].get('edge_penalty_power', 1.0)

        # NEW: dV sparsity penalty. L1 on the dV latent toward its centre (z=0.5),
        # which maps EXACTLY to physical dV=0 in both configs/mogi_paras.json and
        # configs/mogi_paras_symdV.json. This lets the encoder return dV=0 for
        # quiescent (no-deformation) epochs instead of the degenerate
        # "large dV at a very deep depth" solution. Default 0.0 -> feature off,
        # so existing configs and runs are unchanged.
        self.dv_l1_weight = config['trainer']['phys_vae'].get('dv_l1_weight', 0.0)
        # NEW: dV L2 (quadratic) sparsity penalty toward the same physical-dV=0 centre.
        # Complements/replaces the L1 form. Because the dV<->z map is affine, a physical
        # spike (e.g. dV=2e8 -> z=1.0, |z-z0|=0.5) sits FURTHER from centre than the
        # genuine rise (dV=8e7 -> z=0.7, |z-z0|=0.2). L1's gradient is constant in
        # |z-z0| so it cannot crush a saturated spike without also dragging the rise;
        # L2's gradient grows with |z-z0|, so it penalises the z=1.0 spike ~6x harder
        # than the z=0.7 rise (0.5^2 vs 0.2^2) and pushes saturated outliers back much
        # more strongly. Default 0.0 -> feature off; L1 and L2 can be combined.
        self.dv_l2_weight = config['trainer']['phys_vae'].get('dv_l2_weight', 0.0)
        # NEW: dv_penalty_space controls the SPACE the dV penalty acts in.
        #   'z' (default, legacy) -> penalise |z_dV - z0| in post-sigmoid (0,1) space.
        #       Problem: the optimiser only moves the pre-sigmoid u (z = sigmoid(u)), so the
        #       gradient carries a z(1-z) factor that -> 0 at the rail (z->0/1) -- exactly the
        #       dV-railing failure mode we want to fix. A larger weight cannot pull a railed
        #       dV back because it just multiplies a ~0 gradient.
        #   'u' -> penalise |u_dV - u0| in pre-sigmoid space, u0 = logit(z0). u is unbounded
        #       and linear, so the pull is a constant, non-vanishing force even at the rail.
        # Default 'z' keeps every existing config/run byte-identical.
        self.dv_penalty_space = config['trainer']['phys_vae'].get('dv_penalty_space', 'z')
        if self.dv_penalty_space not in ('z', 'u'):
            raise ValueError(
                f"dv_penalty_space must be 'z' or 'u', got {self.dv_penalty_space!r}")
        # Locate the dV dimension in the physics decoder's parameter ordering, and
        # the z-value that maps to physical dV=0 for THIS config.
        #   physical_dV(z) = (z*(max-min)+min) * scale + shift   (see rescale())
        # Solving physical_dV(z0)=0 gives z0 = ((-shift/scale) - min)/(max-min).
        # We penalise |z_dV - z0| so the term pulls toward physical dV=0 regardless
        # of the config's min/max/scale/shift (z0=0.5 for the symmetric/default Mogi
        # configs, but differs when custom min/max are used with default scale/shift).
        # The amplitude parameter is named 'dV' for Mogi and Sun69 but 'opening' for
        # Okada, whose paras file has no 'dV' key at all. Resolving only 'dV' made BOTH
        # this penalty AND temporal_smoothness_include_dv silent no-ops on every Okada
        # config -- the loss term was configured, logged and ignored. We therefore look
        # for 'dV' first and fall back to 'opening', keeping the attribute names (_dv_*)
        # for backward compatibility. Mogi/Sun69 configs resolve exactly as before, so
        # existing runs are bit-identical.
        # Stays None only for no_phy mode or a source with neither parameter.
        self._dv_dim = None
        self._dv_name = None
        self._dv_z_center = 0.5
        # Pre-sigmoid target for the u-space penalty: u0 = logit(z0). Default logit(0.5)=0.
        self._dv_u_center = 0.0
        if (not self.no_phy) and hasattr(self.model, 'physics_model') \
                and hasattr(self.model.physics_model, 'z_phy_ranges'):
            z_phy_param_names = list(self.model.physics_model.z_phy_ranges.keys())
            amp_name = next((nm for nm in ('dV', 'opening') if nm in z_phy_param_names), None)
            if amp_name is not None:
                self._dv_name = amp_name
                self._dv_dim = z_phy_param_names.index(amp_name)
                dv_range = self.model.physics_model.z_phy_ranges[amp_name]
                dv_min = float(dv_range['min'])
                dv_max = float(dv_range['max'])
                if amp_name == 'dV':
                    # Mogi/Sun69: physical dV = (rescaled dV)*scale + shift, so the
                    # scale/shift defaults below reproduce the legacy hardcoded map
                    # (see Physics_Mogi.rescale / Physics_Sun69.rescale).
                    dv_scale = float(dv_range.get('scale', 1e5))
                    dv_shift = float(dv_range.get('shift', -1e7))
                else:
                    # Okada 'opening' is a PLAIN linear rescale in
                    # Physics_Okada.rescale -- no scale/shift transform is applied to
                    # it -- so the identity map (scale=1, shift=0) is the correct one
                    # here. Using the dV defaults would solve for the wrong z0.
                    dv_scale = float(dv_range.get('scale', 1.0))
                    dv_shift = float(dv_range.get('shift', 0.0))
                # z that yields physical amplitude = 0
                self._dv_z_center = ((-dv_shift / dv_scale) - dv_min) / (dv_max - dv_min)
                # Pre-sigmoid target u0 = logit(z0) for the u-space penalty. Clamp z0 off the
                # 0/1 asymptotes first, since logit(0/1)=+-inf (np is already imported; avoids
                # a new `import math`). For the symmetric symdV configs z0=0.5 -> u0=0.
                _zc = min(max(self._dv_z_center, 1e-6), 1.0 - 1e-6)
                self._dv_u_center = float(np.log(_zc / (1.0 - _zc)))
                # dV=0 must be reachable within the (0,1) latent range, else the L1/L2
                # penalty would pull toward a spurious nonzero dV. Fail loudly.
                if (self.dv_l1_weight > 0.0 or self.dv_l2_weight > 0.0) \
                        and not (0.0 <= self._dv_z_center <= 1.0):
                    raise ValueError(
                        f"dv_l1_weight/dv_l2_weight>0 but physical {amp_name}=0 is not reachable "
                        f"for this config: z0={self._dv_z_center:.4f} lies outside [0,1] "
                        f"({amp_name} min={dv_min}, max={dv_max}, scale={dv_scale}, "
                        f"shift={dv_shift}). Adjust the paras so {amp_name}=0 is in range.")
        
        # Multi-source decoders (physics 'MultiMogi_LOS'): z_phy holds K blocks of the paras-file
        # parameters, so the amplitude dim repeats every n_per_source dims. _dv_dims lists every
        # source's amplitude index; for single-source physics it is [_dv_dim] (or []), so the
        # dV penalties below give bit-identical values to the old single-index code.
        self._n_sources = int(getattr(getattr(self.model, 'physics_model', None), 'n_sources', 1))
        if self._dv_dim is None:
            self._dv_dims = []
        else:
            n_per_source = len(self.model.physics_model.z_phy_ranges)
            self._dv_dims = [k * n_per_source + self._dv_dim for k in range(self._n_sources)]
        if self._n_sources > 1:
            unsupported = [name for name, on in [
                ('arch.phys_vae.temporal_prior', config['arch']['phys_vae'].get('temporal_prior', {}).get('enabled', False)),
                ('arch.phys_vae.informed_prior', config['arch']['phys_vae'].get('informed_prior', {}).get('enabled', False)),
                ('arch.phys_vae.iterative_refinement', config['arch']['phys_vae'].get('iterative_refinement', {}).get('enabled', False)),
            ] if on]
            if unsupported:
                raise ValueError(f"n_sources={self._n_sources} is not supported together with {unsupported} "
                                 "(these assume a single 4-parameter source)")

        # Optional per-acquisition NUISANCE offset + planar ramp (standard in geodetic inversion).
        # MintPy-style LOS maps carry an arbitrary constant (reference pixel) and long-wavelength
        # tilts (orbit / large-scale APS) that a point source cannot model; with plain MSE the
        # source is then dragged into a deep corner position to mimic them. With
        # nuisance_ramp=True the reconstruction loss is computed on residuals with their
        # least-squares offset + ramp removed, r -> (I - G G^+) r, G = [1, x_km, y_km] over the
        # observation points -- exactly equivalent to fitting 3 free nuisance parameters per
        # acquisition. Point/MLP path only. Default off -> legacy configs unchanged.
        self.nuisance_ramp = bool(config['trainer']['phys_vae'].get('nuisance_ramp', False))
        if self.nuisance_ramp:
            if config['arch']['args'].get('encoder_type', 'mlp') != 'mlp':
                raise ValueError("nuisance_ramp is implemented for the point/MLP path only")
            if config['arch']['phys_vae'].get('iterative_refinement', {}).get('enabled', False):
                raise ValueError("nuisance_ramp is not wired into the iterative-refinement per-pass loss")
            x_km = self.model.physics_model.model.x.double() / 1e3
            y_km = self.model.physics_model.model.y.double() / 1e3
            G = torch.stack([torch.ones_like(x_km), x_km, y_km], dim=1)          # [N, 3]
            self._ramp_G = G.float().to(self.device)
            self._ramp_G_pinv = torch.linalg.pinv(G).float().to(self.device)     # [3, N]
            print(f"  [nuisance_ramp] ON: per-acquisition offset + planar ramp projected out of the "
                  f"reconstruction residual ({G.shape[0]} points)")

        # Per-phase wall-clock timing (utils/timing.py): data fetch, encode, physics decode,
        # residual decode, loss, backward, optimiser step, bookkeeping, validation, checkpoint.
        # Written per epoch to <save_dir>/timing_epochs.csv and summarised in compute_usage.json.
        self.timer = PhaseTimer(enabled=config['trainer'].get('timing', {}).get('enabled', True))
        if not self.no_phy:
            self.timer.attach_module(self.model.physics_model, 'physics_decode',
                                     within='decode', remainder_name='residual_decode')
        self.timing_csv = os.path.join(str(self.checkpoint_dir), 'timing_epochs.csv')
        self.n_samples_seen = 0

        # NEW: Temporal smoothness regularization for the source geometry params.
        self.temporal_smoothness_weight = config['trainer']['phys_vae'].get('temporal_smoothness_weight', 0.0)
        # Which leading z_phy dims are the "location/geometry" params to keep
        # temporally smooth (amplitude is intentionally left free to vary).
        #   Mogi  : [:3] = xcen, ycen, d            (dV free)
        #   Okada : [:7] = xoff,yoff,depth,strike,dip,length,width  (opening free)
        physics = config['arch']['args'].get('physics', 'Mogi')
        if physics in ('Okada', 'Okada_LOS'):
            self.temporal_smoothness_dims = 7
        else:  # Mogi / Mogi_LOS (and any other point-source style decoder)
            self.temporal_smoothness_dims = 3
        # Never request more dims than exist.
        self.temporal_smoothness_dims = min(self.temporal_smoothness_dims, self.dim_z_phy)
        # NEW: temporal-smoothness FORM. 'variance' (default, legacy) penalises the variance
        # of each smoothed param across the window -> zero only if the param is CONSTANT
        # (correct for a static source). 'curvature' penalises the discrete second time
        # difference (z[t+1]-2z[t]+z[t-1])^2 -> zero for ANY linear trend, so a constant-
        # velocity source move or a linear dV ramp is not penalised, only jitter/kinks.
        self.temporal_smoothness_mode = config['trainer']['phys_vae'].get('temporal_smoothness_mode', 'variance')
        if self.temporal_smoothness_mode not in ('variance', 'curvature'):
            raise ValueError(
                f"temporal_smoothness_mode must be 'variance' or 'curvature', "
                f"got '{self.temporal_smoothness_mode}'.")
        # NEW: also smooth the dV (amplitude) dim, not just geometry. Off by default so the
        # amplitude stays free (legacy behaviour). Only meaningful with 'curvature' when dV
        # genuinely ramps; with 'variance' it assumes the amplitude is constant. Reuses
        # self._dv_dim ('dV' for Mogi/Sun69, 'opening' for Okada)
        # (set above; None for no_phy / Okada -> this flag is then a safe no-op).
        self.temporal_smoothness_include_dv = config['trainer']['phys_vae'].get('temporal_smoothness_include_dv', False)
        # Explicit list of smoothed z_phy dims. Single source: the leading geometry dims (+ dV).
        # K sources: the location/depth dims (xcen, ycen, d) of EVERY source (+ every dV).
        if self._n_sources > 1:
            n_per_source = len(self.model.physics_model.z_phy_ranges)
            self._smoothed_dims = [k * n_per_source + j for k in range(self._n_sources) for j in range(3)]
        else:
            self._smoothed_dims = list(range(self.temporal_smoothness_dims))
        if self.temporal_smoothness_include_dv:
            self._smoothed_dims += [d for d in self._dv_dims if d not in self._smoothed_dims]
        if self._n_sources > 1:
            print(f"  [multi-source] n_sources={self._n_sources}: dV dims {self._dv_dims}, "
                  f"temporal-smoothness dims {self._smoothed_dims}")
        # Announce the active regularizer (easy to miss in a config diff). NOTE on weight
        # scale: rec_loss = mse_loss = SUM over the ~16384 pixels then mean over batch
        # (model/loss.py), so it is O(1e3-1e4); the smoothness term is a mean over only a
        # few params x time points, so it is O(1e-2). The weight must therefore be O(1e3-1e4)
        # to contribute a non-trivial ~1-10% of rec_loss - a weight of O(0.1-1) is effectively
        # OFF. Tune by watching (temporal_smoothness_weight * temporal_smoothness) vs rec_loss.
        if self.temporal_smoothness_weight > 0.0 and not self.no_phy:
            k_ts = len(self._smoothed_dims)
            print(f"  [temporal_smoothness] ON: mode={self.temporal_smoothness_mode}, "
                  f"weight={self.temporal_smoothness_weight}, include_dv={self.temporal_smoothness_include_dv}, "
                  f"smoothed dims={k_ts} (geometry[:{self.temporal_smoothness_dims}]"
                  f"{'+' + str(self._dv_name) if (self.temporal_smoothness_include_dv and self._dv_dim is not None) else ''})")

        # NEW: EMA prior configuration
        self.use_ema_prior = config['trainer']['phys_vae'].get('use_ema_prior', False)

        # NEW: Iterative refinement encoder (default disabled -> original single-pass behaviour)
        iter_cfg = config['arch']['phys_vae'].get('iterative_refinement', {})
        self.use_iterative = iter_cfg.get('enabled', False)
        # Weight on the per-pass (physics-only) reconstruction loss accumulated across passes.
        self.iter_loss_weight = iter_cfg.get('iter_loss_weight', 1.0)

        # NEW: gradient clipping
        self.grad_clip_norm = config['trainer'].get('grad_clip_norm', 1.0)

        # trackers
        self.train_metrics = MetricTracker(
            'loss', 'rec_loss', 'kl_loss',
            'syn_data_loss',  # diagnostic
            'residual_loss',  # L2 difference between raw and corrected output
            'residual_rel_diff',  # relative difference as percentage
            'ortho_penalty',  # orthogonality penalty for basis matrix
            'coeff_penalty',  # coefficient L2 penalty
            'delta_penalty',  # delta L2 penalty
            'edge_penalty',  # edge penalty for z_phy
            'dv_l1_penalty',  # L1 sparsity penalty on the dV latent (toward dV=0)
            'dv_l2_penalty',  # L2 (quadratic) sparsity penalty on the dV latent (toward dV=0)
            'temporal_smoothness',  # temporal smoothness penalty for Mogi source
            'c_norm',  # norm of coefficient vector
            'delta_norm',  # norm of residual vector
            's_norm',  # norm of scale parameters
            'basis_quality',  # ||B^T B - I||_F
            # NEW: Capacity control metrics
            'kl_u_phy',  # Physics KL in u-space
            'kl_z_aux',  # Auxiliary KL
            'beta_z_phy',  # Beta weight for z_phy KL term
            'beta_z_aux',  # Beta weight for z_aux KL term
            'capacity_target',  # Current capacity target C_t
            'capacity_penalty',  # Capacity penalty term
            # NEW: EMA prior metrics
            'ema_prior_mean_norm',  # L2 norm of EMA prior mean
            'ema_prior_std_mean',  # Mean of EMA prior standard deviations
            # NEW: Iterative refinement
            'iter_loss',  # mean per-pass physics reconstruction loss
            *[m.__name__ for m in metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('rec_loss', 'kl_loss', 'residual_loss', 'residual_rel_diff', *[m.__name__ for m in metric_ftns], writer=self.writer)

        self.data_key = config['trainer']['input_key']
        self.target_key = config['trainer']['output_key']
        self.input_const_keys = config['trainer'].get('input_const_keys', None)

        self.stablize_grad = config['trainer']['stablize_grad']
        self.stablize_count = 0

        # NEW (optional): running stats of u per-dim
        self.log_u_stats = True

        # NEW: Store initial learning rate for pretraining
        self.initial_lr = self.optimizer.param_groups[0]['lr']

        # ====================================================================
        # NEW: Per-sample CAUSAL TEMPORAL PRIOR (Option A).
        # Replaces the static N(0,I) physics prior with a temporal-continuity
        # prior: for epoch t the u_phy KL is pulled toward epoch (t-1)'s CURRENT
        # estimate, held in an online EMA buffer. This couples the otherwise
        # independent per-epoch (amortized) InSAR inversions into a Gaussian
        # random-walk over the parameter track, in ONE training run.
        # Scope: Stage-A per-epoch point/MLP InSAR path (one sample == one epoch,
        # dataset carries a scalar 'epoch_idx'). Not for the CNN or seq paths.
        # ====================================================================
        tp_cfg = config['arch']['phys_vae'].get('temporal_prior', {})
        self.use_temporal_prior = tp_cfg.get('enabled', False)
        self.tp_ema_decay = float(tp_cfg.get('ema_decay', 0.9))  # buffer smoothing across steps
        if self.use_temporal_prior and not self.no_phy:
            # The temporal prior acts ONLY through the physics-KL term, which is weighted
            # by beta_z_phy (=0 unless use_kl_term_z_phy=True with beta_max_z_phy>0). Fail
            # loud instead of silently training a zero-weighted (inert) temporal prior.
            if not self.use_kl_term_z_phy or self.beta_max_z_phy <= 0.0:
                raise ValueError(
                    "temporal_prior.enabled=True requires trainer.phys_vae.use_kl_term_z_phy=True "
                    f"and beta_max_z_phy>0 (got use_kl_term_z_phy={self.use_kl_term_z_phy}, "
                    f"beta_max_z_phy={self.beta_max_z_phy}); otherwise the temporal KL is "
                    "zero-weighted and the prior is silently inert.")
            # NOTE: the EMA buffer below is NOT persisted in checkpoints, so resuming
            # (-r) cold-starts it to zeros. This feature is designed for ONE uninterrupted
            # training run; resume will transiently re-warm the buffer.
            # Prior std per parameter (temporal-coupling strength). Reuse the existing
            # location/geometry vs amplitude split: the leading temporal_smoothness_dims
            # (Mogi [:3], Okada [:7]) are slowly-varying geometry -> tight std (strong
            # pull); the rest (dV / opening) are genuinely episodic -> loose std so the
            # encoder can still track fast amplitude change per epoch.
            tp_prior_std = float(tp_cfg.get('prior_std', 1.0))
            tp_loc_std = float(tp_cfg.get('loc_std', tp_prior_std))
            tp_amp_std = float(tp_cfg.get('amp_std', tp_prior_std))
            k_loc = self.temporal_smoothness_dims
            std_vec = np.full(self.dim_z_phy, tp_prior_std, dtype=np.float64)
            std_vec[:k_loc] = tp_loc_std   # location/geometry group (tight)
            std_vec[k_loc:] = tp_amp_std   # amplitude group (loose)
            # Guard a zero/negative std; then convert to log-variance and clamp to the
            # SAME [-9, 5] range kldiv_normal_normal (utils/utils.py) clamps to, so the
            # requested std is exactly what training uses (no silent floor/cap).
            std_vec = np.clip(std_vec, 1e-3, None)
            tp_lnvar = np.clip(2.0 * np.log(std_vec), -9.0, 5.0).astype(np.float32)
            self.temporal_prior_lnvar = torch.tensor(tp_lnvar, device=self.device)  # (dim_z_phy,)

            # Online buffer of per-epoch u-space MEANS, indexed by chronological epoch
            # ordinal. Init to zeros: in u-space sigmoid(0)=0.5 = center of every
            # parameter's [min,max] box, so the temporal prior STARTS identical to the
            # old N(0,I) center prior and becomes temporal as the buffer fills.
            n_epoch_total = len(self.data_loader.dataset)
            self.temporal_prior_mean = torch.zeros(
                n_epoch_total, self.dim_z_phy, device=self.device)

            # Every epoch ordinal must appear in the TRAINING loader, else its buffer
            # slot never updates and its neighbour's prior goes stale. BaseDataLoader
            # splits validation by randomly holding out indices, so require a 0 split
            # (InSAR configs use a separate data_dir_valid).
            val_split = config['data_loader']['args'].get('validation_split', 0.0)
            if val_split and float(val_split) > 0.0:
                raise ValueError(
                    f"temporal_prior requires validation_split=0 (got {val_split}); "
                    "otherwise held-out epochs never update their buffer slot. "
                    "Use a separate data_dir_valid.")
            std_report = np.exp(0.5 * tp_lnvar).round(3).tolist()
            print(f"  [temporal_prior] enabled: {n_epoch_total} epochs, "
                  f"ema_decay={self.tp_ema_decay}, loc dims [:{k_loc}], "
                  f"prior std per param = {std_report}")
        else:
            self.temporal_prior_mean = None
            self.temporal_prior_lnvar = None

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        
        # NEW: Reset learning rate to initial value during pretraining
        if epoch < self.epochs_pretrain:
            # Pretraining phase: use initial learning rate (no scheduling)
            if self.optimizer.param_groups[0]['lr'] != self.initial_lr:
                self.optimizer.param_groups[0]['lr'] = self.initial_lr
                self.logger.info(f"Epoch {epoch}: Using initial LR for pretraining: {self.initial_lr}")
        else:
            # Training phase: let scheduler handle learning rate
            if epoch == self.epochs_pretrain:
                self.logger.info(f"Epoch {epoch}: Starting training phase, scheduler will control LR")
        
        sequence_len = None
        # Compute separate beta values for z_phy and z_aux when not in pretraining stage
        if not self.no_phy and epoch >= self.epochs_pretrain and self.use_kl_term_z_phy:
            # FIXED: Beta should start from 0 when training begins
            training_epoch = epoch - self.epochs_pretrain
            beta_z_phy = self.beta_max_z_phy * self._linear_annealing_epoch(training_epoch, warmup_epochs=self.beta_warmup)
        else:
            beta_z_phy = 0.0  # No KL loss during pretraining or when use_kl_term_z_phy=False
            
        if epoch >= self.epochs_pretrain and self.use_kl_term_z_aux:
            training_epoch = epoch - self.epochs_pretrain
            beta_z_aux = self.beta_max_z_aux * self._linear_annealing_epoch(training_epoch, warmup_epochs=self.beta_warmup)
        else:
            beta_z_aux = 0.0  # No KL loss during pretraining or when use_kl_term_z_aux=False

        # NEW: accumulators for u-stats
        u_sum = None
        u2_sum = None
        u_count = 0

        t_mark = time.perf_counter()
        for batch_idx, data_dict in enumerate(self.data_loader):
            data = data_dict[self.data_key].to(self.device)
            self.timer.add('data_fetch', time.perf_counter() - t_mark)
            input_const = {k: data_dict[k].to(self.device) for k in self.input_const_keys} if self.input_const_keys else None
            
            # Get time features if available
            time_feats = data_dict.get('time_feats', None)
            if time_feats is not None:
                time_feats = time_feats.to(self.device)
                if time_feats.dim() == 3:  # For sequence data
                    time_feats = time_feats.view(-1, time_feats.size(-1))

            # Chronological epoch ordinal per sample (Stage-A point/MLP InSAR path only),
            # used by the causal temporal prior. Absent on GPS/CSV/seq/CNN paths.
            epoch_idx = data_dict.get('epoch_idx', None)
            if epoch_idx is not None:
                epoch_idx = epoch_idx.to(self.device).long()
            if self.use_temporal_prior and epoch_idx is None:
                raise ValueError(
                    "temporal_prior is enabled but the batch has no 'epoch_idx'. "
                    "It is only supported on the Stage-A per-epoch point/MLP InSAR "
                    "path (DisplacementInSARh5); disable temporal_prior otherwise.")

            # Full-field CNN path: data is a [B, C, H, W] image with a coherence 'mask'.
            rec_mask = data_dict.get('mask', None)
            if rec_mask is not None:
                rec_mask = rec_mask.to(self.device)
            is_image = (data.dim() == 4)

            if data.dim() == 3:
                # Sequence format: (batch_size, seq_len, features)
                sequence_len = data.size(1)
                data = data.view(-1, data.size(-1))

            self.optimizer.zero_grad()

            # Encode (u-space stats) with optional time features.
            # Iterative refinement: run K passes and keep every pass's stats for the per-pass
            # reconstruction loss; the final pass feeds the usual draw/decode/KL path below.
            # During pretraining the loss is purely the synthetic bootstrap, so the iterative
            # multi-pass encode would just waste K-1 forward passes -> use a single pass then.
            with self.timer.phase('encode'):
                if self.use_iterative and not self.no_phy and epoch >= self.epochs_pretrain:
                    all_z_phy_stats, z_aux_stat = self.model.encode_iterative(data, time_feats)
                    z_phy_stat = all_z_phy_stats[-1]
                else:
                    all_z_phy_stats = None
                    z_phy_stat, z_aux_stat = self.model.encode(data, time_feats)

            # NEW: Update EMA prior statistics (if enabled) - only after pretraining phase
            if self.use_ema_prior and not self.no_phy and epoch > 1 + self.epochs_pretrain:
                self.model.update_ema_prior(z_phy_stat['mean'], z_phy_stat['lnvar'])

            # Draw + decode
            # Use separate deterministic sampling for each latent variable based on their KL term settings
            hard_z_phy = not self.use_kl_term_z_phy  # Use deterministic sampling when KL term is disabled
            hard_z_aux = not self.use_kl_term_z_aux  # Use deterministic sampling when KL term is disabled
            with self.timer.phase('encode'):
                z_phy, z_aux = self.model.draw(z_phy_stat, z_aux_stat, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)
            # 'decode' is split by the physics forward hook into physics_decode + residual_decode.
            with self.timer.phase('decode'):
                x_PB, x_P, y, delta, c = self.model.decode(z_phy, z_aux, epoch=epoch, epochs_pretrain=self.epochs_pretrain, full=True, const=input_const)
            t_mark = time.perf_counter()

            # Losses - Get separate KL terms for flexible combination
            # For the CNN image path, flatten the target image (row-major over H,W to match
            # the grid physics decoder's cell order) and mask incoherent cells in the loss.
            if is_image:
                rec_target = data.reshape(data.size(0), -1)
                rec_mask_flat = rec_mask.reshape(rec_mask.size(0), -1) if rec_mask is not None else None
            else:
                rec_target, rec_mask_flat = data, None
            # Nuisance offset + ramp: project both target and prediction (linear), so the MSE is
            # that of the ramp-removed residual.
            x_PB_for_loss = x_PB
            if self.nuisance_ramp:
                rec_target, x_PB_for_loss = self._remove_ramp(rec_target), self._remove_ramp(x_PB)
            # epoch_idx activates the causal temporal prior in _vae_loss (training only;
            # validation stays on the standard N(0,I)/EMA prior with epoch_idx=None).
            rec_loss, kl_u_phy, kl_z_aux = self._vae_loss(rec_target, z_phy_stat, z_aux_stat, x_PB_for_loss, mask=rec_mask_flat, epoch_idx=epoch_idx)

            # NEW: Update the causal temporal-prior buffer with this batch's u-space
            # posterior means (detached), EMA-smoothed per epoch ordinal. Done AFTER
            # _vae_loss so that the prior lookup buffer[t-1] reads only values written by
            # EARLIER batches/epochs (strict causality) — updating before the loss would
            # let a same-batch neighbour t-1 leak its current-step estimate into t's prior.
            # Done every batch (incl. pretraining) so neighbours are warm when the u_phy KL
            # switches on. buffer[e] <- d*buffer[e] + (1-d)*mu_e.
            if self.use_temporal_prior and not self.no_phy and epoch_idx is not None:
                # No-replacement sampler => each epoch ordinal appears at most once per
                # batch, so the indexed assignment below is well-defined. Assert it so a
                # future WeightedRandomSampler / repeat-augmentation can't silently make
                # the update pick an arbitrary duplicate write.
                assert epoch_idx.unique().numel() == epoch_idx.numel(), (
                    "temporal_prior buffer update requires unique epoch_idx per batch "
                    "(no-replacement sampler); got duplicates.")
                with torch.no_grad():
                    mu = z_phy_stat['mean'].detach()                 # (n, dim_z_phy)
                    d = self.tp_ema_decay
                    prev = self.temporal_prior_mean[epoch_idx]       # (n, dim_z_phy)
                    self.temporal_prior_mean[epoch_idx] = d * prev + (1.0 - d) * mu

            # NEW: Iterative refinement per-pass reconstruction loss.
            # Supervise only the INTERMEDIATE passes (0 .. K-2) with a physics-only
            # reconstruction (decode through the physics model, no residual correction).
            # The FINAL pass (K-1) is deliberately excluded here because it is already scored
            # by the full-ELBO rec_loss above (physics + low-rank residual); supervising it
            # physics-only too would double-count it and pit the physics fit against the
            # residual that is meant to absorb model incompleteness. Linearly increasing
            # weight w_k = (k+1)/K nudges later passes to fit better than earlier ones.
            # z_phy_k uses the deterministic posterior mean (MAP) as the refinement target,
            # which matches the hard_z_phy=True default of the standard (KL-off) configs.
            iter_loss = torch.tensor(0.0, device=data.device)
            if all_z_phy_stats is not None and epoch >= self.epochs_pretrain:
                K = len(all_z_phy_stats)
                num_inter = K - 1  # intermediate passes only
                for k in range(num_inter):
                    z_phy_k = torch.sigmoid(all_z_phy_stats[k]['mean'])           # u-space -> (0,1)
                    x_P_k = self.model.physics_model(z_phy_k, const=input_const)  # physics-only render
                    if rec_mask_flat is not None:
                        sq_err_k = (x_P_k - rec_target).pow(2) * rec_mask_flat
                        rec_k = sq_err_k.sum() / rec_mask_flat.sum().clamp(min=1.0)
                    else:
                        rec_k = self.criterion(x_P_k, rec_target)
                    iter_loss = iter_loss + ((k + 1) / K) * rec_k
                if num_inter > 0:
                    iter_loss = iter_loss / num_inter

            # Calculate weighted KL loss
            kl_loss = beta_z_phy * kl_u_phy + beta_z_aux * kl_z_aux
            
            # ========================================================================
            # KL LOSS COMBINATION: Choose between original and capacity control modes
            # ========================================================================
            # Original mode: Combine KL terms first, then average (exactly as before)NOTE: effects of this term to be studied further
            # kl_loss = kl_u_phy + kl_z_aux

            # Compute L2 difference between raw physics output and corrected output
            residual_loss = torch.sum((x_PB - x_P).pow(2), dim=1).mean()
            # Also compute relative difference as percentage of raw output magnitude
            residual_rel_diff = torch.mean(torch.abs(x_PB - x_P) / (torch.abs(x_P) + 1e-8)) * 100.0

            # IMPROVED: Low-rank residual regularization terms
            if not self.no_phy and epoch >= self.epochs_pretrain and self.config['arch']['phys_vae']['dim_z_aux'] > 0:
                # Orthogonality penalty: λ_B ||B^T B - I||_F^2
                ortho_penalty = self.model.dec.orthogonality_penalty()
                
                # Coefficient penalty: λ_c ||c||_2^2
                coeff_penalty = torch.sum(c.pow(2), dim=1).mean()
                
                # Delta penalty: λ_Δ ||δ||_2^2
                delta_penalty = torch.sum(delta.pow(2), dim=1).mean()
            else:
                ortho_penalty = torch.tensor(0.0, device=data.device)
                coeff_penalty = torch.tensor(0.0, device=data.device)
                delta_penalty = torch.tensor(0.0, device=data.device)

            # NEW: Edge penalty for z_phy to avoid extreme values (0, 1)
            edge_penalty = self._edge_penalty(z_phy)

            # NEW: dV sparsity penalty. Only the dV dim is penalised; geometry
            # (xcen, ycen, depth) is left untouched. No-op when dV dim is absent.
            #   dv_penalty_space='z' (legacy): |z_dV - z0| in post-sigmoid (0,1) space.
            #       z_phy = sigmoid(u), so the gradient wrt the optimised u carries a z(1-z)
            #       slope factor that vanishes at the rail (z->0/1) -- powerless exactly where
            #       dV rails, and no larger weight can fix a ~0 gradient.
            #   dv_penalty_space='u': |u_dV - u0| in pre-sigmoid space, u0 = self._dv_u_center
            #       (= logit(z0)). u is unbounded/linear -> constant, non-vanishing pull that
            #       reaches a railed dV. z_phy_stat['mean'] IS the raw u-space posterior mean.
            _use_u = (self.dv_penalty_space == 'u')
            # Summed over sources (one source -> identical to the old single-dim value), so each
            # of K sources feels the same pull toward dV=0 it would feel alone.
            if self.dv_l1_weight > 0.0 and self._dv_dims:
                if _use_u:
                    dv_l1_penalty = (z_phy_stat['mean'][..., self._dv_dims] - self._dv_u_center).abs().sum(dim=-1).mean()
                elif z_phy is not None:
                    dv_l1_penalty = (z_phy[..., self._dv_dims] - self._dv_z_center).abs().sum(dim=-1).mean()
                else:
                    dv_l1_penalty = torch.tensor(0.0, device=data.device)
            else:
                dv_l1_penalty = torch.tensor(0.0, device=data.device)

            # dV L2 (quadratic) penalty toward the same centre, in the same space as L1.
            if self.dv_l2_weight > 0.0 and self._dv_dims:
                if _use_u:
                    dv_l2_penalty = (z_phy_stat['mean'][..., self._dv_dims] - self._dv_u_center).pow(2).sum(dim=-1).mean()
                elif z_phy is not None:
                    dv_l2_penalty = (z_phy[..., self._dv_dims] - self._dv_z_center).pow(2).sum(dim=-1).mean()
                else:
                    dv_l2_penalty = torch.tensor(0.0, device=data.device)
            else:
                dv_l2_penalty = torch.tensor(0.0, device=data.device)

            # NEW: Temporal smoothness regularization for Mogi source parameters
            temporal_smoothness = self._temporal_smoothness_loss(z_phy, sequence_len)

            # Stage A: synthetic bootstrap (u-target)
            if not self.no_phy and epoch < self.epochs_pretrain:
                synthetic_data_loss = self._synthetic_data_loss(data.shape[0])
                loss = self.synthetic_data_loss_weight * synthetic_data_loss
                # Initialize capacity control variables for metrics
                capacity_penalty = torch.tensor(0.0, device=data.device)
                C_t = 0.0
            else:
                # ========================================================================
                # LOSS CALCULATION: Choose between original and capacity control modes
                # ========================================================================
                if self.use_capacity_control and not self.no_phy and epoch >= self.epochs_pretrain and self.use_kl_term_z_phy:
                    # --- CAPACITY CONTROL MODE ---
                    training_epoch = epoch - self.epochs_pretrain
                    warm = max(1, self.beta_warmup)  # your existing warmup (e.g., 50)
                    
                    C_t = min(self.C_max, self.C_max * training_epoch / warm)   # 0 -> C_max
                    capacity_penalty = self.C_gamma * (kl_u_phy - C_t)**2
                    
                    # Auxiliary KL with separate control
                    aux_term = beta_z_aux * kl_z_aux
                    
                    # Capacity control loss
                    loss = (rec_loss
                            + capacity_penalty
                            + aux_term
                            + self.ortho_penalty_weight * ortho_penalty
                            + self.coeff_penalty_weight * coeff_penalty
                            + self.delta_penalty_weight * delta_penalty
                            + self.edge_penalty_weight * edge_penalty
                            + self.dv_l1_weight * dv_l1_penalty
                            + self.dv_l2_weight * dv_l2_penalty
                            + self.temporal_smoothness_weight * temporal_smoothness
                            + self.iter_loss_weight * iter_loss)
                else:
                    # --- ORIGINAL MODE ---
                    # Original loss function with combined KL loss
                    loss = (rec_loss +
                           kl_loss +
                           self.ortho_penalty_weight * ortho_penalty +
                           self.coeff_penalty_weight * coeff_penalty +
                           self.delta_penalty_weight * delta_penalty +
                           self.edge_penalty_weight * edge_penalty +
                           self.dv_l1_weight * dv_l1_penalty +
                           self.dv_l2_weight * dv_l2_penalty +
                           self.temporal_smoothness_weight * temporal_smoothness +
                           self.iter_loss_weight * iter_loss)
                    
                    # Initialize capacity control variables for metrics (when not using capacity control)
                    C_t = 0.0
                    capacity_penalty = torch.tensor(0.0, device=data.device)

            self.timer.add('loss', time.perf_counter() - t_mark)
            with self.timer.phase('backward'):
                loss.backward()
            t_mark = time.perf_counter()

            # NEW: gradient clipping for stability
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            if self.stablize_grad:
                self._grad_stablizer(epoch, batch_idx, loss.item())

            self.optimizer.step()
            self.timer.add('optimizer_step', time.perf_counter() - t_mark)
            t_mark = time.perf_counter()
            self.n_samples_seen += data.shape[0]

            # Update metrics
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('rec_loss', rec_loss.item())
            self.train_metrics.update('kl_loss', kl_loss.item())
            self.train_metrics.update('residual_loss', residual_loss.item())
            self.train_metrics.update('residual_rel_diff', residual_rel_diff.item())
            
            self.train_metrics.update('kl_u_phy', kl_u_phy.item())
            self.train_metrics.update('kl_z_aux', kl_z_aux.item())
            self.train_metrics.update('beta_z_phy', beta_z_phy)
            self.train_metrics.update('beta_z_aux', beta_z_aux)
            
            self.train_metrics.update('capacity_target', C_t)
            self.train_metrics.update('capacity_penalty', capacity_penalty.item())
            
            # NEW: Track edge penalty
            self.train_metrics.update('edge_penalty', edge_penalty.item())

            # NEW: Track dV sparsity penalty
            self.train_metrics.update('dv_l1_penalty', dv_l1_penalty.item())
            self.train_metrics.update('dv_l2_penalty', dv_l2_penalty.item())

            # NEW: Track temporal smoothness
            self.train_metrics.update('temporal_smoothness', temporal_smoothness.item())

            # NEW: Track iterative-refinement per-pass reconstruction loss
            self.train_metrics.update('iter_loss', iter_loss.item())

            # NEW: Track EMA prior metrics (only after pretraining phase when EMA starts updating)
            if self.use_ema_prior and not self.no_phy and hasattr(self.model, 'ema_mean') and epoch > 1 + self.epochs_pretrain:
                ema_mean_norm = torch.norm(self.model.ema_mean).item()
                ema_var = (self.model.ema_m2 - self.model.ema_mean**2).clamp(self.model.ema_min_var, self.model.ema_max_var)
                ema_std_mean = torch.mean(torch.sqrt(ema_var + 1e-8)).item()
                self.train_metrics.update('ema_prior_mean_norm', ema_mean_norm)
                self.train_metrics.update('ema_prior_std_mean', ema_std_mean)
            else:
                self.train_metrics.update('ema_prior_mean_norm', 0.0)
                self.train_metrics.update('ema_prior_std_mean', 1.0)
            
            if not self.no_phy and epoch < self.epochs_pretrain:
                self.train_metrics.update('syn_data_loss', synthetic_data_loss.item())
            else:
                # IMPROVED: Track all residual metrics
                self.train_metrics.update('ortho_penalty', ortho_penalty.item())
                self.train_metrics.update('coeff_penalty', coeff_penalty.item())
                self.train_metrics.update('delta_penalty', delta_penalty.item())
                
                # Track norms for monitoring
                c_norm = torch.norm(c, dim=1).mean().item() if c.numel() > 0 else 0.0
                delta_norm = torch.norm(delta, dim=1).mean().item()
                s_norm = torch.norm(self.model.dec.s).item() if hasattr(self.model.dec, 's') else 0.0
                basis_quality = torch.norm(torch.matmul(self.model.dec.B.T, self.model.dec.B) - torch.eye(self.model.dec.B.shape[1], device=self.model.dec.B.device), p='fro').item() if hasattr(self.model.dec, 'B') else 0.0
                
                self.train_metrics.update('c_norm', c_norm)
                self.train_metrics.update('delta_norm', delta_norm)
                self.train_metrics.update('s_norm', s_norm)
                self.train_metrics.update('basis_quality', basis_quality)

            # Optional u-stats
            if not self.no_phy and self.log_u_stats:
                u = z_phy_stat['mean'].detach()
                if u_sum is None:
                    u_sum = u.sum(dim=0)
                    u2_sum = (u**2).sum(dim=0)
                else:
                    u_sum += u.sum(dim=0)
                    u2_sum += (u**2).sum(dim=0)
                u_count += u.size(0)

            # Logging
            if batch_idx % self.config['trainer']['log_step'] == 0:
                if self.use_capacity_control:
                    kl_aux_str = f"KL_aux {kl_z_aux.item():.6f}" if self.dim_z_aux > 0 else "KL_aux 0.0"
                    log_str = (f"Train Ep {epoch} [{batch_idx}/{len(self.data_loader)}] "
                              f"Loss {loss.item():.6f} Rec {rec_loss.item():.6f} "
                              f"KL_u {kl_u_phy.item():.6f} {kl_aux_str} "
                              f"beta_z_phy {beta_z_phy:.3f} beta_z_aux {beta_z_aux:.3f} "
                              f"C_t {C_t:.3f} Cap_pen {capacity_penalty.item():.6f} "
                              f"Edge_pen {edge_penalty.item():.6f} "
                              f"Residual {residual_loss.item():.6f} "
                              f"residual_rel_diff {residual_rel_diff.item():.2f}%")
                else:
                    log_str = (f"Train Ep {epoch} [{batch_idx}/{len(self.data_loader)}] "
                              f"Loss {loss.item():.6f} Rec {rec_loss.item():.6f} "
                              f"KL(beta_z_phy={beta_z_phy:.3f},beta_z_aux={beta_z_aux:.3f}) {kl_loss.item():.6f} "
                              f"Edge_pen {edge_penalty.item():.6f} "
                              f"Residual {residual_loss.item():.6f} "
                              f"residual_rel_diff {residual_rel_diff.item():.2f}%")
                
                # Add EMA prior info to logging (only after pretraining phase when EMA starts updating)
                if self.use_ema_prior and not self.no_phy and hasattr(self.model, 'ema_mean') and epoch > 1 + self.epochs_pretrain:
                    ema_mean_norm = torch.norm(self.model.ema_mean).item()
                    ema_var = (self.model.ema_m2 - self.model.ema_mean**2).clamp(self.model.ema_min_var, self.model.ema_max_var)
                    ema_std_mean = torch.mean(torch.sqrt(ema_var + 1e-8)).item()
                    log_str += f" EMA_mean_norm {ema_mean_norm:.3f} EMA_std_mean {ema_std_mean:.3f}"
                
                if not self.no_phy and epoch >= self.epochs_pretrain:
                    log_str += f" Ortho {ortho_penalty.item():.6f} Coeff {coeff_penalty.item():.6f} Delta {delta_penalty.item():.6f}"
                
                # log_str += " (Gate: correction strength, Residual: physics vs corrected L2 diff)"
                self.logger.info(log_str)
            self.timer.add('bookkeeping', time.perf_counter() - t_mark)
            t_mark = time.perf_counter()

        log = self.train_metrics.result()

        # Log epoch summary including residual_loss and current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        if self.use_capacity_control:
            kl_aux_summary = f"KL_aux: {log['kl_z_aux']:.6f}" if self.dim_z_aux > 0 else "KL_aux: 0.000000"
            summary_str = (f"Epoch {epoch} Summary - "
                          f"Loss: {log['loss']:.6f}, "
                          f"Rec: {log['rec_loss']:.6f}, "
                          f"KL_u: {log['kl_u_phy']:.6f}, "
                          f"{kl_aux_summary}, "
                          f"C_t: {log['capacity_target']:.3f}, "
                          f"Cap_pen: {log['capacity_penalty']:.6f}, "
                          f"Edge_pen: {log['edge_penalty']:.6f}, "
                          f"Residual: {log['residual_loss']:.6f}, "
                          f"residual_rel_diff: {log['residual_rel_diff']:.2f}%, "
                          f"LR: {current_lr:.6f}")
        else:
            summary_str = (f"Epoch {epoch} Summary - "
                          f"Loss: {log['loss']:.6f}, "
                          f"Rec: {log['rec_loss']:.6f}, "
                          f"KL: {log['kl_loss']:.6f}, "
                          f"Edge_pen: {log['edge_penalty']:.6f}, "
                          f"Residual: {log['residual_loss']:.6f}, "
                          f"residual_rel_diff: {log['residual_rel_diff']:.2f}%, "
                          f"LR: {current_lr:.6f}")
        
        # Add r(t) and tau monitoring (only when residual components exist)
        if not self.no_phy and self.dim_z_aux > 0:
            r_value = self.model.dec.get_r(epoch, self.epochs_pretrain)
            tau_value = self.model.dec.get_tau(epoch, self.epochs_pretrain)
            summary_str += f", r(t): {r_value:.3f}, tau: {tau_value:.3f}"
        
        # Add residual penalties if available
        if not self.no_phy and epoch >= self.epochs_pretrain:
            if 'ortho_penalty' in log:
                summary_str += f", Ortho: {log['ortho_penalty']:.6f}"
            if 'coeff_penalty' in log:
                summary_str += f", Coeff: {log['coeff_penalty']:.6f}"
            if 'delta_penalty' in log:
                summary_str += f", Delta: {log['delta_penalty']:.6f}"
        
        # Add EMA prior info to summary (only after pretraining phase when EMA starts updating)
        if self.use_ema_prior and not self.no_phy and 'ema_prior_mean_norm' in log and epoch > 1 + self.epochs_pretrain:
            summary_str += f", EMA_mean_norm: {log['ema_prior_mean_norm']:.3f}, EMA_std_mean: {log['ema_prior_std_mean']:.3f}"
        
        # summary_str += " (Gate: correction strength, Residual: physics vs corrected L2 diff)"
        self.logger.info(summary_str)

        # wandb logs
        wandb.log({f'train/{key}': value for key, value in log.items()})
        wandb.log({'train/lr': self.optimizer.param_groups[0]['lr']})
        wandb.log({'train/epoch': epoch})
        wandb.log({'train/beta_z_phy': beta_z_phy})
        wandb.log({'train/beta_z_aux': beta_z_aux})
        
        # Additional context for x_p_diff interpretation
        # if log['gate_mean'] > 0.5:
        #     wandb.log({'train/correction_comment': 'High correction activity (gate > 0.5)'})
        # else:
        #     wandb.log({'train/correction_comment': 'Low correction activity (gate < 0.5)'})

        # per-dim u stats (epoch-aggregated)
        if not self.no_phy and self.log_u_stats and u_count > 0:
            u_mean = (u_sum / u_count).cpu().numpy()
            u_var = (u2_sum / u_count).cpu().numpy() - u_mean**2
            u_std = np.sqrt(np.maximum(u_var, 1e-12))
            wandb.log({f'train/u_mean_dim{i}': float(u_mean[i]) for i in range(len(u_mean))})
            wandb.log({f'train/u_std_dim{i}': float(u_std[i]) for i in range(len(u_std))})

        # Validation
        if self.do_validation:
            with self.timer.phase('validation'):
                val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            wandb.log({f'val/{key}': value for key, value in val_log.items()})
            wandb.log({'val/epoch': epoch})
            
            # Additional context for validation residual_loss interpretation
            if 'val_residual_rel_diff' in val_log:
                if val_log['val_residual_rel_diff'] > 10.0:
                    wandb.log({'val/correction_comment': 'High correction impact (>10% change)'})
                elif val_log['val_residual_rel_diff'] > 5.0:
                    wandb.log({'val/correction_comment': 'Moderate correction impact (5-10% change)'})
                else:
                    wandb.log({'val/correction_comment': 'Low correction impact (<5% change)'})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self._write_timing_row(epoch)
        return log

    def _remove_ramp(self, values):
        """(I - G G^+) applied row-wise: removes each row's least-squares offset + planar ramp."""
        return values - (values @ self._ramp_G_pinv.T) @ self._ramp_G.T

    TIMING_COLUMNS = ['epoch', 'data_fetch', 'encode', 'physics_decode', 'residual_decode', 'loss',
                      'backward', 'optimizer_step', 'bookkeeping', 'validation', 'checkpoint_save']

    def _write_timing_row(self, epoch):
        """Append this epoch's per-phase seconds to timing_epochs.csv and log a one-line summary.
        checkpoint_save is charged to the epoch in which the save completes (base trainer saves
        after _train_epoch returns, so it appears in the following row)."""
        if not self.timer.enabled:
            return
        row = self.timer.epoch_row(epoch)
        new_file = not os.path.exists(self.timing_csv)
        with open(self.timing_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.TIMING_COLUMNS, restval=0.0, extrasaction='ignore')
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        parts = ' | '.join(f"{k} {v:.2f}s" for k, v in row.items() if k != 'epoch')
        self.logger.info(f"Epoch {epoch} timing: {parts}")

    def _save_checkpoint(self, epoch, save_best=False, epoch_file=True):
        """
        Override to save tau/r values for inference. epoch_file=False skips checkpoint-epoch<N>.pth (mid-run best
        saves, see BaseTrainer.train).
        """
        t_save0 = time.perf_counter()
        arch = type(self.model).__name__
        
        # Get current tau/r values
        tau_r_values = None
        if not self.no_phy and self.dim_z_aux > 0:
            tau_r_values = self.model.dec.get_current_tau_r(epoch, self.epochs_pretrain)
        
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config,
            'tau_r_values': tau_r_values  # Save tau/r values for inference
        }
        
        if epoch_file:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, os.path.join(PARENT_DIR, filename))
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, os.path.join(PARENT_DIR, best_path))
            self.logger.info("Saving current best: model_best.pth (epoch {}) ...".format(epoch))
        self.timer.add('checkpoint_save', time.perf_counter() - t_save0)

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()

        total_rec_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, data_dict in enumerate(self.valid_data_loader):
                try:
                    data = data_dict[self.data_key].to(self.device)
                    input_const = {k: data_dict[k].to(self.device) for k in self.input_const_keys} if self.input_const_keys else None
                    
                    # Get time features if available
                    time_feats = data_dict.get('time_feats', None)
                    if time_feats is not None:
                        time_feats = time_feats.to(self.device)
                        if time_feats.dim() == 3:  # For sequence data
                            time_feats = time_feats.view(-1, time_feats.size(-1))
                    
                    rec_mask = data_dict.get('mask', None)
                    if rec_mask is not None:
                        rec_mask = rec_mask.to(self.device)
                    is_image = (data.dim() == 4)

                    if data.dim() == 3:
                        # Sequence format: (batch_size, seq_len, features)
                        sequence_len = data.size(1)
                        data = data.view(-1, data.size(-1))

                    # Use same hard_z settings as training
                    hard_z_phy = not self.use_kl_term_z_phy  # Use deterministic sampling when KL term is disabled
                    hard_z_aux = not self.use_kl_term_z_aux  # Use deterministic sampling when KL term is disabled
                    # Get full model output to compute residual_loss.
                    # Iterative refinement: run passes to convergence (matches inference behaviour).
                    if self.use_iterative and not self.no_phy:
                        z_phy_stat, z_aux_stat, z_phy, z_aux = self.model.encode_iterative_infer(
                            data, time_feats, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)
                    else:
                        z_phy_stat, z_aux_stat = self.model.encode(data, time_feats)
                        z_phy, z_aux = self.model.draw(z_phy_stat, z_aux_stat, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)
                    x_PB, x_P, y, delta, c = self.model.decode(z_phy, z_aux, epoch=epoch, epochs_pretrain=self.epochs_pretrain, full=True, const=input_const)
                    
                    # Use unified VAE loss function (same as training)
                    if is_image:
                        rec_target = data.reshape(data.size(0), -1)
                        rec_mask_flat = rec_mask.reshape(rec_mask.size(0), -1) if rec_mask is not None else None
                    else:
                        rec_target, rec_mask_flat = data, None
                    x_PB_for_loss = x_PB
                    if self.nuisance_ramp:
                        rec_target, x_PB_for_loss = self._remove_ramp(rec_target), self._remove_ramp(x_PB)
                    rec_loss, kl_u_phy, kl_z_aux = self._vae_loss(rec_target, z_phy_stat, z_aux_stat, x_PB_for_loss, mask=rec_mask_flat)
                    kl_loss = kl_u_phy + kl_z_aux # Original behavior: combine first, then average
                    
                    # Compute residual_loss for validation (L2 difference)
                    residual_loss = torch.sum((x_PB - x_P).pow(2), dim=1).mean()
                    # Also compute relative difference as percentage
                    residual_rel_diff = torch.mean(torch.abs(x_PB - x_P) / (torch.abs(x_P) + 1e-8)) * 100.0

                    self.valid_metrics.update('rec_loss', rec_loss.item())
                    self.valid_metrics.update('kl_loss', kl_loss.item())
                    self.valid_metrics.update('residual_loss', residual_loss.item())
                    self.valid_metrics.update('residual_rel_diff', residual_rel_diff.item())
                    
                    total_rec_loss += rec_loss.item()
                    total_kl_loss += kl_loss.item()
                    num_batches += 1

                except Exception as e:
                    self.logger.warning(f"Error in validation batch {batch_idx}: {e}")
                    continue

        avg_rec_loss = total_rec_loss / max(num_batches, 1)
        avg_kl_loss = total_kl_loss / max(num_batches, 1)
        
        # Compute average residual_loss for validation logging
        val_metrics = self.valid_metrics.result()
        avg_residual_loss = val_metrics['residual_loss']
        avg_residual_rel_diff = val_metrics['residual_rel_diff']
        self.logger.info(f"Validation Epoch: {epoch} Rec Loss: {avg_rec_loss:.6f} KL Loss: {avg_kl_loss:.6f} Residual: {avg_residual_loss:.6f} residual_rel_diff: {avg_residual_rel_diff:.2f}% (Physics vs Corrected)")
        return self.valid_metrics.result()

    def _vae_loss(self, data, z_phy_stat, z_aux_stat, x, pretrain=False, mask=None, epoch_idx=None):
        """
        VAE loss function that returns separate KL terms for flexible combination.
        Returns: rec_loss, kl_u_phy, kl_z_aux (separate terms for both modes)

        mask : optional [B, D] (1=valid, 0=incoherent). When given (CNN full-field path),
               reconstruction is masked-MSE over valid cells only, so incoherent pixels do
               not contribute. When None, the configured criterion is used (point/GPS path).
        epoch_idx : optional (n,) LongTensor of chronological epoch ordinals. When given AND
               temporal_prior is enabled, the physics-KL prior MEAN is set per-sample to
               epoch (t-1)'s current estimate from the temporal buffer (causal temporal
               prior), instead of the static N(0,I)/EMA prior. Pass None (default) to keep
               the standard prior (validation path, or non-InSAR-point paths).
        """
        if mask is not None:
            sq_err = (x - data).pow(2) * mask
            rec_loss = sq_err.sum() / mask.sum().clamp(min=1.0)
        else:
            # Use the configured loss function for reconstruction loss
            rec_loss = self.criterion(x, data)

        n = data.shape[0]
        prior_u_phy_stat, prior_z_aux_stat = self.model.priors(n, self.device)

        # Physics KL in u-space (per-sample, not averaged yet)
        if not self.no_phy:
            if self.use_temporal_prior and epoch_idx is not None:
                # Causal temporal prior: pull epoch t toward epoch (t-1)'s current guess.
                # Prior mean = buffer[t-1]; prior lnvar = fixed loc/amp split.
                prev = (epoch_idx - 1).clamp(min=0)
                prior_mean = self.temporal_prior_mean[prev].clone()      # (n, dim_z_phy)
                # .repeat (not .expand) so the epoch-0 rows below are writable.
                prior_lnvar = self.temporal_prior_lnvar.unsqueeze(0).repeat(n, 1)  # (n, dim_z_phy)
                # Epoch 0 (MintPy reference date, displacement identically 0) has no
                # predecessor -> full standard-normal cold start N(0,1): mean 0 AND
                # std 1 (lnvar 0). Must reset BOTH; leaving the tight loc lnvar here would
                # wrongly pin epoch 0's geometry to the box centre with the strongest pull
                # in the whole series.
                is_epoch0 = (epoch_idx == 0)
                prior_mean[is_epoch0] = 0.0
                prior_lnvar[is_epoch0] = 0.0
                KL_u_phy = kldiv_normal_normal(
                    z_phy_stat['mean'], z_phy_stat['lnvar'],
                    prior_mean, prior_lnvar
                ).mean()
            else:
                KL_u_phy = kldiv_normal_normal(
                    z_phy_stat['mean'], z_phy_stat['lnvar'],
                    prior_u_phy_stat['mean'], prior_u_phy_stat['lnvar']
                ).mean()     # a scalar value, average of per-sample KL
        else:
            KL_u_phy = torch.zeros(n, device=self.device).mean()

        # Auxiliary KL (per-sample, not averaged yet)
        if pretrain or self.config['arch']['phys_vae']['dim_z_aux'] == 0:
            KL_z_aux = torch.zeros(n, device=self.device).mean()
        else:
            KL_z_aux = kldiv_normal_normal(
                z_aux_stat['mean'], z_aux_stat['lnvar'],
                prior_z_aux_stat['mean'], prior_z_aux_stat['lnvar']
            ).mean()     # a scalar value, average of per-sample KL

        # Return separate KLs for flexible combination
        return rec_loss, KL_u_phy, KL_z_aux


    def _synthetic_data_loss(self, batch_size):
        """
        Synthetic inversion loss in u-space:
        sample z~Uniform(0,1), simulate y, infer u_mean, and match to logit(z).
        
        This loss pretrains the encoder to learn the inverse mapping from physics
        outputs back to latent parameters, establishing a good initialization
        before introducing KL divergence terms.
        """
        if not self.no_phy:
            self.model.eval()
            with torch.no_grad():
                z = torch.rand((batch_size, self.dim_z_phy), device=self.device).clamp(1e-4, 1-1e-4)
                synthetic_y = self.model.generate_physonly(z)  # physics-only
            self.model.train()
            target_u = torch.log(z) - torch.log1p(-z)  # logit(z)

            if self.use_iterative:
                # Mirror the iterative encoder during pretraining: refine the u-estimate over
                # K passes, matching each pass to logit(z) with linearly increasing weight.
                # No init jitter here (unlike encode_iterative's training path) -- pretraining
                # is the deterministic bootstrap that establishes a clean inverse mapping.
                K = self.model.num_passes_train
                u_phy_est = torch.zeros(batch_size, self.dim_z_phy, device=self.device, dtype=synthetic_y.dtype)
                total_loss = torch.tensor(0.0, device=self.device)
                for k in range(K):
                    synthetic_features = self.model.enc.func_feat(synthetic_y, u_phy_est=u_phy_est)
                    inferred_u_phy = self.model.enc.func_z_phy_mean(synthetic_features)
                    pass_loss = torch.sum((inferred_u_phy - target_u).pow(2), dim=1).mean()
                    total_loss = total_loss + ((k + 1) / K) * pass_loss
                    u_phy_est = inferred_u_phy.detach().clamp(-10.0, 10.0)  # detach + clamp between passes
                return total_loss / K

            synthetic_features = self.model.enc.func_feat(synthetic_y)
            inferred_u_phy = self.model.enc.func_z_phy_mean(synthetic_features)  # u-mean
            return torch.sum((inferred_u_phy - target_u).pow(2), dim=1).mean()
        else:
            return torch.zeros(1, device=self.device)

    def _edge_penalty(self, z_phy):
        """
        Edge penalty to encourage z_phy to avoid extreme values (0, 1).
        
        Args:
            z_phy: Physical variables tensor of shape (batch_size, dim_z_phy)
            
        Returns:
            Edge penalty: λ_edge * [-log(z) - log(1-z)]^power
        """
        if self.no_phy or z_phy is None:
            return torch.tensor(0.0, device=self.device)
        
        # Clamp z_phy to avoid numerical issues (fixed eps value)
        eps = 1e-6
        z_clamped = z_phy.clamp(eps, 1.0 - eps)
        
        # Compute edge penalty: -log(z) - log(1-z)
        edge_penalty = -torch.log(z_clamped) - torch.log(1.0 - z_clamped)
        
        # Apply power for fine-grained control
        edge_penalty = edge_penalty.pow(self.edge_penalty_power)
        
        # Return mean penalty across batch and dimensions
        return edge_penalty.mean()

    def _grad_stablizer(self, epoch, batch_idx, loss):
        para_grads = [v.grad.data for v in self.model.parameters(
        ) if v.grad is not None and torch.isnan(v.grad).any()]
        if len(para_grads) > 0:
            epsilon = 1e-7
            for v in para_grads:
                rand_values = torch.rand_like(v, dtype=torch.float)*epsilon
                mask = torch.isnan(v) | v.eq(0)
                v[mask] = rand_values[mask]
            self.stablize_count += 1
            self.logger.info(
                'epoch: {}, batch: {}, loss: {}, stablize count: {}'.format(
                    epoch, batch_idx, loss, self.stablize_count)
            )

    def _linear_annealing_epoch(self, epoch, warmup_epochs=30):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 1.0

    def _temporal_smoothness_loss(self, z_phy, sequence_len):
        """
        Temporal smoothness regularization for the source parameters, computed over the
        leading geometry dims (and optionally dV) across each temporal sequence window.

        The number of smoothed geometry dims is set in __init__ by physics type:
            Mogi  -> 3 (xcen, ycen, d;            dV free unless include_dv)
            Okada -> 7 (xoff,yoff,depth,strike,dip,length,width; opening free)
        If self.temporal_smoothness_include_dv is True and an amplitude dim exists
        (self._dv_dim -- 'dV' for Mogi/Sun69, 'opening' for Okada;
        Mogi/Sun69; None for Okada), the dV dim is added to the smoothed set.

        Two forms (self.temporal_smoothness_mode):
          'variance'  -> mean over dims of Var_t(z)   [zero only if param is CONSTANT]
          'curvature' -> mean over dims,t of (z[t+1]-2z[t]+z[t-1])^2   [zero for any
                         LINEAR trend; only jitter/kinks penalised]. Needs seq_len >= 3.

        Args:
            z_phy: Physical variables tensor of shape (batch_size * seq_len, dim_z_phy),
                   z in (0,1) (sigmoid of the u-space latent).
            sequence_len: Length of temporal sequences (None if not using sequences)

        Returns:
            Scalar temporal-smoothness penalty (0 tensor when inapplicable).
        """
        if self.no_phy or z_phy is None or sequence_len is None:
            return torch.tensor(0.0, device=self.device)

        # Reshape z_phy back to sequence format: (batch_size, seq_len, dim_z_phy)
        batch_size = z_phy.size(0) // sequence_len
        if batch_size * sequence_len != z_phy.size(0):
            # If not evenly divisible, we can't apply temporal smoothness
            return torch.tensor(0.0, device=self.device)

        z_phy_seq = z_phy.view(batch_size, sequence_len, -1)  # (batch_size, seq_len, dim_z_phy)

        # Which parameter dims to keep smooth: leading geometry dims, optionally plus dV.
        # Built in __init__ (geometry dims of every source, + dV dims if include_dv).
        coords = z_phy_seq[:, :, self._smoothed_dims]  # (batch_size, seq_len, n_smoothed)

        if self.temporal_smoothness_mode == 'curvature':
            # Discrete second time difference; zero for any straight line / constant, so a
            # constant-velocity source move or a linear dV ramp is NOT penalised.
            # NOTE (why z-space, not u-space): the decoder z->physical map is AFFINE
            # (model_phys_smpl.rescale: z*(max-min)+min, geometry x1000, dV x scale+shift),
            # so "linear in physical units" <=> "linear in z". The synthetic truths are
            # piecewise-linear in physical units, hence linear in z, hence zero z-curvature.
            # Penalising in u-space (u=logit(z)) would instead penalise the true linear ramp,
            # because logit is nonlinear. So z-space is correct HERE; revisit only if any
            # param's z->physical map is ever made nonlinear (e.g. log-depth).
            if sequence_len < 3:
                # Curvature is undefined for windows shorter than 3 epochs.
                return torch.tensor(0.0, device=self.device)
            second_diff = coords[:, 2:, :] - 2.0 * coords[:, 1:-1, :] + coords[:, :-2, :]
            return second_diff.pow(2).mean()

        # 'variance' mode (default): variance across the seq_len (time) axis per param.
        coord_variance = torch.var(coords, dim=1)  # (batch_size, n_smoothed)
        return coord_variance.mean()
