import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from base import PARENT_DIR
import os
import json


def _resolve_dv_physical_endpoints(config):
    """
    Resolve the physical dV values at the latent extremes (z=0 and z=1) implied by a
    config's Mogi/Sun69 paras JSON, i.e. the actual dV physical mapping the decoder uses:
        dV_phys(z) = (z*(max-min)+min) * scale + shift   (scale/shift default 1e5/-1e7)
    Returns (dV_at_z0, dV_at_z1) rounded to drop float noise, or None if the physics has
    no dV parameter (e.g. Okada) or the paras file can't be read. Comparing the resulting
    endpoints (not the raw min/max/scale/shift) means two equivalent reparameterizations
    of the SAME linear map are treated as equal. Used to guard resume against a silent dV
    mapping change (see model/model_phys_smpl.py Physics_Mogi.rescale).
    """
    try:
        args = config['arch']['args']
    except (KeyError, TypeError):
        return None
    paras_path = next((args[k] for k in ('mogi_paras', 'sun69_paras') if k in args), None)
    if paras_path is None:
        return None
    full = os.path.join(PARENT_DIR, paras_path)
    if not os.path.exists(full):
        return None
    with open(full) as fh:
        ranges = json.load(fh)
    dv = ranges.get('dV')
    if dv is None:
        return None
    minv, maxv = float(dv['min']), float(dv['max'])
    scale = float(dv.get('scale', 1e5))
    shift = float(dv.get('shift', -1e7))
    dv_z0 = minv * scale + shift
    dv_z1 = maxv * scale + shift
    return (round(dv_z0, 3), round(dv_z1, 3))


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance              
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic.

        model_best.pth policy (trainer.best_checkpoint, default "lowest_monitor", 2026-10-22):
          lowest_monitor  model_best.pth is written the moment the monitored metric (e.g. val_rec_loss) reaches a new
                          best, so it always holds the best state seen -- also when a warm-restart LR jump later throws
                          the model out of it. The final model stays in checkpoint-epoch<N>.pth. With monitor "off" (or
                          the metric missing) model_best.pth falls back to the final model. model_best_info.json records
                          which epoch it is.
          final           legacy behaviour (all runs before 2026-10-22): model_best.pth = the final epoch.
        """
        policy = self.config['trainer'].get('best_checkpoint', 'lowest_monitor')
        if policy == 'final':
            return self._train_legacy_final_best()
        if policy != 'lowest_monitor':
            raise ValueError(f"trainer.best_checkpoint must be 'lowest_monitor' or 'final', got {policy!r}")
        not_improved_count = 0
        last_saved_epoch = None
        best_epoch = None
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch}
            log.update(result)
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # --- Save model_best.pth immediately whenever the monitored metric improves ---
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. model_best.pth will be the final "
                                        "model.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best_epoch = epoch
                    self._save_checkpoint(epoch, save_best=True, epoch_file=False)
                    self._write_best_info(epoch, 'lowest_monitor')
                else:
                    not_improved_count += 1
                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. Training stops "
                                     "(model_best.pth = epoch {}).".format(self.early_stop, best_epoch))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False)
                last_saved_epoch = epoch

        # --- Final model always kept as checkpoint-epoch<N>.pth ---
        if last_saved_epoch != epoch:
            self._save_checkpoint(epoch, save_best=False)
        if best_epoch is None:                 # no monitor / metric missing: best = final (as before)
            self._save_checkpoint(epoch, save_best=True, epoch_file=False)
            self._write_best_info(epoch, 'final (no monitored metric)')
            best_epoch = epoch
        self.logger.info("Training completed. model_best.pth = epoch {} ({} {}); final model = "
                         "checkpoint-epoch{}.pth".format(best_epoch, self.monitor, self.mnt_best, epoch))

    def _write_best_info(self, epoch, how):
        """Record which epoch model_best.pth holds (provenance for every downstream reader)."""
        info = {'epoch': int(epoch), 'policy': how, 'monitor': self.monitor,
                'monitor_value': float(self.mnt_best) if self.mnt_mode != 'off' else None}
        with open(os.path.join(PARENT_DIR, str(self.checkpoint_dir / 'model_best_info.json')), 'w') as handle:
            json.dump(info, handle, indent=2)

    def _train_legacy_final_best(self):
        """
        Legacy training loop (trainer.best_checkpoint = "final"): model_best.pth = final epoch, unchanged from before
        2026-10-22, for reproducing earlier runs exactly.
        """
        not_improved_count = 0
        last_saved_epoch = None
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    # When early stopping, save the best model from the last saved checkpoint
                    if last_saved_epoch is not None:
                        self.logger.info("Saving best model from epoch {} as model_best.pth".format(last_saved_epoch))
                        # Load the last saved checkpoint and save it as model_best
                        self._load_and_save_best(last_saved_epoch)
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
                last_saved_epoch = epoch
        
        # Save final checkpoint at the end of training if not already saved
        if epoch % self.save_period != 0:
            self._save_checkpoint(epoch, save_best=best)
            last_saved_epoch = epoch
        
        # Always save the last model as the best model at the end of training
        self.logger.info("Training completed. Saving final model as model_best.pth ...")
        self._save_checkpoint(epoch, save_best=True)
        self._write_best_info(epoch, 'final (legacy)')

    def _save_checkpoint(self, epoch, save_best=False, epoch_file=True):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, also write the state to 'model_best.pth'
        :param epoch_file: if False, skip checkpoint-epoch<N>.pth (used for mid-run best saves, to save disk)
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if epoch_file:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, os.path.join(PARENT_DIR, filename))
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, os.path.join(PARENT_DIR, best_path))
            self.logger.info("Saving current best: model_best.pth (epoch {}) ...".format(epoch))

    def _load_and_save_best(self, epoch):
        """
        Load a checkpoint from a specific epoch and save it as model_best.pth
        
        :param epoch: epoch number of the checkpoint to load
        """
        try:
            checkpoint_path = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            checkpoint = torch.load(os.path.join(PARENT_DIR, checkpoint_path), map_location='cpu')
            
            # Save as model_best.pth
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(checkpoint, os.path.join(PARENT_DIR, best_path))
            self.logger.info("Loaded checkpoint from epoch {} and saved as model_best.pth".format(epoch))
        except Exception as e:
            self.logger.warning("Failed to load checkpoint from epoch {}: {}".format(epoch, str(e)))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")

        # Guard: refuse to resume if the dV physical mapping differs from the checkpoint's.
        # The encoder's u-space is calibrated to the dV affine active at TRAINING time; a
        # checkpoint trained under one mapping (e.g. legacy 1e5/-1e7) resumed under another
        # (e.g. configs/mogi_paras_symdV.json, symmetric about 0) silently reinterprets the
        # same learned latent as a very different physical volume -- no shape mismatch, no
        # error, just wrong physics. Compare the resulting z=0/z=1 endpoints and fail loudly.
        ckpt_dv = _resolve_dv_physical_endpoints(checkpoint['config'])
        cur_dv = _resolve_dv_physical_endpoints(self.config)
        if ckpt_dv is not None and cur_dv is not None and ckpt_dv != cur_dv:
            raise ValueError(
                "Refusing to resume: the dV physical mapping differs between the checkpoint "
                f"and the current config. Checkpoint dV endpoints (z=0, z=1) = {ckpt_dv} m^3; "
                f"current config = {cur_dv} m^3. The encoder u-space is calibrated to the "
                "training-time mapping, so resuming under a different one corrupts recovered "
                "volumes. Retrain from scratch, or point --config at a paras file whose dV "
                "affine matches the checkpoint.")

        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
