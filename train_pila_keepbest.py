#!/usr/bin/env python
# Usage:       python train_pila_keepbest.py -c <config.json>     (same arguments as train_pila.py)
# Description: train_pila_delayedpen.py (penalty schedule + per_epoch_sources.csv) PLUS "keep the lowest-loss epoch":
#              whenever the epoch's validation reconstruction loss (val_rec_loss, PILA's own misfit on the album -- never
#              the truth) reaches a new minimum, the model is saved to <run models dir>/lowest_loss/model_lowest_loss.pth.
#              After training, that checkpoint is dumped with dump_multisource_fit.py to lowest_loss/fit_K<k>.npz.
#              WHY: base_trainer always overwrites model_best.pth with the FINAL epoch, and in the K=2 SNR-12 study
#              several seeds separated the two sources mid-run at a lower loss, then a warm-restart LR jump threw them
#              back into the trap. The final-epoch outputs (model_best.pth, fit_K2.npz) are untouched, so a run is also
#              directly comparable to the earlier runs of the same arm.
# Date:        2026-10-22

import json
import os
import runpy
import subprocess
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import train_pila_delayedpen  # noqa: E402,F401  (importing applies its _train_epoch / draw patches)
from trainer import PhysVAETrainerSMPL  # noqa: E402

MONITOR_KEY = 'val_rec_loss'
_scheduled_train_epoch = PhysVAETrainerSMPL._train_epoch       # the delayed-penalty wrapper
_run_state = {'models_dir': None}


def keep_lowest_train_epoch(self, epoch):
    """Run the scheduled epoch, then save the model if val_rec_loss is a new minimum (lowest_loss/)."""
    result = _scheduled_train_epoch(self, epoch)
    if MONITOR_KEY not in result:
        raise KeyError(f"{MONITOR_KEY} missing from the epoch log; keep-lowest needs a validation loader")
    loss = float(result[MONITOR_KEY])
    best_dir = os.path.join(str(self.config.save_dir), 'lowest_loss')
    _run_state['models_dir'] = str(self.config.save_dir)
    if loss < getattr(self, '_lowest_val_rec_loss', float('inf')):
        self._lowest_val_rec_loss = loss
        os.makedirs(best_dir, exist_ok=True)
        state = {'arch': type(self.model).__name__, 'epoch': epoch, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'monitor_best': loss, 'config': self.config}
        torch.save(state, os.path.join(best_dir, 'model_lowest_loss.pth'))
        with open(os.path.join(best_dir, 'lowest_loss.json'), 'w') as handle:
            json.dump({'epoch': epoch, MONITOR_KEY: loss, 'train_rec_loss': float(result.get('rec_loss', float('nan')))},
                      handle, indent=2)
        self.logger.info(f"[keep_lowest] epoch {epoch}: new lowest {MONITOR_KEY} {loss:.4f} -> lowest_loss/")
    return result


PhysVAETrainerSMPL._train_epoch = keep_lowest_train_epoch       # in memory only

if __name__ == '__main__':
    runpy.run_path(os.path.join(PROJECT_ROOT, 'train_pila.py'), run_name='__main__')
    # --- Dump the lowest-loss model the same way train_pila.py dumps the final one ---
    models_dir = _run_state['models_dir']
    checkpoint = os.path.join(models_dir, 'lowest_loss', 'model_lowest_loss.pth') if models_dir else None
    if not checkpoint or not os.path.exists(checkpoint):
        raise RuntimeError(f"keep-lowest: no lowest-loss checkpoint was written (models dir {models_dir})")
    n_sources = int(torch.load(checkpoint, map_location='cpu', weights_only=False)['config']['arch']['args'].get('n_sources', 1))
    out_npz = os.path.join(models_dir, 'lowest_loss', f'fit_K{n_sources}.npz')
    print(f"[keep_lowest] Dumping {checkpoint} -> {out_npz}")
    subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'dump_multisource_fit.py'), '--resume', checkpoint,
                    '--out', out_npz], check=True, cwd=PROJECT_ROOT)
