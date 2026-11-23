# Description: Keep the lowest-validation-loss epoch (from legacy train_pila_keepbest.py, logic copied verbatim).
#              Whenever val_rec_loss (PILA's own misfit on the album -- never the truth) reaches a new minimum, the
#              model is saved to <run models dir>/lowest_loss/model_lowest_loss.pth (+ lowest_loss.json). After
#              training, run_pila dumps it with methods/pila/dump_fit.py to lowest_loss/fit_K<k>.npz.
#              Config: trainer.extensions.keep_lowest = true (forced on by basin_hop, as in the legacy chain).
# Date:        2026-11-23
import json
import os

import torch

MONITOR_KEY = 'val_rec_loss'


def make_keep_lowest_epoch(trainer, inner_epoch):
    """Wrap one trainer's epoch: run it, then save the model if val_rec_loss is a new minimum (lowest_loss/)."""
    self = trainer

    def keep_lowest_train_epoch(epoch):
        result = inner_epoch(epoch)
        if MONITOR_KEY not in result:
            raise KeyError(f"{MONITOR_KEY} missing from the epoch log; keep-lowest needs a validation loader")
        loss = float(result[MONITOR_KEY])
        best_dir = os.path.join(str(self.config.save_dir), 'lowest_loss')
        if loss < getattr(self, '_lowest_val_rec_loss', float('inf')):
            self._lowest_val_rec_loss = loss
            os.makedirs(best_dir, exist_ok=True)
            state = {'arch': type(self.model).__name__, 'epoch': epoch, 'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(), 'monitor_best': loss, 'config': self.config}
            torch.save(state, os.path.join(best_dir, 'model_lowest_loss.pth'))
            with open(os.path.join(best_dir, 'lowest_loss.json'), 'w') as handle:
                json.dump({'epoch': epoch, MONITOR_KEY: loss,
                           'train_rec_loss': float(result.get('rec_loss', float('nan')))}, handle, indent=2)
            self.logger.info(f"[keep_lowest] epoch {epoch}: new lowest {MONITOR_KEY} {loss:.4f} -> lowest_loss/")
        return result
    return keep_lowest_train_epoch
