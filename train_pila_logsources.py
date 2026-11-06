#!/usr/bin/env python
# Usage:       python train_pila_logsources.py -c <config.json>     (same arguments as train_pila.py)
# Description: Runs train_pila.py UNCHANGED (standard penalties, no schedule) but writes one row per epoch to
#              <run models dir>/per_epoch_sources.csv: train rec loss, per-slot median xcen/ycen/depth (km),
#              dV (m^3) of the encoder MEAN over that epoch's training batches, and median u-space sigma.
#              Reuses the logger of train_pila_delayedpen.py; its delayed-penalty _train_epoch patch is replaced
#              here by a logging-only wrapper, and its draw patch is a pass-through at noise scale 1 (stash only).
#              No existing file is edited.
# Date:        2026-10-20

import os
import runpy
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import train_pila_delayedpen as delayed  # noqa: E402  (importing applies its patches; _train_epoch is re-patched below)
from trainer import PhysVAETrainerSMPL  # noqa: E402


def logging_train_epoch(self, epoch):
    """Unchanged _train_epoch plus the per-epoch source row (penalty factor recorded as 1 = full standard weights)."""
    self.model._epoch_stash = {'mean': [], 'lnvar': []}
    result = delayed._original_train_epoch(self, epoch)
    delayed.write_source_row(self, epoch, 1.0, result)
    self.model._epoch_stash = None
    return result


PhysVAETrainerSMPL._train_epoch = logging_train_epoch       # in memory only
if __name__ == '__main__':
    runpy.run_path(os.path.join(PROJECT_ROOT, 'train_pila.py'), run_name='__main__')
