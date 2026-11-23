# Description: Optional training extensions for PILA, switched on by CONFIG keys instead of separate launcher
#              scripts (replaces legacy train_pila_{delayedpen,logsources,keepbest,basinhop,biasinit}.py, which
#              patched the classes at import time and chained each other through runpy).
#
#   Every extension is switched on EXPLICITLY in trainer.extensions; its settings stay where they always were.
#   A settings block WITHOUT the switch is ignored (with a warning), exactly as legacy train_pila.py ignored it,
#   so legacy configs that carried leftover blocks reproduce their original launcher.
#
#   switch (trainer.extensions)   settings block                     legacy launcher
#   delayed_penalty: true         trainer.phys_vae.delayed_penalty   train_pila_delayedpen.py
#   log_per_epoch_sources: true   -                                  train_pila_logsources.py
#   keep_lowest: true             -                                  train_pila_keepbest.py (+ delayed_penalty)
#   basin_hop: true               trainer.phys_vae.basin_hop         train_pila_basinhop.py (+ delayed_penalty;
#                                                                    forces keep_lowest)
#   arch.phys_vae.init_sources_mode: "bias_only"                      train_pila_biasinit.py
#
#   Difference from legacy (more permissive, only affects configs that used to crash): legacy keepbest/basinhop
#   REQUIRED a delayed_penalty block (they imported the delayed-penalty launcher); here keep_lowest / basin_hop run
#   without it, then with no schedule, no draw wrapper and no per_epoch_sources.csv.
#
#   Wrapping order (innermost first) is the legacy import chain, so numerics are identical:
#     original _train_epoch -> (delayed penalty | source log) -> keep_lowest -> basin_hop
#   The z_phy draw wrapper (stash + noise scale; a pass-through at scale 1) is installed whenever the delayed
#   penalty or the source log is on, as the legacy imports did.
# Date:        2026-11-23
import warnings

import torch

from pila.extensions.source_log import install_draw_wrapper, make_logging_epoch
from pila.extensions.penalty_schedule import make_delayed_penalty_epoch
from pila.extensions.keep_lowest import make_keep_lowest_epoch
from pila.extensions.basin_hop import make_basin_hop_epoch, finish_basin_hop


def active_extensions(config):
    """
    Decide which extensions a config switches on.

    Input:  config (ConfigParser or dict).
    Output: dict of booleans: delayed_penalty, log_sources, keep_lowest, basin_hop.
    """
    phys = config['trainer']['phys_vae']
    flags = config['trainer'].get('extensions', {}) or {}
    unknown = set(flags) - {'delayed_penalty', 'log_per_epoch_sources', 'keep_lowest', 'basin_hop'}
    if unknown:
        raise ValueError(f"unknown trainer.extensions keys {sorted(unknown)}")
    delayed = bool(flags.get('delayed_penalty', False))
    basin_hop = bool(flags.get('basin_hop', False))
    if delayed and not phys.get('delayed_penalty'):
        raise ValueError("trainer.extensions.delayed_penalty is on but trainer.phys_vae.delayed_penalty is missing")
    if basin_hop and not phys.get('basin_hop'):
        raise ValueError("trainer.extensions.basin_hop is on but trainer.phys_vae.basin_hop is missing")
    for block, switched_on in (('delayed_penalty', delayed), ('basin_hop', basin_hop)):
        if phys.get(block) and not switched_on:
            warnings.warn(f"trainer.phys_vae.{block} is present but trainer.extensions.{block} is not true: IGNORED "
                          f"(legacy train_pila.py behaviour)")
    return dict(delayed_penalty=delayed,
                log_sources=bool(flags.get('log_per_epoch_sources', False)),
                keep_lowest=bool(flags.get('keep_lowest', False)) or basin_hop,     # legacy: basinhop imported keepbest
                basin_hop=basin_hop)


def install_extensions(trainer, config):
    """
    Wrap trainer._train_epoch (and model.draw) on the INSTANCE according to the config. Returns the active dict.
    The model must not be wrapped in DataParallel (all PILA runs are single-device).
    """
    active = active_extensions(config)
    if any(active.values()) and isinstance(trainer.model, torch.nn.DataParallel):
        # instance wrappers bind to the un-replicated module; legacy class patches did not have this limitation
        raise ValueError("training extensions do not support DataParallel; run single-device (n_gpu <= 1)")
    epoch_fn = trainer._train_epoch                              # bound original method
    if active['delayed_penalty'] or active['log_sources']:
        install_draw_wrapper(trainer.model)
    if active['delayed_penalty']:
        epoch_fn = make_delayed_penalty_epoch(trainer, epoch_fn)
    elif active['log_sources']:
        epoch_fn = make_logging_epoch(trainer, epoch_fn)
    if active['keep_lowest']:
        epoch_fn = make_keep_lowest_epoch(trainer, epoch_fn)
    if active['basin_hop']:
        epoch_fn = make_basin_hop_epoch(trainer, epoch_fn)
    trainer._train_epoch = epoch_fn                              # base_trainer.train() calls self._train_epoch
    return active
