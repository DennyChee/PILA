#!/usr/bin/env python
# Usage:       from utils.timing import PhaseTimer
#              timer = PhaseTimer(enabled=True)
#              with timer.phase('encode'):
#                  ...
#              timer.attach_module(model.physics_model, 'physics_decode',
#                                  within='decode', remainder_name='residual_decode')
#              row = timer.epoch_row(epoch)                                 # per-epoch totals (s)
# Description: Lightweight wall-clock phase timer for PILA training/inference, so the cost of
#              data fetch, encoding, physics decoding, residual decoding, loss, backward and the
#              optimiser step can be compared across runs (e.g. K = 1, 2, 3 sources) from the logs.
# Notes:       time.perf_counter() per phase, accumulated per epoch and over the run. On GPU it calls
#              torch.cuda.synchronize() at phase boundaries so asynchronous kernels are charged to
#              the right phase. Overhead ~ a few microseconds per call (per batch, not per op).
# Date:        2026-10-06

import contextlib
import time
from collections import OrderedDict

import torch


class PhaseTimer:
    """Accumulate wall-clock seconds per named phase, per epoch and in total."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.epoch_s = OrderedDict()            # phase -> seconds in the current epoch
        self.total_s = OrderedDict()            # phase -> seconds over the run
        self.counts = OrderedDict()             # phase -> number of timed calls (run)
        self._hook_start = {}
        self._open = set()                      # phases currently open (for nested hook timing)
        self._nested = {}                       # child phase -> parent phase it is carved out of

    @staticmethod
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def add(self, name, seconds, count=1):
        """Add `seconds` to phase `name` (epoch and run totals)."""
        self.epoch_s[name] = self.epoch_s.get(name, 0.0) + seconds
        self.total_s[name] = self.total_s.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + count

    @contextlib.contextmanager
    def phase(self, name):
        """Context manager timing one block as phase `name`."""
        if not self.enabled:
            yield
            return
        self._sync()
        t0 = time.perf_counter()
        self._open.add(name)
        try:
            yield
        finally:
            self._sync()
            self._open.discard(name)
            self.add(name, time.perf_counter() - t0)

    def attach_module(self, module, name, within, remainder_name):
        """
        Time every forward() of `module` that runs INSIDE the open phase `within`, as phase `name`
        (forward hooks). The parent phase is then reported split into `name` and `remainder_name`
        (= within - name), e.g. decode -> physics_decode + residual_decode. Calls outside `within`
        (e.g. during validation) are not counted here.
        """
        if not self.enabled or module is None:
            return
        self._nested[name] = (within, remainder_name)

        def pre_hook(_mod, _inp):
            if within in self._open:
                self._sync()
                self._hook_start[name] = time.perf_counter()

        def post_hook(_mod, _inp, _out):
            if name in self._hook_start:
                self._sync()
                self.add(name, time.perf_counter() - self._hook_start.pop(name))

        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(post_hook)

    def _split_nested(self, seconds_by_phase):
        """Replace each parent phase by (child, parent - child) so phases do not double count."""
        out = OrderedDict(seconds_by_phase)
        for child, (parent, remainder) in self._nested.items():
            if parent in out:
                parent_s = out.pop(parent)
                out[remainder] = max(parent_s - out.get(child, 0.0), 0.0)
        return out

    def epoch_row(self, epoch):
        """Return {'epoch': e, phase: seconds, ...} for the epoch just finished and reset epoch totals."""
        row = OrderedDict(epoch=epoch)
        row.update((k, round(v, 4)) for k, v in self._split_nested(self.epoch_s).items())
        self.epoch_s = OrderedDict()
        return row

    def summary(self, n_samples_seen=None):
        """Run totals (s), fractions of the timed total, and per-call / per-sample milliseconds."""
        totals = self._split_nested(self.total_s)
        counts = dict(self.counts)
        for child, (parent, remainder) in self._nested.items():
            counts[remainder] = counts.get(parent, 1)
        total = sum(totals.values()) or 1e-12
        out = OrderedDict()
        for name, seconds in totals.items():
            out[f'{name}_sec'] = round(seconds, 3)
            out[f'{name}_frac'] = round(seconds / total, 4)
            out[f'{name}_ms_per_call'] = round(1e3 * seconds / max(counts.get(name, 1), 1), 4)
            if n_samples_seen:
                out[f'{name}_ms_per_sample'] = round(1e3 * seconds / n_samples_seen, 5)
        out['timed_total_sec'] = round(total, 3)
        return out
