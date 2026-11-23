# Description: Delayed-penalty schedule (from legacy train_pila_delayedpen.py, logic copied verbatim).
#              Config: trainer.phys_vae.delayed_penalty = {"off_epochs": N, "ramp_epochs": M, "kl_stays_off": false}
#                epochs 1..N      : penalties OFF (curvature 0, dV-L1 0, z_phy KL off -> deterministic z_phy)
#                epochs N+1..N+M  : weights ramp linearly 0 -> full config value; z_phy sampling noise ramps with them
#                epochs > N+M     : full standard penalties
#              kl_stays_off=true: curvature/dV-L1 follow the schedule, KL and sampling stay off for the whole run.
#              The switch is a fixed epoch, never a truth-based criterion (honest inversion).
#              Also writes per_epoch_sources.csv (see source_log.py).
# Assumptions: epochs_pretrain = 0; the KL linear warm-up has reached 1 by epoch N.
# Date:        2026-11-23
from pila.extensions.source_log import write_source_row


def penalty_ramp_factor(epoch, off_epochs, ramp_epochs):
    """
    Fraction (0-1) of the full penalty weights to apply at this epoch.

    Inputs:  epoch (int, 1-based as in base_trainer), off_epochs (int) penalty-free epochs,
             ramp_epochs (int) linear ramp length (0 = switch on at once).
    Output:  float in [0, 1].
    """
    if epoch <= off_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, (epoch - off_epochs) / float(ramp_epochs))


def make_delayed_penalty_epoch(trainer, inner_epoch):
    """Wrap one trainer's epoch: set this epoch's penalty weights from the schedule, run it, log per-slot sources."""
    self = trainer                                   # keep the legacy body readable (it was a method)

    def delayed_penalty_train_epoch(epoch):
        if not hasattr(self, '_delayed_full_weights'):
            schedule = self.config['trainer']['phys_vae'].get('delayed_penalty')
            if not schedule:
                raise ValueError("delayed penalty needs trainer.phys_vae.delayed_penalty in the config")
            if self.epochs_pretrain != 0:
                raise ValueError(f"delayed penalty assumes epochs_pretrain=0 (got {self.epochs_pretrain})")
            self._delayed_schedule = (int(schedule['off_epochs']), int(schedule['ramp_epochs']))
            self._kl_stays_off = bool(schedule.get('kl_stays_off', False))
            self._delayed_full_weights = dict(
                temporal_smoothness_weight=self.temporal_smoothness_weight, dv_l1_weight=self.dv_l1_weight,
                use_kl_term_z_phy=self.use_kl_term_z_phy, beta_max_z_phy=self.beta_max_z_phy)
            print(f"[delayed_penalty] schedule off={self._delayed_schedule[0]} ramp={self._delayed_schedule[1]} "
                  f"kl_stays_off={self._kl_stays_off}, full weights {self._delayed_full_weights}")
        factor = penalty_ramp_factor(epoch, *self._delayed_schedule)
        full = self._delayed_full_weights
        self.temporal_smoothness_weight = full['temporal_smoothness_weight'] * factor
        self.dv_l1_weight = full['dv_l1_weight'] * factor
        # KL fully OFF (deterministic z_phy) until the ramp starts; then scale beta. Isolating arm: never on.
        kl_factor = 0.0 if self._kl_stays_off else factor
        self.use_kl_term_z_phy = full['use_kl_term_z_phy'] and kl_factor > 0.0
        self.beta_max_z_phy = full['beta_max_z_phy'] * kl_factor
        self.model._z_phy_noise_scale = kl_factor       # sampling noise ramps with the KL weight
        self.logger.info(f"[delayed_penalty] epoch {epoch}: factor={factor:.3f} curvature={self.temporal_smoothness_weight:.1f} "
                         f"dv_l1={self.dv_l1_weight:.1f} kl_on={self.use_kl_term_z_phy} beta_max={self.beta_max_z_phy:.3f}")
        self.model._epoch_stash = {'mean': [], 'lnvar': []}
        result = inner_epoch(epoch)
        write_source_row(self, epoch, factor, result)
        self.model._epoch_stash = None
        return result
    return delayed_penalty_train_epoch
