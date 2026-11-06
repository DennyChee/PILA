#!/usr/bin/env python
# Usage:       python run_k2_tune_pen.py --album <npz> --seed 1 --epochs 300 --tag k2tune_noisyE1pen_wr100_seed1 \
#                  [--save-root saved/multisource_k2_tune] [--lr-scheduler '{...}'] [--dry-run]
#              (all run_k2_tune.py flags are accepted)
# Description: Same as run_k2_tune.py (honest K=2 from scratch, no truth) but with the STANDARD penalties ON:
#              temporal curvature 15000 (include_dv, seq loader), dV-L1 300 in u-space, z_phy KL beta 0.5 (30-epoch
#              warm-up) -- i.e. make_sequential_config's standard recipe with no trainer overrides. Implemented by
#              clearing run_k2_tune.NO_PENALTY_OVERRIDES before calling its main(), so run_k2_tune.py is unchanged.
# Date:        2026-10-09

import run_k2_tune

# --- Keep the standard-recipe penalties: an empty override dict leaves cfg['trainer']['phys_vae'] untouched ---
run_k2_tune.NO_PENALTY_OVERRIDES = {}

if __name__ == '__main__':
    print("[run_k2_tune_pen] standard penalties ON (curvature 15000, dV-L1 300 u-space, KL beta 0.5)")
    run_k2_tune.main()
