# Usage:       from utils.amp_box import write_data_scaled_paras
#              paras_path, info = write_data_scaled_paras(config, out_dir)   # config = PILA config dict
#              python -m utils.amp_box --config configs/synth/<cfg>.json --out-dir <dir>   (CLI, prints the box)
# Description: DATA-SCALED amplitude prior box ("the jug"). PILA divides both the data and the decoder output by
#              the scene's global LOS std, so surface LOS being linear in the amplitude parameter (Mogi/Sun69 dV,
#              Okada opening) means a box proportional to that std makes the inversion invariant to the scene's
#              absolute amplitude (verified on the K=1 weak album: box scaling == data scaling, 2026-10-22).
#                Mogi / MultiMogi / Sun69 : dV box = +-2e8 m^3 * std_mm / 12 mm
#                Okada                    : opening box = +-opening_ref * std_mm / 12 mm, where opening_ref is the
#                                           opening of a MID-PRIOR dike whose peak |LOS| equals that of the reference
#                                           Mogi (dV 2e8 m^3 at mid-prior depth) on the same points ("exchange rate").
#              Only the data and the prior midpoints are used -- never the truth. The box is written as an ordinary
#              paras JSON so every downstream reader (trainer, dump, plots, resume guard) sees the same numbers.
# Date:        2026-10-22

import argparse
import json
import os

import numpy as np

STANDARD_DV_MAX_M3 = 2.0e8      # configs/mogi_paras_symdV.json box = the 12 mm anchor
ANCHOR_STD_MM = 12.0            # data std at which the data-scaled box equals the standard box

# arch.args key -> (amplitude parameter name, decoder family)
PARAS_KEYS = {'mogi_paras': ('dV', 'mogi'), 'sun69_paras': ('dV', 'sun69'), 'okada_paras': ('opening', 'okada')}


def _paras_key(config):
    """Return the single amplitude-bearing paras key present in arch.args (mogi/sun69/okada)."""
    keys = [key for key in PARAS_KEYS if key in config['arch']['args']]
    if len(keys) != 1:
        raise ValueError(f"expected exactly one of {list(PARAS_KEYS)} in arch.args, found {keys}")
    return keys[0]


def _resolve(path, project_root):
    """Absolute path of a repo-relative or absolute config path."""
    return path if os.path.isabs(path) else os.path.join(project_root, path)


def scene_std_mm(config):
    """Global LOS std (mm) of the scene = x_scale_global of the SAME memoized loader the decoder uses."""
    from model.model_phys_smpl import _load_insar_data
    return float(_load_insar_data(config).x_scale_global)


def okada_opening_ref_m(config, okada_paras, mogi_paras):
    """
    Opening (m) of a mid-prior Okada dike whose peak |LOS| equals that of a Mogi source of dV = STANDARD_DV_MAX_M3
    at mid-prior depth, both centred at the prior-midpoint location, on the scene's own points/LOS vectors.

    Inputs : config (for points + LOS), okada_paras / mogi_paras dicts (km, deg, m; Mogi dV in m^3)
    Output : (opening_ref_m, peak_mogi_mm, peak_okada_per_m_mm)
    LOS is linear in opening, so opening_ref = peak_mogi / peak_okada(opening = 1 m).
    """
    import torch
    from model.model_phys_smpl import _load_insar_inputs, project_enu_to_los
    from physics.mogi.mogi import Mogi
    from physics.okada.okada_dike import OkadaDike

    x_east_m, y_north_m, los_e, los_n, los_u, _, _, n_points = _load_insar_inputs(config)
    los_e, los_n, los_u = los_e.cpu(), los_n.cpu(), los_u.cpu()
    mid = lambda paras, key: 0.5 * (paras[key]['min'] + paras[key]['max'])   # prior midpoint

    # Reference Mogi: the standard box edge (2e8 m^3) at mid-prior location/depth (km -> m)
    mogi = Mogi(x_east_m, y_north_m)
    enu_mogi_mm = mogi.run(xcen=torch.tensor([mid(mogi_paras, 'xcen') * 1e3]),
                           ycen=torch.tensor([mid(mogi_paras, 'ycen') * 1e3]),
                           d=torch.tensor([mid(mogi_paras, 'd') * 1e3]),
                           dV=torch.tensor([STANDARD_DV_MAX_M3]))
    peak_mogi_mm = float(project_enu_to_los(enu_mogi_mm, los_e, los_n, los_u, n_points).abs().max())

    # Mid-prior dike with a 1 m opening (km -> m for xoff/yoff/depth/length/width; deg kept)
    okada = OkadaDike(x_east_m, y_north_m)
    kwargs = {key: torch.tensor([mid(okada_paras, key) * (1e3 if key in ('xoff', 'yoff', 'depth', 'length', 'width') else 1.0)])
              for key in ('xoff', 'yoff', 'depth', 'strike', 'dip', 'length', 'width')}
    enu_okada_mm = okada.run(**kwargs, opening=torch.tensor([1.0]))
    peak_okada_per_m_mm = float(project_enu_to_los(enu_okada_mm, los_e, los_n, los_u, n_points).abs().max())
    return peak_mogi_mm / peak_okada_per_m_mm, peak_mogi_mm, peak_okada_per_m_mm


def write_data_scaled_paras(config, out_dir, project_root=None, anchor_std_mm=ANCHOR_STD_MM,
                            mogi_reference='configs/mogi_paras_symdV.json'):
    """
    Write a copy of the config's paras JSON with a symmetric, data-scaled amplitude box.

    Inputs : config        PILA config dict (arch.args.insar must point at the scene actually trained on)
             out_dir       directory for the new paras JSON
             anchor_std_mm data std (mm) at which the box equals the standard box (12 mm)
    Output : (absolute paras JSON path, info dict with std_mm, box, param, family)
    The source paras file must already use the symmetric map (Mogi/Sun69: scale 1, shift 0, min = -max;
    Okada: min = -max) so that zero amplitude stays at z = 0.5 and only the width changes.
    """
    project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paras_key = _paras_key(config)
    param, family = PARAS_KEYS[paras_key]
    paras_src = _resolve(config['arch']['args'][paras_key], project_root)
    with open(paras_src) as handle:
        paras = json.load(handle)

    # --- Refuse anything but a symmetric map: the box must only change width, never move the zero ---
    entry = paras[param]
    if family in ('mogi', 'sun69') and (entry.get('scale', 1e5) != 1 or entry.get('shift', -1e7) != 0):
        raise ValueError(f"{paras_src}: {param} must use the symmetric map (scale 1, shift 0)")
    if entry['min'] != -entry['max']:
        raise ValueError(f"{paras_src}: {param} box must be symmetric (min = -max), got {entry['min']}, {entry['max']}")

    std_mm = scene_std_mm(config)
    info = {'family': family, 'param': param, 'std_mm': std_mm, 'anchor_std_mm': anchor_std_mm, 'paras_src': paras_src}
    if family == 'okada':
        with open(_resolve(mogi_reference, project_root)) as handle:
            mogi_paras = json.load(handle)
        opening_ref_m, peak_mogi_mm, peak_okada_per_m_mm = okada_opening_ref_m(config, paras, mogi_paras)
        box = opening_ref_m * std_mm / anchor_std_mm
        info.update(opening_ref_m=opening_ref_m, peak_mogi_ref_mm=peak_mogi_mm, peak_okada_per_m_mm=peak_okada_per_m_mm)
    else:
        box = STANDARD_DV_MAX_M3 * std_mm / anchor_std_mm
    entry['min'], entry['max'] = -box, box
    info['box'] = box

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(paras_src))[0]
    paras_path = os.path.abspath(os.path.join(out_dir, f"{stem}_ampbox.json"))
    with open(paras_path, 'w') as handle:
        json.dump(paras, handle, indent=4)
    return paras_path, info


def main():
    parser = argparse.ArgumentParser(description='Compute and write the data-scaled amplitude box for a config.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    with open(args.config) as handle:
        config = json.load(handle)
    paras_path, info = write_data_scaled_paras(config, args.out_dir)
    print(json.dumps({**info, 'paras_path': paras_path}, indent=2))


if __name__ == '__main__':
    main()
