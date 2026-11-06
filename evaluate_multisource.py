#!/usr/bin/env python
# Usage:       python evaluate_multisource.py --run saved/multisource/clean_detr/<MMDD_HHMMSS> \
#                  [--ckpt model_best.pth] [--split test] [--out <run>/eval_test]
# Description: Step-4 evaluation of a Stage-1 multi-source model on the FIXED test (or val)
#              set of its data dir. Writes metrics JSON and figures:
#                fig_count_confusion      true K vs predicted K (queries with P(exist) > 0.5)
#                fig_existence_calibration  reliability of P(exist): predicted vs observed rate
#                fig_param_recovery       predicted vs true x, y, depth, dV for found sources
#                fig_uncertainty_zscores  (truth - mean)/sigma per parameter vs N(0,1)
#                fig_examples             input | predicted Mogi sum | residual, with true
#                                         (black) and predicted (coloured by P) sources
# Scientific notes:
#   * Predicted sources are matched to true sources with the same Hungarian cost used in
#     training; "found" = matched query with P(exist) > 0.5.
#   * The predicted field re-renders found sources with the torch Mogi model (same LOS
#     geometry as the data), so the residual shows what the network did not explain.
#   * Units: x, y, depth in km; dV in m^3; LOS in mm (positive toward satellite).
# Date:        2026-10-05

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from model.multisource import MultiSourceNet, SetCriterion, TargetCodec
from synthetic.multisource.dataset import FixedSceneDataset, load_stage1_meta
from synthetic.multisource.torch_generator import TorchSceneGenerator, mogi_los_mm_torch

PARAM_LABELS = ['East (km)', 'North (km)', 'Depth (km)', 'log10|dV| (m³)']


@torch.no_grad()
def run_inference(model, loader, codec, n_queries, k_max):
    """Predict on every scene; return per-scene arrays and matched (true, pred) pairs."""
    matcher = SetCriterion(codec, n_queries, k_max, assignment='hungarian')
    keep = {'p_exist': [], 'pred_phys': [], 'mean': [], 'std': [], 'assigned': [], 'k': [],
            'params': [], 'exist': [], 'peak_ratio': [], 'x': []}
    for batch in loader:
        outputs, _ = model(batch['x'])
        pred = outputs[-1]
        targets, sign = codec.encode(batch['params'])
        keep['assigned'].append(matcher.match(pred, targets, sign, batch['exist']))
        keep['p_exist'].append(pred['exist_logit'].sigmoid())
        keep['pred_phys'].append(codec.decode(pred['mean'], pred['sign_logit'].sigmoid()))
        keep['mean'].append(pred['mean'])
        cov = pred['scale_tril'] @ pred['scale_tril'].transpose(-1, -2)
        keep['std'].append(torch.diagonal(cov, dim1=-2, dim2=-1).sqrt())   # marginal sigma
        for key in ['k', 'params', 'exist', 'x']:
            keep[key].append(batch[key])
        keep['peak_ratio'].append(batch['extra'][..., 0])
    return {key: torch.cat(val).numpy() for key, val in keep.items()}


def collect_pairs(res, codec):
    """Matched pairs for found sources plus per-query existence labels."""
    n_scenes, n_queries = res['p_exist'].shape
    label = np.zeros((n_scenes, n_queries))
    found_true, found_pred, z_scores, peak = [], [], [], []
    targets_norm, _ = codec.encode(torch.from_numpy(res['params']))
    targets_norm = targets_norm.numpy()
    for b in range(n_scenes):
        for j in range(int(res['k'][b])):
            q = int(res['assigned'][b, j])
            label[b, q] = 1.0
            if res['p_exist'][b, q] > 0.5:
                found_true.append(res['params'][b, j])
                found_pred.append(res['pred_phys'][b, q])
                z_scores.append((targets_norm[b, j] - res['mean'][b, q]) / res['std'][b, q])
                peak.append(res['peak_ratio'][b, j])
    return label, np.array(found_true), np.array(found_pred), np.array(z_scores), np.array(peak)


def fig_confusion(res, k_max, out_dir):
    pred_k = (res['p_exist'] > 0.5).sum(1)
    n_q = res['p_exist'].shape[1]
    mat = np.zeros((k_max + 1, n_q + 1), int)
    for true_k, pk in zip(res['k'], pred_k):
        mat[int(true_k), int(pk)] += 1
    frac = mat / mat.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(frac, cmap='viridis', vmin=0, vmax=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f'{mat[i, j]}\n{frac[i, j]:.0%}', ha='center', va='center',
                    color='w' if frac[i, j] < 0.6 else 'k', fontsize=9)
    ax.set_xlabel('Predicted K (queries with P(exist) > 0.5)'); ax.set_ylabel('True K')
    ax.set_xticks(range(n_q + 1)); ax.set_yticks(range(k_max + 1))
    ax.set_title(f'Count confusion (accuracy {np.mean(pred_k == res["k"]):.1%})')
    cb = fig.colorbar(im, ax=ax); cb.set_label('Fraction of scenes with this true K')
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig_count_confusion.png'), dpi=150); plt.close(fig)
    return mat


def fig_calibration(res, label, out_dir):
    p = res['p_exist'].ravel(); y = label.ravel()
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
    centres, rates, counts = [], [], []
    for i in range(10):
        sel = idx == i
        if sel.sum():
            centres.append(p[sel].mean()); rates.append(y[sel].mean()); counts.append(sel.sum())
    ece = sum(c * abs(r - m) for c, r, m in zip(counts, rates, centres)) / len(p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot([0, 1], [0, 1], 'k--', label='perfect calibration')
    axes[0].plot(centres, rates, 'o-', label=f'model (ECE {ece:.3f})')
    axes[0].set_xlabel('Predicted P(exist)'); axes[0].set_ylabel('Observed fraction that are real sources')
    axes[0].set_title('Existence calibration (all queries)'); axes[0].legend()
    axes[1].hist([p[y == 1], p[y == 0]], bins=edges, label=['matched to a true source', 'unmatched'],
                 stacked=True)
    axes[1].set_yscale('log'); axes[1].set_xlabel('Predicted P(exist)'); axes[1].set_ylabel('Queries (log)')
    axes[1].set_title('P(exist) distribution'); axes[1].legend()
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig_existence_calibration.png'), dpi=150); plt.close(fig)
    return float(ece)


def fig_recovery(found_true, found_pred, peak, out_dir):
    fig, axes = plt.subplots(1, 5, figsize=(25, 4.8))
    true_cols = [found_true[:, 0], found_true[:, 1], found_true[:, 2], np.log10(np.abs(found_true[:, 3]))]
    pred_cols = [found_pred[:, 0], found_pred[:, 1], found_pred[:, 2], np.log10(np.abs(found_pred[:, 3]))]
    for ax, t, p, lab in zip(axes[:4], true_cols, pred_cols, PARAM_LABELS):
        sc = ax.scatter(t, p, c=np.log10(peak), s=4, cmap='viridis')
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lims, lims, 'k--', lw=1)
        ax.set_xlabel(f'True {lab}'); ax.set_ylabel(f'Predicted {lab}')
        ax.set_title(f'{lab}: median |err| {np.median(np.abs(p - t)):.3f}')
    cb = fig.colorbar(sc, ax=axes[3]); cb.set_label('log10 peak ratio')
    sign_ok = np.sign(found_true[:, 3]) == np.sign(found_pred[:, 3])
    loc_err = np.linalg.norm(found_pred[:, :2] - found_true[:, :2], axis=1)
    axes[4].scatter(peak, loc_err, c=found_true[:, 2], s=4, cmap='viridis')
    axes[4].set_xscale('log'); axes[4].set_yscale('log')
    axes[4].set_xlabel('Peak ratio max|LOS| / σ_ref'); axes[4].set_ylabel('Horizontal location error (km)')
    axes[4].set_title(f'Location error vs amplitude (sign accuracy {sign_ok.mean():.1%})')
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig_param_recovery.png'), dpi=150); plt.close(fig)


def fig_zscores(z_scores, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    grid = np.linspace(-4, 4, 200)
    coverage = {}
    for i, (ax, lab) in enumerate(zip(axes, PARAM_LABELS)):
        z = np.clip(z_scores[:, i], -6, 6)
        ax.hist(z, bins=60, range=(-6, 6), density=True, alpha=0.7, label='model')
        ax.plot(grid, np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), 'k--', label='N(0,1) ideal')
        cov1, cov2 = np.mean(np.abs(z_scores[:, i]) < 1), np.mean(np.abs(z_scores[:, i]) < 2)
        coverage[lab] = {'within_1sigma': float(cov1), 'within_2sigma': float(cov2)}
        ax.set_title(f'{lab}\n|z|<1: {cov1:.0%} (ideal 68%), |z|<2: {cov2:.0%} (95%)')
        ax.set_xlabel('z = (truth − mean) / σ (normalised units)'); ax.set_ylabel('Density'); ax.legend()
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig_uncertainty_zscores.png'), dpi=150); plt.close(fig)
    return coverage


def fig_examples(res, meta, out_dir, n_per_k=2):
    gen = TorchSceneGenerator(meta)
    scale_mm = meta['input_scaler']['std_mm']
    k_max = meta['prior']['k_max']
    rows = [(k, b) for k in range(k_max + 1) for b in np.where(res['k'] == k)[0][:n_per_k]]
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4.2 * len(rows)), squeeze=False)
    extent = [-20, 20, -20, 20]
    for r, (k, b) in enumerate(rows):
        obs_mm = res['x'][b, 0] * scale_mm
        on = np.where(res['p_exist'][b] > 0.5)[0]
        pred = torch.from_numpy(res['pred_phys'][b, on]).double()
        if len(on):
            fields = mogi_los_mm_torch(gen.xE_m.double(), gen.yN_m.double(), gen.los_enu.double(),
                                       pred[None, :, 0] * 1e3, pred[None, :, 1] * 1e3,
                                       pred[None, :, 2] * 1e3, pred[None, :, 3])
            pred_mm = fields.sum(1)[0].view(*gen.shape).numpy()
        else:
            pred_mm = np.zeros_like(obs_mm)
        vmax = max(np.abs(obs_mm).max(), np.abs(pred_mm).max(), 1e-3)   # K=0: input is all zero
        for c, (panel, title) in enumerate([(obs_mm, 'Input (true sources black)'),
                                            (pred_mm, 'Predicted Mogi sum (P>0.5 queries)'),
                                            (obs_mm - pred_mm, 'Residual = input − predicted')]):
            ax = axes[r, c]
            im = ax.imshow(panel, cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent, origin='upper')
            cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.set_label('LOS (mm)')
            for j in range(k):
                tp = res['params'][b, j]
                ax.plot(tp[0], tp[1], '^' if tp[3] > 0 else 'v', mfc='none', mec='k', ms=10, mew=1.5)
            for q in range(res['p_exist'].shape[1]):
                pp, prob = res['pred_phys'][b, q], res['p_exist'][b, q]
                if prob > 0.05:
                    ax.plot(pp[0], pp[1], 'o', mfc='none', mec=plt.cm.autumn(1 - prob), ms=14, mew=2)
                    if c == 0:
                        ax.text(pp[0] + 1.2, pp[1] + 1.2, f'q{q}: {prob:.2f}', fontsize=8,
                                color=plt.cm.autumn(1 - prob))
            ax.set_title(f'{title}\ntrue K={k}, predicted K={len(on)} (scene {b})' if c == 0 else title,
                         fontsize=9)
            ax.set_xlabel('East (km)'); ax.set_ylabel('North (km)')
    fig.suptitle('Example test scenes: ▲▼ true sources (black); circles = queries with P(exist) > 0.05, '
                 'red = confident, yellow = unsure', y=1.0)
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig_examples.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='run dir with config.json and checkpoints')
    parser.add_argument('--ckpt', default='model_best.pth')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    ckpt_path = os.path.join(args.run, args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    out_dir = args.out or os.path.join(args.run, f'eval_{args.split}')
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/4] Loading checkpoint {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['config']
    model = MultiSourceNet(**cfg['model'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    codec = TargetCodec(**cfg.get('codec', {}))
    meta = load_stage1_meta(cfg['data_dir'])
    print(f"  step {ckpt['step'] + 1}, training-time val metrics: {ckpt.get('metrics', {})}")

    print(f"[2/4] Inference on the fixed {args.split} set...")
    ds = FixedSceneDataset(os.path.join(cfg['data_dir'], f'{args.split}.npz'), meta)
    res = run_inference(model, DataLoader(ds, batch_size=250), codec,
                        cfg['model']['n_queries'], meta['prior']['k_max'])
    print(f"  {len(res['k'])} scenes")

    print("[3/4] Metrics and figures...")
    label, found_true, found_pred, z_scores, peak = collect_pairs(res, codec)
    mat = fig_confusion(res, meta['prior']['k_max'], out_dir)
    ece = fig_calibration(res, label, out_dir)
    fig_recovery(found_true, found_pred, peak, out_dir)
    coverage = fig_zscores(z_scores, out_dir)
    fig_examples(res, meta, out_dir)

    pred_k = (res['p_exist'] > 0.5).sum(1)
    loc_err = np.linalg.norm(found_pred[:, :2] - found_true[:, :2], axis=1)
    metrics = {
        'checkpoint': ckpt_path, 'split': args.split, 'n_scenes': int(len(res['k'])),
        'count_accuracy': float(np.mean(pred_k == res['k'])),
        'count_accuracy_by_true_k': {int(k): float(np.mean(pred_k[res['k'] == k] == k))
                                     for k in np.unique(res['k'])},
        'recall': float(len(found_true) / max(1, res['k'].sum())),
        'existence_ece': ece,
        'median_loc_err_km': float(np.median(loc_err)),
        'p90_loc_err_km': float(np.percentile(loc_err, 90)),
        'median_depth_err_km': float(np.median(np.abs(found_pred[:, 2] - found_true[:, 2]))),
        'median_log10dV_err': float(np.median(np.abs(np.log10(np.abs(found_pred[:, 3]))
                                                     - np.log10(np.abs(found_true[:, 3]))))),
        'sign_accuracy': float(np.mean(np.sign(found_pred[:, 3]) == np.sign(found_true[:, 3]))),
        'uncertainty_coverage': coverage, 'confusion_matrix': mat.tolist(),
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("[4/4] Summary:")
    for key in ['count_accuracy', 'recall', 'existence_ece', 'median_loc_err_km', 'p90_loc_err_km',
                'median_depth_err_km', 'median_log10dV_err', 'sign_accuracy']:
        print(f"  {key:22s} {metrics[key]:.4f}")
    print(f"Done. Output saved to {out_dir}/")


if __name__ == '__main__':
    main()
