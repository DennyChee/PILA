#!/usr/bin/env python
# Usage:       python train_multisource.py --config configs/multisource/clean_detr.json
#              (optional: --steps N to override, --resume path/to/model_last.pth)
# Description: Stage-1 SUPERVISED training of the multi-source Mogi network (model/multisource.py).
#              Training scenes are generated on the fly in batches by the torch generator
#              (noise-free, same prior as the Stage-1 data dir); validation uses the fixed
#              NumPy-generated val.npz. Logs per-eval metrics to CSV and saves checkpoints to
#              saved/multisource/<name>/<MMDD_HHMMSS>/.
# Metrics (validation, last decoder layer, Hungarian-matched for every model):
#   count_acc    fraction of scenes with #(P(exist) > 0.5) == true K
#   recall       fraction of true sources whose matched query has P(exist) > 0.5
#   recall_near  fraction of true sources with ANY 'on' query within 2 km (matching-independent)
#   dup_rate     fraction of K>=1 scenes with an extra 'on' query within 2 km of a true source
#                (slot collapse / duplicate detection)
#   err_xy_km, err_depth_km, err_log10dV   median absolute errors over recalled sources
#   sign_acc     inflation/deflation accuracy over recalled sources
# Date:        2026-10-05

import argparse
import csv
import json
import math
import os
import shutil
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.multisource import MultiSourceNet, SetCriterion, TargetCodec
from synthetic.multisource.dataset import FixedSceneDataset, load_stage1_meta
from synthetic.multisource.torch_generator import TorchSceneGenerator


def lr_at(step, cfg):
    """Linear warm-up then cosine decay to 5% of the peak learning rate."""
    warm, total, peak = cfg['warmup_steps'], cfg['steps'], cfg['lr']
    if step < warm:
        return peak * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return peak * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, criterion, loader, codec, step):
    """Validation loss and metrics (see header). Always uses Hungarian matching."""
    model.eval()
    hungarian = SetCriterion(codec, criterion.perms.max().item() + 1, criterion.perms.shape[1],
                             assignment='hungarian')
    n_scenes = n_count_ok = n_true = n_recalled = n_dup_scenes = n_k_pos = n_near = 0
    loss_sum, n_batches = 0.0, 0
    err_xy, err_depth, err_logdv, sign_ok = [], [], [], []
    for batch in loader:
        outputs, _ = model(batch['x'])
        loss, _ = criterion(outputs, batch['params'], batch['exist'], step)
        loss_sum += loss.item()
        n_batches += 1
        pred = outputs[-1]
        targets, sign = codec.encode(batch['params'])
        exist = batch['exist']
        assigned = hungarian.match(pred, targets, sign, exist)          # [B,K]
        p_exist = pred['exist_logit'].sigmoid()
        on = p_exist > 0.5
        n_scenes += len(exist)
        n_count_ok += int((on.sum(1) == batch['k']).sum())

        phys_pred = codec.decode(pred['mean'], pred['sign_logit'].sigmoid())   # [B,Q,4]
        for b in range(len(exist)):
            k = int(batch['k'][b])
            matched_on = set()
            for j in range(k):
                q = int(assigned[b, j])
                n_true += 1
                if on[b, q]:
                    n_recalled += 1
                    matched_on.add(q)
                    true_p = batch['params'][b, j]
                    err_xy.append(float(torch.linalg.norm(phys_pred[b, q, :2] - true_p[:2])))
                    err_depth.append(float((phys_pred[b, q, 2] - true_p[2]).abs()))
                    err_logdv.append(float((torch.log10(phys_pred[b, q, 3].abs())
                                            - torch.log10(true_p[3].abs())).abs()))
                    sign_ok.append(bool((phys_pred[b, q, 3] > 0) == (true_p[3] > 0)))
            for j in range(k):                       # matching-independent recall
                on_xy = phys_pred[b, on[b], :2]
                if on_xy.shape[0] and float(torch.linalg.norm(on_xy - batch['params'][b, j, :2], dim=-1).min()) < 2.0:
                    n_near += 1
            if k >= 1:
                n_k_pos += 1
                extra_on = [q for q in range(on.shape[1]) if on[b, q] and q not in matched_on]
                true_xy = batch['params'][b, :k, :2]
                dup = any(float(torch.linalg.norm(true_xy - phys_pred[b, q, :2], dim=-1).min()) < 2.0
                          for q in extra_on)
                n_dup_scenes += int(dup)
    model.train()
    med = lambda v: float(np.median(v)) if v else float('nan')   # noqa: E731
    return {'val_loss': loss_sum / n_batches, 'count_acc': n_count_ok / n_scenes,
            'recall': n_recalled / max(1, n_true), 'recall_near': n_near / max(1, n_true),
            'dup_rate': n_dup_scenes / max(1, n_k_pos),
            'err_xy_km': med(err_xy), 'err_depth_km': med(err_depth),
            'err_log10dV': med(err_logdv), 'sign_acc': float(np.mean(sign_ok)) if sign_ok else float('nan')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--steps', type=int, default=None, help='override train.steps (e.g. smoke test)')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(args.config)
    with open(args.config) as f:
        cfg = json.load(f)
    tcfg = cfg['train']
    if args.steps is not None:
        tcfg['steps'] = args.steps
    torch.set_num_threads(int(tcfg.get('threads', os.cpu_count() or 1)))
    torch.manual_seed(tcfg['seed'])

    run_dir = os.path.join(cfg['out_root'], cfg['name'], datetime.now().strftime('%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=False)
    shutil.copy(args.config, os.path.join(run_dir, 'config.json'))
    print(f"[1/4] Run dir: {run_dir}  (threads={torch.get_num_threads()})")

    print(f"[2/4] Data: on-the-fly torch generator + fixed val from {cfg['data_dir']}...")
    meta = load_stage1_meta(cfg['data_dir'])
    generator = TorchSceneGenerator(meta)
    val_ds = FixedSceneDataset(os.path.join(cfg['data_dir'], 'val.npz'), meta)
    n_val = min(len(val_ds), tcfg.get('n_val', len(val_ds)))
    val_loader = DataLoader(torch.utils.data.Subset(val_ds, range(n_val)), batch_size=250)
    print(f"  val scenes used: {n_val}")

    print("[3/4] Building model and criterion...")
    codec = TargetCodec(**cfg.get('codec', {}))
    model = MultiSourceNet(**cfg['model'])
    criterion = SetCriterion(codec, cfg['model']['n_queries'], meta['prior']['k_max'], **cfg['criterion'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg['lr'], weight_decay=tcfg['weight_decay'])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params / 1e6:.2f} M; assignment={criterion.assignment}; "
          f"depletion={cfg['model'].get('depletion', False)}")
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step'] + 1
        print(f"  resumed from {args.resume} at step {start_step}")

    print(f"[4/4] Training {tcfg['steps']} steps x batch {tcfg['batch_size']}...")
    log_path = os.path.join(run_dir, 'log.csv')
    log_fields = ['step', 'lr', 'train_loss', 'exist', 'param', 'nll', 'sign', 'sec_per_step', 'val_loss',
                  'count_acc', 'recall', 'recall_near', 'dup_rate', 'err_xy_km', 'err_depth_km',
                  'err_log10dV', 'sign_acc']
    with open(log_path, 'a', newline='') as f:
        csv.DictWriter(f, log_fields).writeheader()
    # 'best' = highest validation count accuracy, ties broken by recall (val_loss is not used:
    # in the NLL phase it mostly rewards a sharp covariance, not finding the right sources).
    best_score = ckpt.get('best_score', (-1.0, -1.0)) if args.resume else (-1.0, -1.0)
    t_last, running = time.time(), []
    for step in range(start_step, tcfg['steps']):
        for group in optimizer.param_groups:
            group['lr'] = lr_at(step, tcfg)
        batch = generator.sample(tcfg['batch_size'], seed=tcfg['seed'] * 10_000_000 + step)
        outputs, _ = model(batch['x'])
        loss, parts = criterion(outputs, batch['params'], batch['exist'], step)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg['grad_clip'])
        optimizer.step()
        running.append([loss.item(), parts['exist'], parts['param'], parts['nll'], parts['sign']])

        if (step + 1) % tcfg['eval_every'] == 0 or step + 1 == tcfg['steps']:
            sec_per_step = (time.time() - t_last) / len(running)
            mean_run = np.mean(running, axis=0)
            metrics = evaluate(model, criterion, val_loader, codec, step)
            row = {'step': step + 1, 'lr': lr_at(step, tcfg), 'train_loss': mean_run[0],
                   'exist': mean_run[1], 'param': mean_run[2], 'nll': mean_run[3], 'sign': mean_run[4],
                   'sec_per_step': sec_per_step, **metrics}
            with open(log_path, 'a', newline='') as f:
                csv.DictWriter(f, log_fields).writerow(row)
            print(f"  step {step + 1:6d} | loss {mean_run[0]:.3f} | val {metrics['val_loss']:.3f} | "
                  f"count {metrics['count_acc']:.3f} recall {metrics['recall']:.3f}/{metrics['recall_near']:.3f} "
                  f"dup {metrics['dup_rate']:.3f} | "
                  f"xy {metrics['err_xy_km']:.2f} km d {metrics['err_depth_km']:.2f} km "
                  f"logdV {metrics['err_log10dV']:.3f} sign {metrics['sign_acc']:.3f} | {sec_per_step:.2f} s/step")
            score = (metrics['count_acc'], metrics['recall'])
            if score > best_score:
                best_score = score
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                     'config': cfg, 'metrics': metrics, 'best_score': best_score}
            torch.save(state, os.path.join(run_dir, 'model_last.pth'))
            if score == best_score:
                torch.save(state, os.path.join(run_dir, 'model_best.pth'))
            t_last, running = time.time(), []
    print(f"Done. Output saved to {run_dir}/")


if __name__ == '__main__':
    main()
