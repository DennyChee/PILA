#!/usr/bin/env python
# Usage:       from methods.vbica.grouping import load_ic_fields, group_ics, ic_diagnostics
# Description: Which vbICA independent components (ICs) belong to which deforming source. A moving source spreads
#              over several ICs (each IC has ONE fixed map), so per-source signals are rebuilt as SUMS of ICs
#              (exact: S is diagonal, no cross terms) before the per-epoch inversion.
#              The legacy bench chose groups BY HAND from decision sheets (blind or truth-informed); the fresh
#              comparison needs an AUTOMATIC, TRUTH-FREE rule, chosen in Phase 4. This module provides the interface
#              and candidate rules; no rule here reads the truth.
# Rules:
#   'explicit'          groups given by the caller (tests, or a rule decided elsewhere)
#   'each'              every kept IC is its own source (the single-IC route, generalised)
#   'overlap_cluster'   CANDIDATE (Phase 4 to evaluate): (1) drop noise-like ICs whose temporal course is white
#                       (lag-1 autocorrelation < min_lag1) or whose share of the total field energy < min_energy;
#                       (2) agglomerate the rest by spatial-footprint overlap (cosine similarity of |map|) until
#                       n_groups remain (oracle K in the main track). Thresholds are ARGUMENTS, not tuned on truth.
# Conventions: IC field k = u_k(t) v_k^T exactly (rank 1); u, v from its SVD; fields in metres.
# Date:        2026-11-23
import numpy as np
from scipy.io import loadmat


def load_ic_fields(path):
    """Read ic_fields_N<NN>.mat (export_ic_fields.m) into a dict of numpy arrays."""
    data = loadmat(path, squeeze_me=False)
    out = {key: np.asarray(data[key]) for key in ('ic_fields', 'llh', 'mu_epoch', 'obs')}
    out['ic_fields'] = out['ic_fields'].astype(np.float64)                      # [N, T, M] m
    out['mu_epoch'] = out['mu_epoch'].ravel().astype(np.float64)
    for key in ('nfe', 'steps', 'converged', 'max_rel'):
        out[key] = float(np.asarray(data[key]).ravel()[0]) if np.asarray(data[key]).size else float('nan')
    out['ard'] = np.asarray(data['ard']).ravel()
    return out


def ic_diagnostics(ic_fields):
    """
    Truth-free per-IC diagnostics.

    Output: dict with temporal [N, T], spatial [N, M] (unit norm), energy_frac [N], lag1 [N].
    """
    n_comp = ic_fields.shape[0]
    temporal, spatial, energy = [], [], []
    for k in range(n_comp):
        u, s, vt = np.linalg.svd(ic_fields[k], full_matrices=False)
        temporal.append(u[:, 0] * s[0])
        spatial.append(vt[0])
        energy.append(float(np.sum(ic_fields[k] ** 2)))
    temporal, spatial = np.array(temporal), np.array(spatial)
    energy = np.array(energy) / max(np.sum(energy), 1e-300)
    lag1 = np.array([np.corrcoef(series[:-1], series[1:])[0, 1] if np.std(series) > 0 else 0.0 for series in temporal])
    return dict(temporal=temporal, spatial=spatial, energy_frac=energy, lag1=lag1)


def group_ics(ic_fields, rule, n_groups=None, groups=None, min_lag1=0.3, min_energy=0.02):
    """
    Assign ICs to sources.

    Inputs:  ic_fields [N, T, M]; rule ('explicit'|'each'|'overlap_cluster'); n_groups (oracle K for
             'overlap_cluster'); groups (list of lists, 'explicit'); thresholds for 'overlap_cluster'.
    Output:  (list of groups = lists of 0-based IC indices, dict with the decision record).
    """
    n_comp = ic_fields.shape[0]
    diag = ic_diagnostics(ic_fields)
    record = dict(rule=rule, energy_frac=diag['energy_frac'].tolist(), lag1=diag['lag1'].tolist())
    if rule == 'explicit':
        if not groups:
            raise ValueError("rule 'explicit' needs groups")
        flat = [k for group in groups for k in group]
        if len(set(flat)) != len(flat) or not all(0 <= k < n_comp for k in flat):
            raise ValueError(f"invalid explicit groups {groups} for N={n_comp}")
        return [list(group) for group in groups], record
    kept = [k for k in range(n_comp) if diag['lag1'][k] >= min_lag1 and diag['energy_frac'][k] >= min_energy]
    record.update(kept=kept, min_lag1=min_lag1, min_energy=min_energy)
    if rule == 'each':
        return [[k] for k in kept], record
    if rule != 'overlap_cluster':
        raise ValueError(f"unknown rule {rule!r}")
    if n_groups is None or n_groups < 1:
        raise ValueError("'overlap_cluster' needs n_groups (oracle K)")
    clusters = [[k] for k in kept]
    footprint = np.abs(diag['spatial'])
    footprint /= np.maximum(np.linalg.norm(footprint, axis=1, keepdims=True), 1e-300)
    merges = []
    while len(clusters) > n_groups:
        best, pair = -np.inf, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                fa = footprint[clusters[a]].sum(axis=0)
                fb = footprint[clusters[b]].sum(axis=0)
                similarity = float(fa @ fb / max(np.linalg.norm(fa) * np.linalg.norm(fb), 1e-300))
                if similarity > best:
                    best, pair = similarity, (a, b)
        a, b = pair
        merges.append(dict(merged=[clusters[a], clusters[b]], similarity=best))
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    record.update(merges=merges, n_groups=n_groups, short_of_groups=len(clusters) < n_groups)
    return clusters, record
