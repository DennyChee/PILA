#!/usr/bin/env python
# Usage:       from model.multisource import MultiSourceNet, SetCriterion, TargetCodec
# Description: Stage-1 SUPERVISED multi-source Mogi inversion network ("DETR-lite").
#              A CNN encoder turns the LOS map into a 16x16 grid of feature tokens, a small
#              transformer adds global context, and Q learned "detective" queries read the
#              tokens through cross-attention. Each query outputs P(source exists), P(inflation),
#              and a full-covariance Gaussian over normalised (x, y, depth, log10|dV|).
#              Optional Houba (2026) evidence depletion on the cross-attention (config switch):
#              queries are processed one after another and tokens a query attended to are
#              down-weighted for later queries. SetCriterion matches queries to true sources
#              (exhaustive Hungarian over all permutations, or fixed order for the Step-0
#              baseline) and computes existence BCE + sign BCE + parameter loss, with the same
#              loss applied after every decoder layer (DETR auxiliary losses).
# Scientific notes:
#   * Targets are encoded as x/15, y/15 (km), (depth-5.5)/4.5 (km), (log10|dV| - c)/s with
#     the sign of dV predicted separately (a signed log would be bimodal with a gap).
#   * Full 4x4 covariance (Cholesky) so the depth-dV Mogi trade-off is represented.
#   * Empty target slots are never multiplied by a mask: targets are NaN->0 and matched
#     pairs are selected by INDEXING, so NaN cannot reach the gradients.
#   * Standalone: does not import or modify PILA's single-source model code.
# Date:        2026-10-05

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_PARAMS = 4                                  # x, y, depth, log10|dV| (normalised)
N_TRIL = N_PARAMS * (N_PARAMS + 1) // 2       # Cholesky entries


# --- Target encoding ---------------------------------------------------------
class TargetCodec:
    """Physical params (km, km, km, m^3) <-> normalised network targets + sign."""

    def __init__(self, xy_half_km=15.0, depth_mid_km=5.5, depth_half_km=4.5,
                 logdv_mid=6.5, logdv_half=1.5):
        self.xy_half_km = xy_half_km
        self.depth_mid_km, self.depth_half_km = depth_mid_km, depth_half_km
        self.logdv_mid, self.logdv_half = logdv_mid, logdv_half

    def encode(self, params):
        """params [..., 4] physical (NaN allowed) -> (targets [..., 4], sign [...] in {0,1})."""
        params = torch.nan_to_num(params, nan=0.0)
        abs_dV = params[..., 3].abs().clamp_min(1.0)               # avoid log10(0) in empty slots
        targets = torch.stack([params[..., 0] / self.xy_half_km,
                               params[..., 1] / self.xy_half_km,
                               (params[..., 2] - self.depth_mid_km) / self.depth_half_km,
                               (torch.log10(abs_dV) - self.logdv_mid) / self.logdv_half], dim=-1)
        sign = (params[..., 3] > 0).float()
        return targets, sign

    def decode(self, mean, sign_prob):
        """Normalised mean [..., 4] + P(inflation) [...] -> physical params [..., 4]."""
        sign = torch.where(sign_prob >= 0.5, 1.0, -1.0)
        return torch.stack([mean[..., 0] * self.xy_half_km,
                            mean[..., 1] * self.xy_half_km,
                            mean[..., 2] * self.depth_half_km + self.depth_mid_km,
                            sign * 10 ** (mean[..., 3] * self.logdv_half + self.logdv_mid)], dim=-1)


# --- Encoder -----------------------------------------------------------------
def conv_block(in_ch, out_ch, stride):
    """PILA's conv pattern (3x3 conv + BatchNorm + LeakyReLU)."""
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
                         nn.BatchNorm2d(out_ch), nn.LeakyReLU())


def sincos_2d(height, width, dim):
    """Fixed 2-D sin/cos positional encoding [height*width, dim] (half for rows, half cols)."""
    quarter = dim // 4
    freqs = torch.exp(-math.log(10000.0) * torch.arange(quarter) / quarter)
    rows = torch.arange(height)[:, None] * freqs[None]          # [H, quarter]
    cols = torch.arange(width)[:, None] * freqs[None]
    row_enc = torch.cat([rows.sin(), rows.cos()], dim=1)        # [H, dim/2]
    col_enc = torch.cat([cols.sin(), cols.cos()], dim=1)        # [W, dim/2]
    return torch.cat([row_enc[:, None].expand(height, width, -1),
                      col_enc[None].expand(height, width, -1)], dim=-1).reshape(height * width, dim)


class Encoder(nn.Module):
    """
    [B,1,H,W] standardized LOS -> [B, T, d_model] tokens (T = (H/8)*(W/8)).

    Input channels: raw x, asinh(x) (compresses large peaks), and normalised East/North
    coordinates (CoordConv) so absolute location is easy to regress. Two stride-1 convs at
    full resolution before any downsampling keep 2-4 px shallow sources from aliasing.
    """

    def __init__(self, d_model=256, stem_ch=32, stage_ch=(64, 128, 256)):
        super().__init__()
        layers = [conv_block(4, stem_ch, 1), conv_block(stem_ch, stem_ch, 1)]
        in_ch = stem_ch
        for out_ch in stage_ch:                                     # 3 stride-2 stages -> /8
            layers += [conv_block(in_ch, out_ch, 2), conv_block(out_ch, out_ch, 1)]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Linear(in_ch, d_model)
        self.d_model = d_model

    def forward(self, x):
        batch, _, height, width = x.shape
        ys = torch.linspace(1, -1, height, device=x.device)        # row 0 = north = +1
        xs = torch.linspace(-1, 1, width, device=x.device)
        coord_y = ys[:, None].expand(height, width)
        coord_x = xs[None, :].expand(height, width)
        coords = torch.stack([coord_x, coord_y])[None].expand(batch, -1, -1, -1)
        h = self.cnn(torch.cat([x, torch.asinh(x), coords], dim=1))   # [B, C, H/8, W/8]
        grid_h, grid_w = h.shape[-2:]
        tokens = self.proj(h.flatten(2).transpose(1, 2))             # [B, T, d]
        pos = sincos_2d(grid_h, grid_w, self.d_model).to(x.device, x.dtype)
        return tokens, pos


# --- Decoder -----------------------------------------------------------------
class CrossAttention(nn.Module):
    """Multi-head cross-attention (queries -> tokens) with optional evidence depletion."""

    def __init__(self, d_model, nhead, depletion=False, gamma=5.0, eps=1e-4):
        super().__init__()
        self.nhead, self.head_dim = nhead, d_model // nhead
        self.w_q, self.w_k = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.w_v, self.w_o = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.depletion, self.gamma, self.eps = depletion, gamma, eps

    def _split(self, t):
        return t.view(t.shape[0], t.shape[1], self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, query, query_pos, memory, memory_pos, order=None):
        """query [B,Q,d], memory [B,T,d] -> (out [B,Q,d], attn [B,Q,T] head-averaged)."""
        batch, n_query, d_model = query.shape
        q = self._split(self.w_q(query + query_pos))                 # [B,h,Q,hd]
        if not self.depletion:
            k = self._split(self.w_k(memory + memory_pos))
            v = self._split(self.w_v(memory))
            logits = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)   # [B,h,Q,T]
            attn = logits.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(batch, n_query, d_model)
            return self.w_o(out), attn.mean(dim=1)

        # Evidence depletion (Houba 2026): one query at a time; evidence e_l in [eps, 1]
        # scales tokens and biases logits by gamma*log(e_l); after query s attends with
        # alpha_s, e_l <- max(e_l * (1 - alpha_s,l), eps).
        evidence = torch.ones(batch, memory.shape[1], device=memory.device, dtype=memory.dtype)
        outs = [None] * n_query
        attns = [None] * n_query
        for s in (order if order is not None else range(n_query)):
            scaled = evidence[..., None] * memory
            k = self._split(self.w_k(scaled + memory_pos))
            v = self._split(self.w_v(scaled))
            logits = q[:, :, s:s + 1] @ k.transpose(-1, -2) / math.sqrt(self.head_dim)  # [B,h,1,T]
            logits = logits + self.gamma * torch.log(evidence)[:, None, None, :]
            attn = logits.softmax(dim=-1)
            outs[s] = (attn @ v).transpose(1, 2).reshape(batch, 1, d_model)
            attns[s] = attn.mean(dim=1)                              # [B,1,T]
            evidence = (evidence * (1 - attns[s][:, 0])).clamp_min(self.eps)
        return self.w_o(torch.cat(outs, dim=1)), torch.cat(attns, dim=1)


class DecoderLayer(nn.Module):
    """Pre-norm DETR decoder layer: query self-attention, cross-attention, feed-forward."""

    def __init__(self, d_model, nhead, ff_dim, depletion, gamma):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = CrossAttention(d_model, nhead, depletion, gamma)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model))
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d_model) for _ in range(3))

    def forward(self, tgt, query_pos, memory, memory_pos, order=None):
        h = self.norm1(tgt)
        tgt = tgt + self.self_attn(h + query_pos, h + query_pos, h)[0]   # queries talk: avoid duplicates
        cross_out, attn = self.cross_attn(self.norm2(tgt), query_pos, memory, memory_pos, order)
        tgt = tgt + cross_out
        tgt = tgt + self.ff(self.norm3(tgt))
        return tgt, attn


class MultiSourceNet(nn.Module):
    """CNN encoder + transformer + Q query decoder with per-layer prediction heads."""

    def __init__(self, n_queries=4, d_model=256, nhead=8, n_enc_layers=2, n_dec_layers=3,
                 ff_dim=512, stem_ch=32, stage_ch=(64, 128, 256), depletion=False, gamma=5.0):
        super().__init__()
        self.encoder = Encoder(d_model, stem_ch, tuple(stage_ch))
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff_dim, dropout=0.0,
                                               batch_first=True, norm_first=True)
        # Final LayerNorm: pre-norm layers leave the output un-normalised otherwise.
        self.context = nn.TransformerEncoder(enc_layer, n_enc_layers, norm=nn.LayerNorm(d_model))
        self.query_pos = nn.Parameter(torch.randn(n_queries, d_model) * 0.1)  # learned queries
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, ff_dim, depletion, gamma)
                                     for _ in range(n_dec_layers)])
        self.out_norm = nn.LayerNorm(d_model)

        def head(out_dim):
            return nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, out_dim))
        self.head_exist, self.head_sign = head(1), head(1)
        self.head_mean, self.head_tril = head(N_PARAMS), head(N_TRIL)
        self.n_queries, self.depletion = n_queries, depletion
        self.tril_idx = torch.tril_indices(N_PARAMS, N_PARAMS)

    def _predict(self, h):
        """Heads on decoder state [B,Q,d] -> dict of per-query predictions."""
        h = self.out_norm(h)
        raw = self.head_tril(h)                                       # [B,Q,10]
        batch, n_query = raw.shape[:2]
        tril = torch.zeros(batch, n_query, N_PARAMS, N_PARAMS, device=h.device, dtype=h.dtype)
        tril[..., self.tril_idx[0], self.tril_idx[1]] = raw
        diag = torch.diagonal(tril, dim1=-2, dim2=-1)
        # Diagonal std = 0.01 + softplus(raw): positive, smooth, never gradient-dead (a hard
        # clamp would freeze a std stuck at its bound). Floor 0.01 normalised units.
        tril = tril - torch.diag_embed(diag) + torch.diag_embed(0.01 + F.softplus(diag))
        return {'exist_logit': self.head_exist(h)[..., 0], 'sign_logit': self.head_sign(h)[..., 0],
                'mean': self.head_mean(h), 'scale_tril': tril}

    def forward(self, x):
        tokens, pos = self.encoder(x)
        memory = self.context(tokens + pos)
        batch = x.shape[0]
        query_pos = self.query_pos[None].expand(batch, -1, -1)
        tgt = torch.zeros_like(query_pos)
        # Random query order in training (Houba: avoids a fixed 'first query wins' bias).
        order = (torch.randperm(self.n_queries).tolist() if (self.training and self.depletion)
                 else None)
        outputs, attns = [], []
        for layer in self.layers:
            tgt, attn = layer(tgt, query_pos, memory, pos[None], order)
            outputs.append(self._predict(tgt))
            attns.append(attn)
        return outputs, attns


# --- Loss --------------------------------------------------------------------
class SetCriterion(nn.Module):
    """
    Match queries to true sources and compute the loss.

    assignment='hungarian': exhaustive search over all injective maps of the K_max true
        slots onto the Q queries (Q=4, K_max=3 -> 24 candidates) minimising a no-grad cost
        (-P(exist) + 5*L1(x,y) + 2*L1(depth, log|dV|) + |P(infl) - sign|); empty true slots
        contribute nothing, so K=0 means every query should say 'no source'.
    assignment='fixed': query j <-> true slot j (Step-0 baseline, slots ordered by amplitude).
    Parameter loss: L1 on the means ALWAYS, plus (after `nll_warmup_steps`) a full-covariance
    Gaussian NLL evaluated on DETACHED means, weighted 0.1 and ramped in linearly over
    `nll_ramp_steps`. Detaching stops the 1/sigma^2-scaled NLL gradient from swamping the
    existence/sign heads and the shared encoder; the NLL then trains only the covariance.
    Applied to every decoder layer's output (auxiliary losses) and averaged.
    """

    def __init__(self, codec, n_queries, k_max, assignment='hungarian', nll_warmup_steps=2000,
                 nll_ramp_steps=2000, nll_weight=0.1, w_exist=1.0, w_param=1.0, w_sign=0.5):
        super().__init__()
        self.codec, self.assignment = codec, assignment
        self.nll_warmup_steps, self.nll_ramp_steps, self.nll_weight = nll_warmup_steps, nll_ramp_steps, nll_weight
        self.w_exist, self.w_param, self.w_sign = w_exist, w_param, w_sign
        perms = list(itertools.permutations(range(n_queries), k_max))
        self.register_buffer('perms', torch.tensor(perms, dtype=torch.long))   # [P, K]

    @torch.no_grad()
    def match(self, pred, targets, sign, exist):
        """Return assigned query index per true slot, [B, K] (meaningless where ~exist)."""
        batch, k_max = exist.shape
        if self.assignment == 'fixed':
            return torch.arange(k_max, device=exist.device)[None].expand(batch, -1)
        p_exist = pred['exist_logit'].sigmoid()                         # [B,Q]
        p_sign = pred['sign_logit'].sigmoid()
        diff = (pred['mean'][:, :, None] - targets[:, None]).abs()      # [B,Q,K,4]
        cost = (-p_exist[:, :, None] + 5 * diff[..., :2].sum(-1) + 2 * diff[..., 2:].sum(-1)
                + (p_sign[:, :, None] - sign[:, None]).abs())           # [B,Q,K]
        perms = self.perms                                              # [P,K]
        # cost of permutation p = sum_j exist_j * cost[b, perms[p, j], j]
        gathered = cost[:, perms, torch.arange(k_max, device=cost.device)[None]]   # [B,P,K]
        total = (gathered * exist[:, None].float()).sum(-1)             # exist is bool, no NaN here
        return perms[total.argmin(dim=1)]                               # [B,K]

    def layer_loss(self, pred, targets, sign, exist, step):
        assigned = self.match(pred, targets, sign, exist)               # [B,K]
        batch, n_query = pred['exist_logit'].shape
        exist_target = torch.zeros(batch, n_query, device=targets.device)
        b_idx, k_idx = exist.nonzero(as_tuple=True)                     # matched pairs by INDEX
        q_idx = assigned[b_idx, k_idx]
        exist_target[b_idx, q_idx] = 1.0
        loss_exist = F.binary_cross_entropy_with_logits(pred['exist_logit'], exist_target)

        n_matched = b_idx.numel()
        if n_matched == 0:                                              # all K=0 batch
            return loss_exist * self.w_exist, {'exist': loss_exist.item(), 'param': 0.0, 'nll': 0.0, 'sign': 0.0}
        mean_m = pred['mean'][b_idx, q_idx]                             # [M,4]
        target_m = targets[b_idx, k_idx]
        loss_l1 = (mean_m - target_m).abs().sum(-1).mean()
        ramp = min(1.0, max(0.0, (step - self.nll_warmup_steps) / max(1, self.nll_ramp_steps)))
        if ramp > 0:
            dist = torch.distributions.MultivariateNormal(mean_m.detach(),
                                                          scale_tril=pred['scale_tril'][b_idx, q_idx])
            loss_nll = -dist.log_prob(target_m).mean()
        else:
            loss_nll = torch.zeros((), device=mean_m.device)
        loss_param = loss_l1 + ramp * self.nll_weight * loss_nll
        loss_sign = F.binary_cross_entropy_with_logits(pred['sign_logit'][b_idx, q_idx], sign[b_idx, k_idx])
        total = self.w_exist * loss_exist + self.w_param * loss_param + self.w_sign * loss_sign
        return total, {'exist': loss_exist.item(), 'param': loss_l1.item(), 'nll': loss_nll.item(),
                       'sign': loss_sign.item()}

    def forward(self, outputs, params, exist, step):
        targets, sign = self.codec.encode(params)
        totals, parts = [], []
        for pred in outputs:                                            # auxiliary losses
            total, part = self.layer_loss(pred, targets, sign, exist, step)
            totals.append(total)
            parts.append(part)
        return torch.stack(totals).mean(), parts[-1]
