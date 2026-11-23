"""
Multi-source unsupervised domain adaptation for stroke-work regression.

Architecture 4: as architecture 3 (msda_conv_ctx.py), plus the intervention labels of the
SOURCE animals as privileged information during training (never for the target).

  Loading condition (class) of a beat:
    baseline     intervention_phase == 1 (before anything happens), whatever the intervention
    I<code>      intervention code from --interv_classes (default 1, 2, 3, 4, 10), other phases
    ignored      unknown codes and codes in --ignore_interv (label -1, no classification loss;
                 the beat is still used for the stroke work)
  Supervised routing (--route_sup, weight --route_weight, default 1.0: the loss only reaches
  the routing branches, since the routing context is an input, so its weight mainly sets how
  fast the routing learns and does not trade off against the stroke work in W0 or the heads):
    expert   K = number of classes (column K_used; --K applies to probe/none and to runs
             without DRT); pi of every DRT layer is trained with cross-entropy to
             give the loading condition, so that each basis Phi_k becomes the expert of one
             condition (default)
    probe    K stays --K; a linear probe on log pi must predict the condition, so pi has to
             carry the information without fixing one basis per condition
    none     no routing supervision (architecture 3)
  Auxiliary head (--aux_weight, default 0 = off): a classifier on the features of G
  predicts the condition (supervises the features in general instead of the routing).
  Both losses are added to Step A on the labelled source beats only. On the target, the
  routing is predicted from the input. The target's conditions are used only to report how
  well its routing / auxiliary head recognise them (test_route_acc, test_aux_acc).
  Diagnostics per run: src_route_acc / src_aux_acc on the labelled source beats (training
  data: can the conditions be learnt at all?), <split>_majority = accuracy of always
  predicting the most frequent class, <split>_route_top = share of the most frequent
  predicted routing class (close to 1 = practically constant routing).
  Without DRT (source_only, mcd, oracle) there is no routing to supervise; the auxiliary
  head still works if switched on.
  Data: one file per animal, --interv_pattern (default I_<animal>.csv), with a header and
  the columns --interv_col / --phase_col, one row per beat in the same order as X_<animal>.csv.
  With --route_sup none --aux_weight 0 this file reproduces msda_conv_ctx.py exactly.

Architecture 3 (unchanged here): the DRT routing is decided per set of beats instead of per
window. Each DRT layer computes Lambda(x) and pi(x) from a summary
of the last --ctx_len beats (routing context), while the convolution (W0, Phi_k) still
covers only the short window. The kernel mixture therefore follows the current state of the
animal (e.g. the loading condition) rather than the last few beats.

Everything else is identical to msda_conv.py: protocol, target split, windows, MCD steps,
BatchNorm handling, self-training, output format. With --ctx_len 0 this file reproduces
msda_conv.py exactly (routing from the window mean, as in Li et al.).

Routing context
  context(t) = mean (or mean and SD, --ctx_stats) of the standardised features of beats
  t-N+1 .. t, N = --ctx_len (causal; the first beats are padded with the first beat, as
  the windows). It is computed once per animal and stored next to each window as extra
  input channels, so that batching, pseudo-labelling, AdaBN and checkpoints work unchanged;
  the backbone splits it off again. All DRT layers route from the same context, through
  their own branch. Without DRT (source_only, mcd, oracle) the context is not used.
  The adaptation split is the same for every --ctx_len: windows that contain a test beat
  are removed as before, but the context of an adaptation beat may include the features
  (never the labels) of earlier test beats. Removing those as well would drop up to N-1
  beats after every test block and change the adaptation split with N.

Windows
  The prediction for beat t uses beats t-L+1 .. t (causal, the label is the stroke work of
  the last beat). The first L-1 beats of a recording are padded with the first beat, so
  every beat gets a prediction and the evaluated beats are the same as in msda_mlp.py.
  Beats with negative stroke work are removed as labels but kept as input context, so
  that windows never jump over missing beats.
  Target split: the same 50-block split as msda_mlp.py (identical test beats). Windows of
  the adaptation split that contain a test beat are removed, so the network never sees a
  test beat during training, not even unlabelled.

Dynamic convolution (DRTConv1d)
  W(x) = W0 + Lambda(x) W0 + sum_k pi_k(x) Phi_k        (Li et al. 2021, Eq. 5)
  W0 is a k-wide temporal kernel. Lambda(x) and pi(x) come from one shared branch fed by
  the input averaged over the window (Li et al. 2021, Fig. 3). The bases Phi_k are
  --phi_kernel wide (default 1, as in the released code of Li et al.) and aligned with the
  most recent beat of W0's field. A window's feature vector is the output at its last beat.

Options (as in msda_mlp.py): --source_mode, --norm, --K, --target_bn, --drop_features.
Options of msda_conv.py: --window (0 = receptive field), --kernel_size, --phi_kernel,
--channels. New: --ctx_len (0 = window mean as in architecture 2), --ctx_stats.

Configurations (--configs): source_only, drt, mcd, mcd_drt, mcd_drt_st, oracle.
Hyperparameters: see msda_mlp.py; the defaults are the same.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

ARCH = "conv_drt_interv"
# the five animals of the dataset initially provided for this thesis
MAIN_ANIMALS = ["anim1", "anim3", "anim4", "anim8", "anim10"]
DEFAULT_SEEDS = [100, 200, 74, 15, 984]

CONFIGS = {
    "source_only": dict(mcd=False, drt=False, st=False, oracle=False),
    "drt": dict(mcd=False, drt=True, st=False, oracle=False),
    "mcd": dict(mcd=True, drt=False, st=False, oracle=False),
    "mcd_drt": dict(mcd=True, drt=True, st=False, oracle=False),
    "mcd_drt_st": dict(mcd=True, drt=True, st=True, oracle=False),
    "oracle": dict(mcd=False, drt=False, st=False, oracle=True),
}

# =============================================================================
# Data
# =============================================================================

_CACHE: dict = {}


def load_animal(data_dir: str, aid: str):
    """All beats of one animal in temporal order (nothing removed)."""
    key = (data_dir, aid)
    if key not in _CACHE:
        X = np.loadtxt(os.path.join(data_dir, f"X_{aid}.csv"), delimiter=",",
                       dtype=np.float32, ndmin=2)
        y = np.loadtxt(os.path.join(data_dir, f"Y_{aid}.csv"), delimiter=",",
                       dtype=np.float32, ndmin=1).reshape(-1)
        if len(X) != len(y):
            raise ValueError(f"{aid}: {len(X)} feature rows but {len(y)} labels")
        _CACHE[key] = (X, y)
    return _CACHE[key]


def labelled_rows(y, drop_negative):
    """Rows whose stroke work is used as label (all rows serve as input context)."""
    return np.flatnonzero(y >= 0) if drop_negative else np.arange(len(y))


def make_windows(X, rows, L):
    """Causal windows ending at the given rows: [len(rows), C, L], channels first as
    Conv1d expects. The first L-1 beats are padded with the first beat. Called on one
    animal at a time, so no window spans two animals."""
    Xp = np.concatenate([np.repeat(X[:1], L - 1, axis=0), X], axis=0)
    W = np.lib.stride_tricks.sliding_window_view(Xp, L, axis=0)  # [T, C, L], W[t] ends at t
    return np.ascontiguousarray(W[rows], dtype=np.float32)


def routing_context(X, N, stats):
    """Causal summary of the last N beats for every beat: [T, C] (mean) or [T, 2C]
    (mean and SD). The first N-1 beats are padded with the first beat."""
    Xp = np.concatenate([np.repeat(X[:1], N - 1, axis=0), X], axis=0).astype(np.float64)

    def rolling_mean(a):
        cs = np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)], axis=0)
        return (cs[N:] - cs[:-N]) / N  # row t: beats t-N+1 .. t

    mean = rolling_mean(Xp)
    if stats == "mean":
        return mean.astype(np.float32)
    sd = np.sqrt(np.maximum(rolling_mean(Xp ** 2) - mean ** 2, 0.0))
    return np.concatenate([mean, sd], axis=1).astype(np.float32)


def make_inputs(Xn, rows, args):
    """Windows [n, C, L]; with a routing context, the context of each window's last beat
    is appended as extra channels (constant over the window): [n, C + D, L]."""
    W = make_windows(Xn, rows, args.window)
    if args.ctx_len == 0:
        return W
    ctx = routing_context(Xn, args.ctx_len, args.ctx_stats)[rows]
    return np.ascontiguousarray(np.concatenate(
        [W, np.repeat(ctx[:, :, None], args.window, axis=2)], axis=1))


def load_conditions(args, aid, n):
    """Loading condition per beat (all beats, as X): 0 = baseline (phase 1), 1.. = the
    interventions of --interv_classes, -1 = ignored. None if the file does not exist."""
    import pandas as pd
    path = os.path.join(args.data_dir, args.interv_pattern.format(aid=aid))
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    for col in (args.interv_col, args.phase_col):
        if col not in df.columns:
            raise ValueError(f"{path}: column '{col}' not found (columns: {list(df.columns)})")
    if len(df) != n:
        raise ValueError(f"{path}: {len(df)} rows, but X_{aid}.csv has {n} beats")
    interv = df[args.interv_col].to_numpy()
    phase = df[args.phase_col].to_numpy()
    c = np.full(n, -1, dtype=np.int64)
    for j, code in enumerate(interv_codes(args), start=1):
        c[interv == code] = j
    c[phase == 1] = 0
    return c


def interv_codes(args):
    ignore = {float(v) for v in args.ignore_interv.split(",") if v}
    return [float(v) for v in args.interv_classes.split(",") if v and float(v) not in ignore]


def class_names(args):
    return ["baseline"] + [f"I{int(c)}" for c in interv_codes(args)]


_REPORTED: set = set()


def report_conditions(aid, c, names):
    if aid in _REPORTED:
        return
    _REPORTED.add(aid)
    counts = ", ".join(f"{n}={int((c == j).sum())}" for j, n in enumerate(names))
    print(f"  [{aid}] conditions: {counts}, ignored={int((c < 0).sum())}")


def split_target(n: int, test_ratio: float, mode: str, seed: int, n_blocks: int):
    """Adaptation / test indices of the target animal (identical to msda_mlp.py).

    block: the time series is cut into n_blocks contiguous blocks and a random subset of
    blocks (test_ratio of them) forms the test split. random: i.i.d. split of beats.
    """
    rng = np.random.default_rng(seed)
    if mode == "random":
        perm = rng.permutation(n)
        n_test = int(round(n * test_ratio))
        return np.sort(perm[n_test:]), np.sort(perm[:n_test])
    blocks = np.array_split(np.arange(n), n_blocks)
    n_test = max(1, int(round(test_ratio * n_blocks)))
    test_ids = set(rng.choice(n_blocks, n_test, replace=False).tolist())
    test = np.concatenate([b for i, b in enumerate(blocks) if i in test_ids])
    adapt = np.concatenate([b for i, b in enumerate(blocks) if i not in test_ids])
    return adapt, test


@dataclass
class Stats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, a):
        return cls(a.mean(0), a.std(0) + 1e-8)

    def apply(self, a):
        return ((a - self.mean) / self.std).astype(np.float32)


@dataclass
class Fold:
    target: str
    source_ids: list
    src_X: list  # standardised windows [N, C, L] per source domain
    src_y: list  # standardised stroke work per source domain
    adapt_X: np.ndarray  # windows, unlabelled for training
    adapt_y: np.ndarray  # raw stroke work, evaluation only
    test_X: np.ndarray
    test_y: np.ndarray  # raw stroke work, evaluation only
    y_stats: Stats
    adapt_idx: np.ndarray  # row indices in the full recording of the target
    test_idx: np.ndarray
    n_adapt_dropped: int  # adaptation windows removed because they contain a test beat
    n_feat: int  # input features; further channels hold the routing context
    src_c: list  # loading condition per source beat (-1 = ignored)
    adapt_c: np.ndarray  # conditions of the target, evaluation only (-1 if unknown)
    test_c: np.ndarray
    n_classes: int


def kept_columns(args, n_cols):
    drop = {int(i) for i in args.drop_features.split(",") if i != ""}
    if any(i < 0 or i >= n_cols for i in drop):
        raise ValueError(f"--drop_features out of range for {n_cols} columns")
    return [i for i in range(n_cols) if i not in drop]


def build_fold(args, target: str, sources: list, oracle: bool) -> Fold:
    L = args.window
    Xt, yt = load_animal(args.data_dir, target)
    cols = kept_columns(args, Xt.shape[1])
    Xt = Xt[:, cols]

    # same split as msda_mlp.py: computed on the labelled beats, mapped back to rows
    lab = labelled_rows(yt, args.drop_negative)
    a, t = split_target(len(lab), args.test_ratio, args.split, args.split_seed, args.n_blocks)
    adapt_rows, test_rows = lab[a], lab[t]

    # remove adaptation windows that contain a test beat
    is_test = np.zeros(len(yt), dtype=np.int64)
    is_test[test_rows] = 1
    csum = np.concatenate([[0], np.cumsum(is_test)])
    first = np.maximum(adapt_rows - L + 1, 0)
    clean = (csum[adapt_rows + 1] - csum[first]) == 0
    n_dropped = int((~clean).sum())
    adapt_rows = adapt_rows[clean]

    names = class_names(args)
    need = args.route_weight > 0 or args.aux_weight > 0

    def conditions(aid, n):
        c = load_conditions(args, aid, n)
        if c is None:
            if need and aid != target:
                raise FileNotFoundError(
                    f"{args.interv_pattern.format(aid=aid)} not found in {args.data_dir}; "
                    f"needed for --route_weight / --aux_weight")
            return np.full(n, -1, dtype=np.int64)
        report_conditions(aid, c, names)
        return c

    ct = conditions(target, len(yt))
    if oracle:  # upper reference: labelled data of the target animal itself
        src = [(target, Xt, yt, adapt_rows)]
        src_cond = [ct]
    else:
        src, src_cond = [], []
        for aid in sources:
            X, y = load_animal(args.data_dir, aid)
            src.append((aid, X[:, cols], y, labelled_rows(y, args.drop_negative)))
            src_cond.append(conditions(aid, len(y)))

    # statistics on the labelled beats (as in msda_mlp.py), applied to all beats
    y_stats = Stats.fit(np.concatenate([y[r] for _, _, y, r in src]))
    if args.norm == "global" or oracle:
        x_stats = Stats.fit(np.concatenate([X[r] for _, X, _, r in src]))
        src_stats = [x_stats] * len(src)
        tgt_stats = x_stats
    else:  # per_animal
        src_stats = [Stats.fit(X[r]) for _, X, _, r in src]
        tgt_stats = Stats.fit(Xt[adapt_rows])  # unlabelled adaptation split only

    src_X = [make_inputs(s.apply(X), r, args) for s, (_, X, _, r) in zip(src_stats, src)]
    src_y = [y_stats.apply(y[r]) for _, _, y, r in src]
    Xt_n = tgt_stats.apply(Xt)
    return Fold(target=target,
                source_ids=[aid for aid, _, _, _ in src],
                src_X=src_X, src_y=src_y,
                adapt_X=make_inputs(Xt_n, adapt_rows, args), adapt_y=yt[adapt_rows],
                test_X=make_inputs(Xt_n, test_rows, args), test_y=yt[test_rows],
                y_stats=y_stats, adapt_idx=adapt_rows, test_idx=test_rows,
                n_adapt_dropped=n_dropped, n_feat=len(cols),
                src_c=[c[r] for c, (_, _, _, r) in zip(src_cond, src)],
                adapt_c=ct[adapt_rows], test_c=ct[test_rows], n_classes=len(names))


class BatchStream:
    """Endless random mini-batches; reshuffled after each pass over the data."""

    def __init__(self, tensors, batch_size, generator):
        self.t = tensors
        self.n = len(tensors[0])
        self.bs = min(batch_size, self.n)
        self.gen = generator
        self._shuffle()

    def _shuffle(self):
        self.perm = torch.randperm(self.n, generator=self.gen)
        self.pos = 0

    def next(self):
        if self.pos + self.bs > self.n:
            self._shuffle()
        idx = self.perm[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return tuple(t[idx.to(t.device)] for t in self.t)


# =============================================================================
# Model
# =============================================================================

class DRTConv1d(nn.Module):
    """Causal temporal convolution with the combined dynamic residual of Li et al. (2021),
    Eq. 5:

        W(x) = W0 + Lambda(x) W0 + sum_k pi_k(x) Phi_k

    Input [B, C_in, L] -> output [B, C_out, L-k+1] ('valid' convolution: output position
    j sees beats j .. j+k-1, no padding). Lambda(x) (diagonal, sigmoid) and pi(x) (softmax
    over K) come from one shared branch fed by the input averaged over time (Li et al.,
    Fig. 3). The bases Phi_k are phi_kernel wide and aligned with the most recent beat of
    W0's field. With drt=False the layer is the static convolution W0 * x + b. The
    residual parameters are always created, so that W0 gets the same initialisation with
    and without DRT for a given seed (and the same as DRTLinear for kernel_size=1).

    route_dim > 0: the branch is fed with an external routing context [B, route_dim]
    (summary of a longer stretch of beats) instead of the window mean.
    """

    def __init__(self, in_dim, out_dim, kernel_size=3, phi_kernel=1, K=4, drt=True,
                 basis_scale=0.1, min_hidden=16, route_dim=0):
        super().__init__()
        if not 1 <= phi_kernel <= kernel_size:
            raise ValueError("phi_kernel must be between 1 and kernel_size")
        self.K, self.drt, self.out_dim = K, drt, out_dim
        self.k, self.k_phi = kernel_size, phi_kernel
        self.last_logits = None
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        # basis kernels Phi_k: small, non-zero and distinct (zero init gives the router
        # no gradient and identical updates for all bases)
        self.basis = nn.Parameter(torch.empty(K, out_dim, in_dim, phi_kernel))
        with torch.no_grad():
            for k in range(K):
                nn.init.kaiming_uniform_(self.basis[k], a=np.sqrt(5))
            self.basis.mul_(basis_scale / np.sqrt(K))

        def make_branch(b_in):
            hidden = min(max(b_in // 16, min_hidden), b_in)
            return (nn.Sequential(nn.Linear(b_in, hidden), nn.ReLU()),
                    nn.Linear(hidden, out_dim), nn.Linear(hidden, K))

        # The window-mean branch is always created first, so that all other parameters
        # (W0, Phi_k, later layers, heads) get exactly the same initialisation as in
        # msda_conv.py. The context branch is then created on a forked random state, which
        # leaves the global random stream untouched.
        self.branch, self.to_lambda, self.to_pi = make_branch(in_dim)
        if route_dim:
            with torch.random.fork_rng(devices=[]):
                self.branch, self.to_lambda, self.to_pi = make_branch(route_dim)

    def forward(self, x, ctx=None):  # [B, C_in, L]
        z0 = F.conv1d(x, self.weight)  # [B, C_out, L-k+1]
        b = self.bias.unsqueeze(-1)
        if not self.drt:
            return z0 + b
        h = self.branch(x.mean(-1) if ctx is None else ctx)  # window mean or context
        lam = torch.sigmoid(self.to_lambda(h))  # [B, C_out]
        logits = self.to_pi(h)  # [B, K]
        self.last_logits = logits  # kept with gradient for the routing supervision
        pi = torch.softmax(logits, dim=-1)
        z = F.conv1d(x, self.basis.reshape(self.K * self.out_dim, -1, self.k_phi))
        z = z[..., self.k - self.k_phi:]  # align last beats
        z = z.reshape(z.size(0), self.K, self.out_dim, -1)  # Phi_k * x
        return z0 + lam.unsqueeze(-1) * z0 + (pi[:, :, None, None] * z).sum(1) + b


class ConvBackbone(nn.Module):
    """Feature generator G: DRT convolutions with BatchNorm and ReLU; the feature vector of
    a window is the output at its last beat. Receptive field 1 + n_layers * (k - 1).

    Input [B, n_feat + ctx_dim, L]: the first n_feat channels are the beats, the remaining
    ctx_dim channels the routing context (constant over the window), which every DRT layer
    receives. ctx_dim = 0: routing from the window mean (architecture 2)."""

    def __init__(self, in_dim, channels, kernel_size, phi_kernel, K, drt, basis_scale,
                 ctx_dim=0):
        super().__init__()
        self.n_feat, self.ctx_dim = in_dim, ctx_dim
        layers, c_in = [], in_dim
        for c in channels:  # same creation order as msda_conv.py (identical initialisation)
            layers += [DRTConv1d(c_in, c, kernel_size, phi_kernel, K, drt, basis_scale,
                                 route_dim=ctx_dim),
                       nn.BatchNorm1d(c), nn.ReLU()]
            c_in = c
        self.layers = nn.ModuleList(layers)
        self.out_dim = c_in
        self.receptive_field = 1 + len(channels) * (kernel_size - 1)

    def forward(self, x):
        ctx = x[:, self.n_feat:, -1] if self.ctx_dim else None
        h = x[:, :self.n_feat]
        for m in self.layers:
            h = m(h, ctx) if isinstance(m, DRTConv1d) else m(h)
        return h[..., -1]


class Head(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, f):
        return self.net(f).squeeze(-1)


class MSDAModel(nn.Module):
    """G plus n_pairs head pairs (F1, F2). Prediction = mean over all heads."""

    def __init__(self, in_dim, channels, kernel_size, phi_kernel, n_pairs, K, drt,
                 basis_scale, ctx_dim=0, n_classes=0, route_sup="none", aux=False):
        super().__init__()
        self.G = ConvBackbone(in_dim, channels, kernel_size, phi_kernel, K, drt, basis_scale,
                              ctx_dim)
        self.heads = nn.ModuleList([Head(self.G.out_dim) for _ in range(2 * n_pairs)])
        self.n_pairs = n_pairs
        self.drt_layers = [m for m in self.G.modules() if isinstance(m, DRTConv1d)]
        # created after G and the heads, so that their initialisation is unchanged
        self.route_sup = route_sup if drt else "none"
        self.probes = (nn.ModuleList([nn.Linear(K, n_classes) for _ in self.drt_layers])
                       if self.route_sup == "probe" else None)
        self.aux_head = (nn.Sequential(nn.Linear(self.G.out_dim, 64), nn.ReLU(),
                                       nn.Linear(64, n_classes)) if aux else None)

    def extra_parameters(self):
        mods = [m for m in (self.probes, self.aux_head) if m is not None]
        return [p for m in mods for p in m.parameters()]

    def route_class_logits(self):
        """Class logits of every DRT layer from its last forward pass."""
        if self.route_sup == "expert":
            return [l.last_logits for l in self.drt_layers]
        if self.route_sup == "probe":
            return [p(torch.log_softmax(l.last_logits, -1))
                    for p, l in zip(self.probes, self.drt_layers)]
        return []

    def pair(self, k):
        return self.heads[2 * k], self.heads[2 * k + 1]

    def all_heads(self, feat):  # [2 * n_pairs, B]
        return torch.stack([h(feat) for h in self.heads])

    @torch.no_grad()
    def all_heads_eval(self, X, batch_size=2048):
        self.eval()
        return torch.cat([self.all_heads(self.G(X[i:i + batch_size]))
                          for i in range(0, len(X), batch_size)], dim=1)

    @torch.no_grad()
    def predict(self, X):
        return self.all_heads_eval(X).mean(0)


@contextmanager
def frozen_bn(module):
    """BatchNorm in eval mode: normalise with the running source statistics and do not
    update them. Used for every forward pass of target data, so that BN statistics come
    from the source domains only."""
    bns = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    modes = [m.training for m in bns]
    for m in bns:
        m.eval()
    try:
        yield
    finally:
        for m, mode in zip(bns, modes):
            m.train(mode)


@contextmanager
def target_batch_bn(module):
    """BatchNorm uses the statistics of the current (target) batch but does not update the
    running statistics, which therefore stay pure source statistics."""
    bns = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [(m.training, m.momentum) for m in bns]
    for m in bns:
        m.train()
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, (mode, mom) in zip(bns, saved):
            m.train(mode)
            m.momentum = mom


def target_pass(module, args):
    return target_batch_bn(module) if args.target_bn == "target" else frozen_bn(module)


@torch.no_grad()
def with_target_bn_stats(model, X, batch_size=512):
    """Copy of the model whose BatchNorm statistics are recomputed on the target data X
    (AdaBN). All other parameters are unchanged."""
    for layer in model.modules():  # stored routing logits cannot be deep-copied
        if isinstance(layer, DRTConv1d):
            layer.last_logits = None
    m = copy.deepcopy(model)
    bns = [b for b in m.G.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm)]
    for b in bns:
        b.reset_running_stats()
        b.momentum = None  # cumulative average over all batches
    m.eval()
    for b in bns:
        b.train()
    for i in range(0, len(X), batch_size):
        m.G(X[i:i + batch_size])
    m.eval()
    return m


# =============================================================================
# Losses
# =============================================================================

def discrepancy(p1, p2, kind):
    d = (p1 - p2).abs()
    return d.mean() if kind == "l1" else (1.0 - torch.exp(-d)).mean()


def source_loss(model, feats, ys, per_domain):
    """Mean over source domains of the MSE of both heads of the responsible pair."""
    losses = []
    for j, (f, y) in enumerate(zip(feats, ys)):
        h1, h2 = model.pair(j if per_domain else 0)
        losses.append(F.mse_loss(h1(f), y) + F.mse_loss(h2(f), y))
    return torch.stack(losses).mean()


def target_terms(model, feat_t, kind, cap):
    """Discrepancy and (capped) variance penalty on target features, averaged over pairs."""
    dis, vpen = [], []
    for k in range(model.n_pairs):
        h1, h2 = model.pair(k)
        p1, p2 = h1(feat_t), h2(feat_t)
        dis.append(discrepancy(p1, p2, kind))
        vpen.append(-(p1.var().clamp(max=cap) + p2.var().clamp(max=cap)))
    return torch.stack(dis).mean(), torch.stack(vpen).mean()


def class_loss(logits, c):
    """Cross-entropy over the beats with a known condition (0 if there are none)."""
    keep = c >= 0
    if not bool(keep.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[keep], c[keep])


def privileged_losses(model, feat, c):
    """Routing supervision (mean over DRT layers) and auxiliary head on source beats.
    Must be called right after the forward pass of G on these beats."""
    route = [class_loss(z, c) for z in model.route_class_logits()]
    l_route = torch.stack(route).mean() if route else None
    l_aux = class_loss(model.aux_head(feat), c) if model.aux_head is not None else None
    return l_route, l_aux


def pseudo_loss(model, feat, y):
    losses = []
    for k in range(model.n_pairs):
        h1, h2 = model.pair(k)
        losses.append(F.mse_loss(h1(feat), y) + F.mse_loss(h2(feat), y))
    return torch.stack(losses).mean()


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


# =============================================================================
# Training
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def da_weight(epoch, args):
    """0 during warm-up (pure source training), then linear ramp to 1."""
    if epoch <= args.warmup_epochs:
        return 0.0
    if args.ramp_epochs <= 0:
        return 1.0
    return min(1.0, (epoch - args.warmup_epochs) / args.ramp_epochs)


@torch.no_grad()
def make_pseudo_labels(model, X, keep_quantile):
    """Pseudo-label = mean of all heads; confidence = agreement of the heads of each pair.
    Keeps the target windows whose discrepancy is at most the given quantile."""
    P = model.all_heads_eval(X)  # [2 * n_pairs, N]
    pseudo = P.mean(0)
    d = (P[0::2] - P[1::2]).abs().mean(0)
    thr = torch.quantile(d, keep_quantile)
    keep = d <= thr
    return X[keep], pseudo[keep], float(keep.float().mean())


def run_epochs(model, opt_g, opt_f, opt_x, scheds, streams, trg_stream, pseudo_stream,
               epochs, cfg, args, per_domain, steps, device, fold, phase):
    for epoch in epochs:
        w = da_weight(epoch, args) if cfg["mcd"] else 0.0
        log = defaultdict(float)

        for _ in range(steps):
            batches = [s.next() for s in streams]
            xs = [b[0] for b in batches]
            ys = [b[1] for b in batches]
            cs = torch.cat([b[2] for b in batches])
            sizes = [len(x) for x in xs]

            # ---- Step A: all networks on the labelled source data (+ pseudo-labels),
            #      plus the privileged losses on the source conditions
            model.train()
            feat_all = model.G(torch.cat(xs))
            feats = torch.split(feat_all, sizes)
            l_src = source_loss(model, feats, ys, per_domain)
            loss_a = l_src
            l_route, l_aux = privileged_losses(model, feat_all, cs)
            if l_route is not None:
                loss_a = loss_a + args.route_weight * l_route
                log["route_ce"] += l_route.item()
            if l_aux is not None:
                loss_a = loss_a + args.aux_weight * l_aux
                log["aux_ce"] += l_aux.item()
            if pseudo_stream is not None:
                xp, yp = pseudo_stream.next()
                with target_pass(model.G, args):
                    l_ps = pseudo_loss(model, model.G(xp), yp)
                loss_a = loss_a + args.st_weight * l_ps
                log["pseudo_mse"] += l_ps.item()
            opt_g.zero_grad();
            opt_f.zero_grad()
            if opt_x is not None:
                opt_x.zero_grad()
            loss_a.backward()
            opt_g.step();
            opt_f.step()
            if opt_x is not None:
                opt_x.step()
            log["src_mse"] += l_src.item()

            if w == 0.0:  # warm-up: identical to source-only training
                continue
            (xt,) = trg_stream.next()

            # ---- Step B: heads maximise the target discrepancy, G fixed
            with torch.no_grad():
                with frozen_bn(model.G):
                    fs = torch.split(model.G(torch.cat(xs)), sizes)
                with target_pass(model.G, args):
                    ft = model.G(xt)
                log["trg_pred_std"] += model.all_heads(ft).std(1).mean().item()
            dis, vpen = target_terms(model, ft, args.discrepancy, args.vpen_cap)
            loss_b = source_loss(model, fs, ys, per_domain) + w * (args.gamma * vpen - args.eta * dis)
            opt_f.zero_grad()
            loss_b.backward()
            opt_f.step()

            # ---- Step C: G minimises the target discrepancy, heads fixed (n times)
            set_requires_grad(model.heads, False)
            for _ in range(args.num_k):
                with target_pass(model.G, args):
                    ft = model.G(xt)
                dis, vpen = target_terms(model, ft, args.discrepancy, args.vpen_cap)
                loss_c = w * (args.eta * dis + args.gamma * vpen)
                opt_g.zero_grad()
                loss_c.backward()
                opt_g.step()
            set_requires_grad(model.heads, True)
            log["trg_dis"] += dis.item()

        if scheds is not None:
            for s in scheds:
                s.step()

        if args.log_every and (epoch % args.log_every == 0 or epoch == epochs[-1]):
            msg = (f"    [{phase}] ep {epoch:3d}  w={w:.2f}  "
                   f"src_mse={log['src_mse'] / steps:.4f}")
            if cfg["mcd"] and w > 0:
                msg += (f"  trg_dis={log['trg_dis'] / steps:.4f}"
                        f"  trg_pred_std={log['trg_pred_std'] / steps:.2f}")
            if pseudo_stream is not None:
                msg += f"  pseudo_mse={log['pseudo_mse'] / steps:.4f}"
            if "route_ce" in log:
                msg += f"  route_ce={log['route_ce'] / steps:.3f}"
            if "aux_ce" in log:
                msg += f"  aux_ce={log['aux_ce'] / steps:.3f}"
            if args.monitor_target:  # debugging only: uses target labels
                msg += (f"  [label-based] test_rmse="
                        f"{evaluate(eval_model(model, fold, args, device), fold, 'test', device)['rmse']:.1f}")
            print(msg, flush=True)


def eval_model(model, fold, args, device):
    """Model used for target predictions: with --target_bn target, the BatchNorm statistics
    are recomputed on the unlabelled adaptation split of the target."""
    if args.target_bn == "target":
        return with_target_bn_stats(model, torch.as_tensor(fold.adapt_X, device=device))
    return model


@torch.no_grad()
def evaluate(model, fold, split, device):
    X = torch.as_tensor(fold.test_X if split == "test" else fold.adapt_X, device=device)
    y = fold.test_y if split == "test" else fold.adapt_y
    pred = model.predict(X).cpu().numpy() * fold.y_stats.std + fold.y_stats.mean
    err = pred - y
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r = float(pearsonr(y, pred)[0]) if np.std(pred) > 0 else float("nan")
    return {"rmse": float(np.sqrt((err ** 2).mean())),
            "mae": float(np.abs(err).mean()),
            "r2": float(1.0 - (err ** 2).sum() / ss_tot),
            "r": r}


@torch.no_grad()
def condition_accuracy(model, X, c, batch_size=2048):
    """How well the routing of the last DRT layer and the auxiliary head recognise the
    loading condition of the beats X (evaluation only; target conditions are never used
    for training).
      route_acc / aux_acc   share of beats whose condition is predicted correctly
      route_top             share of the most frequent predicted routing class
                            (close to 1: the routing predicts practically one class)
      majority              share of the most frequent true class, i.e. the accuracy of
                            always predicting that class"""
    keep = c >= 0
    out = {"route_acc": float("nan"), "aux_acc": float("nan"),
           "route_top": float("nan"), "majority": float("nan")}
    if not keep.any():
        return out
    ct = torch.as_tensor(c)
    out["majority"] = float(torch.bincount(ct[keep]).max()) / int(keep.sum())
    model.eval()
    route, aux = [], []
    for i in range(0, len(X), batch_size):
        feat = model.G(X[i:i + batch_size])
        z = model.route_class_logits()
        if z:
            route.append(z[-1].argmax(1).cpu())
        if model.aux_head is not None:
            aux.append(model.aux_head(feat).argmax(1).cpu())
    if route:
        pr = torch.cat(route)[keep]
        out["route_acc"] = float((pr == ct[keep]).float().mean())
        out["route_top"] = float(torch.bincount(pr).max()) / len(pr)
    if aux:
        out["aux_acc"] = float((torch.cat(aux)[keep] == ct[keep]).float().mean())
    return out


def parse_channels(s):
    return [int(c) for c in s.split(",") if c]


def train_run(args, cfg_name, fold, seed, device):
    cfg = CONFIGS[cfg_name]
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    per_domain = args.source_mode == "per_domain" and not cfg["oracle"]
    n_pairs = len(fold.src_X) if per_domain else 1
    expert = args.route_sup == "expert" and args.route_weight > 0 and cfg["drt"]
    K = fold.n_classes if expert else args.K
    model = MSDAModel(fold.n_feat, parse_channels(args.channels),
                      args.kernel_size, args.phi_kernel, n_pairs, K,
                      cfg["drt"], args.basis_scale,
                      ctx_dim=fold.src_X[0].shape[1] - fold.n_feat,
                      n_classes=fold.n_classes,
                      route_sup=args.route_sup if args.route_weight > 0 else "none",
                      aux=args.aux_weight > 0).to(device)

    opt_g = torch.optim.SGD(model.G.parameters(), lr=args.lr, momentum=0.9)
    opt_f = torch.optim.SGD(model.heads.parameters(), lr=args.lr_f, momentum=0.9)
    # probes and auxiliary head: own optimiser, stepped in Step A only
    extra = model.extra_parameters()
    opt_x = torch.optim.SGD(extra, lr=args.lr_f, momentum=0.9) if extra else None
    opts = [o for o in (opt_g, opt_f, opt_x) if o is not None]
    scheds = [torch.optim.lr_scheduler.MultiStepLR(o, milestones=args.schedule, gamma=0.1)
              for o in opts]

    tens = lambda a: torch.as_tensor(a, device=device)
    n_src = sum(len(X) for X in fold.src_X)
    if args.source_mode == "pooled" or cfg["oracle"]:
        streams = [BatchStream((tens(np.concatenate(fold.src_X)), tens(np.concatenate(fold.src_y)),
                                tens(np.concatenate(fold.src_c))), args.batch_size, gen)]
    else:  # balanced / per_domain: equal share of each source animal in every batch
        bs = max(2, args.batch_size // len(fold.src_X))
        streams = [BatchStream((tens(X), tens(y), tens(c)), bs, gen)
                   for X, y, c in zip(fold.src_X, fold.src_y, fold.src_c)]
    steps = max(1, n_src // args.batch_size)
    # separate generators: the source batch order is identical with and without MCD/ST
    trg_gen = torch.Generator().manual_seed(seed + 1)
    trg_stream = BatchStream((tens(fold.adapt_X),), args.batch_size, trg_gen) if cfg["mcd"] else None

    run_epochs(model, opt_g, opt_f, opt_x, scheds, streams, trg_stream, None,
               list(range(1, args.epochs + 1)), cfg, args, per_domain, steps,
               device, fold, "main")

    keep_rate = float("nan")
    if cfg["st"]:
        Xp, yp, keep_rate = make_pseudo_labels(eval_model(model, fold, args, device),
                                               tens(fold.adapt_X), args.st_keep_quantile)
        print(f"    [st] kept {keep_rate:.0%} of the adaptation windows as pseudo-labels")
        for o, lr in ((opt_g, args.lr), (opt_f, args.lr_f), (opt_x, args.lr_f)):
            if o is None:
                continue
            for g in o.param_groups:
                g["lr"] = lr * args.st_lr_scale
        pseudo_stream = BatchStream((Xp, yp), args.batch_size,
                                    torch.Generator().manual_seed(seed + 2))
        run_epochs(model, opt_g, opt_f, opt_x, None, streams, trg_stream, pseudo_stream,
                   list(range(args.epochs + 1, args.epochs + args.st_epochs + 1)),
                   cfg, args, per_domain, steps, device, fold, "st")

    result = {"st_keep_rate": keep_rate, "K_used": K}
    final = eval_model(model, fold, args, device)
    for split in ("test", "adapt"):
        if cfg["oracle"] and split == "adapt":
            continue  # adaptation split is the oracle's training data
        for k, v in evaluate(final, fold, split, device).items():
            result[f"{split}_{k}"] = v
        X = torch.as_tensor(fold.test_X if split == "test" else fold.adapt_X, device=device)
        c = fold.test_c if split == "test" else fold.adapt_c
        for k, v in condition_accuracy(final, X, c).items():
            result[f"{split}_{k}"] = v
    # the same on the labelled source beats (training data): shows whether the conditions
    # can be learnt at all; compared with the target it shows whether they transfer
    if not cfg["oracle"]:
        Xs = torch.as_tensor(np.concatenate(fold.src_X), device=device)
        for k, v in condition_accuracy(final, Xs, np.concatenate(fold.src_c)).items():
            result[f"src_{k}"] = v

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg_name,
                    "args": vars(args), "target": fold.target, "sources": fold.source_ids,
                    "y_mean": fold.y_stats.mean, "y_std": fold.y_stats.std,
                    "adapt_idx": fold.adapt_idx, "test_idx": fold.test_idx,
                    "n_pairs": n_pairs, "drt": cfg["drt"], "window": args.window,
                    "kernel_size": args.kernel_size, "phi_kernel": args.phi_kernel,
                    "channels": args.channels, "ctx_len": args.ctx_len,
                    "ctx_stats": args.ctx_stats, "n_feat": fold.n_feat,
                    "route_sup": args.route_sup, "route_weight": args.route_weight,
                    "aux_weight": args.aux_weight, "class_names": class_names(args),
                    "K": K, "n_classes": fold.n_classes},
                   os.path.join(args.save_dir,
                                f"{ARCH}_{args.route_sup}{args.route_weight}_aux{args.aux_weight}_"
                                f"N{args.ctx_len}_L{args.window}_k{args.kernel_size}_"
                                f"{args.source_mode}_{args.norm}_K{K}_"
                                f"{cfg_name}_{fold.target}_seed{seed}.pt"))
    return result


# =============================================================================
# Experiment loop and output
# =============================================================================

FIELDS = ["run_id", "arch", "protocol", "config", "source_mode", "norm", "target_bn", "K",
          "window", "kernel_size", "phi_kernel", "channels", "ctx_len", "ctx_stats",
          "route_sup", "route_weight", "aux_weight", "ignore_interv", "n_classes", "K_used",
          "split", "n_blocks", "discrepancy", "num_k", "eta", "gamma", "vpen_cap",
          "warmup_epochs", "ramp_epochs", "epochs", "st_keep_quantile", "st_weight",
          "drop_features", "target",
          "sources", "seed", "test_rmse", "test_mae", "test_r2", "test_r",
          "adapt_rmse", "adapt_mae", "adapt_r2", "adapt_r", "st_keep_rate",
          "test_route_acc", "adapt_route_acc", "test_aux_acc", "adapt_aux_acc",
          "src_route_acc", "src_aux_acc", "test_majority", "adapt_majority", "src_majority",
          "test_route_top", "adapt_route_top", "src_route_top"]


def check_out_file(path):
    """Rows are appended to --out. A file written by an older version has other columns,
    so new rows would be shifted against its header."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        header = next(csv.reader(f), [])
    if header and header != FIELDS:
        raise SystemExit(f"{path} has other columns than this version writes "
                         f"(older version?); choose a new --out file")


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def summarize(path, run_id):
    import pandas as pd
    df = pd.read_csv(path)
    df = df[df["run_id"] == run_id]
    if df.empty:
        return
    per_target = (df.groupby(["config", "target"])[["test_rmse", "test_r2"]]
                  .agg(["mean", "std"]).round(3))
    print("\nPer target (mean and std over seeds, held-out test split):")
    print(per_target.to_string())
    seed_mean = df.groupby(["config", "target"])[["test_rmse", "test_r2"]].mean()
    overall = seed_mean.groupby("config").agg(["mean", "std"]).round(3)
    print("\nAcross targets (mean and std of the per-target means):")
    print(overall.to_string())
    cols = ["src_route_acc", "test_route_acc", "test_majority", "test_route_top"]
    acc = df[df["test_route_acc"].notna()]
    if not acc.empty:
        print("\nCondition recognition by the routing (mean over seeds): source (training "
              "data) vs target test split; majority = always the most frequent class, "
              "route_top = share of the most frequent predicted class")
        print(acc.groupby(["config", "target"])[cols].mean().round(3).to_string())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # data and protocol
    p.add_argument("--data_dir",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "Tierdaten"))
    p.add_argument("--protocol", choices=["loao", "test"], default="loao")
    p.add_argument("--main_animals", default=",".join(MAIN_ANIMALS))
    p.add_argument("--test_animals", default="",
                   help="comma-separated untouched animals for --protocol test")
    p.add_argument("--targets", default="", help="optional subset of targets to run")
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--drop_negative", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--split", choices=["block", "random"], default="block")
    p.add_argument("--n_blocks", type=int, default=50)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=0,
                   help="fixed across training seeds, so seed variance is training variance")
    # options without architecture change
    p.add_argument("--source_mode", choices=["pooled", "balanced", "per_domain"], default="pooled")
    p.add_argument("--norm", choices=["global", "per_animal"], default="global")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--target_bn", choices=["source", "target"], default="source")
    p.add_argument("--drop_features", default="",
                   help="comma-separated 0-based column indices to remove, e.g. 0,1,7,8,9")
    # architecture
    p.add_argument("--channels", default="256,128",
                   help="output channels of the DRT convolutions (as the MLP: 256,128)")
    p.add_argument("--kernel_size", type=int, default=3, help="temporal width of W0")
    p.add_argument("--phi_kernel", type=int, default=1,
                   help="temporal width of the bases Phi_k (released code of Li et al.: 1)")
    p.add_argument("--window", type=int, default=0,
                   help="beats per window; 0 = receptive field 1 + n_layers*(kernel_size-1)")
    p.add_argument("--basis_scale", type=float, default=0.1)
    # routing context
    p.add_argument("--ctx_len", type=int, default=50,
                   help="beats summarised for the DRT routing; 0 = window mean (msda_conv.py)")
    p.add_argument("--ctx_stats", choices=["mean", "meanstd"], default="mean",
                   help="summary of the routing context: mean, or mean and SD")
    # intervention labels (privileged information, source animals only)
    p.add_argument("--interv_pattern", default="I_{aid}.csv",
                   help="file with the intervention labels per animal ({aid} = animal)")
    p.add_argument("--interv_col", default="intervention")
    p.add_argument("--phase_col", default="intervention_phase")
    p.add_argument("--interv_classes", default="1,2,3,4,10",
                   help="intervention codes that become classes (phase 1 = baseline class)")
    p.add_argument("--ignore_interv", default="",
                   help="intervention codes to ignore in the classification losses, e.g. 4")
    p.add_argument("--route_sup", choices=["expert", "probe", "none"], default="expert",
                   help="expert: K = number of classes, pi predicts the condition; "
                        "probe: linear probe on log pi; none: architecture 3")
    p.add_argument("--route_weight", type=float, default=1.0,
                   help="weight of the routing supervision (0 = off); it only reaches the "
                        "routing branches, not W0, Phi_k or the heads")
    p.add_argument("--aux_weight", type=float, default=0.0,
                   help="weight of the auxiliary condition head on the features (0 = off)")
    # optimisation
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="feature generator")
    p.add_argument("--lr_f", type=float, default=5e-3, help="heads")
    p.add_argument("--schedule", type=int, nargs="+", default=[110, 135, 145])
    # MCD
    p.add_argument("--discrepancy", choices=["bounded", "l1"], default="bounded")
    p.add_argument("--num_k", type=int, default=4)
    p.add_argument("--eta", type=float, default=0.1, help="discrepancy weight")
    p.add_argument("--gamma", type=float, default=0.0,
                   help="variance penalty weight (0 = as in Saito et al.)")
    p.add_argument("--vpen_cap", type=float, default=1.0,
                   help="prediction variance above this value is not rewarded "
                        "(standardised stroke work has variance ~1)")
    p.add_argument("--warmup_epochs", type=int, default=100)
    p.add_argument("--ramp_epochs", type=int, default=20)
    # self-training
    p.add_argument("--st_epochs", type=int, default=30)
    p.add_argument("--st_keep_quantile", type=float, default=0.5)
    p.add_argument("--st_weight", type=float, default=1.0)
    p.add_argument("--st_lr_scale", type=float, default=0.1)
    # output
    p.add_argument("--out", default="results_conv_interv.csv")
    p.add_argument("--save_dir", default="", help="save one checkpoint per run (for SHAP/CRAFT)")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--monitor_target", action="store_true",
                   help="DEBUG ONLY: print target test RMSE during training")
    args = p.parse_args()

    rf = 1 + len(parse_channels(args.channels)) * (args.kernel_size - 1)
    if args.window == 0:
        args.window = rf
    if args.ctx_len < 0:
        raise ValueError("--ctx_len must be >= 0")
    if args.route_sup == "none":
        args.route_weight = 0.0
    if args.route_sup == "expert" and args.route_weight > 0:
        print(f"--route_sup expert: DRT runs use K = number of classes "
              f"({len(class_names(args))}: {', '.join(class_names(args))}); runs without DRT "
              f"keep K = {args.K}, so their initialisation equals msda_conv_ctx.py")
    if args.window < rf:
        raise ValueError(f"--window {args.window} is shorter than the receptive field ({rf})")
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_animals = [a for a in args.main_animals.split(",") if a]
    configs = [c for c in args.configs.split(",") if c]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    for c in configs:
        if c not in CONFIGS:
            raise ValueError(f"unknown config {c}; choose from {list(CONFIGS)}")

    if args.protocol == "loao":
        targets = main_animals
        sources_of = lambda t: [a for a in main_animals if a != t]
    else:
        targets = [a for a in args.test_animals.split(",") if a]
        if not targets:
            raise ValueError("--protocol test needs --test_animals")
        sources_of = lambda t: main_animals
    if args.targets:
        targets = [t for t in targets if t in args.targets.split(",")]

    if args.monitor_target:
        print("WARNING: --monitor_target prints label-based target metrics. "
              "Do not use them to make any decision.")

    check_out_file(args.out)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    print(f"run {run_id}: {ARCH}, channels={args.channels}, kernel_size={args.kernel_size}, "
          f"phi_kernel={args.phi_kernel}, window={args.window}, ctx_len={args.ctx_len} "
          f"({args.ctx_stats}), route_sup={args.route_sup} ({args.route_weight}), "
          f"aux_weight={args.aux_weight}, ignore_interv={args.ignore_interv or 'none'}, "
          f"protocol={args.protocol}, "
          f"source_mode={args.source_mode}, norm={args.norm}, target_bn={args.target_bn}, "
          f"K={args.K}, split={args.split} ({args.n_blocks} blocks), gamma={args.gamma}, "
          f"num_k={args.num_k}, drop_features={args.drop_features or 'none'}, device={device}")

    total = len(configs) * len(targets) * len(seeds)
    i = 0
    for cfg_name in configs:
        for target in targets:
            fold = build_fold(args, target, sources_of(target), CONFIGS[cfg_name]["oracle"])
            if fold.n_adapt_dropped:
                print(f"  [{target}] removed {fold.n_adapt_dropped} adaptation windows "
                      f"that contain test beats")
            for seed in seeds:
                i += 1
                print(f"\n[{i}/{total}] {cfg_name} | target={target} | "
                      f"sources={','.join(fold.source_ids)} | seed={seed}", flush=True)
                res = train_run(args, cfg_name, fold, seed, device)
                print(f"    test: RMSE={res['test_rmse']:.1f}  R2={res['test_r2']:.3f}  "
                      f"r={res['test_r']:.3f}")
                append_row(args.out, {
                    "run_id": run_id, "arch": ARCH, "protocol": args.protocol,
                    "config": cfg_name, "source_mode": args.source_mode, "norm": args.norm,
                    "target_bn": args.target_bn, "K": args.K, "window": args.window,
                    "kernel_size": args.kernel_size, "phi_kernel": args.phi_kernel,
                    "channels": args.channels, "ctx_len": args.ctx_len,
                    "ctx_stats": args.ctx_stats, "route_sup": args.route_sup,
                    "route_weight": args.route_weight, "aux_weight": args.aux_weight,
                    "ignore_interv": args.ignore_interv or "none",
                    "n_classes": fold.n_classes, "split": args.split,
                    "n_blocks": args.n_blocks, "discrepancy": args.discrepancy,
                    "num_k": args.num_k, "eta": args.eta, "gamma": args.gamma,
                    "vpen_cap": args.vpen_cap, "warmup_epochs": args.warmup_epochs,
                    "ramp_epochs": args.ramp_epochs, "epochs": args.epochs,
                    "st_keep_quantile": args.st_keep_quantile, "st_weight": args.st_weight,
                    "drop_features": args.drop_features or "none",
                    "target": target, "sources": ";".join(fold.source_ids), "seed": seed,
                    **res})

    summarize(args.out, run_id)


if __name__ == "__main__":
    main()