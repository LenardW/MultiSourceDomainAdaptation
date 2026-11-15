#!/usr/bin/env python3
"""
Multi-source unsupervised domain adaptation for stroke-work regression.

Architecture 1: MLP backbone whose linear layers are DRT layers with the
combined dynamic residual (Li et al. 2021, Eq. 5), two regression heads per
head pair (MCD, Saito et al. 2018), optional self-training (Li et al. 2021,
Sec. 4.3). Each animal is one domain.

Options that do not change the architecture:
  --source_mode   pooled      all source animals in one batch stream (ERM on the pooled data)
                  balanced    every batch contains equal shares of each source animal and the
                              loss is the mean of the per-animal losses (Csurka et al., Eq. 7.1)
                  per_domain  like balanced, but one head pair per source animal; the target
                              prediction is the mean over all heads
  --norm          global      features standardised with the pooled source statistics
                  per_animal  every animal standardised with its own feature statistics
                              (target: statistics of its unlabelled adaptation split)
                  The stroke work is always standardised with the pooled source statistics.
  --K             number of DRT basis matrices
  --target_bn     source      target data are normalised with the BatchNorm statistics of the
                              source data (no adaptation through BN)
                  target      target data are normalised with target statistics: batch
                              statistics in the MCD steps, statistics of the unlabelled
                              adaptation split for pseudo-labelling and evaluation (AdaBN,
                              Li et al. 2017; Csurka et al. 2022, p. 49)

Configurations (--configs):
  source_only, drt, mcd, mcd_drt, mcd_drt_st  : component ablation
  oracle                                      : trained on the labelled adaptation split of
                                                the target animal itself (upper reference)

Protocols (--protocol):
  loao   leave-one-animal-out over --main_animals (default: the 10 animals without speed
         ramps); every animal is the target once, all others are the sources
  test   targets = --test_animals, sources = all --main_animals (e.g. the speed-ramp
         animals as targets with the non-ramp animals as sources)

Ablation design: the defaults below are THE method (fixed in advance). Every ablation
changes exactly one setting relative to these defaults; results are not carried forward.

Features: --drop_features removes input columns by index (0-based), e.g. the five
features that need invasive pressures.

Target data: the target animal is split into an unlabelled adaptation split (used by MCD and
self-training without labels) and a held-out test split. Labels of the target are used
ONLY for the final evaluation. The default split uses contiguous time blocks, because
neighbouring heartbeats are almost identical and a random split would leak them.

Hyperparameters and where they come from
  K = 4                          Li et al. 2021, Sec. 4.2 (default; results insensitive to K)
  DRT residual = combination     Li et al. 2021, Eq. 5 (fixed in this file)
  MCD steps A/B/C, source loss
    kept in step B               Saito et al. 2018, Sec. 3.3
  num_k (n) = 4                  Saito et al. 2018, Sec. 4.2 (n in {2,3,4}; accuracy rises
                                 with n and saturates)
  SGD, momentum 0.9, lr 1e-3,
    batch size 64                Li et al. 2021, Sec. 4.1 (DomainNet setting)
  ST: retrain on pseudo-labelled
    target + source samples      Li et al. 2021, Sec. 4.3
  ST confidence = head agreement own choice (regression has no class probability)
  discrepancy = 1 - exp(-|d|)    own choice (bounded regression analogue of Saito Eq. 1);
                                 --discrepancy l1 gives |d|, the direct analogue
  gamma = 0 (no variance penalty) as in Saito et al.; an unbounded variance penalty diverged
                                 for a target outside the source support, so a penalty is only
                                 available with a cap (--vpen_cap)
  target split: 50 time blocks   avoids leakage between neighbouring beats, keeps the test
                                 set representative of the whole recording
  lr_f, epochs, schedule         set on the source data (stable source-only training)
  eta, gamma, warmup, ramp,
    st_* parameters              selected on the main animals (protocol loao)
"""
from __future__ import annotations

import argparse
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

ARCH = "mlp_drt_linear"
# animals without speed ramps (intervention 4); anim3 has only short ramp sections and
# shows no concept shift


# MAIN_ANIMALS = ["anim3", "anim4", "anim7", "anim8", "anim10",
#                 "anim11", "anim13", "anim16", "anim19", "anim20"]
# MAIN_ANIMALS = ["anim3", "anim4", "anim8", "anim10",
#                 "anim11", "anim16", "anim20"]
MAIN_ANIMALS = ["anim1", "anim3", "anim4", "anim8", "anim10"]
# RAMP_ANIMALS = ["anim1", "anim107", "anim108", "anim111", "anim113"]
DEFAULT_SEEDS = [100, 200, 74, 15, 984]

CONFIGS = {
    "source_only": dict(mcd=False, drt=False, st=False, oracle=False),
    "drt":         dict(mcd=False, drt=True,  st=False, oracle=False),
    "mcd":         dict(mcd=True,  drt=False, st=False, oracle=False),
    "mcd_drt":     dict(mcd=True,  drt=True,  st=False, oracle=False),
    "mcd_drt_st":  dict(mcd=True,  drt=True,  st=True,  oracle=False),
    "oracle":      dict(mcd=False, drt=False, st=False, oracle=True),
}


# =============================================================================
# Data
# =============================================================================

_CACHE: dict = {}


def load_animal(data_dir: str, aid: str, drop_negative: bool):
    """Rows are heartbeats in temporal order. Beats with negative stroke work are removed."""
    key = (data_dir, aid, drop_negative)
    if key not in _CACHE:
        X = np.loadtxt(os.path.join(data_dir, f"X_{aid}.csv"), delimiter=",",
                       dtype=np.float32, ndmin=2)
        y = np.loadtxt(os.path.join(data_dir, f"Y_{aid}.csv"), delimiter=",",
                       dtype=np.float32, ndmin=1).reshape(-1)
        if len(X) != len(y):
            raise ValueError(f"{aid}: {len(X)} feature rows but {len(y)} labels")
        if drop_negative:
            keep = y >= 0
            if (~keep).any():
                print(f"  [{aid}] removed {int((~keep).sum())} beats with negative stroke work")
            X, y = X[keep], y[keep]
        _CACHE[key] = (X, y)
    return _CACHE[key]


def split_target(n: int, test_ratio: float, mode: str, seed: int, n_blocks: int):
    """Adaptation / test indices of the target animal.

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
    src_X: list            # standardised features per source domain
    src_y: list            # standardised stroke work per source domain
    adapt_X: np.ndarray    # standardised, unlabelled for training
    adapt_y: np.ndarray    # raw stroke work, evaluation only
    test_X: np.ndarray
    test_y: np.ndarray     # raw stroke work, evaluation only
    y_stats: Stats
    adapt_idx: np.ndarray
    test_idx: np.ndarray


def kept_columns(args, n_cols):
    drop = {int(i) for i in args.drop_features.split(",") if i != ""}
    if any(i < 0 or i >= n_cols for i in drop):
        raise ValueError(f"--drop_features out of range for {n_cols} columns")
    return [i for i in range(n_cols) if i not in drop]


def build_fold(args, target: str, sources: list, oracle: bool) -> Fold:
    Xt, yt = load_animal(args.data_dir, target, args.drop_negative)
    cols = kept_columns(args, Xt.shape[1])
    Xt = Xt[:, cols]
    adapt, test = split_target(len(Xt), args.test_ratio, args.split,
                               args.split_seed, args.n_blocks)

    if oracle:   # upper reference: labelled data of the target animal itself
        src = [(target, Xt[adapt], yt[adapt])]
    else:
        src = []
        for aid in sources:
            X, y = load_animal(args.data_dir, aid, args.drop_negative)
            src.append((aid, X[:, cols], y))

    y_stats = Stats.fit(np.concatenate([y for _, _, y in src]))

    if args.norm == "global" or oracle:
        x_stats = Stats.fit(np.concatenate([X for _, X, _ in src]))
        src_X = [x_stats.apply(X) for _, X, _ in src]
        tgt_stats = x_stats
    else:        # per_animal
        src_X = [Stats.fit(X).apply(X) for _, X, _ in src]
        tgt_stats = Stats.fit(Xt[adapt])          # unlabelled adaptation split only

    Xt_n = tgt_stats.apply(Xt)
    return Fold(target=target,
                source_ids=[aid for aid, _, _ in src],
                src_X=src_X,
                src_y=[y_stats.apply(y) for _, _, y in src],
                adapt_X=Xt_n[adapt], adapt_y=yt[adapt],
                test_X=Xt_n[test], test_y=yt[test],
                y_stats=y_stats, adapt_idx=adapt, test_idx=test)


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

class DRTLinear(nn.Module):
    """Linear layer with the combined dynamic residual of Li et al. (2021), Eq. 5:

        W(x) = W0 + Lambda(x) W0 + sum_k pi_k(x) Phi_k

    Lambda(x) (diagonal, sigmoid) and pi(x) (softmax over K) come from one shared
    lightweight attention branch. With drt=False the layer is the static W0 x + b.
    The residual parameters are always created, so that the static weights get the same
    initialisation with and without DRT for a given seed.
    """

    def __init__(self, in_dim, out_dim, K=4, drt=True, basis_scale=0.1, min_hidden=16):
        super().__init__()
        self.K, self.drt, self.out_dim = K, drt, out_dim
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        # basis matrices Phi_k: small, non-zero and distinct (zero init gives the router
        # no gradient and identical updates for all bases)
        self.basis = nn.Parameter(torch.empty(K, out_dim, in_dim))
        with torch.no_grad():
            for k in range(K):
                nn.init.kaiming_uniform_(self.basis[k], a=np.sqrt(5))
            self.basis.mul_(basis_scale / np.sqrt(K))

        hidden = min(max(in_dim // 16, min_hidden), in_dim)
        self.branch = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.to_lambda = nn.Linear(hidden, out_dim)
        self.to_pi = nn.Linear(hidden, K)

    def forward(self, x):
        z0 = F.linear(x, self.weight)
        if not self.drt:
            return z0 + self.bias
        h = self.branch(x)
        lam = torch.sigmoid(self.to_lambda(h))                    # [B, out]
        pi = torch.softmax(self.to_pi(h), dim=-1)                 # [B, K]
        z = F.linear(x, self.basis.reshape(self.K * self.out_dim, -1))
        z = z.view(-1, self.K, self.out_dim)                      # Phi_k x
        return z0 + lam * z0 + (pi.unsqueeze(-1) * z).sum(1) + self.bias


class Backbone(nn.Module):
    """Feature generator G: 14 -> 256 -> 128 with BatchNorm and ReLU."""

    def __init__(self, in_dim, hidden_dim, K, drt, basis_scale):
        super().__init__()
        self.net = nn.Sequential(
            DRTLinear(in_dim, 256, K, drt, basis_scale), nn.BatchNorm1d(256), nn.ReLU(),
            DRTLinear(256, hidden_dim, K, drt, basis_scale), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, f):
        return self.net(f).squeeze(-1)


class MSDAModel(nn.Module):
    """G plus n_pairs head pairs (F1, F2). Prediction = mean over all heads."""

    def __init__(self, in_dim, hidden_dim, n_pairs, K, drt, basis_scale):
        super().__init__()
        self.G = Backbone(in_dim, hidden_dim, K, drt, basis_scale)
        self.heads = nn.ModuleList([Head(hidden_dim) for _ in range(2 * n_pairs)])
        self.n_pairs = n_pairs

    def pair(self, k):
        return self.heads[2 * k], self.heads[2 * k + 1]

    def all_heads(self, feat):                       # [2 * n_pairs, B]
        return torch.stack([h(feat) for h in self.heads])

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        return self.all_heads(self.G(X)).mean(0)


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
    import copy
    m = copy.deepcopy(model)
    bns = [b for b in m.G.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm)]
    for b in bns:
        b.reset_running_stats()
        b.momentum = None              # cumulative average over all batches
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
def make_pseudo_labels(model, X, keep_quantile, kind):
    """Pseudo-label = mean of all heads; confidence = agreement of the heads of each pair.
    Keeps the target beats whose discrepancy is at most the given quantile."""
    model.eval()
    P = model.all_heads(model.G(X))                      # [2 * n_pairs, N]
    pseudo = P.mean(0)
    d = (P[0::2] - P[1::2]).abs().mean(0)
    thr = torch.quantile(d, keep_quantile)
    keep = d <= thr
    return X[keep], pseudo[keep], float(keep.float().mean())


def run_epochs(model, opt_g, opt_f, scheds, streams, trg_stream, pseudo_stream,
               epochs, cfg, args, per_domain, steps, device, fold, phase):
    for epoch in epochs:
        w = da_weight(epoch, args) if cfg["mcd"] else 0.0
        log = defaultdict(float)

        for _ in range(steps):
            batches = [s.next() for s in streams]
            xs = [b[0] for b in batches]
            ys = [b[1] for b in batches]
            sizes = [len(x) for x in xs]

            # ---- Step A: all networks on the labelled source data (+ pseudo-labels)
            model.train()
            feats = torch.split(model.G(torch.cat(xs)), sizes)
            l_src = source_loss(model, feats, ys, per_domain)
            loss_a = l_src
            if pseudo_stream is not None:
                xp, yp = pseudo_stream.next()
                with target_pass(model.G, args):
                    l_ps = pseudo_loss(model, model.G(xp), yp)
                loss_a = loss_a + args.st_weight * l_ps
                log["pseudo_mse"] += l_ps.item()
            opt_g.zero_grad(); opt_f.zero_grad()
            loss_a.backward()
            opt_g.step(); opt_f.step()
            log["src_mse"] += l_src.item()

            if w == 0.0:          # warm-up: identical to source-only training
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
            if args.monitor_target:    # debugging only: uses target labels
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


def train_run(args, cfg_name, fold, seed, device):
    cfg = CONFIGS[cfg_name]
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    per_domain = args.source_mode == "per_domain" and not cfg["oracle"]
    n_pairs = len(fold.src_X) if per_domain else 1
    model = MSDAModel(fold.src_X[0].shape[1], args.hidden_dim, n_pairs, args.K,
                      cfg["drt"], args.basis_scale).to(device)

    opt_g = torch.optim.SGD(model.G.parameters(), lr=args.lr, momentum=0.9)
    opt_f = torch.optim.SGD(model.heads.parameters(), lr=args.lr_f, momentum=0.9)
    scheds = [torch.optim.lr_scheduler.MultiStepLR(o, milestones=args.schedule, gamma=0.1)
              for o in (opt_g, opt_f)]

    tens = lambda a: torch.as_tensor(a, device=device)
    n_src = sum(len(X) for X in fold.src_X)
    if args.source_mode == "pooled" or cfg["oracle"]:
        streams = [BatchStream((tens(np.concatenate(fold.src_X)), tens(np.concatenate(fold.src_y))),
                               args.batch_size, gen)]
    else:   # balanced / per_domain: equal share of each source animal in every batch
        bs = max(2, args.batch_size // len(fold.src_X))
        streams = [BatchStream((tens(X), tens(y)), bs, gen) for X, y in zip(fold.src_X, fold.src_y)]
    steps = max(1, n_src // args.batch_size)
    # separate generators: the source batch order is identical with and without MCD/ST
    trg_gen = torch.Generator().manual_seed(seed + 1)
    trg_stream = BatchStream((tens(fold.adapt_X),), args.batch_size, trg_gen) if cfg["mcd"] else None

    run_epochs(model, opt_g, opt_f, scheds, streams, trg_stream, None,
               list(range(1, args.epochs + 1)), cfg, args, per_domain, steps,
               device, fold, "main")

    keep_rate = float("nan")
    if cfg["st"]:
        Xp, yp, keep_rate = make_pseudo_labels(eval_model(model, fold, args, device),
                                               tens(fold.adapt_X),
                                               args.st_keep_quantile, args.discrepancy)
        print(f"    [st] kept {keep_rate:.0%} of the adaptation beats as pseudo-labels")
        for o, lr in ((opt_g, args.lr), (opt_f, args.lr_f)):
            for g in o.param_groups:
                g["lr"] = lr * args.st_lr_scale
        pseudo_stream = BatchStream((Xp, yp), args.batch_size,
                                    torch.Generator().manual_seed(seed + 2))
        run_epochs(model, opt_g, opt_f, None, streams, trg_stream, pseudo_stream,
                   list(range(args.epochs + 1, args.epochs + args.st_epochs + 1)),
                   cfg, args, per_domain, steps, device, fold, "st")

    result = {"st_keep_rate": keep_rate}
    final = eval_model(model, fold, args, device)
    for split in ("test", "adapt"):
        if cfg["oracle"] and split == "adapt":
            continue   # adaptation split is the oracle's training data
        for k, v in evaluate(final, fold, split, device).items():
            result[f"{split}_{k}"] = v

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg_name,
                    "args": vars(args), "target": fold.target, "sources": fold.source_ids,
                    "y_mean": fold.y_stats.mean, "y_std": fold.y_stats.std,
                    "adapt_idx": fold.adapt_idx, "test_idx": fold.test_idx,
                    "n_pairs": n_pairs, "drt": cfg["drt"]},
                   os.path.join(args.save_dir,
                                f"{ARCH}_{args.source_mode}_{args.norm}_K{args.K}_"
                                f"{cfg_name}_{fold.target}_seed{seed}.pt"))
    return result


# =============================================================================
# Experiment loop and output
# =============================================================================

FIELDS = ["run_id", "arch", "protocol", "config", "source_mode", "norm", "target_bn", "K",
          "split", "n_blocks", "discrepancy", "num_k", "eta", "gamma", "vpen_cap",
          "warmup_epochs", "ramp_epochs", "epochs", "st_keep_quantile", "st_weight",
          "drop_features", "target",
          "sources", "seed", "test_rmse", "test_mae", "test_r2", "test_r",
          "adapt_rmse", "adapt_mae", "adapt_r2", "adapt_r", "st_keep_rate"]


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
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--basis_scale", type=float, default=0.1)
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
    p.add_argument("--out", default="results.csv")
    p.add_argument("--save_dir", default="", help="save one checkpoint per run (for SHAP/CRAFT)")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--monitor_target", action="store_true",
                   help="DEBUG ONLY: print target test RMSE during training")
    return p.parse_args()


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

    run_id = time.strftime("%Y%m%d-%H%M%S")
    print(f"run {run_id}: {ARCH}, protocol={args.protocol}, source_mode={args.source_mode}, "
          f"norm={args.norm}, target_bn={args.target_bn}, K={args.K}, split={args.split} "
          f"({args.n_blocks} blocks), gamma={args.gamma}, num_k={args.num_k}, "
          f"drop_features={args.drop_features or 'none'}, device={device}")

    total = len(configs) * len(targets) * len(seeds)
    i = 0
    for cfg_name in configs:
        for target in targets:
            fold = build_fold(args, target, sources_of(target), CONFIGS[cfg_name]["oracle"])
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
                    "target_bn": args.target_bn, "K": args.K, "split": args.split,
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