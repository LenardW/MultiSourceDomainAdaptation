"""
Dynamic Residual Transfer (DRT) for tabular regression.

Port of Li et al., "Dynamic Transfer for Multi-Source Domain Adaptation"
(CVPR 2021, arXiv:2103.10583) from image classification to per-animal
hemodynamic regression.

Reference implementation: https://github.com/liyunsheng13/DRT

DRT variants (Table 1 of the paper):
    mode = 'none'          static transfer, DeltaW(x) = 0
    mode = 'channel'       Eq. 3   DeltaW(x) = Lambda(x) W_0
    mode = 'subspace'      Eq. 4   DeltaW(x) = sum_i pi_i(x) Phi_i     <- default
    mode = 'combination'   Eq. 5   both terms

Logged metric families (all per epoch, see METRIC GLOSSARY at the bottom):
    fit/*      how well the model fits what it is trained on
    gen/*      source generalisation gap  -> overfitting
    dom/*      source-to-target gap       -> domain gap
    trg/*      target performance         -> the reported result
    mcd/*      discrepancy game health
    drt/*      routing health
    opt/*      learning rates, gradient norms, timing
"""

import argparse
import csv
import os
import random
import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def set_seed(seed):
    """Seed every RNG that touches the run.

    The original script passed `seed` only to the train/val split, so all five
    "seeds" shared identical weight initialisation and identical batch order.
    The reported +- was a split-variance estimate, not a run-variance estimate.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_csv(path):
    rows = []
    with open(path, "r") as f:
        for row in csv.reader(f):
            rows.append([float(v) for v in row])
    return np.array(rows, dtype=np.float32)


def load_data(x_path, y_path, verbose=True):
    X = load_csv(x_path)
    Y = load_csv(y_path).squeeze(1)
    if verbose:
        print(f"Loaded {len(X)} datapoints from {os.path.basename(x_path)} "
              f"| X {X.shape} Y {Y.shape} | Y range "
              f"[{Y.min():.3f}, {Y.max():.3f}] mean {Y.mean():.3f}")
    return X, Y


class Normalizer:
    """Fit on source training data only.

    The target domain is unlabelled at train time, so target statistics would
    leak information the method is not supposed to have. Fitting on the source
    *training* split (rather than all source data) keeps the source validation
    split clean as an overfitting probe.
    """

    def __init__(self):
        self.X_mean = self.X_std = None
        self.Y_mean = self.Y_std = None

    def fit(self, X, Y):
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-8
        self.Y_mean = Y.mean()
        self.Y_std = Y.std() + 1e-8
        return self

    def transform_X(self, X):
        return (X - self.X_mean) / self.X_std

    def transform_Y(self, Y):
        return (Y - self.Y_mean) / self.Y_std

    def inverse_Y(self, Y_norm):
        return Y_norm * self.Y_std + self.Y_mean


def split_domain(X, Y, val_ratio=0.2, seed=100):
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    n_val = int(n * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (X[train_idx], Y[train_idx]), (X[val_idx], Y[val_idx])


def make_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(Y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last)


# --------------------------------------------------------------------------- #
# DRT layer
# --------------------------------------------------------------------------- #

class DRTLinear(nn.Module):
    """Linear layer with a sample-dependent residual on the weight matrix.

        W(x) = W_0 + Lambda(x) W_0 + sum_i pi_i(x) Phi_i          (Eq. 5)

    Both residual terms are applied in *feature* space rather than by
    materialising a per-sample [B, out, in] weight tensor. Every operation is
    linear in x and shares the same input, so the two are identical:

        (sum_i pi_i Phi_i) x  ==  sum_i pi_i (Phi_i x)

    but the feature-space form is one GEMM instead of a batched matmul over a
    tensor that, for 256->128 at batch 64, is 2.1M floats forward and again in
    backward. This is also what the reference repo does (conv_s1..conv_s4
    applied to the activation, then blended).

    Lambda scales the weight, not the bias, so the bias is added last.
    """

    def __init__(self, in_dim, out_dim, mode="subspace", K=4, reduction=16,
                 rank=None, basis_scale=0.1, bias=True):
        super().__init__()
        assert mode in ("none", "channel", "subspace", "combination")
        self.mode = mode
        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.static = nn.Linear(in_dim, out_dim, bias=bias)
        self.last_pi = None
        self.last_lambda = None

        if mode == "none":
            return

        # Single lightweight branch feeding both heads, as in Fig. 3: the
        # classification version is avg-pool -> FC -> ReLU -> FC, with sigmoid
        # on the Lambda head and softmax on the pi head. Tabular inputs have no
        # spatial extent, so the pooling step drops out.
        hidden = max(in_dim // reduction, 4)
        self.branch = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.ReLU(inplace=True),
        )

        if mode in ("channel", "combination"):
            self.to_lambda = nn.Linear(hidden, out_dim, bias=False)

        if mode in ("subspace", "combination"):
            self.to_pi = nn.Linear(hidden, K, bias=False)
            # The K basis matrices Phi_i, packed into one Linear so all K
            # projections are a single matmul.
            if rank is None:
                self.basis = nn.Linear(in_dim, K * out_dim, bias=False)
            else:
                # Low-rank factorisation of each Phi_i: the tabular analogue of
                # the paper restricting Phi to 1x1 kernels at the narrowest
                # point of the bottleneck. A capacity cap, not a different
                # method.
                self.basis = nn.Sequential(
                    nn.Linear(in_dim, rank, bias=False),
                    nn.Linear(rank, K * out_dim, bias=False),
                )
            self._init_basis(basis_scale)

    def _init_basis(self, scale):
        """Small but non-zero.

        dL/dpi_k = <dL/dout, Phi_k x>, so with Phi_k == 0 the router receives
        exactly zero gradient on the first step. Scaling a normal init down
        keeps the layer near-static at init without blinding the router.
        """
        for m in self.basis.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                m.weight.data.mul_(scale / np.sqrt(self.K))

    def forward(self, x):
        z0 = F.linear(x, self.static.weight)  # W_0 x, bias held back

        if self.mode == "none":
            self.last_pi = None
            self.last_lambda = None
            return z0 + self.static.bias if self.static.bias is not None else z0

        h = self.branch(x)
        out = z0

        if self.mode in ("channel", "combination"):
            lam = torch.sigmoid(self.to_lambda(h))            # [B, out_dim]
            # Eq. 3: W(x) = W_0 + Lambda(x) W_0 = (I + Lambda(x)) W_0.
            # A residual boost, not SE-style gating (lam * z0 alone, which can
            # only attenuate).
            out = out + lam * z0
            self.last_lambda = lam.detach()

        if self.mode in ("subspace", "combination"):
            pi = torch.softmax(self.to_pi(h), dim=-1)          # [B, K]
            z = self.basis(x).view(-1, self.K, self.out_dim)   # [B, K, out_dim]
            out = out + (z * pi.unsqueeze(-1)).sum(dim=1)
            self.last_pi = pi.detach()

        if self.static.bias is not None:
            out = out + self.static.bias
        return out


class DynamicBackbone(nn.Module):
    """Two-layer feature extractor.

    `dynamic_layers` mirrors the paper's placement: on ResNet-101 the 7x7 stem
    stays static and residuals are added only to the 3x3 conv inside each
    bottleneck. With 14 input features a per-sample dynamic first layer is close
    to fitting a separate model per animal, so layer 1 is static by default.
    """

    def __init__(self, in_dim=14, hidden_dim=128, mode="none", K=4,
                 rank=16, reduction=16, dynamic_layers=(2,)):
        super().__init__()
        m1 = mode if 1 in dynamic_layers else "none"
        m2 = mode if 2 in dynamic_layers else "none"

        self.layer1 = DRTLinear(in_dim, 256, mode=m1, K=K, rank=rank,
                                reduction=reduction)
        self.bn1, self.relu1 = nn.BatchNorm1d(256), nn.ReLU(inplace=True)

        self.layer2 = DRTLinear(256, hidden_dim, mode=m2, K=K, rank=rank,
                                reduction=reduction)
        self.bn2, self.relu2 = nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.relu1(self.bn1(self.layer1(x)))
        h = self.relu2(self.bn2(self.layer2(h)))
        return h

    def drt_layers(self):
        return [m for m in self.modules()
                if isinstance(m, DRTLinear) and m.mode != "none"]

    def routing_layers(self):
        return [m for m in self.drt_layers()
                if m.mode in ("subspace", "combination")]

    def param_groups(self, lr):
        """Static params at lr, residual-path params at 10x lr.

        The paper gives newly introduced parameters 10x the base learning rate
        (`get_10xparams` in the repo). There the split is "present in the
        ImageNet checkpoint or not"; here the equivalent is static vs dynamic,
        since there is no pretrained dynamic backbone to inherit.
        """
        dyn, static = [], []
        for name, p in self.named_parameters():
            is_dyn = any(k in name for k in
                         (".branch.", ".to_pi.", ".to_lambda.", ".basis."))
            (dyn if is_dyn else static).append(p)
        groups = [{"params": static, "lr": lr}]
        if dyn:
            groups.append({"params": dyn, "lr": 10 * lr})
        return groups


class RegressorHead(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

def discrepancy_loss(pred1, pred2, kind="bounded"):
    """MCD's L1 discrepancy, adapted to regression.

    In classification both heads output a simplex point, so |p1 - p2| is bounded
    by 2 and maximising it is safe. Two regression heads can separate without
    limit, so the raw L1 version lets Step B win by pushing the heads apart
    forever. `bounded` saturates at 1 and keeps the game finite.
    """
    d = torch.abs(pred1 - pred2)
    if kind == "l1":
        return d.mean()
    return torch.mean(1.0 - torch.exp(-d))


def variance_penalty(pred1, pred2):
    """Negative output variance. Minimising it stops the heads collapsing to a
    constant prediction, which is the degenerate way to make the discrepancy
    small in Step C."""
    return -(torch.var(pred1) + torch.var(pred2))


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(G, F1, F2, loader, device, normalizer):
    """Metrics for one split, plus two health signals.

    The heads are averaged. The repo's `test()` reports the max-logit selection
    over F1 and F2 as its headline number, so combining both heads is the
    paper's own protocol rather than an extra advantage; regression has no
    argmax, so the mean is the analogue.

    Returns a dict:
        mae, rmse, r2, corr   standard metrics on the original scale
        disagree              mean |F1 - F2| in normalised units. This is the
                              quantity MCD minimises, measured out of sample.
        pred_std_ratio        std(pred) / std(target). Near 0 means the model
                              has collapsed to predicting the mean, which can
                              produce a deceptively small MAE.
    """
    G.eval(); F1.eval(); F2.eval()
    p1s, p2s, targets = [], [], []
    for X_b, Y_b in loader:
        feat = G(X_b.to(device))
        p1s.append(F1(feat).cpu()); p2s.append(F2(feat).cpu())
        targets.append(Y_b)

    p1 = torch.cat(p1s).numpy().ravel()
    p2 = torch.cat(p2s).numpy().ravel()
    preds = (p1 + p2) / 2.0
    targets = torch.cat(targets).numpy().ravel()

    preds_o = normalizer.inverse_Y(preds)
    targets_o = normalizer.inverse_Y(targets)

    ss_res = np.sum((targets_o - preds_o) ** 2)
    ss_tot = np.sum((targets_o - targets_o.mean()) ** 2)

    return {
        "mae": float(np.mean(np.abs(preds_o - targets_o))),
        "rmse": float(np.sqrt(np.mean((preds_o - targets_o) ** 2))),
        "r2": float(1 - ss_res / (ss_tot + 1e-8)),
        "corr": float(pearsonr(targets, preds)[0]),
        "disagree": float(np.mean(np.abs(p1 - p2))),
        "pred_std_ratio": float(preds.std() / (targets.std() + 1e-8)),
        # normalised-scale MSE, directly comparable with the running training
        # loss. A large gap between the two on the same data means the
        # BatchNorm running statistics disagree with the batch statistics.
        "mse_norm": float(np.mean((preds - targets) ** 2)),
    }


# --------------------------------------------------------------------------- #
# Routing diagnostics
# --------------------------------------------------------------------------- #

@torch.no_grad()
def collect_pi(G, loader, device):
    """Concatenate pi across all routing layers for every sample.

    This is the paper's Pi = {pi_i^l(x)} vector (Sec. 4.5) used for the t-SNE in
    Fig. 4. Returns [N, n_routing_layers * K], or None in static/channel mode.
    """
    layers = G.routing_layers()
    if not layers:
        return None
    was_training = G.training
    G.eval()
    chunks = []
    for X_b, _ in loader:
        G(X_b.to(device))
        chunks.append(torch.cat([l.last_pi for l in layers], dim=1).cpu())
    if was_training:
        G.train()
    return torch.cat(chunks).numpy()


def routing_entropy(pi, K):
    """Mean per-sample entropy of pi, normalised to [0, 1].

    1.0 = uniform blend for every sample, the router carries no information.
    0.0 = hard one-hot routing.
    """
    pi = np.clip(pi.reshape(-1, K), 1e-8, 1.0)
    return float(-(pi * np.log(pi)).sum(axis=1).mean() / np.log(K))


def routing_spread(pi, K):
    """Std of pi across samples, averaged over the K coordinates.

    Complements entropy. A router could route confidently but identically for
    every sample (low entropy, zero spread) -- that is still a static layer.
    Only spread > 0 means routing is actually sample-dependent.
    """
    return float(pi.reshape(-1, K).std(axis=0).mean())


def routing_usage(pi, K):
    """Batch-mean pi per basis, i.e. how much each Phi_i is used overall.

    A basis whose usage decays to ~0 is dead: its gradient is proportional to
    pi_k, so once unused it stops learning and K is effectively smaller.
    """
    return pi.reshape(-1, K).mean(axis=0)


@torch.no_grad()
def basis_similarity(layer):
    """Mean pairwise |cosine| between the K basis matrices Phi_i.

    Nothing in DRT's loss encourages the Phi_i to differ. If they converge,
    sum_i pi_i Phi_i is constant in pi and routing silently degrades to a static
    layer with K times the parameters. Values near 1 mean collapse.
    """
    if layer.mode not in ("subspace", "combination"):
        return None
    if isinstance(layer.basis, nn.Sequential):
        W = layer.basis[1].weight @ layer.basis[0].weight
    else:
        W = layer.basis.weight
    W = W.view(layer.K, layer.out_dim, layer.in_dim).flatten(1)
    W = F.normalize(W, dim=1)
    sim = (W @ W.t()).abs()
    off = sim[~torch.eye(layer.K, dtype=torch.bool, device=sim.device)]
    return float(off.mean())


@torch.no_grad()
def residual_magnitude(layer):
    """||DeltaW|| / ||W_0||, the share of the weight that is dynamic.

    Near 0 means the residual path is inert regardless of what pi does; large
    values mean the static anchor has stopped mattering. Uses the mean routing
    coefficient, so this is the average-case correction, not a worst case.
    """
    if layer.mode not in ("subspace", "combination"):
        return None
    if isinstance(layer.basis, nn.Sequential):
        W = layer.basis[1].weight @ layer.basis[0].weight
    else:
        W = layer.basis.weight
    Phi = W.view(layer.K, layer.out_dim, layer.in_dim)
    delta = Phi.mean(dim=0)  # uniform pi as the reference point
    return float(delta.norm() / (layer.static.weight.norm() + 1e-8))


@torch.no_grad()
def lambda_stats(layer):
    """Mean and std of the most recent Lambda(x).

    Mean near 0 means the channel term is off (W -> W_0); mean near 1 means it
    is doubling the weight uniformly, which is a rescaling BatchNorm would undo
    anyway. Std is what makes it sample-dependent.
    """
    if layer.mode not in ("channel", "combination") or layer.last_lambda is None:
        return None
    lam = layer.last_lambda
    return float(lam.mean()), float(lam.std(dim=0).mean())


def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().norm()) ** 2
    return total ** 0.5


class frozen_bn_stats:
    """Normalise with batch statistics but stop updating the running averages.

    Steps B and C push data through G several times per batch (once for source
    and once for target in B, then num_k times for target in C). In train mode
    every one of those forward passes updates the BatchNorm running statistics,
    so with num_k=4 the running stats are fitted on roughly 5 parts target to 2
    parts source -- an unintended AdaBN effect whose strength depends on num_k.
    It would improve target metrics on its own and confound every ablation row.

    Setting momentum to 0 leaves the normalisation maths untouched (batch
    statistics are still used, so the discrepancy is computed exactly as MCD
    specifies) while freezing accumulation, so the running stats come only from
    the supervised Step A pass. `--bn_update all` restores the old behaviour as
    an explicit ablation.
    """

    def __init__(self, module, active=True):
        self.active = active
        self.bns = [m for m in module.modules()
                    if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        self.saved = []

    def __enter__(self):
        if self.active:
            self.saved = [bn.momentum for bn in self.bns]
            for bn in self.bns:
                bn.momentum = 0.0
        return self

    def __exit__(self, *exc):
        if self.active:
            for bn, mom in zip(self.bns, self.saved):
                bn.momentum = mom
        return False


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def log_epoch_diagnostics(G, args, val_loader, device, log):
    """Routing / channel diagnostics, added in place to `log`."""
    layers = G.drt_layers()
    if not layers:
        return

    pi = collect_pi(G, val_loader, device)
    if pi is not None:
        log["drt/pi_entropy"] = routing_entropy(pi, args.K)
        log["drt/pi_spread"] = routing_spread(pi, args.K)
        usage = routing_usage(pi, args.K)
        log["drt/pi_usage_min"] = float(usage.min())
        log["drt/pi_usage_max"] = float(usage.max())
        for k in range(args.K):
            log[f"drt/pi_usage_{k}"] = float(usage[k])

    for i, layer in enumerate(layers):
        sim = basis_similarity(layer)
        if sim is not None:
            log[f"drt/basis_sim_l{i}"] = sim
        mag = residual_magnitude(layer)
        if mag is not None:
            log[f"drt/residual_frac_l{i}"] = mag
        ls = lambda_stats(layer)
        if ls is not None:
            log[f"drt/lambda_mean_l{i}"], log[f"drt/lambda_std_l{i}"] = ls


def run_epochs(G, F1, F2, opt_g, opt_f, sch_g, sch_f, loaders, normalizer,
               device, args, n_epochs, epoch_offset, use_mcd, phase,
               pseudo_loader=None, best=None):
    """One training phase.

    Metric names do not change between phase 1 and the self-training phase, so
    every curve in MLflow is continuous across the whole run; `opt/phase` marks
    where the second round starts. (Prefixing the ST metrics would split each
    curve into two disconnected series.)
    """
    src_loader = loaders["src_train"]
    trg_loader = loaders["trg_train"]

    metrics = None
    freeze_bn = (args.bn_update == "source")
    trg_iter = iter(trg_loader) if use_mcd else None
    pseudo_iter = iter(pseudo_loader) if pseudo_loader is not None else None
    head_params = list(F1.parameters()) + list(F2.parameters())

    for ep in range(1, n_epochs + 1):
        gstep = epoch_offset + ep
        t0 = time.time()
        G.train(); F1.train(); F2.train()

        # Linear ramp on the domain-adaptation terms. Not in the paper, which
        # activates alignment from step 0; kept because with a few thousand
        # loops per animal the discrepancy game destabilises an untrained
        # backbone. Set --warmup_epochs 0 for the paper's schedule.
        if gstep <= args.warmup_epochs:
            da_w = 0.0
        else:
            da_w = min(1.0, (gstep - args.warmup_epochs) / max(args.da_ramp, 1))

        acc = {"mse1": 0.0, "mse2": 0.0, "st": 0.0, "n": 0,
               "dis_t": 0.0, "dis_s": 0.0, "gn_g": 0.0, "gn_f": 0.0, "nb": 0}

        for X_s, Y_s in src_loader:
            bsz = X_s.size(0)
            X_s, Y_s = X_s.to(device), Y_s.to(device)

            X_t = None
            if use_mcd:
                try:
                    X_t, _ = next(trg_iter)
                except StopIteration:
                    trg_iter = iter(trg_loader)
                    X_t, _ = next(trg_iter)
                X_t = X_t.to(device)

            # ---- Step A: supervised risk on the pooled source domains ----
            feat_s = G(X_s)
            loss_s1 = F.mse_loss(F1(feat_s), Y_s)
            loss_s2 = F.mse_loss(F2(feat_s), Y_s)
            loss_a = loss_s1 + loss_s2

            # Pseudo-labelled target samples join the source loss, as the paper
            # describes: "used to train DRT again with source samples".
            if pseudo_iter is not None:
                try:
                    Xp, Yp = next(pseudo_iter)
                except StopIteration:
                    pseudo_iter = iter(pseudo_loader)
                    Xp, Yp = next(pseudo_iter)
                if Xp.size(0) > 1:
                    Xp, Yp = Xp.to(device), Yp.to(device)
                    feat_p = G(Xp)
                    l_st = F.mse_loss(F1(feat_p), Yp) + F.mse_loss(F2(feat_p), Yp)
                    loss_a = loss_a + args.st_weight * l_st
                    acc["st"] += float(l_st) * bsz

            opt_g.zero_grad(); opt_f.zero_grad()
            loss_a.backward()
            acc["gn_g"] += grad_norm(G.parameters())
            acc["gn_f"] += grad_norm(head_params)
            opt_g.step(); opt_f.step()

            acc["mse1"] += loss_s1.item() * bsz
            acc["mse2"] += loss_s2.item() * bsz
            acc["n"] += bsz
            acc["nb"] += 1

            if not use_mcd:
                continue

            # ---- Step B: heads maximise target discrepancy ----
            # Features are detached, so no gradient can reach G and no
            # requires_grad juggling is needed here.
            with frozen_bn_stats(G, freeze_bn), torch.no_grad():
                feat_s_d = G(X_s)
                feat_t_d = G(X_t)

            pt1, pt2 = F1(feat_t_d), F2(feat_t_d)
            dis = discrepancy_loss(pt1, pt2, args.dis_loss)
            vpen = variance_penalty(pt1, pt2)
            ps1, ps2 = F1(feat_s_d), F2(feat_s_d)
            loss_b = (F.mse_loss(ps1, Y_s) + F.mse_loss(ps2, Y_s)
                      + da_w * (args.gamma * vpen - args.eta * dis))

            opt_f.zero_grad()
            loss_b.backward()
            opt_f.step()

            # Discrepancy on source is the control: it should stay low. If it
            # rises with the target discrepancy, the heads are separating
            # everywhere rather than isolating target samples outside the
            # source support, and the MCD signal is meaningless.
            acc["dis_s"] += float(discrepancy_loss(ps1, ps2, args.dis_loss).detach())

            # ---- Step C: backbone minimises target discrepancy, num_k times --
            # MCD repeats the generator update per batch (num_k = 4 in Saito et
            # al.); a single step lets the heads outrun the backbone.
            for p in head_params:
                p.requires_grad_(False)
            with frozen_bn_stats(G, freeze_bn):
                for _ in range(args.num_k):
                    feat_t = G(X_t)
                    p1, p2 = F1(feat_t), F2(feat_t)
                    loss_c = da_w * (
                        args.eta * discrepancy_loss(p1, p2, args.dis_loss)
                        + args.gamma * variance_penalty(p1, p2))
                    opt_g.zero_grad()
                    loss_c.backward()
                    opt_g.step()
                with torch.no_grad():
                    feat_t = G(X_t)
                    acc["dis_t"] += float(discrepancy_loss(
                        F1(feat_t), F2(feat_t), args.dis_loss))
            for p in head_params:
                p.requires_grad_(True)

        sch_g.step(); sch_f.step()
        n, nb = max(acc["n"], 1), max(acc["nb"], 1)

        # ------------------------------------------------------------------ #
        # Per-epoch evaluation on every split
        # ------------------------------------------------------------------ #
        do_eval = (gstep % args.eval_every == 0) or (ep == n_epochs)
        if not do_eval:
            continue

        m_src_tr = evaluate(G, F1, F2, loaders["src_train_eval"], device, normalizer)
        m_src_va = evaluate(G, F1, F2, loaders["src_val"], device, normalizer)
        m_trg_va = evaluate(G, F1, F2, loaders["trg_val"], device, normalizer)
        m_trg_tr = (evaluate(G, F1, F2, loaders["trg_train_eval"], device, normalizer)
                    if loaders.get("trg_train_eval") is not None else None)
        metrics = m_trg_va

        log = {
            # running training loss (in normalised units, as optimised)
            "fit/mse_head1": acc["mse1"] / n,
            "fit/mse_head2": acc["mse2"] / n,
            # in-sample fit
            "fit/src_train_mse_eval": m_src_tr["mse_norm"],
            "fit/bn_eval_gap": (m_src_tr["mse_norm"]
                                - 0.5 * (acc["mse1"] + acc["mse2"]) / n),
            "fit/src_train_r2": m_src_tr["r2"],
            "fit/src_train_rmse": m_src_tr["rmse"],
            # source generalisation -> overfitting
            "gen/src_val_r2": m_src_va["r2"],
            "gen/src_val_rmse": m_src_va["rmse"],
            "gen/gap_r2": m_src_tr["r2"] - m_src_va["r2"],
            "gen/gap_rmse": m_src_va["rmse"] - m_src_tr["rmse"],
            # source -> target -> domain gap
            "dom/gap_r2": m_src_va["r2"] - m_trg_va["r2"],
            "dom/gap_rmse": m_trg_va["rmse"] - m_src_va["rmse"],
            # the reported result
            "trg/r2": m_trg_va["r2"],
            "trg/mae": m_trg_va["mae"],
            "trg/rmse": m_trg_va["rmse"],
            "trg/corr": m_trg_va["corr"],
            "trg/pred_std_ratio": m_trg_va["pred_std_ratio"],
            # MCD health
            "mcd/da_weight": da_w,
            "mcd/disagree_trg_val": m_trg_va["disagree"],
            "mcd/disagree_src_val": m_src_va["disagree"],
            # optimisation
            "opt/lr_g": opt_g.param_groups[0]["lr"],
            "opt/lr_f": opt_f.param_groups[0]["lr"],
            "opt/grad_norm_g": acc["gn_g"] / nb,
            "opt/grad_norm_f": acc["gn_f"] / nb,
            "opt/epoch_seconds": time.time() - t0,
            "opt/phase": phase,
        }
        if acc["dis_t"]:
            log["mcd/discrepancy_trg"] = acc["dis_t"] / nb
            log["mcd/discrepancy_src"] = acc["dis_s"] / nb
        if pseudo_iter is not None:
            log["fit/mse_pseudo"] = acc["st"] / n
        if m_trg_tr is not None:
            # Target-train labels are held out of every loss and used here only
            # as a diagnostic. The gap against trg/r2 separates "the target
            # domain is hard" from "this validation split is hard".
            log["trg/train_r2"] = m_trg_tr["r2"]
            log["trg/split_gap_r2"] = m_trg_tr["r2"] - m_trg_va["r2"]

        if gstep % args.pi_log_every == 0 or ep == n_epochs:
            log_epoch_diagnostics(G, args, loaders["trg_val"], device, log)

        # Running best, so the curve shows the ceiling reached so far rather
        # than only the final epoch.
        if best is not None:
            if m_trg_va["r2"] > best["r2"]:
                best.update(r2=m_trg_va["r2"], corr=m_trg_va["corr"],
                            rmse=m_trg_va["rmse"], mae=m_trg_va["mae"],
                            epoch=gstep)
            log["trg/r2_best_so_far"] = best["r2"]

        mlflow.log_metrics(log, step=gstep)

        print(f"  ep {gstep:03d} da={da_w:.2f} "
              f"src_tr={m_src_tr['r2']:+.3f} src_va={m_src_va['r2']:+.3f} "
              f"(gap {log['gen/gap_r2']:+.3f}) | "
              f"trg={m_trg_va['r2']:+.3f} rmse={m_trg_va['rmse']:.3f} "
              f"corr={m_trg_va['corr']:+.3f}")

    return metrics


@torch.no_grad()
def make_pseudo_labels(G, F1, F2, X_trg, device, args):
    """One-shot pseudo-labelling of the unlabelled target split.

    The paper runs self-training as a second training round after DRT has
    converged, not as an online per-batch step: pseudo-labels above a confidence
    threshold train DRT again together with source samples. Regression has no
    softmax confidence, so head agreement stands in -- which is also what MCD
    already optimises, making it the natural proxy.

    The threshold is a quantile of the observed discrepancy rather than an
    absolute value, so the keep rate is stable across animals and scales.
    """
    G.eval(); F1.eval(); F2.eval()
    Xt = torch.tensor(X_trg, dtype=torch.float32).to(device)
    p1s, p2s = [], []
    for i in range(0, len(Xt), 512):
        feat = G(Xt[i:i + 512])
        p1s.append(F1(feat)); p2s.append(F2(feat))
    p1, p2 = torch.cat(p1s), torch.cat(p2s)

    pseudo = ((p1 + p2) / 2.0).cpu().numpy()
    disc = torch.abs(p1 - p2).cpu().numpy()
    thr = float(np.quantile(disc, args.st_keep_quantile))
    keep = disc <= thr
    return pseudo, keep, thr


def train(args, run_name=None):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_X, target_Y = load_data(
        os.path.join(args.data_dict_path, args.target_x_dict),
        os.path.join(args.data_dict_path, args.target_y_dict),
        verbose=args.verbose)

    sources = []
    for xs, ys in zip(args.source_x_dicts.split(","),
                      args.source_y_dicts.split(",")):
        sources.append(load_data(os.path.join(args.data_dict_path, xs),
                                 os.path.join(args.data_dict_path, ys),
                                 verbose=args.verbose))

    # Source domains are pooled with no domain labels anywhere: the point of
    # dynamic transfer is that it "turns multi-source domains into a
    # single-source domain".
    X_src = np.concatenate([X for X, _ in sources], axis=0)
    Y_src = np.concatenate([Y for _, Y in sources], axis=0)

    # Source is split too, so overfitting is separable from the domain gap.
    (src_tr_X, src_tr_Y), (src_va_X, src_va_Y) = split_domain(
        X_src, Y_src, val_ratio=args.src_val_ratio, seed=args.seed)

    normalizer = Normalizer().fit(src_tr_X, src_tr_Y)
    src_tr_Xn = normalizer.transform_X(src_tr_X)
    src_tr_Yn = normalizer.transform_Y(src_tr_Y)
    src_va_Xn = normalizer.transform_X(src_va_X)
    src_va_Yn = normalizer.transform_Y(src_va_Y)

    trg_Xn = normalizer.transform_X(target_X)
    trg_Yn = normalizer.transform_Y(target_Y)
    (trg_tr_X, trg_tr_Y), (trg_va_X, trg_va_Y) = split_domain(
        trg_Xn, trg_Yn, val_ratio=args.val_ratio, seed=args.seed)

    # trg_tr_Y is never used in any loss. It feeds trg_train_eval only.
    trg_tr_Y_unused = np.zeros(len(trg_tr_X), dtype=np.float32)

    bs = args.batch_size
    loaders = {
        "src_train": make_loader(src_tr_Xn, src_tr_Yn, bs),
        "src_train_eval": make_loader(src_tr_Xn, src_tr_Yn, bs,
                                      shuffle=False, drop_last=False),
        "src_val": make_loader(src_va_Xn, src_va_Yn, bs,
                               shuffle=False, drop_last=False),
        "trg_train": make_loader(trg_tr_X, trg_tr_Y_unused, bs),
        "trg_val": make_loader(trg_va_X, trg_va_Y, bs,
                               shuffle=False, drop_last=False),
        "trg_train_eval": (make_loader(trg_tr_X, trg_tr_Y, bs, shuffle=False,
                                       drop_last=False)
                           if args.diag_target_train else None),
    }

    G = DynamicBackbone(in_dim=args.input_dim, hidden_dim=args.hidden_dim,
                        mode=args.drt_mode, K=args.K, rank=args.rank,
                        reduction=args.reduction,
                        dynamic_layers=tuple(args.dynamic_layers)).to(device)
    F1 = RegressorHead(args.hidden_dim).to(device)
    F2 = RegressorHead(args.hidden_dim).to(device)

    opt_g = optim.SGD(G.param_groups(args.lr), lr=args.lr, momentum=0.9,
                      weight_decay=args.weight_decay)
    opt_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()),
                      lr=args.lr_f, momentum=0.9, weight_decay=args.weight_decay)
    sch_g = optim.lr_scheduler.MultiStepLR(opt_g, args.schedule, gamma=0.1)
    sch_f = optim.lr_scheduler.MultiStepLR(opt_f, args.schedule, gamma=0.1)

    best = {"r2": -np.inf, "corr": np.nan, "rmse": np.nan, "mae": np.nan,
            "epoch": -1}

    with mlflow.start_run(run_name=run_name, nested=args.nested_runs):
        mlflow.log_params({k: (str(v) if isinstance(v, (list, tuple)) else v)
                           for k, v in vars(args).items()})
        n_static = sum(p.numel() for n_, p in G.named_parameters()
                       if n_.startswith(("layer1.static", "layer2.static", "bn")))
        mlflow.log_params({
            "n_params_G": sum(p.numel() for p in G.parameters()),
            "n_params_G_static": n_static,
            "n_src_train": len(src_tr_Xn), "n_src_val": len(src_va_Xn),
            "n_trg_train": len(trg_tr_X), "n_trg_val": len(trg_va_X),
        })

        metrics = run_epochs(G, F1, F2, opt_g, opt_f, sch_g, sch_f, loaders,
                             normalizer, device, args, args.epochs, 0,
                             use_mcd=args.MCD_activated, phase=0, best=best)

        # ---- Self-training: second round, as in the paper ----
        if args.ST_activated:
            pseudo_all, keep, thr = make_pseudo_labels(G, F1, F2, trg_tr_X,
                                                       device, args)
            Xp, Yp = trg_tr_X[keep], pseudo_all[keep]
            keep_rate = float(keep.mean())
            st_log = {"st/keep_rate": keep_rate, "st/threshold": thr,
                      "st/n_pseudo": float(len(Xp))}

            # Pseudo-label quality against the held-out target train labels.
            # Measurement only -- these never enter a loss. If the kept subset
            # is no more accurate than the discarded one, head agreement is not
            # tracking correctness and any ST gain is confirmation bias.
            if args.diag_target_train and keep.sum() > 1 and (~keep).sum() > 1:
                err = np.abs(pseudo_all - trg_tr_Y) * normalizer.Y_std
                st_log["st/pseudo_mae_kept"] = float(err[keep].mean())
                st_log["st/pseudo_mae_dropped"] = float(err[~keep].mean())
                st_log["st/selection_gain"] = float(err[~keep].mean()
                                                    - err[keep].mean())
                ss_res = np.sum((trg_tr_Y[keep] - pseudo_all[keep]) ** 2)
                ss_tot = np.sum((trg_tr_Y[keep] - trg_tr_Y[keep].mean()) ** 2)
                st_log["st/pseudo_r2_kept"] = float(1 - ss_res / (ss_tot + 1e-8))

            mlflow.log_metrics(st_log)
            print(f"  [ST] kept {keep_rate:.1%} of target samples "
                  f"({len(Xp)} pseudo-labelled, thr={thr:.4f})")

            if len(Xp) > args.batch_size:
                pseudo_loader = make_loader(Xp, Yp, args.batch_size,
                                            shuffle=True, drop_last=True)
                for g in opt_g.param_groups:
                    g["lr"] = g["lr"] * args.st_lr_scale
                for g in opt_f.param_groups:
                    g["lr"] = g["lr"] * args.st_lr_scale
                sch_g = optim.lr_scheduler.MultiStepLR(opt_g, [], gamma=0.1)
                sch_f = optim.lr_scheduler.MultiStepLR(opt_f, [], gamma=0.1)
                metrics = run_epochs(G, F1, F2, opt_g, opt_f, sch_g, sch_f,
                                     loaders, normalizer, device, args,
                                     args.st_epochs, args.epochs,
                                     use_mcd=args.MCD_activated, phase=1,
                                     pseudo_loader=pseudo_loader, best=best)

        # ---- dump pi for the Fig. 4 style t-SNE ----
        if args.pi_dump_dir and G.routing_layers():
            os.makedirs(args.pi_dump_dir, exist_ok=True)
            pi_val = collect_pi(G, loaders["trg_val"], device)
            pi_src = collect_pi(G, loaders["src_val"], device)
            if pi_val is not None:
                tag = (f"{args.target_x_dict.replace('.csv', '')}"
                       f"_{args.drt_mode}_seed{args.seed}")
                path = os.path.join(args.pi_dump_dir, f"pi_{tag}.npz")
                np.savez_compressed(path, pi_target=pi_val, pi_source=pi_src,
                                    K=args.K)
                mlflow.log_artifact(path)

        # Final and best, side by side. The paper's numbers are best-epoch
        # (`best_acc` in the repo); reporting final as well makes it clear
        # whether the gap is large enough to matter.
        mlflow.log_metrics({
            "final/r2": metrics["r2"], "final/corr": metrics["corr"],
            "final/rmse": metrics["rmse"], "final/mae": metrics["mae"],
            "best/r2": best["r2"], "best/corr": best["corr"],
            "best/rmse": best["rmse"], "best/mae": best["mae"],
            "best/epoch": float(best["epoch"]),
        })

    if args.report == "best":
        return best["r2"], best["corr"]
    return metrics["r2"], metrics["corr"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="DRT for tabular regression")
    p.add_argument("--data_dict_path", type=str,
                   default="C:/Users/leona/PythonScripts/MasterThesis/Tierdaten/")
    p.add_argument("--target_x_dict", type=str, default="X_anim1.csv")
    p.add_argument("--target_y_dict", type=str, default="Y_anim1.csv")
    p.add_argument("--source_x_dicts", type=str,
                   default="X_anim10.csv,X_anim3.csv,X_anim4.csv,X_anim8.csv")
    p.add_argument("--source_y_dicts", type=str,
                   default="Y_anim10.csv,Y_anim3.csv,Y_anim4.csv,Y_anim8.csv")

    # optimisation
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--warmup_epochs", type=int, default=100)
    p.add_argument("--da_ramp", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lr_f", type=float, default=0.005)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--schedule", type=int, nargs="+", default=[110, 135, 145])
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--src_val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=100)

    # architecture
    p.add_argument("--input_dim", type=int, default=14)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--drt_mode", type=str, default="none",
                   choices=["none", "channel", "subspace", "combination"])
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--rank", type=int, default=16,
                   help="low-rank factorisation of Phi_i; -1 for full rank")
    p.add_argument("--reduction", type=int, default=16)
    p.add_argument("--dynamic_layers", type=int, nargs="+", default=[2])

    # MCD
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.001)
    p.add_argument("--num_k", type=int, default=4)
    p.add_argument("--dis_loss", type=str, default="bounded",
                   choices=["bounded", "l1"])
    p.add_argument("--bn_update", type=str, default="source",
                   choices=["source", "all"],
                   help="'source': BatchNorm running stats accumulate only in "
                        "the supervised step. 'all': every forward pass "
                        "updates them, an implicit AdaBN whose strength "
                        "depends on num_k.")

    # self-training
    p.add_argument("--st_epochs", type=int, default=30)
    p.add_argument("--st_keep_quantile", type=float, default=0.5)
    p.add_argument("--st_weight", type=float, default=1.0)
    p.add_argument("--st_lr_scale", type=float, default=0.1)

    # switches
    p.add_argument("--MCD_activated", type=str2bool, default=False)
    p.add_argument("--DRT_activated", type=str2bool, default=False)
    p.add_argument("--ST_activated", type=str2bool, default=False)

    # logging / reporting
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--pi_log_every", type=int, default=5)
    p.add_argument("--pi_dump_dir", type=str, default="pi_dumps")
    p.add_argument("--diag_target_train", type=str2bool, default=True,
                   help="evaluate on the target train split for diagnostics; "
                        "these labels never enter any loss")
    p.add_argument("--report", type=str, default="final",
                   choices=["final", "best"])
    p.add_argument("--nested_runs", type=str2bool, default=True)
    p.add_argument("--verbose", type=str2bool, default=False)
    p.add_argument("--experiment", type=str, default="DRT_multisource")
    p.add_argument("--results_file", type=str, default="drt_training_results.txt")

    a = p.parse_args()
    if a.rank is not None and a.rank < 0:
        a.rank = None
    return a


def main():
    args = parse_args()
    mlflow.set_experiment(args.experiment)

    seeds = [100, 200, 74, 15, 984]

    x_dicts = [args.target_x_dict] + args.source_x_dicts.split(",")
    y_dicts = [args.target_y_dict] + args.source_y_dicts.split(",")
    animal_pool = list(zip(x_dicts, y_dicts))

    # Rows 3-5 reproduce Table 1 of the paper on this data.
    configs = [
        {"MCD_activated": False, "drt_mode": "none", "ST_activated": False},
        {"MCD_activated": True, "drt_mode": "none", "ST_activated": False},
        {"MCD_activated": True, "drt_mode": "channel", "ST_activated": False},
        {"MCD_activated": True, "drt_mode": "subspace", "ST_activated": False},
        {"MCD_activated": True, "drt_mode": "combination", "ST_activated": False},
        {"MCD_activated": True, "drt_mode": "subspace", "ST_activated": True},
    ]

    for config in configs:
        args.MCD_activated = config["MCD_activated"]
        args.drt_mode = config["drt_mode"]
        args.DRT_activated = config["drt_mode"] != "none"
        args.ST_activated = config["ST_activated"]

        summary, all_r2, all_corr = {}, [], []
        cfg_name = (f"cfg_{config['drt_mode']}_mcd{int(args.MCD_activated)}"
                    f"_st{int(args.ST_activated)}")

        with mlflow.start_run(run_name=cfg_name):
            mlflow.log_params(config)

            for target_x, target_y in animal_pool:
                name = target_x.replace("X_", "").replace(".csv", "")
                srcs = [p for p in animal_pool if p != (target_x, target_y)]
                args.target_x_dict, args.target_y_dict = target_x, target_y
                args.source_x_dicts = ",".join(p[0] for p in srcs)
                args.source_y_dicts = ",".join(p[1] for p in srcs)

                r2s, corrs = [], []
                for s in seeds:
                    args.seed = s
                    print(f"[{cfg_name}] target={name} seed={s}")
                    r2, corr = train(args, run_name=f"{name}_seed{s}")
                    r2s.append(r2); corrs.append(corr)

                summary[name] = (float(np.mean(r2s)), float(np.std(r2s, ddof=1)),
                                 float(np.mean(corrs)),
                                 float(np.std(corrs, ddof=1)))
                all_r2.append(np.mean(r2s)); all_corr.append(np.mean(corrs))
                print(f"Finished target {name}: R2 {summary[name][0]:.4f} "
                      f"+- {summary[name][1]:.4f}")

            mlflow.log_metrics({"mean_R2": float(np.mean(all_r2)),
                                "mean_Corr": float(np.mean(all_corr))})

        with open(args.results_file, "a", encoding="utf-8") as f:
            f.write(f"Results for configuration: {config}\n" + "-" * 40 + "\n")
            for name, (m_r2, s_r2, m_c, s_c) in summary.items():
                f.write(f"Target: {name}\n R2: {m_r2:.4f} +- {s_r2:.4f}\n"
                        f" Correlation: {m_c:.4f} +- {s_c:.4f}\n" + "-" * 40 + "\n")
            f.write(f"Average across all targets:\n"
                    f" R2: {np.mean(all_r2):.4f} +- {np.std(all_r2, ddof=1):.4f}\n"
                    f" Correlation: {np.mean(all_corr):.4f} "
                    f"+- {np.std(all_corr, ddof=1):.4f}\n" + "=" * 40 + "\n\n")


# --------------------------------------------------------------------------- #
# METRIC GLOSSARY
#
# fit/mse_head1,2       running source MSE during the epoch (normalised units)
# fit/src_train_r2      R2 on the source training split, eval mode
# fit/mse_pseudo        MSE on pseudo-labels (ST phase only)
#
# fit/bn_eval_gap       eval-mode source-train MSE minus the running train-mode
#                       MSE, same data, same units. Large positive values mean
#                       the BatchNorm running statistics disagree with the batch
#                       statistics -- usually too few epochs, or target data
#                       leaking into the running stats (see --bn_update).
#
# gen/src_val_r2        R2 on held-out source data
# gen/gap_r2            src_train_r2 - src_val_r2. Overfitting, in-domain.
#                       Rises when Phi capacity is too large for the data.
#
# dom/gap_r2            src_val_r2 - trg_r2. The domain gap proper. This is
#                       what DRT + MCD are supposed to shrink. If gen/gap_r2 is
#                       flat and dom/gap_r2 falls, the method is working; if
#                       dom/gap_r2 is flat and gen/gap_r2 grows, the DRT gain
#                       is capacity, not transfer.
#
# trg/r2, mae, rmse, corr    the reported result (target validation split)
# trg/train_r2          same model on the target *training* split, whose labels
#                       are never used in any loss. Diagnostic only.
# trg/split_gap_r2      trg/train_r2 - trg/r2. Large values mean the 20% split
#                       is unrepresentative, not that adaptation failed.
# trg/pred_std_ratio    std(pred)/std(target). Near 0 = collapsed to the mean.
# trg/r2_best_so_far    running maximum, the repo's `best_acc` equivalent.
#
# mcd/da_weight         the warmup ramp
# mcd/discrepancy_trg   Step C discrepancy on target, should fall
# mcd/discrepancy_src   same on source, the control: should stay low. Both
#                       rising means the heads separate everywhere and the MCD
#                       signal is meaningless.
# mcd/disagree_*        out-of-sample |F1 - F2|
#
# drt/pi_entropy        normalised entropy of pi; 1.0 = uninformative router
# drt/pi_spread         std of pi across samples; 0 = not sample-dependent
# drt/pi_usage_k        mean pi_k; a coordinate decaying to 0 is a dead basis
# drt/basis_sim_lN      mean pairwise |cos| between Phi_i; ->1 means collapse
# drt/residual_frac_lN  ||DeltaW|| / ||W_0||; ~0 means the dynamic path is inert
# drt/lambda_mean,std   channel-attention term, when active
#
# st/keep_rate          fraction of target samples pseudo-labelled
# st/pseudo_mae_kept    error of the pseudo-labels actually used, in original
#                       units, scored against held-out target train labels
# st/pseudo_mae_dropped same for the samples the filter rejected
# st/selection_gain     dropped - kept. Positive means head agreement really
#                       does select the accurate samples; near zero means the
#                       filter is not doing anything and ST is self-confirming.
#
# opt/*                 learning rates, gradient norms, seconds/epoch, phase
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()