"""
Dynamic Residual Transfer (DRT) for tabular regression.

Port of Li et al., "Dynamic Transfer for Multi-Source Domain Adaptation"
(CVPR 2021, arXiv:2103.10583) from image classification to per-animal
hemodynamic regression.

Reference implementation: https://github.com/liyunsheng13/DRT

Structure of the DRT variants (Table 1 of the paper):
    mode = 'none'          static transfer, DeltaW(x) = 0
    mode = 'channel'       Eq. 3   DeltaW(x) = Lambda(x) W_0
    mode = 'subspace'      Eq. 4   DeltaW(x) = sum_i pi_i(x) Phi_i     <- paper default
    mode = 'combination'   Eq. 5   both terms
"""

import argparse
import csv
import json
import os
import random

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed):
    """
    set seed for every rng
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

# Data

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
    """
    fits on source data only
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


# Architecture

class DRTLinear(nn.Module):
    """Linear layer with a sample-dependent residual on the weight matrix.

        W(x) = W_0 + Lambda(x) W_0 + sum_i pi_i(x) Phi_i          (Eq. 5)

    Both residual terms are applied in *feature* space rather than by
    materialising a per-sample [B, out, in] weight tensor. Since every operation
    is linear in x and shares the same input, the two are algebraically
    identical:

        (sum_i pi_i Phi_i) x  ==  sum_i pi_i (Phi_i x)

    but the feature-space form is one GEMM instead of a batched matmul over a
    tensor that, for a 256->128 layer at batch 64, is 2.1M floats forward and
    the same again in backward. This is also what the reference repo does
    (`conv_s1..conv_s4` applied to the activation, then blended).

    Lambda scales the *weight*, not the bias, so the bias is added after the
    residual terms.
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

        # Single lightweight branch feeding both heads, as in Fig. 3 of the
        # paper: the classification version is avg-pool -> FC -> ReLU -> FC,
        # with sigmoid on the Lambda head and softmax on the pi head. Tabular
        # inputs have no spatial extent, so the pooling step drops out.
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
                # Low-rank factorisation of each Phi_i. This is the tabular
                # analogue of the paper restricting Phi to 1x1 kernels on the
                # narrowest layer of the bottleneck: a capacity cap, not a
                # different method.
                self.basis = nn.Sequential(
                    nn.Linear(in_dim, rank, bias=False),
                    nn.Linear(rank, K * out_dim, bias=False),
                )
            self._init_basis(basis_scale)

    def _init_basis(self, scale):
        """Small but non-zero.

        d L / d pi_k = <d L / d out, Phi_k x>, so with Phi_k == 0 the router
        receives exactly zero gradient on the first step. Scaling a normal
        init down keeps the layer near-static at initialisation without
        blinding the router.
        """
        mods = [m for m in self.basis.modules() if isinstance(m, nn.Linear)]
        for m in mods:
            nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
            m.weight.data.mul_(scale / np.sqrt(self.K))

    def dynamic_parameters(self):
        """Parameters of the residual path only (for the 10x lr group)."""
        for name, p in self.named_parameters():
            if not name.startswith("static."):
                yield p

    def forward(self, x):
        z0 = F.linear(x, self.static.weight)  # W_0 x, bias held back

        if self.mode == "none":
            self.last_pi = None
            self.last_lambda = None
            return z0 + self.static.bias if self.static.bias is not None else z0

        h = self.branch(x)
        out = z0

        if self.mode in ("channel", "combination"):
            lam = torch.sigmoid(self.to_lambda(h))          # [B, out_dim]
            # Eq. 3: W(x) = W_0 + Lambda(x) W_0 = (I + Lambda(x)) W_0.
            # A residual *boost*, not SE-style gating (which would be lam * z0
            # alone and can only attenuate).
            out = out + lam * z0
            self.last_lambda = lam.detach()

        if self.mode in ("subspace", "combination"):
            pi = torch.softmax(self.to_pi(h), dim=-1)        # [B, K]
            z = self.basis(x).view(-1, self.K, self.out_dim)  # [B, K, out_dim]
            out = out + (z * pi.unsqueeze(-1)).sum(dim=1)
            self.last_pi = pi.detach()

        if self.static.bias is not None:
            out = out + self.static.bias
        return out


class DynamicBackbone(nn.Module):
    """Two-layer feature extractor.

    `dynamic_layers` mirrors the paper's placement choice: on ResNet-101 the
    7x7 stem stays static and residuals are added only to the 3x3 conv inside
    each bottleneck. With 14 input features, a per-sample dynamic first layer
    is close to fitting a separate model per animal, so layer 1 is static by
    default.
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

    def param_groups(self, lr):
        """Static params at lr, residual-path params at 10x lr.

        The paper gives newly introduced parameters 10x the base learning rate
        (`get_10xparams` in the repo). There it separates by "present in the
        ImageNet checkpoint or not"; here the equivalent split is static vs.
        dynamic, since we have no pretrained dynamic backbone to inherit.
        """
        dyn, static = [], []
        for name, p in self.named_parameters():
            (dyn if (".branch." in name or ".to_pi." in name
                     or ".to_lambda." in name or ".basis." in name)
             else static).append(p)
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


# Losses

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
    """Negative output variance; minimising it keeps the heads from collapsing
    to a constant prediction, which is the degenerate way to make the
    discrepancy term small in Step C."""
    return -(torch.var(pred1) + torch.var(pred2))


# --------------------------------------------------------------------------- #
# Routing diagnostics
# --------------------------------------------------------------------------- #

@torch.no_grad()
def collect_pi(G, loader, device):
    """Concatenate pi across all dynamic layers for every sample.

    This is the paper's Pi = {pi_i^l(x)} vector (Sec. 4.5) used for the t-SNE
    in Fig. 4. Returns [N, n_dynamic_layers * K], or None in static mode.
    """
    layers = G.drt_layers()
    layers = [l for l in layers if l.mode in ("subspace", "combination")]
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
    """Mean entropy of pi, normalised to [0, 1].

    1.0 means the router emits a uniform blend for every sample and carries no
    information. 0.0 means hard one-hot routing. A value pinned at 1.0 is the
    signal that the dynamic path is inert.
    """
    pi = np.clip(pi.reshape(-1, K), 1e-8, 1.0)
    ent = -(pi * np.log(pi)).sum(axis=1).mean()
    return float(ent / np.log(K))


@torch.no_grad()
def basis_similarity(layer):
    """Mean pairwise |cosine| between the K basis matrices Phi_i.

    Nothing in DRT's loss encourages the Phi_i to differ. If they converge,
    sum_i pi_i Phi_i is constant in pi and routing silently degrades to a
    static layer with K times the parameters. Values near 1 mean collapse.
    """
    if layer.mode not in ("subspace", "combination"):
        return None
    if isinstance(layer.basis, nn.Sequential):
        W = layer.basis[1].weight @ layer.basis[0].weight   # [K*out, in]
    else:
        W = layer.basis.weight
    W = W.view(layer.K, layer.out_dim, layer.in_dim).flatten(1)
    W = F.normalize(W, dim=1)
    sim = (W @ W.t()).abs()
    off = sim[~torch.eye(layer.K, dtype=torch.bool, device=sim.device)]
    return float(off.mean())


#Eval

@torch.no_grad()
def evaluate(G, F1, F2, loader, device, normalizer):
    """Average the two heads.

    The repo's `test()` picks whichever head has the larger logit at its argmax
    and reports that as the headline number, so combining both heads is the
    paper's own protocol rather than an extra advantage. Regression has no
    argmax to compare, so the mean is the natural analogue.
    """
    G.eval(); F1.eval(); F2.eval()
    preds, targets = [], []
    for X_b, Y_b in loader:
        feat = G(X_b.to(device))
        preds.append(((F1(feat) + F2(feat)) / 2.0).cpu())
        targets.append(Y_b)

    preds = torch.cat(preds).numpy().ravel()
    targets = torch.cat(targets).numpy().ravel()

    preds_orig = normalizer.inverse_Y(preds)
    targets_orig = normalizer.inverse_Y(targets)

    mae = float(np.mean(np.abs(preds_orig - targets_orig)))
    rmse = float(np.sqrt(np.mean((preds_orig - targets_orig) ** 2)))
    ss_res = np.sum((targets_orig - preds_orig) ** 2)
    ss_tot = np.sum((targets_orig - targets_orig.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    corr = float(pearsonr(targets, preds)[0])
    return mae, rmse, r2, corr


# Train


def run_epochs(G, F1, F2, opt_g, opt_f, sch_g, sch_f, src_loader, trg_loader,
               val_loader, normalizer, device, args, n_epochs, epoch_offset,
               use_mcd, pseudo_loader=None, log_prefix=""):
    """One training phase. `pseudo_loader` supplies labelled target samples
    during the self-training phase."""

    metrics = None
    trg_iter = iter(trg_loader) if use_mcd else None
    pseudo_iter = iter(pseudo_loader) if pseudo_loader is not None else None

    for ep in range(1, n_epochs + 1):
        gstep = epoch_offset + ep
        G.train(); F1.train(); F2.train()

        # Linear ramp on the domain-adaptation terms. Not in the paper, which
        # activates the alignment loss from step 0; kept because with a few
        # thousand loops per animal the discrepancy game destabilises an
        # untrained backbone. Set --warmup_epochs 0 for the paper's schedule.
        if gstep <= args.warmup_epochs:
            da_w = 0.0
        else:
            da_w = min(1.0, (gstep - args.warmup_epochs) / max(args.da_ramp, 1))

        acc = {"mse1": 0.0, "mse2": 0.0, "dis": 0.0, "n": 0, "nb": 0}

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

            # ---- Step A: supervised risk on the union of source domains ----
            feat_s = G(X_s)
            loss_s1 = F.mse_loss(F1(feat_s), Y_s)
            loss_s2 = F.mse_loss(F2(feat_s), Y_s)
            loss_a = loss_s1 + loss_s2

            # Pseudo-labelled target samples join the source loss, exactly as
            # the paper describes: "used to train DRT again with source samples".
            if pseudo_iter is not None:
                try:
                    Xp, Yp = next(pseudo_iter)
                except StopIteration:
                    pseudo_iter = iter(pseudo_loader)
                    Xp, Yp = next(pseudo_iter)
                if Xp.size(0) > 1:
                    Xp, Yp = Xp.to(device), Yp.to(device)
                    feat_p = G(Xp)
                    loss_a = loss_a + args.st_weight * (
                        F.mse_loss(F1(feat_p), Yp) + F.mse_loss(F2(feat_p), Yp))

            opt_g.zero_grad(); opt_f.zero_grad()
            loss_a.backward()
            opt_g.step(); opt_f.step()

            acc["mse1"] += loss_s1.item() * bsz
            acc["mse2"] += loss_s2.item() * bsz
            acc["n"] += bsz

            if not use_mcd:
                continue

            # ---- Step B: heads maximise target discrepancy ----
            # Features are detached, so no gradient reaches G and no
            # requires_grad juggling is needed.
            with torch.no_grad():
                feat_s_d = G(X_s)
                feat_t_d = G(X_t)
            feat_s_d, feat_t_d = feat_s_d.detach(), feat_t_d.detach()

            dis = discrepancy_loss(F1(feat_t_d), F2(feat_t_d), args.dis_loss)
            vpen = variance_penalty(F1(feat_t_d), F2(feat_t_d))
            loss_b = (F.mse_loss(F1(feat_s_d), Y_s)
                      + F.mse_loss(F2(feat_s_d), Y_s)
                      + da_w * (args.gamma * vpen - args.eta * dis))

            opt_f.zero_grad()
            loss_b.backward()
            opt_f.step()

            # ---- Step C: backbone minimises target discrepancy, num_k times ----
            # MCD runs the generator update repeatedly per batch (num_k = 4 in
            # Saito et al.); a single step lets the heads outrun the backbone.
            for p in list(F1.parameters()) + list(F2.parameters()):
                p.requires_grad_(False)
            for _ in range(args.num_k):
                feat_t = G(X_t)
                dis_c = discrepancy_loss(F1(feat_t), F2(feat_t), args.dis_loss)
                vpen_c = variance_penalty(F1(feat_t), F2(feat_t))
                loss_c = da_w * (args.eta * dis_c + args.gamma * vpen_c)
                opt_g.zero_grad()
                loss_c.backward()
                opt_g.step()
            for p in list(F1.parameters()) + list(F2.parameters()):
                p.requires_grad_(True)

            acc["dis"] += float(dis_c.detach())
            acc["nb"] += 1

        sch_g.step(); sch_f.step()

        n = max(acc["n"], 1)
        mae, rmse, r2, corr = evaluate(G, F1, F2, val_loader, device, normalizer)
        metrics = (mae, rmse, r2, corr)

        log = {f"{log_prefix}MSE1": acc["mse1"] / n,
               f"{log_prefix}MSE2": acc["mse2"] / n,
               f"{log_prefix}MAE": mae, f"{log_prefix}RMSE": rmse,
               f"{log_prefix}R2": r2, f"{log_prefix}Corr": corr,
               f"{log_prefix}da_weight": da_w}
        if acc["nb"]:
            log[f"{log_prefix}discrepancy"] = acc["dis"] / acc["nb"]

        # ---- routing diagnostics ----
        if gstep % args.pi_log_every == 0:
            pi = collect_pi(G, val_loader, device)
            if pi is not None:
                log[f"{log_prefix}pi_entropy"] = routing_entropy(pi, args.K)
                for i, layer in enumerate(G.drt_layers()):
                    s = basis_similarity(layer)
                    if s is not None:
                        log[f"{log_prefix}basis_sim_l{i}"] = s
        mlflow.log_metrics(log, step=gstep)

        print(f"  ep {gstep:03d} da={da_w:.2f} "
              f"MSE1={acc['mse1']/n:.4f} MSE2={acc['mse2']/n:.4f} "
              f"MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} corr={corr:.4f}")

    return metrics


@torch.no_grad()
def make_pseudo_labels(G, F1, F2, X_trg, device, args):
    """One-shot pseudo-labelling of the unlabelled target split.

    The paper runs self-training as a *second training round* after DRT has
    converged, not as an online per-batch step: pseudo-labels above a
    confidence threshold are used to train DRT again together with source
    samples. Regression has no softmax confidence, so head agreement stands in
    for it -- which is also the quantity MCD already optimises, making it the
    natural confidence proxy here.

    The threshold is a quantile of the observed discrepancy rather than an
    absolute number, so the keep rate is stable across animals and scales.
    """
    G.eval(); F1.eval(); F2.eval()
    Xt = torch.tensor(X_trg, dtype=torch.float32).to(device)
    preds1, preds2 = [], []
    for i in range(0, len(Xt), 512):
        feat = G(Xt[i:i + 512])
        preds1.append(F1(feat)); preds2.append(F2(feat))
    p1, p2 = torch.cat(preds1), torch.cat(preds2)

    pseudo = ((p1 + p2) / 2.0).cpu().numpy()
    disc = torch.abs(p1 - p2).cpu().numpy()
    thr = np.quantile(disc, args.st_keep_quantile)
    keep = disc <= thr
    return X_trg[keep], pseudo[keep], float(keep.mean())


def train(args, run_name=None):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_X, target_Y = load_data(
        os.path.join(args.data_dict_path, args.target_x_dict),
        os.path.join(args.data_dict_path, args.target_y_dict),
        verbose=args.verbose)

    source_domains = []
    for xs, ys in zip(args.source_x_dicts.split(","),
                      args.source_y_dicts.split(",")):
        source_domains.append(load_data(
            os.path.join(args.data_dict_path, xs),
            os.path.join(args.data_dict_path, ys), verbose=args.verbose))

    # Source domains are pooled with no domain labels anywhere -- the whole
    # point of dynamic transfer is that it "turns multi-source domains into a
    # single-source domain".
    X_src = np.concatenate([X for X, _ in source_domains], axis=0)
    Y_src = np.concatenate([Y for _, Y in source_domains], axis=0)

    normalizer = Normalizer().fit(X_src, Y_src)
    X_src_n = normalizer.transform_X(X_src)
    Y_src_n = normalizer.transform_Y(Y_src)
    trg_X_n = normalizer.transform_X(target_X)
    trg_Y_n = normalizer.transform_Y(target_Y)

    (trg_tr_X, _), (trg_va_X, trg_va_Y) = split_domain(
        trg_X_n, trg_Y_n, val_ratio=args.val_ratio, seed=args.seed)
    # trg_tr_Y is deliberately discarded: the target training split is
    # unlabelled as far as the method is concerned.
    trg_tr_Y_dummy = np.zeros(len(trg_tr_X), dtype=np.float32)

    src_loader = make_loader(X_src_n, Y_src_n, args.batch_size)
    trg_loader = make_loader(trg_tr_X, trg_tr_Y_dummy, args.batch_size)
    val_loader = make_loader(trg_va_X, trg_va_Y, args.batch_size,
                             shuffle=False, drop_last=False)

    G = DynamicBackbone(in_dim=args.input_dim, hidden_dim=args.hidden_dim,
                        mode=args.drt_mode, K=args.K, rank=args.rank,
                        reduction=args.reduction,
                        dynamic_layers=tuple(args.dynamic_layers)).to(device)
    F1 = RegressorHead(args.hidden_dim).to(device)
    F2 = RegressorHead(args.hidden_dim).to(device)

    opt_g = optim.SGD(G.param_groups(args.lr), lr=args.lr,
                      momentum=0.9, weight_decay=args.weight_decay)
    opt_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()),
                      lr=args.lr_f, momentum=0.9, weight_decay=args.weight_decay)
    sch_g = optim.lr_scheduler.MultiStepLR(opt_g, args.schedule, gamma=0.1)
    sch_f = optim.lr_scheduler.MultiStepLR(opt_f, args.schedule, gamma=0.1)

    with mlflow.start_run(run_name=run_name, nested=args.nested_runs):
        mlflow.log_params({k: (str(v) if isinstance(v, (list, tuple)) else v)
                           for k, v in vars(args).items()})
        mlflow.log_param("n_params_G", sum(p.numel() for p in G.parameters()))

        metrics = run_epochs(
            G, F1, F2, opt_g, opt_f, sch_g, sch_f, src_loader, trg_loader,
            val_loader, normalizer, device, args, args.epochs, 0,
            use_mcd=args.MCD_activated)

        # ---- Self-training: second round, as in the paper ----
        if args.ST_activated:
            Xp, Yp, keep_rate = make_pseudo_labels(G, F1, F2, trg_tr_X,
                                                   device, args)
            mlflow.log_metric("st_keep_rate", keep_rate)
            print(f"  [ST] kept {keep_rate:.1%} of target samples "
                  f"({len(Xp)} pseudo-labelled)")
            if len(Xp) > args.batch_size:
                pseudo_loader = make_loader(Xp, Yp, args.batch_size,
                                            shuffle=True, drop_last=True)
                for g in opt_g.param_groups:
                    g["lr"] = args.lr * args.st_lr_scale
                for g in opt_f.param_groups:
                    g["lr"] = args.lr_f * args.st_lr_scale
                sch_g = optim.lr_scheduler.MultiStepLR(opt_g, [], gamma=0.1)
                sch_f = optim.lr_scheduler.MultiStepLR(opt_f, [], gamma=0.1)
                metrics = run_epochs(
                    G, F1, F2, opt_g, opt_f, sch_g, sch_f, src_loader,
                    trg_loader, val_loader, normalizer, device, args,
                    args.st_epochs, args.epochs,
                    use_mcd=args.MCD_activated, pseudo_loader=pseudo_loader,
                    log_prefix="st_")

        # ---- dump pi for the Fig. 4 style t-SNE ----
        if args.pi_dump_dir and args.drt_mode in ("subspace", "combination"):
            os.makedirs(args.pi_dump_dir, exist_ok=True)
            pi_val = collect_pi(G, val_loader, device)
            if pi_val is not None:
                tag = (f"{args.target_x_dict.replace('.csv','')}"
                       f"_{args.drt_mode}_seed{args.seed}")
                path = os.path.join(args.pi_dump_dir, f"pi_{tag}.npz")
                np.savez_compressed(path, pi=pi_val, K=args.K)
                mlflow.log_artifact(path)

    mae, rmse, r2, corr = metrics
    return r2, corr


# Args


def parse_args():
    p = argparse.ArgumentParser(description="DRT for tabular regression")
    p.add_argument("--data_dict_path", type=str,
                   default="Tierdaten/")
    p.add_argument("--target_x_dict", type=str, default="X_anim1.csv")
    p.add_argument("--target_y_dict", type=str, default="Y_anim1.csv")
    p.add_argument("--source_x_dicts", type=str,
                   default="X_anim10.csv,X_anim3.csv,X_anim4.csv,X_anim8.csv")
    p.add_argument("--source_y_dicts", type=str,
                   default="Y_anim10.csv,Y_anim3.csv,Y_anim4.csv,Y_anim8.csv")

    # optimisation
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--warmup_epochs", type=int, default=80)
    p.add_argument("--da_ramp", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lr_f", type=float, default=0.005)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--schedule", type=int, nargs="+", default=[110, 135, 145])
    p.add_argument("--val_ratio", type=float, default=0.2)
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

    # self-training
    p.add_argument("--st_epochs", type=int, default=30)
    p.add_argument("--st_keep_quantile", type=float, default=0.5)
    p.add_argument("--st_weight", type=float, default=1.0)
    p.add_argument("--st_lr_scale", type=float, default=0.1)

    # switches
    p.add_argument("--MCD_activated", type=str2bool, default=False)
    p.add_argument("--DRT_activated", type=str2bool, default=False)
    p.add_argument("--ST_activated", type=str2bool, default=False)

    # logging
    p.add_argument("--pi_log_every", type=int, default=10)
    p.add_argument("--pi_dump_dir", type=str, default="pi_dumps")
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

    # (MCD, DRT mode, ST). Row 3 reproduces Table 1 of the paper on this data.
    configs = [
        {"MCD_activated": False, "drt_mode": "none",        "ST_activated": False},
        {"MCD_activated": True,  "drt_mode": "none",        "ST_activated": False},
        {"MCD_activated": True,  "drt_mode": "channel",     "ST_activated": False},
        {"MCD_activated": True,  "drt_mode": "subspace",    "ST_activated": False},
        {"MCD_activated": True,  "drt_mode": "combination", "ST_activated": False},
        {"MCD_activated": True,  "drt_mode": "subspace",    "ST_activated": True},
    ]

    for config in configs:
        args.MCD_activated = config["MCD_activated"]
        args.drt_mode = config["drt_mode"]
        args.DRT_activated = config["drt_mode"] != "none"
        args.ST_activated = config["ST_activated"]

        summary, all_r2, all_corr = {}, [], []

        with mlflow.start_run(run_name=f"cfg_{config['drt_mode']}"
                                       f"_mcd{int(args.MCD_activated)}"
                                       f"_st{int(args.ST_activated)}"):
            mlflow.log_params(config)

            for target_x, target_y in animal_pool:
                name = target_x.replace("X_", "").replace(".csv", "")
                sources = [p for p in animal_pool if p != (target_x, target_y)]

                args.target_x_dict, args.target_y_dict = target_x, target_y
                args.source_x_dicts = ",".join(p[0] for p in sources)
                args.source_y_dicts = ",".join(p[1] for p in sources)

                r2s, corrs = [], []
                for s in seeds:
                    args.seed = s
                    print(f"[{config['drt_mode']} mcd={args.MCD_activated} "
                          f"st={args.ST_activated}] target={name} seed={s}")
                    r2, corr = train(args, run_name=f"{name}_seed{s}")
                    r2s.append(r2); corrs.append(corr)

                summary[name] = (float(np.mean(r2s)), float(np.std(r2s, ddof=1)),
                                 float(np.mean(corrs)), float(np.std(corrs, ddof=1)))
                all_r2.append(np.mean(r2s)); all_corr.append(np.mean(corrs))
                print(f"Finished target {name}: "
                      f"R2 {summary[name][0]:.4f} +- {summary[name][1]:.4f}")

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


if __name__ == "__main__":
    main()