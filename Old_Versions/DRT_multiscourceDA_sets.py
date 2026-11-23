import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
from scipy.stats import pearsonr
import mlflow
from drt_pi import dump_pi
from shap_analysis import run_shap_for_fold
from plots import plot_fold

def set_seed(seed):
    """
    set seed for every rng
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_csv(path):

    rows = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            parsed = [float(v) for v in row]
            rows.append(parsed)
    return np.array(rows, dtype=np.float32)

def load_data(x_path, y_path):
    X = load_csv(x_path)
    Y = load_csv(y_path).squeeze(1)

    print(f"Loaded {len(X)} Datapoints from {x_path}")
    print(f"  X shape: {X.shape}  |  Y shape: {Y.shape}")
    print(f"  Y range: [{Y.min():.3f}, {Y.max():.3f}]  mean: {Y.mean():.3f}")
    return X, Y

class Normalizer:
    def __init__(self):
        self.X_mean = self.X_std = None
        self.Y_mean = self.Y_std = None

    def fit(self, X, Y):
        self.X_mean = X.mean(axis=0)
        self.X_std  = X.std(axis=0) + 1e-8
        self.Y_mean = Y.mean()
        self.Y_std  = Y.std() + 1e-8
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

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    ds  = TensorDataset(X_t, Y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last)

def pool_segments(h, segments=None):
    """Average the routing features over each set, then broadcast back.

    `segments` lists the row counts the batch is made of (one entry per source
    animal, in the order they were concatenated); None means the whole batch is
    a single set. Rows are never reordered, so the same list is valid for every
    layer of the backbone.
    """
    if segments is None:
        return h.mean(0, keepdim=True).expand_as(h)
    return torch.cat([p.mean(0, keepdim=True).expand_as(p)
                      for p in torch.split(h, segments)], 0)

def chunk_sizes(sizes, S):
    """Split per-domain row counts into routing sets of S rows each.

    sizes lists the rows each domain contributes to the batch, in concatenation
    order. S > 1 splits every domain's rows into sets of S (a domain whose
    batch is not a multiple of S ends with one shorter set); S == -1 keeps one
    set per domain; S <= 1 means routing is per sample and segments are unused.
    """
    if S == -1:
        return list(sizes)
    if S <= 1:
        return None
    out = []
    for n in sizes:
        out += [S] * (n // S)
        if n % S:
            out.append(n % S)
    return out

class DynamicLinear(nn.Module):
    def __init__(self, in_dim, out_dim, K=4, reduction=16, drt_activated=True, min_hidden=16, basis_scale=0.1,
                 set_size=0):
        super().__init__()
        self.last_pi = None
        self.last_lam = None
        self.K = K
        self.drt_activated = drt_activated
        self.in_dim = in_dim
        self.out_dim = out_dim

        # per-set routing: one pi/Lambda for a whole set of samples instead of
        # one per sample. `set_size` is how many rows share a routing vector;
        # `segments` is set from outside only for a batch that concatenates
        # several domains, and any batch whose length it does not match is
        # chunked here instead, so a stale list can never be applied. `frozen`
        # holds a routing vector computed once from a reference set, used at
        # evaluation so a prediction does not depend on batch composition.
        self.set_size = set_size
        self.set_routing = set_size not in (0, 1)
        self.segments = None
        self.frozen = None

        #static component
        self.static_weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.static_bias = nn.Parameter(torch.Tensor(out_dim))
        nn.init.kaiming_uniform_(self.static_weight, a=np.sqrt(5))
        nn.init.zeros_(self.static_bias)

        # basis matrices for subspace routing
        self.residual_weights = nn.Parameter(torch.Tensor(K, out_dim, in_dim))
        # Small but non-zero and mutually distinct. With Phi_k == 0 the router
        # gets exactly zero gradient on the first step, and all K bases receive
        # identical gradients thereafter, which drives them together.


        # nn.init.zeros_(self.residual_weights)
        with torch.no_grad():
            for k in range(K):
                nn.init.kaiming_uniform_(self.residual_weights[k], a=np.sqrt(5))
            self.residual_weights.mul_(basis_scale / np.sqrt(K))

        # ONE shared lightweight branch feeds both Λ(x) and π(x),
        # mirroring the single attention branch in the paper's Fig. 3
        #hidden = max(in_dim // reduction, 4)
        hidden = min(max(in_dim // reduction, min_hidden), in_dim) # first layer get 14, second gets 16
        self.branch = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.to_lambda = nn.Linear(hidden, out_dim)   # -> sigmoid, Λ(x)
        self.to_pi     = nn.Linear(hidden, K)          # -> softmax, π(x)

    def forward(self, x):
        if not self.drt_activated:
            self.last_pi = None
            return F.linear(x, self.static_weight, self.static_bias)


        z0 = F.linear(x, self.static_weight)              # W_0 x, bias held back
        if self.frozen is not None:
            # routing fixed from a reference set
            lam = self.frozen[0].to(x.device).expand(x.size(0), -1)
            pi  = self.frozen[1].to(x.device).expand(x.size(0), -1)
        else:
            h = self.branch(x)
            if self.set_routing:
                # every row of a set gets the same W(x), so pi/Lambda describe
                # the set (animal, regime) rather than the single beat
                seg = self.segments
                if seg is None or sum(seg) != x.size(0):
                    # a batch from a single domain: chunk it here
                    seg = chunk_sizes([x.size(0)], self.set_size)
                h = pool_segments(h, seg)
            lam = torch.sigmoid(self.to_lambda(h))       # [B, out_dim]
            pi  = torch.softmax(self.to_pi(h), dim=-1)    # [B, K]
        self.last_lam = lam.detach()
        self.last_pi = pi.detach()

        # Lambda(x) W_0 x  ==  lam * (W_0 x), since Lambda is diagonal
        out = z0 + lam * z0

        # all K projections in one GEMM: [B, K*out] -> [B, K, out]
        z = F.linear(x, self.residual_weights.reshape(self.K * self.out_dim,
                                                      self.in_dim))
        z = z.view(-1, self.K, self.out_dim)
        out = out + (z * pi.unsqueeze(-1)).sum(dim=1)

        return out + self.static_bias

        # lam_term = lam.unsqueeze(-1) * self.static_weight.unsqueeze(0)     # Λ(x)W_0, per sample
        # pi_term  = torch.einsum('bk,kij->bij', pi, self.residual_weights)  # Σπ_i(x)Φ_i
        #
        # weight = self.static_weight.unsqueeze(0) + lam_term + pi_term
        # return torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1) + self.static_bias

class DynamicBackbone(nn.Module):
    def __init__(self, in_dim=14, hidden_dim=128, K=4, drt_activated=True, basis_scale=0.1,
                 set_size=0):
        super().__init__()
        self.layer1 = DynamicLinear(in_dim, 256, K=K, drt_activated=drt_activated, basis_scale=basis_scale,
                                    set_size=set_size)
        self.bn1, self.relu1 = nn.BatchNorm1d(256), nn.ReLU()

        self.layer2 = DynamicLinear(256, hidden_dim, K=K, drt_activated=drt_activated, basis_scale=basis_scale,
                                    set_size=set_size)
        self.bn2, self.relu2 = nn.BatchNorm1d(hidden_dim), nn.ReLU()

    def set_segments(self, segments):
        """Routing sets of a batch that concatenates several domains.

        None (or a list that does not match the batch length) lets each layer
        chunk the batch by its own set size instead.
        """
        self.layer1.segments = segments
        self.layer2.segments = segments

    def forward(self, x):
        h = self.relu1(self.bn1(self.layer1(x)))
        h = self.relu2(self.bn2(self.layer2(h)))
        return h

@torch.no_grad()
def freeze_routing(G, X_ref, device, n_rows=0, max_rows=8192):
    """Fix pi/Lambda from one reference set for the forward passes that follow.

    With per-set routing a batched forward would otherwise make each row's
    weights depend on which other rows share its batch, exactly like BatchNorm
    in train mode. Evaluation, pseudo-labelling and the post-hoc tools instead
    use a single routing vector computed over a reference set (the target
    animal's unlabelled rows).

    n_rows > 0 estimates it from that many rows, evenly spaced over the whole
    recording, so the estimate carries the same sampling noise as the sets the
    router was trained on; n_rows == 0 uses the whole reference set.
    """
    layers = [G.layer1, G.layer2]
    if not layers[0].drt_activated:
        return
    was_training = G.training
    G.eval()
    for m in layers:
        m.frozen = None
    X = torch.as_tensor(X_ref, dtype=torch.float32)
    if 0 < n_rows < len(X):
        X = X[torch.linspace(0, len(X) - 1, n_rows).long()]   # deterministic
    else:
        X = X[:max_rows]
    G.set_segments([len(X)])                 # the reference is ONE set
    G(X.to(device))
    G.set_segments(None)
    for m in layers:
        m.frozen = (m.last_lam[:1].clone(), m.last_pi[:1].clone())
    if was_training:
        G.train()

def unfreeze_routing(G):
    G.layer1.frozen = None
    G.layer2.frozen = None

class RegressorHead(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, feat):
        return self.net(feat).squeeze(-1)

class MeanHead(nn.Module):
    """Averages several heads into one callable head.

    Only used so that tools expecting a single (F1, F2) pair -- SHAP, the
    prediction plots -- keep working in `pairs` mode, where there is one head
    pair per source animal.
    """
    def __init__(self, heads):
        super().__init__()
        self.heads = nn.ModuleList(heads)

    def forward(self, feat):
        return torch.stack([h(feat) for h in self.heads], 0).mean(0)

def flat_heads(heads):
    """[(F1, F2), ...] -> flat list of every head."""
    return [h for pair in heads for h in pair]

def ensemble(heads, feat, weights=None):
    """Prediction of the ensemble.

    Each source contributes the mean of its own pair; the sources are then
    combined uniformly or with MMPD's weights w_k (Eq. 12).
    """
    per_source = torch.stack([(p[0](feat) + p[1](feat)) / 2.0
                              for p in heads], 0)              # [M, B]
    if weights is None:
        return per_source.mean(0)
    return (per_source * weights.to(per_source.device).view(-1, 1)).sum(0)

def mean_discrepancy(heads, feat, kind="bounded"):
    """Mean over pairs of the within-pair discrepancy."""
    return sum(discrepancy_loss(F1(feat), F2(feat), kind)
               for F1, F2 in heads) / len(heads)

def discrepancy_loss(pred1, pred2, kind="bounded"):
    """Disagreement of a predictor pair on one batch.

    bounded: 1 - exp(-|d|), saturates at 1, so a pair that is already far
             apart gets almost no gradient in Step B.
    l2:      MMPD's Eq. 6, the mean squared difference, unbounded.
    """
    d = pred1 - pred2
    if kind == "l2":
        return torch.mean(d ** 2)
    return torch.mean(1.0 - torch.exp(-torch.abs(d)))

@torch.no_grad()
def mmpd_weights(G, heads, src_loaders, trg_loader, device, args, max_batches=8):
    """MMPD's weighted ensemble schema (Eq. 8-11).

    A source is well aligned with the target when its pair predicts its OWN
    source data accurately (small error_k) and disagrees little on the target
    (small dis_k), so alpha_k = beta / (error_k * dis_k), normalised with a
    softmax. The paper found uniform averaging unsatisfying because sources
    differ in how well they match the target.

    Both quantities are label-free with respect to the target, so nothing here
    peeks at target labels. Returns None for uniform averaging.
    """
    if args.ens_mode == "uniform" or len(heads) < 2 or len(heads) != len(src_loaders):
        return None

    G.eval()
    for h in flat_heads(heads):
        h.eval()

    err = torch.zeros(len(heads))
    dis = torch.zeros(len(heads))
    trg_iter = iter(trg_loader)
    src_iters = [iter(l) for l in src_loaders]
    nb = 0

    for _ in range(max_batches):
        (X_t, _), trg_iter = next_batch(trg_iter, trg_loader)
        feat_t = G(X_t.to(device))
        for k, loader in enumerate(src_loaders):
            (X_k, Y_k), src_iters[k] = next_batch(src_iters[k], loader)
            feat_k = G(X_k.to(device))
            pred_k = (heads[k][0](feat_k) + heads[k][1](feat_k)) / 2.0
            err[k] += float(F.mse_loss(pred_k, Y_k.to(device)))          # Eq. 9
            dis[k] += float(torch.mean((heads[k][0](feat_t)
                                        - heads[k][1](feat_t)) ** 2))    # Eq. 10
        nb += 1

    alpha = args.ens_beta / (err / nb * dis / nb + 1e-8)                  # Eq. 8
    return torch.softmax(alpha, 0)                                        # Eq. 11

def variance_penalty(pred1, pred2):
    return -(torch.var(pred1) + torch.var(pred2))


@torch.no_grad()
def evaluate(G, heads, loader, device, normalizer, weights=None):
    G.eval()
    for h in flat_heads(heads):
        h.eval()

    preds, targets = [], []

    for X_b, Y_b in loader:
        X_b, Y_b = X_b.to(device), Y_b.to(device)
        feat = G(X_b)
        # ensemble over the per-source pairs, uniform or MMPD-weighted
        pred = ensemble(heads, feat, weights)
        preds.append(pred.cpu())
        targets.append(Y_b.cpu())

    preds   = torch.cat(preds).numpy().flatten()
    targets = torch.cat(targets).numpy().flatten()

    # denormalise for interpretable metrics
    preds_orig   = normalizer.inverse_Y(preds)
    targets_orig = normalizer.inverse_Y(targets)

    mae  = np.mean(np.abs(preds_orig - targets_orig))
    rmse = np.sqrt(np.mean((preds_orig - targets_orig) ** 2))
    # R2 on original scale
    ss_res = np.sum((targets_orig - preds_orig) ** 2)
    ss_tot = np.sum((targets_orig - targets_orig.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    correlation, _ = pearsonr(targets, preds)
    return mae, rmse, r2, correlation

@torch.no_grad()
def basis_similarity(layer):
    """Mean pairwise |cosine| between the K basis matrices Phi_i.

    Nothing in DRT's loss encourages the Phi_i to differ. If they converge,
    sum_i pi_i Phi_i is constant in pi and routing silently degrades to a
    static layer with K times the parameters. Values near 1 mean collapse.
    """
    W = F.normalize(layer.residual_weights.flatten(1), dim=1)   # [K, out*in]
    sim = (W @ W.t()).abs()
    off = sim[~torch.eye(layer.K, dtype=torch.bool, device=sim.device)]
    return float(off.mean())

@torch.no_grad()
def make_pseudo_labels(G, heads, X_trg, device, keep_quantile):
    """
    one-shot pseudo-labelling of the unlabelled target split
    pseudo-labels above a confidence threshold are used to train DRT
    again together with source samples
    uses head agreement instead of softmax confidence
    threshold is a quantile of the observed discrepancy
    """
    G.eval()
    for h in flat_heads(heads):
        h.eval()
    Xt = torch.tensor(X_trg, dtype=torch.float32).to(device)

    means, discs = [], []
    for i in range(0, len(Xt), 512):
        feat = G(Xt[i:i + 512])
        means.append(ensemble(heads, feat))
        # confidence: mean within-pair disagreement over all pairs
        discs.append(torch.stack([torch.abs(F1(feat) - F2(feat))
                                  for F1, F2 in heads], 0).mean(0))

    pseudo = torch.cat(means).cpu().numpy()
    disc = torch.cat(discs).cpu().numpy()
    thr = float(np.quantile(disc, keep_quantile))
    keep = disc <= thr
    return X_trg[keep], pseudo[keep], float(keep.mean()), thr

def next_batch(it, loader):
    """Next batch from a cycling iterator; returns the batch and the iterator."""
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it

def run_phase(G, heads, opt_g, opt_f, sch_g, sch_f, src_loaders, trg_loader,
              val_loader, normalizer, device, args, n_epochs, epoch_offset,
              phase, pseudo_loader=None):
    """
    Training Phase

    `src_loaders` holds one loader per source animal in balanced/pairs mode and
    a single pooled loader in combine mode. The epoch is driven by the longest
    source loader; shorter animals are cycled, so every animal contributes the
    same number of samples per step regardless of recording length.

    identical metric names in all phases as well as epoch counter
    metric phase marks beginn self-training
    """
    metrics = None
    trg_iter = iter(trg_loader) if args.MCD_activated else None
    pseudo_iter = iter(pseudo_loader) if pseudo_loader is not None else None
    head_params = [p for h in flat_heads(heads) for p in h.parameters()]

    # pairs mode: one head pair per source animal, so each pair sees only its
    # own animal in Step A. Otherwise the single pair sees every animal.
    paired = len(heads) == len(src_loaders) and len(heads) > 1

    # DRT routing granularity: S rows share one W(x). 0/1 is per sample.
    S = args.routing_set_size
    routing_on = S not in (0, 1)
    # rows the frozen estimate is built from at evaluation: the same count as a
    # training set (match) or the whole reference (full, and always for S == -1)
    eval_rows = 0 if args.routing_eval == "full" else max(S, 0)

    steps_per_epoch = max(len(l) for l in src_loaders)

    for epoch in range(1, n_epochs + 1):
        gstep = epoch_offset + epoch
        G.train()
        for h in flat_heads(heads):
            h.train()

        # Discrepancy warmup factor
        if gstep <= args.warmup_epochs:
            da_weight = 0.0
        else:
            da_weight = min(1.0, (gstep - args.warmup_epochs) / 20.0)

        stats = {"mse1": 0.0, "mse2": 0.0, "st": 0.0, "dis": 0.0,
                 "dis_s": 0.0, "vpen": 0.0, "n": 0, "nb": 0}

        src_iters = [iter(l) for l in src_loaders]

        for _ in range(steps_per_epoch):
            xs, ys = [], []
            for i, loader in enumerate(src_loaders):
                (X_i, Y_i), src_iters[i] = next_batch(src_iters[i], loader)
                xs.append(X_i.to(device)); ys.append(Y_i.to(device))
            bsz = sum(x.size(0) for x in xs)
            sizes = [x.size(0) for x in xs]

            if args.MCD_activated:
                (X_t, _), trg_iter = next_batch(trg_iter, trg_loader)
                X_t = X_t.to(device)

            # ---- Step A: supervised risk, plus pseudo-labelled target ----
            for p in G.parameters():
                p.requires_grad = True
            for p in head_params:
                p.requires_grad = True

            # ONE forward pass over all animals: separate passes would give each
            # animal its own BatchNorm statistics, i.e. unintended per-domain
            # normalisation.
            G.set_segments(chunk_sizes(sizes, S))   # sets within each animal
            feats = torch.split(G(torch.cat(xs, 0)), sizes)
            G.set_segments(None)       # single-domain batches chunk themselves

            if paired:
                # each pair is trained only on its own animal, which is what
                # makes the pairs genuinely different (M3SDA-beta's structure)
                l1 = sum(F.mse_loss(hp[0](f), y)
                         for hp, f, y in zip(heads, feats, ys)) / len(heads)
                l2 = sum(F.mse_loss(hp[1](f), y)
                         for hp, f, y in zip(heads, feats, ys)) / len(heads)
            else:
                # the single pair sees every animal; mean over animals rather
                # than over pooled samples, so a long recording cannot dominate
                F1, F2 = heads[0]
                l1 = sum(F.mse_loss(F1(f), y) for f, y in zip(feats, ys)) / len(feats)
                l2 = sum(F.mse_loss(F2(f), y) for f, y in zip(feats, ys)) / len(feats)

            loss_s1, loss_s2 = l1, l2
            loss_stepA = loss_s1 + loss_s2

            # Pseudo-labelled target samples join the source loss, as the paper
            # describes: "used to train DRT again with source samples".
            if pseudo_iter is not None:
                (Xp, Yp), pseudo_iter = next_batch(pseudo_iter, pseudo_loader)
                if Xp.size(0) > 1:
                    Xp, Yp = Xp.to(device), Yp.to(device)
                    feat_p = G(Xp)
                    l_st = sum(F.mse_loss(h(feat_p), Yp)
                               for h in flat_heads(heads)) / len(heads)
                    loss_stepA = loss_stepA + args.st_weight * l_st
                    stats["st"] += float(l_st) * bsz

            opt_g.zero_grad(); opt_f.zero_grad()
            loss_stepA.backward()
            opt_g.step(); opt_f.step()

            # Accumulated here, before Step B reassigns the source losses.
            stats["mse1"] += loss_s1.item() * bsz
            stats["mse2"] += loss_s2.item() * bsz
            stats["n"] += bsz
            stats["nb"] += 1

            # ---- Steps B and C: MCD only ----
            if args.MCD_activated:
                for h in flat_heads(heads):      # dropout off: the discrepancy
                    h.eval()                      # must measure real disagreement

                # Step B: heads maximise target discrepancy, backbone frozen.
                for p in G.parameters():
                    p.requires_grad = False
                for p in head_params:
                    p.requires_grad = True

                G.set_segments(chunk_sizes(sizes, S))
                feats_d = [f.detach() for f in
                           torch.split(G(torch.cat(xs, 0)), sizes)]
                G.set_segments(None)
                feat_t_d = G(X_t).detach()

                if paired:
                    src_term = sum(
                        F.mse_loss(hp[0](f), y) + F.mse_loss(hp[1](f), y)
                        for hp, f, y in zip(heads, feats_d, ys)) / len(heads)
                else:
                    F1, F2 = heads[0]
                    src_term = sum(F.mse_loss(F1(f), y) + F.mse_loss(F2(f), y)
                                   for f, y in zip(feats_d, ys)) / len(feats_d)

                # every pair marks the support of its own animal; the generator
                # must later pull the target inside all of them at once
                dis = mean_discrepancy(heads, feat_t_d, args.disc_kind)
                vpen = sum(variance_penalty(F1(feat_t_d), F2(feat_t_d))
                           for F1, F2 in heads) / len(heads)

                # Minimising a negative 'dis' term maximises the discrepancy
                loss_stepB = src_term + da_weight * (args.gamma * vpen
                                                     - args.eta * dis)

                opt_f.zero_grad()
                loss_stepB.backward()
                opt_f.step()

                # Step C: backbone minimises target discrepancy, heads frozen.
                for p in G.parameters():
                    p.requires_grad = True
                for p in head_params:
                    p.requires_grad = False

                for _ in range(args.num_k):
                    feat_t = G(X_t)
                    dis = mean_discrepancy(heads, feat_t, args.disc_kind)
                    vpen = sum(variance_penalty(F1(feat_t), F2(feat_t))
                               for F1, F2 in heads) / len(heads)

                    # Minimising positive 'dis' pulls the target inside the boundary
                    loss_stepC = da_weight * (args.eta * dis + args.gamma * vpen)

                    opt_g.zero_grad()
                    loss_stepC.backward()
                    opt_g.step()

                G.eval()
                with torch.no_grad():
                    G.set_segments(chunk_sizes(sizes, S))
                    feat_s = G(torch.cat(xs, 0))
                    G.set_segments(None)
                    feat_t = G(X_t)
                    stats["dis_s"] += float(mean_discrepancy(heads, feat_s,
                                                             args.disc_kind))
                    stats["dis"]   += float(mean_discrepancy(heads, feat_t,
                                                             args.disc_kind))
                    stats["vpen"]  += float(
                        sum(variance_penalty(F1(feat_t), F2(feat_t))
                            for F1, F2 in heads) / len(heads))
                G.train()
                for h in flat_heads(heads):
                    h.train()

        sch_g.step(); sch_f.step()

        n, nb = max(stats["n"], 1), max(stats["nb"], 1)
        # MMPD's w_k, recomputed each epoch from a few batches (cheap)
        weights = mmpd_weights(G, heads, src_loaders, trg_loader, device, args)
        if routing_on:
            # routing from the unlabelled adaptation split, not from the val
            # batches, so the score does not depend on batch composition
            freeze_routing(G, trg_loader.dataset.tensors[0], device,
                           n_rows=eval_rows)
        mae, rmse, r2, corr = evaluate(G, heads, val_loader, device, normalizer,
                                       weights)
        if routing_on:
            unfreeze_routing(G)
        metrics = (mae, rmse, r2, corr)

        log = {
            "MSE1": stats["mse1"] / n,
            "MSE2": stats["mse2"] / n,
            "MAE": mae, "RMSE": rmse, "R2": r2, "Corr": corr,
            "da_weight": da_weight,
            "phase": phase,
            "lr_g": opt_g.param_groups[0]["lr"],
            "lr_f": opt_f.param_groups[0]["lr"],
        }
        if args.MCD_activated:
            log["discrepancy_src"] = stats["dis_s"] / nb
            log["discrepancy_trg"] = stats["dis"] / nb
            log["discrepancy_ratio"] = stats["dis"]/ nb / (stats["dis_s"]/ nb + 1e-8)
            log["variance_penalty"] = stats["vpen"] / nb
        if weights is not None:
            for k, w in enumerate(weights.tolist()):
                log[f"source_weight_{k}"] = w
        if pseudo_iter is not None:
            log["mse_pseudo"] = stats["st"] / n
        if args.DRT_activated:
            for i, layer in enumerate([G.layer1, G.layer2], start=1):
                log[f"basis_sim_l{i}"] = basis_similarity(layer)
                log[f"basis_norm_l{i}"] = float(
                    layer.residual_weights.norm() /
                    (layer.static_weight.norm() + 1e-8))
        mlflow.log_metrics(log, step=gstep)

        print(f"  ep {gstep:03d} da={da_weight:.2f} "
              f"MSE1={stats['mse1']/n:.4f} MSE2={stats['mse2']/n:.4f} "
              f"MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} Corr={corr:.4f}")

    return metrics

def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load target domain and labels
    target_x_path = os.path.join(args.data_dict_path, args.target_x_dict)
    target_y_path = os.path.join(args.data_dict_path, args.target_y_dict)
    target_X, target_Y = load_data(target_x_path, target_y_path)

    # load source domains
    source_domains = []
    x_src_dicts = args.source_x_dicts.split(",")
    y_src_dicts = args.source_y_dicts.split(",")

    for i in range(len(x_src_dicts)):
        source_x_path = os.path.join(args.data_dict_path, x_src_dicts[i])
        source_y_path = os.path.join(args.data_dict_path, y_src_dicts[i])
        X_s, Y_s = load_data(source_x_path, source_y_path)
        source_domains.append((X_s, Y_s))

    # concat pooled source
    X_src_all = np.concatenate([X for X, Y in source_domains], axis=0)
    Y_src_all = np.concatenate([Y for X, Y in source_domains], axis=0)

    # normalize on source only
    normalizer = Normalizer().fit(X_src_all, Y_src_all)

    X_src_n = normalizer.transform_X(X_src_all)
    Y_src_n = normalizer.transform_Y(Y_src_all)

    target_X_n = normalizer.transform_X(target_X)
    target_Y_n = normalizer.transform_Y(target_Y)

    # split in train/eval
    (trg_train_X, trg_train_Y), (trg_val_X, trg_val_Y) = split_domain(
        target_X_n, target_Y_n, val_ratio=0.2, seed=args.seed)

    # Dataloader trg_train_Y not used
    if args.ms_mode == "combine":
        # single pooled source domain -- the setting the MCD paper describes
        per = args.batch_size
        src_loaders = [make_loader(X_src_n, Y_src_n, args.batch_size)]
    else:
        # one loader per animal; the per-animal batch size keeps the total
        # number of source samples per step at roughly args.batch_size
        # MMPD underperformed when its source domains were small relative to
        # the target, because each source-specific pair was then trained on too
        # little data. --src_batch_per_domain keeps a usable batch per animal
        # instead of splitting one batch four ways.
        per = (args.src_batch_per_domain or
               max(args.batch_size // len(source_domains), 2))
        src_loaders = [make_loader(normalizer.transform_X(Xs),
                                   normalizer.transform_Y(Ys), per)
                       for Xs, Ys in source_domains]
    trg_loader = make_loader(trg_train_X, trg_train_Y, args.batch_size)
    val_loader = make_loader(trg_val_X, trg_val_Y, args.batch_size,
                             shuffle=False, drop_last=False)

    S = args.routing_set_size
    routing_on = S not in (0, 1)
    eval_rows = 0 if args.routing_eval == "full" else max(S, 0)
    if routing_on:
        if args.ms_mode == "combine":
            print("  [warn] --routing_set_size with --ms_mode combine: the source "
                  "batch mixes animals, so one set spans several of them")
        if S > per:
            print(f"  [warn] --routing_set_size {S} exceeds the {per} source rows "
                  f"per animal per step, so sets are cut to {per} "
                  f"(raise --src_batch_per_domain)")

    G = DynamicBackbone(
        in_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        K= args.K,
        drt_activated=args.DRT_activated,
        basis_scale = args.basis_scale,
        set_size=S,
    ).to(device)

    # pairs: one head pair per source animal (M3SDA-beta's structure), so the
    # pairs differ because the animals differ, not only because of the init
    n_pairs = len(source_domains) if args.ms_mode == "pairs" else 1
    heads = [(RegressorHead(hidden_dim=args.hidden_dim, dropout=args.head_dropout).to(device),
              RegressorHead(hidden_dim=args.hidden_dim, dropout=args.head_dropout).to(device))
             for _ in range(n_pairs)]

    head_ps = [p for h in flat_heads(heads) for p in h.parameters()]
    if args.optimizer == "adam":
        opt_g = optim.Adam(G.parameters(), lr=args.lr)
        opt_f = optim.Adam(head_ps, lr=args.lr_f)
    else:
        opt_g = optim.SGD(G.parameters(), lr=args.lr, momentum=0.9)
        opt_f = optim.SGD(head_ps, lr=args.lr_f, momentum=0.9)

    sch_g = optim.lr_scheduler.MultiStepLR(opt_g, milestones=args.schedule,
                                           gamma=0.1)
    sch_f = optim.lr_scheduler.MultiStepLR(opt_f, milestones=args.schedule,
                                           gamma=0.1)

    run_name = (f"{args.target_x_dict.replace('X_', '').replace('.csv', '')}"
                f"_seed{args.seed}")

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params({k: (str(v) if isinstance(v, (list, tuple)) else v)
                           for k, v in vars(args).items()})
        mlflow.log_param("n_params_G", sum(p.numel() for p in G.parameters()))

        #Phase 1
        metrics = run_phase(G, heads, opt_g, opt_f, sch_g, sch_f,
                            src_loaders, trg_loader, val_loader, normalizer,
                            device, args, args.epochs, 0, phase=0)

        #Phase 2: self-training
        if args.ST_activated:
            if routing_on:
                freeze_routing(G, trg_train_X, device, n_rows=eval_rows)
            Xp, Yp, keep_rate, thr = make_pseudo_labels(
                G, heads, trg_train_X, device, args.st_keep_quantile)
            if routing_on:
                unfreeze_routing(G)

            print(f"  [ST] kept {keep_rate:.1%} of target samples "
                  f"({len(Xp)} pseudo-labelled, thr={thr:.4f})")

            if len(Xp) > args.batch_size:
                pseudo_loader = make_loader(Xp, Yp, args.batch_size)

                # lower learning rate
                # new scheduler with no milestones to stays constant
                for g in opt_g.param_groups:
                    g["lr"] = args.lr * args.st_lr_scale
                for g in opt_f.param_groups:
                    g["lr"] = args.lr_f * args.st_lr_scale
                sch_g = optim.lr_scheduler.MultiStepLR(opt_g, [], gamma=0.1)
                sch_f = optim.lr_scheduler.MultiStepLR(opt_f, [], gamma=0.1)

                metrics = run_phase(G, heads, opt_g, opt_f, sch_g, sch_f,
                                    src_loaders, trg_loader, val_loader,
                                    normalizer, device, args, args.st_epochs,
                                    args.epochs, phase=1,
                                    pseudo_loader=pseudo_loader)
            else:
                print(f"  [ST] skipped: only {len(Xp)} samples kept, "
                      f"need > {args.batch_size}")


        # --------------------------
        # optional post-hoc analysis
        # --------------------------
        # one seed SHAP Values

        # per-set routing: fix pi/Lambda from the whole target animal, so SHAP
        # perturbations and the plots act on the static path with fixed weights
        if routing_on:
            freeze_routing(G, target_X_n, device, n_rows=eval_rows)

        F1 = MeanHead([h[0] for h in heads]).to(device)
        F2 = MeanHead([h[1] for h in heads]).to(device)

        if args.shap and args.seed == args.shap_seed:
            names = (args.feature_names.split(",") if args.feature_names
                     else [f"f{i}" for i in range(args.input_dim)])
            run_shap_for_fold(
                G, F1, F2,
                X_background=trg_train_X,   # variation within this animal
                X_explain=trg_val_X,        # the same split evaluate() scores
                feature_names=names,
                out_dir=args.shap_dir,
                fold=args.target_x_dict.replace("X_", "").replace(".csv", ""),
                seed=args.seed,
                config=f"mcd{int(args.MCD_activated)}drt{int(args.DRT_activated)}st{int(args.ST_activated)}",
                device=device, mlflow=mlflow,
                targets=("mean", "disc_bounded"),
                n_background=args.shap_background,
                n_explain=args.shap_n,
                y_std=float(normalizer.Y_std),
            )

        # residual plots
        if args.seed == args.shap_seed:
            fold = args.target_x_dict.replace("X_", "").replace(".csv", "")
            cfg = (f"mcd{int(args.MCD_activated)}drt{int(args.DRT_activated)}"
                   f"st{int(args.ST_activated)}")
            os.makedirs("Old_Versions/pred_plots", exist_ok=True)
            plot_fold(
                G, F1, F2,
                X=target_X_n,            # full animal, unshuffled -> row order = time
                y=target_Y_n,            # normalized; plot_fold de-normalizes
                animal_id=fold,
                y_mean=float(normalizer.Y_mean),
                y_std=float(normalizer.Y_std),
                device=device,
                out_path=f"Old_Versions/pred_plots/{cfg}_{fold}_seed{args.seed}.png",
            )

        #PIs for t-sne
        # back to live routing: the dump compares pi ACROSS domains, so each
        # domain must compute its own (one vector per set under
        # --routing_set_size, one per sample otherwise)
        if routing_on:
            unfreeze_routing(G)

        if args.pi_dump_dir and args.DRT_activated:
            names = [args.target_x_dict] + args.source_x_dicts.split(",")
            names = [n.replace("X_", "").replace(".csv", "") for n in names]
            domains = [(trg_val_X, normalizer.inverse_Y(trg_val_Y))]
            domains += [(normalizer.transform_X(Xs), Ys)
                        for Xs, Ys in source_domains]

            res = dump_pi(
                G, lambda X, Y: make_loader(X, Y, 512, shuffle=False,
                                            drop_last=False),
                domains, names, device, args.pi_dump_dir, K=args.K,
                target_name=names[0], tag_extra=f"_seed{args.seed}",
                max_per_domain=args.pi_max_per_domain, seed=args.seed)

            if res is not None:
                path, ents, sims = res
                # mlflow.log_artifact(path)
                for i, e in enumerate(ents):
                    mlflow.log_metric(f"final_pi_entropy_l{i + 1}", e)

    mae, rmse, r2, corr = metrics
    return r2, corr

def parse_args():
    p = argparse.ArgumentParser(description="DRT")

    # paths
    p.add_argument("--data_dict_path", type=str, default="Tierdaten/")
    p.add_argument("--target_x_dict", type=str, default="X_anim1.csv")
    p.add_argument("--target_y_dict", type=str, default="Y_anim1.csv")
    p.add_argument("--source_x_dicts", type=str, default="X_anim3.csv,X_anim4.csv,X_anim8.csv,X_anim10.csv")
    p.add_argument("--source_y_dicts", type=str, default="Y_anim3.csv,Y_anim4.csv,Y_anim8.csv,Y_anim10.csv")

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--warmup_epochs", type=int, default=100, help="source-only training")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001, help="backbone lr")
    p.add_argument("--lr_f", type=float, default=0.005, help="head lr")

    p.add_argument("--input_dim", type=int, default=14)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--schedule", type=int, nargs="+", default=[110, 135, 145])
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--head_dropout", type=float, default=0,
                   help="dropout in the regressor heads; 0 = off")

    # DRT
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--basis_scale", type=float, default=0.1, help="init scale of the Phi_i basis matrices")
    p.add_argument("--routing_set_size", type=int, default=0,
                   help="rows of one animal that share a DRT routing vector: "
                        "0 or 1 = one pi/Lambda per sample (unchanged "
                        "behaviour), S > 1 = one per S rows, -1 = one per "
                        "per-domain batch. Capped by --src_batch_per_domain.")
    p.add_argument("--routing_eval", type=str, default="match",
                   choices=["match", "full"],
                   help="frozen routing at evaluation: estimate it from as many "
                        "rows as a training set (match) or from the whole "
                        "reference set (full; also used when S == -1)")
    p.add_argument("--pi_dump_dir", type=str, default="pi_dumps")
    p.add_argument("--pi_max_per_domain", type=int, default=2000)

    #MCD
    p.add_argument("--eta", type=float, default=0.1, help="discrepancy weight")
    p.add_argument("--gamma", type=float, default=0.001, help="variance penalty weight")
    p.add_argument("--ms_mode", type=str, default="combine",
                   choices=["combine", "balanced", "pairs"],
                   help="combine: pooled source ;"
                        "balanced: one batch per animal, one head pair;"
                        "pairs: one head pair per animal")
    p.add_argument("--num_k", type=int, default=1,
                   help="generator updates per batch in Step C (MCD's num_k)")

    # MMPD
    p.add_argument("--mmpd", action="store_true",
                   help="faithful MMPD preset: pairs + squared discrepancy, "
                        "eta=1, no variance penalty, no warmup, Adam, "
                        "weighted ensemble")
    p.add_argument("--disc_kind", type=str, default="bounded",
                   choices=["bounded", "l2"],
                   help="bounded: 1-exp(-|d|); l2: MMPD's mean squared diff")
    p.add_argument("--ens_mode", type=str, default="uniform",
                   choices=["uniform", "mmpd"],
                   help="mmpd: weight each source by 1/(error_k * dis_k)")
    p.add_argument("--ens_beta", type=float, default=1.0,
                   help="beta in MMPD Eq. 8; larger = peakier weights")
    p.add_argument("--src_batch_per_domain", type=int, default=0,
                   help="samples per animal per step; 0 = batch_size // M")
    p.add_argument("--optimizer", type=str, default="sgd",
                   choices=["sgd", "adam"])

    # self-training
    p.add_argument("--st_epochs", type=int, default=30)
    p.add_argument("--st_keep_quantile", type=float, default=0.5)
    p.add_argument("--st_weight", type=float, default=1.0)
    p.add_argument("--st_lr_scale", type=float, default=0.1)

    # switches
    p.add_argument("--MCD_activated", type=bool, default=False, help="If False source only")
    p.add_argument("--DRT_activated", type=bool, default=False)
    p.add_argument("--ST_activated", type=bool, default=False)

    # SHAP
    p.add_argument("--shap", type=bool, default=False, help="post-hoc KernelSHAP after training")
    p.add_argument("--shap_dir", type=str, default="shap_dumps")
    p.add_argument("--shap_n", type=int, default=100, help="target beats to explain")
    p.add_argument("--shap_background", type=int, default=25, help="k-means centroids")
    p.add_argument("--shap_seed", type=int, default=100, help="only this seed gets SHAP")
    p.add_argument("--feature_names", type=str, default="", help="comma-separated, 14 names")

    args = p.parse_args()

    if args.mmpd:       # to replicate the MMPD paper
        args.ms_mode = "pairs"
        args.disc_kind = "l2"
        args.ens_mode = "mmpd"
        args.eta = 1.0
        args.gamma = 0.0
        args.warmup_epochs = 0
        args.optimizer = "adam"
    return args

def main():

    args = parse_args()

    mlflow.set_experiment("DRT_multisource")


    seeds = [100,200,74,15,984]

    x_src_dicts = args.source_x_dicts.split(",")
    y_src_dicts = args.source_y_dicts.split(",")

    x_dicts = [args.target_x_dict] + x_src_dicts
    y_dicts = [args.target_y_dict] + y_src_dicts

    animal_pool = list(zip(x_dicts, y_dicts))

    # MCD DRT ST
    allowed_values = [
        # (False, False, False),
        # (False, True, False),
        (True, False, False),
        (True, True, False),
        (True, True, True)
    ]

    keys = ["MCD_activated", "DRT_activated", "ST_activated"]
    combinations = [dict(zip(keys, v)) for v in allowed_values]

    total_runs = len(combinations) * len(animal_pool) * len(seeds)
    run_idx = 0

    for idx, config in enumerate(combinations, 1):
        args.MCD_activated = config["MCD_activated"]
        args.DRT_activated = config["DRT_activated"]
        args.ST_activated = config["ST_activated"]

        final_summary = {}
        all_target_r2_means = []
        all_target_r2_raw_means = []
        all_target_corr_means = []

        with mlflow.start_run(run_name=f"mcd_{int(args.MCD_activated)}"
                                       f"_drt{int(args.DRT_activated)}"
                                       f"_st{int(args.ST_activated)}"):
            mlflow.log_params({"config_idx": idx, **config})

            for target_x, target_y in animal_pool:

                target_name = target_x.replace("X_", "").replace(".csv", "")

                sources = [pair for pair in animal_pool if pair != (target_x, target_y)]

                sources_x_str = ",".join([pair[0] for pair in sources])
                sources_y_str = ",".join([pair[1] for pair in sources])

                args.target_x_dict = target_x
                args.target_y_dict = target_y
                args.source_x_dicts = sources_x_str
                args.source_y_dicts = sources_y_str

                seed_r2, seed_corr, seed_r2_raw = [], [], []

                for i in range(len(seeds)):
                    args.seed = seeds[i]
                    run_idx += 1
                    print(f"\n[{run_idx}/{total_runs}] "
                          f"mcd={int(args.MCD_activated)} "
                          f"drt={int(args.DRT_activated)} "
                          f"st={int(args.ST_activated)} | "
                          f"target={target_name} | seed={args.seed}",
                          flush=True)

                    run_r2, run_corr = train(args)
                    seed_r2.append(max(run_r2, -1.0))
                    seed_r2_raw.append(run_r2)
                    seed_corr.append(run_corr)

                    # break # only first seed

                mean_r2 = np.mean(seed_r2)
                std_r2 = np.std(seed_r2, ddof=1) # sample std
                mean_raw_r2 = np.mean(seed_r2_raw)
                std_raw_r2 = np.std(seed_r2_raw, ddof=1) # sample std

                mean_corr =  np.mean(seed_corr)
                std_corr = np.std(seed_corr, ddof=1)

                final_summary[target_name] = (mean_r2, std_r2, mean_raw_r2, std_raw_r2, mean_corr, std_corr)
                all_target_r2_means.append(mean_r2)
                all_target_r2_raw_means.append(mean_raw_r2)
                all_target_corr_means.append(mean_corr)
                print(f"Finished Target {target_name}! Result: {mean_r2:.4f} ± {std_r2:.4f}")

                # break # only first animal

            mlflow.log_metrics({"mean_R2": float(np.mean(all_target_r2_means)),
                                "mean_raw_R2": float(np.mean(all_target_r2_raw_means)),
                                "mean_Corr": float(np.mean(all_target_corr_means))})

        output_file = "drt_training_results.txt"

        with open(output_file, "a", encoding="utf-8") as f:

            header = f"Results for configuration: {config}: \n" + "-" * 40 + "\n"
            print(header, end="")
            f.write(header)

            for target_name, (mean_r2, std_r2, mean_raw_r2, std_raw_r2, mean_corr, std_corr) in final_summary.items():
                target_info = f"Target: {target_name}\n R2: {mean_r2:.2f} ± {std_r2:.2f}\n R2 Raw: {mean_raw_r2:.2f} ± {std_raw_r2:.2f}\n Correlation: {mean_corr:.2f} ± {std_corr}\n" + "-" * 40 + "\n"
                print(target_info, end="")
                f.write(target_info)

            average_all_r2 = np.mean(all_target_r2_means, dtype=np.float64)
            std_all_r2 = np.std(all_target_r2_means, ddof=1)

            average_all_raw_r2 = np.mean(all_target_r2_raw_means, dtype=np.float64)
            std_all_raw_r2 = np.std(all_target_r2_raw_means, ddof=1)

            average_all_corr = np.mean(all_target_corr_means, dtype=np.float64)
            std_all_corr = np.std(all_target_corr_means, ddof=1)

            footer = f"Average Across All Targets:\n R2: {average_all_r2:.4f} ± {std_all_r2:.4f}\n R2 Raw: {average_all_raw_r2:.2f} ± {std_all_raw_r2:.2f}\n Correlation: {average_all_corr:.4f} ± {std_all_corr:.4f}\n" + "=" * 40 + "\n\n"
            print(footer, end="")
            f.write(footer)

if __name__ == "__main__":
    main()