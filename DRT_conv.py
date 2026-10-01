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

def make_windows(X, Y, L):
    """Causal sliding windows over ONE animal's beats (rows in time order).

    X: (T, C) -> (T-L+1, C, L), channels first as Conv1d expects.
    Y: (T,)   -> (T-L+1,), strokework of the LAST beat of each window, so the
    model sees only the current and past beats. The first L-1 beats of an
    animal get no prediction. Never call this on concatenated animals.
    """
    Xw = np.lib.stride_tricks.sliding_window_view(X, L, axis=0)  # (T-L+1, C, L)
    return np.ascontiguousarray(Xw, dtype=np.float32), Y[L - 1:]

def split_domain(X, Y, val_ratio=0.2, seed=100, mode="random", gap=0):
    """Split the target windows into the unlabelled adaptation set and val.

    random: as before. Neighbouring windows overlap, so val beats also appear
            inside adaptation windows -- but target labels never enter
            training, so this shares inputs, not labels (transductive DA).
    block:  the last val_ratio of the recording is val, and `gap` windows
            before it are dropped, so the two sets share no beat at all.
    """
    n = len(X)
    n_val = int(n * val_ratio)
    if mode == "block":
        val_idx = np.arange(n - n_val, n)
        train_idx = np.arange(0, max(n - n_val - gap, 0))
    else:
        idx = np.random.default_rng(seed).permutation(n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (X[train_idx], Y[train_idx]), (X[val_idx], Y[val_idx])

def make_loader(X, Y, batch_size, shuffle=True, drop_last=True):

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    ds  = TensorDataset(X_t, Y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last)

class DynamicConv1d(nn.Module):
    """DRT dynamic layer as a causal convolution over the beat axis.

        W(x) = W_0 + Lambda(x) W_0 + sum_i pi_i(x) Phi_i        (paper, Eq. 5)

    Input (B, C_in, L) -> output (B, C_out, L-k+1). 'Valid' convolution: output
    position t sees beats t-k+1..t only, and no padding is involved.

    Paper, Fig. 3: Lambda(x) and pi(x) come from ONE branch fed by the input
    average-pooled over time, so routing is a property of the whole window.

    Official code (model/resnet_dra.py, BottleneckKDRA): the bases Phi_i are
    1-wide kernels running parallel to the static k-wide kernel, and there is
    no Lambda(x) term (SEModule is defined but never used). phi_kernel=1 is
    that default here, aligned on the most recent beat of W_0's field;
    """
    def __init__(self, in_dim, out_dim, kernel_size=3, phi_kernel=1, K=4,
                 reduction=16, drt_activated=True, min_hidden=16,
                 basis_scale=0.1):
        super().__init__()
        assert 1 <= phi_kernel <= kernel_size
        self.last_pi = None
        self.K = K
        self.drt_activated = drt_activated
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k, self.k_phi = kernel_size, phi_kernel

        #static component: W_0 is a (C_out, C_in, k) temporal kernel
        self.static_weight = nn.Parameter(torch.Tensor(out_dim, in_dim, kernel_size))
        self.static_bias = nn.Parameter(torch.Tensor(out_dim))
        nn.init.kaiming_uniform_(self.static_weight, a=np.sqrt(5))
        nn.init.zeros_(self.static_bias)

        # basis kernels for subspace routing: Phi_i is (C_out, C_in, k_phi)
        self.residual_weights = nn.Parameter(torch.Tensor(K, out_dim, in_dim, phi_kernel))
        # Small but non-zero and mutually distinct. With Phi_k == 0 the router
        # gets exactly zero gradient on the first step, and all K bases receive
        # identical gradients thereafter, which drives them together.
        with torch.no_grad():
            for k in range(K):
                nn.init.kaiming_uniform_(self.residual_weights[k], a=np.sqrt(5))
            self.residual_weights.mul_(basis_scale / np.sqrt(K))

        # ONE shared lightweight branch feeds both Λ(x) and π(x),
        # mirroring the single attention branch in the paper's Fig. 3
        hidden = min(max(in_dim // reduction, min_hidden), in_dim) # first layer get 14, second gets 16
        self.branch = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.to_lambda = nn.Linear(hidden, out_dim)  # -> sigmoid, Λ(x)
        self.to_pi     = nn.Linear(hidden, K)                                 # -> softmax, π(x)

    def forward(self, x):                                  # x: [B, C_in, L]
        if not self.drt_activated:
            self.last_pi = None
            return F.conv1d(x, self.static_weight, self.static_bias)  # conv instead of a linear one

        z0 = F.conv1d(x, self.static_weight)               # W_0 * x, bias held back
        h = self.branch(x.mean(dim=-1))                    # global average pool over time
        lam = torch.sigmoid(self.to_lambda(h))              # [B, C_out]
        pi = torch.softmax(self.to_pi(h), dim=-1)          # [B, K]
        self.last_pi = pi.detach()

        # Lambda(x) W_0 * x == lam * (W_0 * x), since Lambda is diagonal
        out = z0 + lam.unsqueeze(-1) * z0 # .unsqueeze(-1) because of time dim

        # all K bases in one conv: [B, K*C_out, L-k_phi+1] -> now con instead of linear
        z = F.conv1d(x, self.residual_weights.reshape(
            self.K * self.out_dim, self.in_dim, self.k_phi))
        z = z[..., self.k - self.k_phi:]                   # align with W_0's output positions
        z = z.reshape(z.size(0), self.K, self.out_dim, -1)
        out = out + (z * pi[:, :, None, None]).sum(dim=1)

        return out + self.static_bias.unsqueeze(-1)

class DynamicBackbone(nn.Module):
    """Two dynamic temporal conv layers. A window's feature vector is the
    output at its last beat. Receptive field: 1 + 2*(kernel_size-1) beats."""
    def __init__(self, in_dim=14, hidden_dim=128, K=4, drt_activated=True, basis_scale=0.1,
                 kernel_size=3, phi_kernel=1):
        super().__init__()
        kw = dict(K=K, drt_activated=drt_activated, basis_scale=basis_scale,
                  kernel_size=kernel_size, phi_kernel=phi_kernel)
        self.receptive_field = 1 + 2 * (kernel_size - 1)

        self.layer1 = DynamicConv1d(in_dim, 128, **kw)
        self.bn1, self.relu1 = nn.BatchNorm1d(128), nn.ReLU()

        self.layer2 = DynamicConv1d(128, hidden_dim, **kw)
        self.bn2, self.relu2 = nn.BatchNorm1d(hidden_dim), nn.ReLU()

    def forward(self, x):                                  # x: [B, C_in, L]
        h = self.relu1(self.bn1(self.layer1(x)))           # [B, 256, L-k+1]
        h = self.relu2(self.bn2(self.layer2(h)))           # [B, hidden, L-2k+2]
        return h[..., -1]                                  # last beat -> [B, hidden]

class FlatWindow(nn.Module):
    """Lets flat-row tools like KernelSHAP call the temporal backbone:
    [B, C*L] -> [B, C, L] -> G."""
    def __init__(self, G, C, L):
        super().__init__()
        self.G, self.C, self.L = G, C, L

    def forward(self, x):
        return self.G(x.reshape(-1, self.C, self.L))

class RegressorHead(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, feat):
        return self.net(feat).squeeze(-1)

def discrepancy_loss(pred1, pred2):
    return torch.mean(1.0 - torch.exp(-torch.abs(pred1 - pred2)))
# def discrepancy_loss(pred1, pred2, kind="bounded"):
#     d = torch.abs(pred1 - pred2)
#     return d.mean() if kind == "l1" else torch.mean(1.0 - torch.exp(-d))

def variance_penalty(pred1, pred2):
    return -(torch.var(pred1) + torch.var(pred2))


@torch.no_grad()
def evaluate(G, F1, F2, loader, device, normalizer):
    G.eval()
    #Router.eval()
    F1.eval()
    F2.eval()

    preds, targets = [], []

    for X_b, Y_b in loader:
        X_b, Y_b = X_b.to(device), Y_b.to(device)
        feat = G(X_b)
        #sub1, sub2 = Router(feat)
        # ensemble: average of the two heads
        pred = (F1(feat) + F2(feat)) / 2.0
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
def make_pseudo_labels(G, F1, F2, X_trg, device, keep_quantile):
    """
    One-shot pseudo-labelling of the unlabelled target split.

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

    p1s, p2s = [], []
    for i in range(0, len(Xt), 512):
        feat = G(Xt[i:i + 512])
        p1s.append(F1(feat)); p2s.append(F2(feat))
    p1, p2 = torch.cat(p1s), torch.cat(p2s)

    pseudo = ((p1 + p2) / 2.0).cpu().numpy()
    disc = torch.abs(p1 - p2).cpu().numpy()
    thr = float(np.quantile(disc, keep_quantile))
    keep = disc <= thr
    return X_trg[keep], pseudo[keep], float(keep.mean()), thr

def run_phase(G, F1, F2, opt_g, opt_f, sch_g, sch_f, src_loader, trg_loader,
              val_loader, normalizer, device, args, n_epochs, epoch_offset,
              phase, pseudo_loader=None):
    """One training phase.

    Metric names are identical in both phases and the epoch counter continues
    across them, so every MLflow curve is continuous over the whole run; the
    `phase` metric marks where the self-training round starts.
    """
    metrics = None
    trg_iter = iter(trg_loader) if args.MCD_activated else None
    pseudo_iter = iter(pseudo_loader) if pseudo_loader is not None else None
    head_params = list(F1.parameters()) + list(F2.parameters())

    for epoch in range(1, n_epochs + 1):
        gstep = epoch_offset + epoch
        G.train(); F1.train(); F2.train()

        # Discrepancy warmup factor
        if gstep <= args.warmup_epochs:
            da_weight = 0.0
        else:
            da_weight = min(1.0, (gstep - args.warmup_epochs) / 20.0)

        stats = {"mse1": 0.0, "mse2": 0.0, "st": 0.0, "dis": 0.0,
                 "dis_s": 0.0, "vpen": 0.0, "n": 0, "nb": 0}

        for X_s, Y_s in src_loader:
            bsz = X_s.size(0)
            X_s, Y_s = X_s.to(device), Y_s.to(device)

            if args.MCD_activated:
                try:
                    X_t, _ = next(trg_iter)
                except StopIteration:
                    trg_iter = iter(trg_loader)
                    X_t, _ = next(trg_iter)
                X_t = X_t.to(device)

            # ---- Step A: supervised risk, plus pseudo-labelled target ----
            for p in G.parameters():
                p.requires_grad = True
            for p in head_params:
                p.requires_grad = True

            feat_s = G(X_s)
            loss_s1 = F.mse_loss(F1(feat_s), Y_s)
            loss_s2 = F.mse_loss(F2(feat_s), Y_s)
            loss_stepA = loss_s1 + loss_s2

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
                    l_st = (F.mse_loss(F1(feat_p), Yp)
                            + F.mse_loss(F2(feat_p), Yp))
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
                # Step B: heads maximise target discrepancy, backbone frozen.
                for p in G.parameters():
                    p.requires_grad = False
                for p in head_params:
                    p.requires_grad = True

                feat_s_d = G(X_s).detach()
                feat_t_d = G(X_t).detach()

                pred_s1, pred_s2 = F1(feat_s_d), F2(feat_s_d)
                pred_t1, pred_t2 = F1(feat_t_d), F2(feat_t_d)

                dis = discrepancy_loss(pred_t1, pred_t2)
                vpen = variance_penalty(pred_t1, pred_t2)

                # stats["dis_s"] += float(discrepancy_loss(pred_s1, pred_s2).detach())

                # Minimising a negative 'dis' term maximises the discrepancy
                loss_stepB = (F.mse_loss(pred_s1, Y_s)
                              + F.mse_loss(pred_s2, Y_s)
                              + da_weight * (args.gamma * vpen
                                             - args.eta * dis))
                # loss_stepB = (F.mse_loss(pred_s1, Y_s)
                #               + F.mse_loss(pred_s2, Y_s)
                #               - da_weight * args.eta * dis)

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
                    pred_t1, pred_t2 = F1(feat_t), F2(feat_t)

                    dis = discrepancy_loss(pred_t1, pred_t2)
                    vpen = variance_penalty(pred_t1, pred_t2)

                    # Minimising positive 'dis' pulls the target inside the boundary
                    loss_stepC = da_weight * (args.eta * dis + args.gamma * vpen)

                    opt_g.zero_grad()
                    loss_stepC.backward()
                    opt_g.step()

                G.eval()
                with torch.no_grad():
                    p1s, p2s = F1(G(X_s)), F2(G(X_s))
                    p1t, p2t = F1(G(X_t)), F2(G(X_t))
                    stats["dis_s"] += float(discrepancy_loss(p1s, p2s))
                    stats["dis"]   += float(discrepancy_loss(p1t, p2t))
                    stats["vpen"]  += float(variance_penalty(p1t, p2t))
                G.train()

        sch_g.step(); sch_f.step()

        n, nb = max(stats["n"], 1), max(stats["nb"], 1)
        mae, rmse, r2, corr = evaluate(G, F1, F2, val_loader, device, normalizer)
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

    # normalize on source only (per beat, before windowing)
    normalizer = Normalizer().fit(X_src_all, Y_src_all)

    # window each animal separately so that no window spans two animals;
    # src_windows keeps raw Y per animal for dump_pi below
    L = args.window
    src_windows = [make_windows(normalizer.transform_X(Xs), Ys, L)
                   for Xs, Ys in source_domains]
    X_src_n = np.concatenate([Xw for Xw, _ in src_windows], axis=0)
    Y_src_n = normalizer.transform_Y(np.concatenate([Yw for _, Yw in src_windows]))

    target_X_n, target_Y_n = make_windows(normalizer.transform_X(target_X),
                                          normalizer.transform_Y(target_Y), L)

    # split in train/eval (gap only used by block mode)
    (trg_train_X, trg_train_Y), (trg_val_X, trg_val_Y) = split_domain(
        target_X_n, target_Y_n, val_ratio=0.2, seed=args.seed,
        mode=args.val_split, gap=L - 1)

    # Dataloader trg_train_Y not used
    src_loader = make_loader(X_src_n, Y_src_n, args.batch_size)
    trg_loader = make_loader(trg_train_X, trg_train_Y, args.batch_size)
    val_loader = make_loader(trg_val_X, trg_val_Y, args.batch_size,
                             shuffle=False, drop_last=False)

    G = DynamicBackbone(
        in_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        K= args.K,
        drt_activated=args.DRT_activated,
        basis_scale = args.basis_scale,
        kernel_size=args.kernel_size,
        phi_kernel=args.phi_kernel,
    ).to(device)
    if L < G.receptive_field:
        raise ValueError(f"--window {L} is shorter than the receptive field "
                         f"({G.receptive_field} beats for kernel_size={args.kernel_size})")

    F1 = RegressorHead(hidden_dim=args.hidden_dim).to(device)
    F2 = RegressorHead(hidden_dim=args.hidden_dim).to(device)

    opt_g = optim.SGD(G.parameters(), lr=args.lr, momentum=0.9)
    opt_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()),
                      lr=args.lr_f, momentum=0.9)

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
        metrics = run_phase(G, F1, F2, opt_g, opt_f, sch_g, sch_f,
                            src_loader, trg_loader, val_loader, normalizer,
                            device, args, args.epochs, 0, phase=0)

        #Phase 2: self-training
        if args.ST_activated:
            Xp, Yp, keep_rate, thr = make_pseudo_labels(
                G, F1, F2, trg_train_X, device, args.st_keep_quantile)

            # mlflow.log_metrics({"st_keep_rate": keep_rate,
            #                     "st_threshold": thr,
            #                     "st_n_pseudo": float(len(Xp))},
            #                    step=args.epochs)
            print(f"  [ST] kept {keep_rate:.1%} of target samples "
                  f"({len(Xp)} pseudo-labelled, thr={thr:.4f})")

            if len(Xp) > args.batch_size:
                pseudo_loader = make_loader(Xp, Yp, args.batch_size)

                # Lower learning rate for the refinement round, and a fresh
                # scheduler with no milestones so it stays constant.
                for g in opt_g.param_groups:
                    g["lr"] = args.lr * args.st_lr_scale
                for g in opt_f.param_groups:
                    g["lr"] = args.lr_f * args.st_lr_scale
                sch_g = optim.lr_scheduler.MultiStepLR(opt_g, [], gamma=0.1)
                sch_f = optim.lr_scheduler.MultiStepLR(opt_f, [], gamma=0.1)

                metrics = run_phase(G, F1, F2, opt_g, opt_f, sch_g, sch_f,
                                    src_loader, trg_loader, val_loader,
                                    normalizer, device, args, args.st_epochs,
                                    args.epochs, phase=1,
                                    pseudo_loader=pseudo_loader)
            else:
                print(f"  [ST] skipped: only {len(Xp)} samples kept, "
                      f"need > {args.batch_size}")


        # ---- post-hoc SHAP (one seed only; ~410k forward rows per beat) ----
        if args.shap and args.seed == args.shap_seed:
            base = (args.feature_names.split(",") if args.feature_names
                     else [f"f{i}" for i in range(args.input_dim)])
            # KernelSHAP needs flat rows, so every (feature, lag) pair becomes
            # one input: 14*L features instead of 14. Sum over lags afterwards
            # for per-feature importance. Order matches reshape(N, C*L).
            names = [f"{n}@t-{L - 1 - j}" for n in base for j in range(L)]
            run_shap_for_fold(
                FlatWindow(G, args.input_dim, L), F1, F2,
                X_background=trg_train_X.reshape(len(trg_train_X), -1),  # variation within this animal
                X_explain=trg_val_X.reshape(len(trg_val_X), -1),         # the same split evaluate() scores
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

        # ---- actual vs predicted + residuals over time (one seed only) ----
        if args.seed == args.shap_seed:
            fold = args.target_x_dict.replace("X_", "").replace(".csv", "")
            cfg = (f"mcd{int(args.MCD_activated)}drt{int(args.DRT_activated)}"
                   f"st{int(args.ST_activated)}")
            os.makedirs("pred_plots", exist_ok=True)
            plot_fold(
                G, F1, F2,
                X=target_X_n,            # full animal as windows, unshuffled -> row order = time
                y=target_Y_n,            # (row i is beat i + L-1); normalized, plot_fold de-normalizes
                animal_id=fold,
                y_mean=float(normalizer.Y_mean),
                y_std=float(normalizer.Y_std),
                device=device,
                out_path=f"pred_plots_conv/{cfg}_{fold}_seed{args.seed}.png",
            )

        #dump Pi
        if args.pi_dump_dir and args.DRT_activated:
            names = [args.target_x_dict] + args.source_x_dicts.split(",")
            names = [n.replace("X_", "").replace(".csv", "") for n in names]
            domains = [(trg_val_X, normalizer.inverse_Y(trg_val_Y))]
            domains += src_windows      # (normalized X windows, raw Y) per source animal

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
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--schedule", type=int, nargs="+", default=[110, 135, 145])
    p.add_argument("--seed", type=int, default=100)

    # DRT
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--basis_scale", type=float, default=0.1, help="init scale of the Phi_i basis matrices")
    p.add_argument("--pi_dump_dir", type=str, default="pi_dumps")
    p.add_argument("--pi_max_per_domain", type=int, default=2000)

    # temporal convolution
    p.add_argument("--window", type=int, default=5,
                   help="beats per window; >= receptive field 1+2*(kernel_size-1)")
    p.add_argument("--kernel_size", type=int, default=3, help="temporal width of W_0")
    p.add_argument("--phi_kernel", type=int, default=3,
                   help="temporal width of the bases Phi_i (released code: 1)")
    p.add_argument("--val_split", type=str, default="random", choices=["random", "block"],
                   help="target train/val split of the windows")

    #MCD
    p.add_argument("--eta", type=float, default=0.1, help="discrepancy weight")
    p.add_argument("--gamma", type=float, default=0.001, help="variance penalty weight")
    p.add_argument("--num_k", type=int, default=1,
                   help="generator updates per batch in Step C (MCD's num_k)")

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
    p.add_argument("--shap_dir", type=str, default="shap_dumps_conv")
    p.add_argument("--shap_n", type=int, default=100, help="target beats to explain")
    p.add_argument("--shap_background", type=int, default=25, help="k-means centroids")
    p.add_argument("--shap_seed", type=int, default=100, help="only this seed gets SHAP")
    p.add_argument("--feature_names", type=str, default="", help="comma-separated, 14 names")

    return p.parse_args()

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
        # (True, False, False),
        (True, True, False),
        # (True, True, True)
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