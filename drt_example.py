"""
DRT (Dynamic Transfer for Multi-Source Domain Adaptation)
Adapted for regression on strokework data.

Expects CSV files with:
  X_anim1.csv  — shape (n_heartbeats, 14)  — one feature per column
  Y_anim1.csv  — shape (n_heartbeats, 1)   — strokework values

Domain simulation: scale X and Y together (physiologically consistent).
"""

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import argparse
import os
from scipy.stats import pearsonr

# ─────────────────────────────────────────────────────────────────────────────
# Feature names (for reference and documentation)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    'p_head_by_lvad_flow',
    'motor_power_by_lvp',
    'auc_lvad_flow_only_pulsatile',
    'auc_lvad_flow_with_non_puls',
    'auc_lvad_current_only_pulsatile',
    'auc_lvad_current_with_non_puls',
    'mean_VAD_speed',
    'mean_pHead',
    'min_pHead',
    'max_pHead',
    'dQdtMax',
    'dQdtMin',
    'dQdtMax_smoothed',
    'dQdtMin_smoothed',
]

N_FEATURES = len(FEATURE_NAMES)   # 14


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path, n_cols):
    """
    Load a CSV file into a numpy float32 array.
    Handles both single-column and multi-column files.
    Skips header rows that cannot be parsed as floats.
    """
    rows = []
    print(path)
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 0:
                continue
            try:
                parsed = [float(v) for v in row[:n_cols]]
                if len(parsed) == n_cols:
                    rows.append(parsed)
            except ValueError:
                continue   # skip header or malformed line
    return np.array(rows, dtype=np.float32)


def load_data(x_path, y_path):
    """Load X (n, 14) and Y (n,) from CSV files."""
    X = load_csv(x_path, n_cols=N_FEATURES)
    Y = load_csv(y_path, n_cols=1).squeeze(1)
    assert len(X) == len(Y), (
        f"X has {len(X)} rows but Y has {len(Y)} rows — they must match."
    )
    print(f"Loaded {len(X)} heartbeats from {x_path}")
    print(f"  X shape: {X.shape}  |  Y shape: {Y.shape}")
    print(f"  Y range: [{Y.min():.3f}, {Y.max():.3f}]  mean: {Y.mean():.3f}")
    return X, Y


def simulate_domain_robust(X, Y, seed = 111):
    """
    Simulates real domain shift by scaling features randomly and independently,
    and applying a non-linear shift to Y.
    """
    rng = np.random.default_rng(seed)
    # Feature-wise random scaling between 0.6 and 1.8
    feature_scales = rng.uniform(0.6, 1.8, size=(1, X.shape[1]))
    X_sim = X * feature_scales

    # Non-linear target shift (simulating different pressure/volume dynamics)
    target_scale = rng.uniform(0.7, 1.5)
    Y_sim = (Y * target_scale) + rng.uniform(-10, 10)

    return X_sim.astype(np.float32), Y_sim.astype(np.float32)


def make_all_domains(X_real, Y_real):
    """
    Returns a list of (X, Y) tuples: first entry is the real data,
    rest are simulated domains at the given scales.
    """
    domains = [(X_real, Y_real)]   # domain 0 = real / original
    for _ in range(6):
        Xs, Ys = simulate_domain_robust(X_real, Y_real)
        domains.append((Xs, Ys))
    return domains

class Normalizer:
    """
    Fit on source domain, apply to all domains.
    Normalises X feature-wise and Y globally.
    Important: fit only on source to avoid target data leakage.
    """
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

def make_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    ds  = TensorDataset(X_t, Y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last)

def split_domain(X, Y, val_ratio=0.2, seed=42):
    """Split a domain into train / val portions."""
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    n_val = int(n * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (X[train_idx], Y[train_idx]), (X[val_idx], Y[val_idx])

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lmbd):
        ctx.lmbd = lmbd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lmbd, None


def grad_reverse(x, lmbd=1.0):
    return GradReverse.apply(x, lmbd)


class StrokeworkBackbone(nn.Module):
    """
    MLP backbone replacing the dynamic ResNet-101.
    Takes 14 heartbeat features, outputs a hidden representation.
    """
    def __init__(self, in_dim=N_FEATURES, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class RegressionHead(nn.Module):
    """
    Regression head replacing ResClassifier.
    Outputs a single continuous strokework prediction.
    Supports gradient reversal for the discrepancy loss.
    """
    def __init__(self, hidden_dim=128, lmbd=0.25):
        super().__init__()
        self.lmbd = lmbd
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, feat, reverse=False):
        x = grad_reverse(feat, self.lmbd) if reverse else feat
        return self.net(x).squeeze(1)


def discrepancy_loss(pred1, pred2):
    """
    Replaces |softmax(F1) - softmax(F2)| from classification DRT.
    For regression: mean absolute difference between the two heads.
    """
    return torch.mean(torch.abs(pred1 - pred2))


def variance_penalty(pred1, pred2):
    """
    Replaces entropy minimisation from classification DRT.
    Penalises degenerate constant predictions on target samples —
    the model should not just predict the mean strokework for everything.
    Negative variance = penalty (we subtract it from the loss).
    """
    return -(torch.var(pred1) + torch.var(pred2))


@torch.no_grad()
def evaluate(G, F1, F2, loader, device, normalizer):
    G.eval(); F1.eval(); F2.eval()
    preds, targets = [], []

    for X_b, Y_b in loader:
        X_b, Y_b = X_b.to(device), Y_b.to(device)
        feat = G(X_b)
        # ensemble: average of the two heads
        pred = (F1(feat) + F2(feat)) / 2.0
        preds.append(pred.cpu())
        targets.append(Y_b.cpu())

    preds   = torch.cat(preds).numpy().flatten()
    targets = torch.cat(targets).numpy().flatten()

    # Denormalise for interpretable metrics
    preds_orig   = normalizer.inverse_Y(preds)
    targets_orig = normalizer.inverse_Y(targets)

    mae  = np.mean(np.abs(preds_orig - targets_orig))
    rmse = np.sqrt(np.mean((preds_orig - targets_orig) ** 2))
    # R² on original scale
    ss_res = np.sum((targets_orig - preds_orig) ** 2)
    ss_tot = np.sum((targets_orig - targets_orig.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    correlation, _ = pearsonr(targets, preds)
    print(f" Correlation (r): {correlation:.4f}")

    return mae, rmse, r2


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"\nDevice: {device}")

    # ── 1. Load the Target Animal Domain ─────────────────────────────────────
    # The target animal is defined by your --x_path and --y_path arguments
    target_X, target_Y = load_data(args.x_path, args.y_path)

    # ── 2. Dynamically Load Selected Real Source Animals ──────────────────────
    # Looks in the same directory as the target data for your specific source files
    base_dir = os.path.dirname(args.x_path)
    source_ids = [3, 4, 8, 10]

    source_domains = []
    print("\nLoading real multi-source domains:")
    for s_id in source_ids:
        x_src_path = os.path.join(base_dir, f"X_anim{s_id}.csv")
        y_src_path = os.path.join(base_dir, f"Y_anim{s_id}.csv")

        # Guard rails to make sure the loop handles missing files gracefully
        if os.path.exists(x_src_path) and os.path.exists(y_src_path):
            print(f"  -> Loading Source Animal {s_id}...")
            X_s, Y_s = load_data(x_src_path, y_src_path)
            source_domains.append((X_s, Y_s))

    # Is concatenate really the correct thing to do?
    X_src_all = np.concatenate([X for X, Y in source_domains], axis=0)
    Y_src_all = np.concatenate([Y for X, Y in source_domains], axis=0)

    # ── 4. Normalise: Fit ONLY on the pooled real sources, apply everywhere ──
    normalizer = Normalizer().fit(X_src_all, Y_src_all)

    X_src_n = normalizer.transform_X(X_src_all)
    Y_src_n = normalizer.transform_Y(Y_src_all)

    target_X_n = normalizer.transform_X(target_X)
    target_Y_n = normalizer.transform_Y(target_Y)

    # ── 5. Split Target Domain into Train (unlabelled) and Val (evaluation) ──
    (trg_train_X, trg_train_Y), (trg_val_X, trg_val_Y) = split_domain(
        target_X_n, target_Y_n, val_ratio=0.2
    )
    # ── 6. Create Balanced DataLoaders ────────────────────────────────────────

    src_loader = make_loader(X_src_n, Y_src_n, args.batch_size)
    trg_loader = make_loader(trg_train_X, trg_train_Y, args.batch_size)
    val_loader = make_loader(trg_val_X, trg_val_Y, args.batch_size,
                             shuffle=False, drop_last=False)

    print(f"\nTotal Merged Source Samples: {len(X_src_n)}")
    print(f"Target Training Samples:     {len(trg_train_X)}")
    print(f"Target Validation Samples:   {len(trg_val_X)}")

    # ── 7. Models and Optimizers ──────────────────────────────────────────────
    G = StrokeworkBackbone(in_dim=N_FEATURES, hidden_dim=args.hidden_dim).to(device)
    F1 = RegressionHead(hidden_dim=args.hidden_dim, lmbd=args.lmbd).to(device)
    F2 = RegressionHead(hidden_dim=args.hidden_dim, lmbd=args.lmbd).to(device)

    opt_g = optim.SGD(G.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    opt_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()), lr=args.lr_f, momentum=0.9, weight_decay=5e-4)

    scheduler_g = optim.lr_scheduler.MultiStepLR(opt_g, milestones=args.schedule, gamma=0.1)
    scheduler_f = optim.lr_scheduler.MultiStepLR(opt_f, milestones=args.schedule, gamma=0.1)

    best_mae = float('inf')
    os.makedirs(args.save_path, exist_ok=True)

    print(f"\nStarting Real MSDA training for {args.epochs} epochs (warmup: {args.warmup_epochs} epochs)\n")

    # ── 8. Training Loop ──────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        G.train();
        F1.train();
        F2.train()

        da_active = epoch > args.warmup_epochs
        da_weight = 1.0 if da_active else 0.0

        trg_iter = iter(trg_loader)
        stats = {"mse1": 0., "mse2": 0., "dis": 0., "var": 0., "n": 0}

        for X_s, Y_s in src_loader:
            try:
                X_t, _ = next(trg_iter)
            except StopIteration:
                trg_iter = iter(trg_loader)
                X_t, _ = next(trg_iter)

            X_s, Y_s = X_s.to(device), Y_s.to(device)
            X_t = X_t.to(device)
            bsz = X_s.size(0)

            # ── Forward Passes ──
            data = torch.cat([X_s, X_t], dim=0)
            feat = G(data)
            feat_s, feat_t = feat[:bsz], feat[bsz:]

            # Standard forward pass (No GRL) for source MSE and target variance
            pred_s1 = F1(feat_s, reverse=False)
            pred_s2 = F2(feat_s, reverse=False)
            pred_t1 = F1(feat_t, reverse=False)
            pred_t2 = F2(feat_t, reverse=False)

            # Supervised loss calculation on true source data
            loss1 = F.mse_loss(pred_s1, Y_s)
            loss2 = F.mse_loss(pred_s2, Y_s)

            # Safe Variance Penalty (Cooperative calculation, kept separate from GRL)
            vpen = variance_penalty(pred_t1, pred_t2)

            # Adversarial Discrepancy (Uses GRL)
            pred_t1_rev = F1(feat_t, reverse=True)
            pred_t2_rev = F2(feat_t, reverse=True)
            dis = discrepancy_loss(pred_t1_rev, pred_t2_rev)

            # Unified Loss Objective
            total = (loss1 + loss2
                     + da_weight * args.gamma * vpen
                     - da_weight * args.eta * dis)

            opt_g.zero_grad();
            opt_f.zero_grad()
            total.backward()

            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(list(F1.parameters()) + list(F2.parameters()), max_norm=1.0)

            opt_g.step();
            opt_f.step()

            stats["mse1"] += loss1.item() * bsz
            stats["mse2"] += loss2.item() * bsz
            stats["dis"] += dis.item() * bsz
            stats["var"] += vpen.item() * bsz
            stats["n"] += bsz

        scheduler_g.step()
        scheduler_f.step()

        n = stats["n"]
        da_str = "ON " if da_active else "OFF"
        print(f"Epoch {epoch:03d}/{args.epochs}  DA:{da_str}  "
              f"MSE1={stats['mse1'] / n:.4f}  MSE2={stats['mse2'] / n:.4f}  "
              f"Dis={stats['dis'] / n:.4f}  Var={stats['var'] / n:.4f}")

        mae, rmse, r2 = evaluate(G, F1, F2, val_loader, device, normalizer)
        print(f"             Val → MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

        if mae < best_mae:
            best_mae = mae
            torch.save({
                "G": G.state_dict(),
                "F1": F1.state_dict(),
                "F2": F2.state_dict(),
                "normalizer": normalizer,
            }, os.path.join(args.save_path, "best_model.pth"))
            print(f"             ★ new best MAE: {best_mae:.4f}")
        print()

    print(f"Done.  Best MAE on target val: {best_mae:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DRT for strokework regression")
    p.add_argument("--x_path",        type=str,   default="C:/Users/leona/PythonScripts/MasterThesis/Tierdaten/X_anim1.csv")
    p.add_argument("--y_path",        type=str,   default="C:/Users/leona/PythonScripts/MasterThesis/Tierdaten/Y_anim1.csv")
    p.add_argument("--save_path",     type=str,   default="output_drt")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--warmup_epochs", type=int,   default=70,
                   help="Epochs of source-only training before DA losses activate")
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=0.001,  help="backbone lr")
    p.add_argument("--lr_f",          type=float, default=0.003,  help="head lr")
    p.add_argument("--lmbd",          type=float, default=0.25,  help="grad reversal λ")
    p.add_argument("--eta",           type=float, default=0.2,   help="discrepancy weight")
    p.add_argument("--gamma",         type=float, default=0.01,  help="variance penalty weight")
    p.add_argument("--hidden_dim",    type=int,   default=128)
    p.add_argument("--schedule",      type=int,   nargs="+", default=[40, 70, 90])
    p.add_argument("--no_cuda",       action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)