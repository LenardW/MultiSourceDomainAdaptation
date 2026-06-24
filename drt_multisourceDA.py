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
import mlflow

def load_csv(path):
    """
    load a CSV file into a numpy float32 array
    """
    rows = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            parsed = [float(v) for v in row]
            rows.append(parsed)
    return np.array(rows, dtype=np.float32)

def load_data(x_path, y_path):
    """
    load X and Y data.
    """
    X = load_csv(x_path)
    Y = load_csv(y_path).squeeze(1)

    print(f"Loaded {len(X)} Datapoints from {x_path}")
    print(f"  X shape: {X.shape}  |  Y shape: {Y.shape}")
    print(f"  Y range: [{Y.min():.3f}, {Y.max():.3f}]  mean: {Y.mean():.3f}")
    return X, Y

class Normalizer:
    """
    fit on source domain, apply to all domains
    normalises X feature-wise and Y globally
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

def split_domain(X, Y, val_ratio=0.2, seed=100):
    """split a domain into train / val portions"""
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    n_val = int(n * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (X[train_idx], Y[train_idx]), (X[val_idx], Y[val_idx])

def make_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    """create DataLoader"""
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    ds  = TensorDataset(X_t, Y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last)

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

# class Backbone(nn.Module):
#     """
#     MLP backbone replacing the dynamic ResNet-101
#     takes 14 features, outputs a hidden representation
#     """
#     def __init__(self, in_dim=14, hidden_dim=128, dropout=0.2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#         )
#
#     def forward(self, x):
#         return self.net(x)

# class DynamicBackbone(nn.Module):
#     def __init__(self, in_dim=14, hidden_dim=128):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU()
#         )
#         # attention network  generates per-sample feature weights
#         self.attention = nn.Sequential(
#             nn.Linear(256, 64), nn.ReLU(),
#             nn.Linear(64, 256), nn.Sigmoid()   # weights in [0,1]
#         )
#         self.out = nn.Sequential(
#             nn.Linear(256, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
#         )
#
#     def forward(self, x):
#         h = self.encoder(x)
#         alpha = self.attention(h)   # input-conditioned weights
#         h = h * alpha               # channel attention
#         return self.out(h)

class DynamicLinear(nn.Module):
    def __init__(self, in_dim, out_dim, K=4):
        super().__init__()
        self.K = K
        self.static_weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.static_bias = nn.Parameter(torch.Tensor(out_dim))
        self.residual_weights = nn.Parameter(torch.Tensor(K, out_dim, in_dim))

        nn.init.kaiming_uniform_(self.static_weight, a=np.sqrt(5))
        nn.init.zeros_(self.static_bias)
        nn.init.zeros_(self.residual_weights)

    def forward(self, x, routing_weights):
        # Dynamically compute weights per sample
        res_w = torch.einsum('bk,kij->bij', routing_weights, self.residual_weights)
        weight = self.static_weight.unsqueeze(0) + res_w  # [B, out_dim, in_dim]

        # Batch linear projection
        out = torch.bmm(x.unsqueeze(1), weight.transpose(1, 2)).squeeze(1)
        out = out + self.static_bias
        return out


class DynamicBackbone(nn.Module):
    def __init__(self, in_dim=14, hidden_dim=128, K=4):
        super().__init__()
        self.K = K

        # 1. Base feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # 2. Routing network: Predicts the π coefficients for the dynamic weights.
        # The DRT paper uses a Softmax over K components so they sum to 1.
        self.routing_net = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, K),
            nn.Softmax(dim=-1)
        )

        # 3. Dynamic alternative to standard nn.Linear
        self.dynamic_layer = DynamicLinear(256, hidden_dim, K=K)

        # 4. Final output processing
        self.bn_out = nn.BatchNorm1d(hidden_dim)
        self.relu_out = nn.ReLU()

    def forward(self, x):
        # Extract base features
        h = self.encoder(x)

        # Generate sample-conditioned routing weights π(x)
        routing_weights = self.routing_net(h)  # Shape: [batch_size, K]

        # Pass features AND routing weights into the Dynamic Layer
        h_dyn = self.dynamic_layer(h, routing_weights)

        return self.relu_out(self.bn_out(h_dyn))

class SubspaceRouter(nn.Module):
    """Split hidden features into two subspaces for F1 and F2."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, feat):
        return self.proj1(feat), self.proj2(feat)


class RegressionHead(nn.Module):
    """
    Regression head replacing ResClassifier
    Outputs a single continuous prediction.
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
    # Restricts the loss cleanly between 0 and 1
    return torch.mean(1.0 - torch.exp(-torch.abs(pred1 - pred2)))

def variance_penalty(pred1, pred2):
    """
    Replaces entropy minimisation from classification DRT.
    Penalises degenerate constant predictions on target samples —
    the model should not just predict the mean for everything.
    Negative variance = penalty (we subtract it from the loss).
    """
    return -(torch.var(pred1) + torch.var(pred2))


@torch.no_grad()
def evaluate(G, router, F1, F2, loader, device, normalizer):
    G.eval()
    router.eval()
    F1.eval()
    F2.eval()

    preds, targets = [], []

    for X_b, Y_b in loader:
        X_b, Y_b = X_b.to(device), Y_b.to(device)
        feat = G(X_b)
        sub1, sub2 = router(feat)
        # ensemble: average of the two heads
        pred = (F1(sub1) + F2(sub2)) / 2.0
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
    return mae, rmse, r2, correlation

def train(args):

    #mlflow loging
    mlflow.set_experiment("DRT_multisource")
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load target domain and labels
    target_x_path = os.path.join(args.data_dict_path, args.target_x_dict)
    target_y_path = os.path.join(args.data_dict_path, args.target_y_dict)
    target_X, target_Y = load_data(target_x_path, target_y_path)


    # load source domains
    source_domains = []
    x_src_dicts =[]
    y_src_dicts =[]

    for x in args.source_x_dicts.split(","):
        x_src_dicts.append(x)

    for y in args.source_y_dicts.split(","):
        y_src_dicts.append(y)

    for i in range(len(x_src_dicts)):
        source_x_path = os.path.join(args.data_dict_path, x_src_dicts[i])
        source_y_path = os.path.join(args.data_dict_path, y_src_dicts[i])
        X_s, Y_s = load_data(source_x_path, source_y_path)
        source_domains.append((X_s, Y_s))

    #concat
    X_src_all = np.concatenate([X for X, Y in source_domains], axis=0)
    Y_src_all = np.concatenate([Y for X, Y in source_domains], axis=0)

    #normalize
    normalizer = Normalizer().fit(X_src_all, Y_src_all)

    X_src_n = normalizer.transform_X(X_src_all)
    Y_src_n = normalizer.transform_Y(Y_src_all)

    target_X_n = normalizer.transform_X(target_X)
    target_Y_n = normalizer.transform_Y(target_Y)

    #split in train/eval
    (trg_train_X, trg_train_Y), (trg_val_X, trg_val_Y) = split_domain(
            target_X_n, target_Y_n, val_ratio=0.2)

    #Dataloader

    src_loader = make_loader(X_src_n, Y_src_n, args.batch_size)
    trg_loader = make_loader(trg_train_X, trg_train_Y, args.batch_size)
    val_loader = make_loader(trg_val_X, trg_val_Y, args.batch_size,
                             shuffle=False, drop_last=False)

    print(f"\nTotal Merged Source Samples: {len(X_src_n)}")
    print(f"Target Training Samples:     {len(trg_train_X)}")
    print(f"Target Validation Samples:   {len(trg_val_X)}")

    G = DynamicBackbone(in_dim=args.input_dim, hidden_dim=args.hidden_dim).to(device)
    F1 = RegressionHead(hidden_dim=args.hidden_dim, lmbd=args.lmbd).to(device)
    F2 = RegressionHead(hidden_dim=args.hidden_dim, lmbd=args.lmbd).to(device)
    router = SubspaceRouter(hidden_dim=args.hidden_dim).to(device)

    opt_g = optim.AdamW(G.parameters(), lr=args.lr, weight_decay=5e-4)
    opt_f = optim.AdamW(list(F1.parameters()) + list(F2.parameters()) + list(router.parameters()), lr=args.lr_f, weight_decay=5e-4)

    scheduler_g = optim.lr_scheduler.MultiStepLR(opt_g, milestones=args.schedule, gamma=0.1)
    scheduler_f = optim.lr_scheduler.MultiStepLR(opt_f, milestones=args.schedule, gamma=0.1)

    best_mae = float('inf')

    print(f"\nStarting Real MSDA training for {args.epochs} epochs (warmup: {args.warmup_epochs} epochs)\n")

    with mlflow.start_run() as run:
        # Log training parameters
        mlflow.log_params(vars(args))
        for epoch in range(1, args.epochs + 1):
            G.train()
            F1.train()
            F2.train()
            router.train()

            # da_active = epoch > args.warmup_epochs
            # da_weight = 1.0 if da_active else 0.0

            if epoch <= args.warmup_epochs:
                da_weight = 0.0
            else:
                da_weight = min(1.0, (epoch - args.warmup_epochs) / 20.0)

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

                # # ── Forward Passes ──
                # data = torch.cat([X_s, X_t], dim=0)
                # feat = G(data)
                # feat_s, feat_t = feat[:bsz], feat[bsz:]
                #
                # # sub1, sub2 = router(feat_t)
                # #
                # # # Standard forward pass (No GRL) for source MSE and target variance
                # # pred_s1 = F1(feat_s, reverse=False)
                # # pred_s2 = F2(feat_s, reverse=False)
                # # pred_t1 = F1(sub1, reverse=False)
                # # pred_t2 = F2(sub2, reverse=False)
                #
                # # Pass both domains through the router
                # sub1_s, sub2_s = router(feat_s)
                # sub1_t, sub2_t = router(feat_t)
                #
                # # Make predictions on identical feature spaces
                # pred_s1 = F1(sub1_s)
                # pred_s2 = F2(sub2_s)
                # pred_t1 = F1(sub1_t)
                # pred_t2 = F2(sub2_t)
                #
                # # Supervised loss calculation on true source data
                # loss1 = F.mse_loss(pred_s1, Y_s)
                # loss2 = F.mse_loss(pred_s2, Y_s)
                #
                # # Safe Variance Penalty (Cooperative calculation, kept separate from GRL)
                # vpen = variance_penalty(pred_t1, pred_t2)
                #
                # # Adversarial Discrepancy (Uses GRL)
                # pred_t1_rev = F1(sub1, reverse=True)
                # pred_t2_rev = F2(sub2, reverse=True)
                # dis = discrepancy_loss(pred_t1_rev, pred_t2_rev)
                #
                # # Unified Loss Objective
                # total = (loss1 + loss2
                #          + da_weight * args.gamma * vpen
                #          - da_weight * args.eta * dis)
                #
                # opt_g.zero_grad()
                # opt_f.zero_grad()
                # total.backward()
                #
                # torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                # torch.nn.utils.clip_grad_norm_(list(F1.parameters()) + list(F2.parameters()) + list(router.parameters()), max_norm=1.0)
                #
                # opt_g.step()
                # opt_f.step()

                # =============================================================
                # STEP A: Train Everything on Source Data (Minimize Regression Error)
                # =============================================================
                feat_s = G(X_s)
                sub1_s, sub2_s = router(feat_s)
                pred_s1 = F1(sub1_s)
                pred_s2 = F2(sub2_s)

                loss_s1 = F.mse_loss(pred_s1, Y_s)
                loss_s2 = F.mse_loss(pred_s2, Y_s)
                loss_stepA = loss_s1 + loss_s2

                opt_g.zero_grad()
                opt_f.zero_grad()
                loss_stepA.backward()
                opt_g.step()
                opt_f.step()

                # =============================================================
                # STEP B: Train Router & Heads (Maximize Target Discrepancy)
                # =============================================================
                # We detach G here because we want to freeze the backbone
                feat_s = G(X_s).detach()
                feat_t = G(X_t).detach()

                sub1_s, sub2_s = router(feat_s)
                sub1_t, sub2_t = router(feat_t)

                pred_s1 = F1(sub1_s)
                pred_s2 = F2(sub2_s)
                pred_t1 = F1(sub1_t)
                pred_t2 = F2(sub2_t)

                loss_s1 = F.mse_loss(pred_s1, Y_s)
                loss_s2 = F.mse_loss(pred_s2, Y_s)
                dis = discrepancy_loss(pred_t1, pred_t2)
                vpen = variance_penalty(pred_t1, pred_t2)

                # Notice the MINUS sign before 'dis'. Minimizing a negative
                # loss maximizes the discrepancy, forcing the heads to find different boundaries.
                loss_stepB = (loss_s1 + loss_s2) + da_weight * (args.gamma * vpen - args.eta * dis)

                opt_g.zero_grad()
                opt_f.zero_grad()
                loss_stepB.backward()
                opt_f.step()  # ONLY optimize Router, F1, and F2

                # =============================================================
                # STEP C: Train Backbone G (Minimize Target Discrepancy)
                # =============================================================
                # We do NOT detach G here. We freeze the router and heads.
                feat_t = G(X_t)
                sub1_t, sub2_t = router(feat_t)
                pred_t1 = F1(sub1_t)
                pred_t2 = F2(sub2_t)

                dis = discrepancy_loss(pred_t1, pred_t2)
                vpen = variance_penalty(pred_t1, pred_t2)

                # Notice the PLUS sign before 'dis'. G wants to minimize
                # discrepancy, pushing target features into areas where F1 and F2 agree.
                loss_step3 = da_weight * (args.eta * dis + args.gamma * vpen)

                opt_g.zero_grad()
                opt_f.zero_grad()
                loss_step3.backward()
                opt_g.step()  # ONLY optimize Backbone G

                stats["mse1"] += loss_s1.item() * bsz
                stats["mse2"] += loss_s2.item() * bsz
                stats["dis"] += dis.item() * bsz
                stats["var"] += vpen.item() * bsz
                stats["n"] += bsz

            scheduler_g.step()
            scheduler_f.step()

            n = stats["n"]
            # da_str = "ON " if da_active else "OFF"
            print(f"Epoch {epoch:03d}/{args.epochs} " #DA:{da_str}  "
                  f"MSE1={stats['mse1'] / n:.4f}  MSE2={stats['mse2'] / n:.4f}  "
                  f"Dis={stats['dis'] / n:.4f}  Var={stats['var'] / n:.4f}")

            mae, rmse, r2, corr = evaluate(G, router, F1, F2, val_loader, device, normalizer)
            print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f} Corr.={corr:.4f}" )

            mlflow.log_metrics(
                {
                    "MSE1": stats['mse1']/n,
                    "MSE2": stats['mse2']/n,
                    "Dis": stats['dis']/n,
                    "Var": stats['var']/n,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R²": r2,
                    "Corr.": corr,
                },
                step=epoch,
            )

            if mae < best_mae:
                best_mae = mae
                print(f"new best MAE: {best_mae:.4f}")

            # Log checkpoint at the end of each epoch
            # mlflow.pytorch.log_model(G, name=f"G_checkpoint_{epoch}")
            # mlflow.pytorch.log_model(F1, name=f"F1_checkpoint_{epoch}")
            # mlflow.pytorch.log_model(F2, name=f"F1_checkpoint_{epoch}")

    print(f"Done. Best MAE on target val: {best_mae:.4f}")

def parse_args():
    p = argparse.ArgumentParser(description="DRT")
    p.add_argument("--data_dict_path", type=str, default="C:/Users/leona/PythonScripts/MasterThesis/Tierdaten/")
    p.add_argument("--target_x_dict", type=str, default="X_anim1.csv")
    p.add_argument("--target_y_dict", type=str, default="Y_anim1.csv")
    p.add_argument("--source_x_dicts", type=str, default="X_anim3.csv,X_anim4.csv,X_anim8.csv,X_anim10.csv")
    p.add_argument("--source_y_dicts", type=str, default="Y_anim3.csv,Y_anim4.csv,Y_anim8.csv,Y_anim10.csv")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--warmup_epochs", type=int, default=50,
                   help="Epochs of source-only training before DA losses activate")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.0005, help="backbone lr")
    p.add_argument("--lr_f", type=float, default=0.001, help="head lr")
    p.add_argument("--lmbd", type=float, default=0.25, help="grad reversal λ")
    p.add_argument("--eta", type=float, default=0.1, help="discrepancy weight")
    p.add_argument("--gamma", type=float, default=0.001, help="variance penalty weight")
    p.add_argument("--input_dim", type=int, default=14)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--schedule", type=int, nargs="+", default=[120, 140, 148])
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)