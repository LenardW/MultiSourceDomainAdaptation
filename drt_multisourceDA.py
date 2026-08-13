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

def set_seed(seed):
    """
    set seed for every rng
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

# class ChannelAttention1d(nn.Module):
#     def __init__(self, dim, reduction=16, drt_activated=True):
#         super().__init__()
#         self.drt_activated = drt_activated
#         # We don't have the Squeeze step here from Squeeze-and-Excitation paper (Hu et al.)
#         # since we don't have spatial feature maps like in CNNs to summarize
#         self.fc = nn.Sequential(
#             nn.Linear(dim, dim // reduction, bias=False), # FC 256 -> 16
#             nn.ReLU(),
#             # forced through a bottleneck to learn relationships
#             nn.Linear(dim // reduction, dim, bias=False), # FC 16 -> 256
#             nn.Sigmoid()    # output [0,1]
#         )
#
#     def forward(self, x):
#         if not self.drt_activated:
#             return x # static transfer: if not activated, nothing happens
#         attention_weights = self.fc(x)
#         # update: W(x) = W_0 + Λ(x) W_0 like in the DRT paper
#         return x + x * attention_weights # not x * attention_weights like in Squeeze-and-Excitation paper (Hu et al.)
#         # that could be a point to test if channel suppression or boosting works better
#         # also called residual gating
#         # can also try out = x * 2 * torch.sigmoid(weights) for [0,2]

class DynamicLinear(nn.Module):
    def __init__(self, in_dim, out_dim, K=4, reduction=16, drt_activated=True, min_hidden=16):
        super().__init__()
        self.K = K
        self.drt_activated = drt_activated

        #static component
        self.static_weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.static_bias = nn.Parameter(torch.Tensor(out_dim))
        nn.init.kaiming_uniform_(self.static_weight, a=np.sqrt(5))
        nn.init.zeros_(self.static_bias)

        # basis matrices for subspace routing
        self.residual_weights = nn.Parameter(torch.Tensor(K, out_dim, in_dim))
        nn.init.zeros_(self.residual_weights)

        # ONE shared lightweight branch feeds both Λ(x) and π(x),
        # mirroring the single attention branch in the paper's Fig. 3
        #hidden = max(in_dim // reduction, 4)
        hidden = min(max(in_dim // reduction, min_hidden), in_dim) # first layer get 14 second gets 16
        self.branch = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.to_lambda = nn.Linear(hidden, out_dim)   # -> sigmoid, Λ(x)
        self.to_pi     = nn.Linear(hidden, K)          # -> softmax, π(x)

    def forward(self, x):
        if not self.drt_activated:
            return F.linear(x, self.static_weight, self.static_bias)

        h = self.branch(x)
        lam = torch.sigmoid(self.to_lambda(h))           # [B, out_dim]
        pi  = torch.softmax(self.to_pi(h), dim=-1)        # [B, K]

        lam_term = lam.unsqueeze(-1) * self.static_weight.unsqueeze(0)     # Λ(x)W_0, per sample
        pi_term  = torch.einsum('bk,kij->bij', pi, self.residual_weights)  # Σπ_i(x)Φ_i

        weight = self.static_weight.unsqueeze(0) + lam_term + pi_term
        return torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1) + self.static_bias

        # # Dynamically blend residual weights per sample based on routing paths
        # res_w = torch.einsum('bk,kij->bij', routing_weights, self.residual_weights)
        # weight = self.static_weight.unsqueeze(0) + res_w  # [B, out_dim, in_dim]
        #
        # # Batch matrix multiplication: applying custom blended weights to each sample
        # return torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1) + self.static_bias

class DynamicBackbone(nn.Module):
    def __init__(self, in_dim=14, hidden_dim=128, K=4, drt_activated=True):
        super().__init__()
        self.layer1 = DynamicLinear(in_dim, 256, K=K, drt_activated=drt_activated)
        self.bn1, self.relu1 = nn.BatchNorm1d(256), nn.ReLU()

        self.layer2 = DynamicLinear(256, hidden_dim, K=K, drt_activated=drt_activated)
        self.bn2, self.relu2 = nn.BatchNorm1d(hidden_dim), nn.ReLU()

    def forward(self, x):
        h = self.relu1(self.bn1(self.layer1(x)))
        h = self.relu2(self.bn2(self.layer2(h)))
        return h


# class DynamicBackbone(nn.Module):
#     def __init__(self, in_dim=14, hidden_dim=128, K=4, drt_activated=True):
#         super().__init__()
#         self.K = K
#         self.drt_activated = drt_activated
#
#         # Feature extractor with Channel Attention
#         self.encoder = nn.Sequential(
#             nn.Linear(in_dim, 256),
#             ChannelAttention1d(256, drt_activated=drt_activated), # dynamic weight combination before BN ReLu
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#
#             nn.Linear(256, 256),
#             ChannelAttention1d(256, drt_activated=drt_activated),
#             nn.BatchNorm1d(256),
#             nn.ReLU()
#         )
#
#         # Predicts routing coefficients
#         self.routing_net = nn.Sequential(
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, K),
#             nn.Softmax(dim=-1)
#         )
#
#         self.dynamic_layer = DynamicLinear(256, hidden_dim, K=K)
#         self.bn_out = nn.BatchNorm1d(hidden_dim)
#         self.relu_out = nn.ReLU()
#
#     def forward(self, x):
#         h = self.encoder(x)
#         if self.drt_activated:
#             routing_weights = self.routing_net(h)
#         else:
#             # Fallback to standard baseline: Zero out routing weights to disable the dynamic path
#             routing_weights = torch.zeros(x.size(0), self.K, device=x.device)
#
#         h_dyn = self.dynamic_layer(h, routing_weights)
#         return self.relu_out(self.bn_out(h_dyn))

# maybe not realy needed
# class SubspaceRouter(nn.Module):
#     def __init__(self, hidden_dim=128):
#         super().__init__()
#         self.proj1 = nn.Linear(hidden_dim, hidden_dim)
#         self.proj2 = nn.Linear(hidden_dim, hidden_dim)
#
#     def forward(self, feat):
#         return self.proj1(feat), self.proj2(feat)


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
    # R² on original scale
    ss_res = np.sum((targets_orig - preds_orig) ** 2)
    ss_tot = np.sum((targets_orig - targets_orig.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    correlation, _ = pearsonr(targets, preds)
    return mae, rmse, r2, correlation

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
            target_X_n, target_Y_n, val_ratio=0.2, seed = args.seed)

    #Dataloader

    src_loader = make_loader(X_src_n, Y_src_n, args.batch_size)
    trg_loader = make_loader(trg_train_X, trg_train_Y, args.batch_size)
    val_loader = make_loader(trg_val_X, trg_val_Y, args.batch_size,
                             shuffle=False, drop_last=False)

    G = DynamicBackbone(
        in_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        K=4,
        drt_activated=args.DRT_activated
    ).to(device)

    #Router = SubspaceRouter(hidden_dim=args.hidden_dim).to(device)
    F1 = RegressorHead(hidden_dim=args.hidden_dim).to(device)
    F2 = RegressorHead(hidden_dim=args.hidden_dim).to(device)

    opt_g = optim.SGD(G.parameters(), lr=args.lr, momentum=0.9)
    opt_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()),lr=args.lr_f, momentum=0.9)

    scheduler_g = optim.lr_scheduler.MultiStepLR(opt_g, milestones=args.schedule, gamma=0.1)
    scheduler_f = optim.lr_scheduler.MultiStepLR(opt_f, milestones=args.schedule, gamma=0.1)

    # We only need to iterate through target batches if MCD or ST is active
    use_target = args.MCD_activated or args.ST_activated

    run_name = (f"{args.target_x_dict.replace('X_', '').replace('.csv', '')}"
                f"_seed{args.seed}")

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params({k: (str(v) if isinstance(v, (list, tuple)) else v)
                           for k, v in vars(args).items()})
        if use_target:
            trg_iter = iter(trg_loader)

        for epoch in range(1, args.epochs + 1):
            G.train()
            F1.train()
            F2.train()
            #Router.train()

            # Discrepancy warmup factor
            if epoch <= args.warmup_epochs:
                da_weight = 0.0
            else:
                da_weight = min(1.0, (epoch - args.warmup_epochs) / 20.0)

            stats = {"mse1": 0.0, "mse2": 0.0, "n": 0}

            for X_s, Y_s in src_loader:
                bsz = X_s.size(0)
                X_s, Y_s = X_s.to(device), Y_s.to(device)

                if use_target:
                    try:
                        X_t, _ = next(trg_iter)
                    except StopIteration:
                        trg_iter = iter(trg_loader)
                        X_t, _ = next(trg_iter)
                    X_t = X_t.to(device)

                # Step A: supervised (and ST)
                # calculate gradients
                for p in G.parameters(): p.requires_grad = True
                for p in  list(F1.parameters()) + list(F2.parameters()):
                    p.requires_grad = True

                feat_s = G(X_s)
                #sub1_s, sub2_s = Router(feat_s)
                pred_s1 = F1(feat_s)
                pred_s2 = F2(feat_s)

                loss_s1 = F.mse_loss(pred_s1, Y_s)
                loss_s2 = F.mse_loss(pred_s2, Y_s)
                loss_stepA = loss_s1 + loss_s2

                #logg params
                st_keep_rate = 0.0

                # Self-Training Block
                if args.ST_activated and epoch > args.warmup_epochs: # use st only after warmup to not train on incorrect predictions
                    # eval mode for turn off batch norm and dropout
                    G.eval()
                    #Router.eval()
                    F1.eval()
                    F2.eval()

                    # dont calculate gradients for pseudo-label generation
                    with torch.no_grad():
                        feat_t_st = G(X_t)
                        #sub1_t_st, sub2_t_st = Router(feat_t_st)
                        pred_t1_st = F1(feat_t_st)
                        pred_t2_st = F2(feat_t_st)

                        # pseudo-label via average
                        pseudo_Y = (pred_t1_st + pred_t2_st) / 2.0

                        # uses consensus instead of confidence ->  distance head predictions
                        discrepancy = torch.abs(pred_t1_st - pred_t2_st)

                        # dynamic threshold
                        st_threshold = Y_s.std().item() * 0.1
                        confident_mask = (discrepancy < st_threshold).squeeze(-1)

                        # for logging
                        st_keep_rate = confident_mask.float().mean().item()

                    # switch back to gradient calculation for training
                    G.train()
                    #Router.train()
                    F1.train()
                    F2.train()

                    if confident_mask.sum() > 1: #only if we have more than two for batchnorm

                        # forward pass with gradients on selected target data
                        feat_t_conf = G(X_t[confident_mask])
                        #sub1_t_conf, sub2_t_conf = Router(feat_t_conf)
                        pred_t1_conf = F1(feat_t_conf)
                        pred_t2_conf = F2(feat_t_conf)

                        loss_st = 0.5 * (
                                F.mse_loss(pred_t1_conf, pseudo_Y[confident_mask]) +
                                F.mse_loss(pred_t2_conf, pseudo_Y[confident_mask])
                        )
                        loss_stepA += da_weight * loss_st # scale depending on warmup for MCD

                opt_g.zero_grad()
                opt_f.zero_grad()
                loss_stepA.backward()
                opt_g.step()
                opt_f.step()

                stats["mse1"] += loss_s1.item() * bsz
                stats["mse2"] += loss_s2.item() * bsz
                stats["n"] += bsz

                # Step B and C only for MCD
                if args.MCD_activated:
                    # Step B: Train Router / Heads (Maximize Target Discrepancy)
                    # Freeze backbone G
                    for p in G.parameters(): p.requires_grad = False
                    for p in list(F1.parameters()) + list(F2.parameters()):
                        p.requires_grad = True

                    feat_s = G(X_s).detach()
                    feat_t = G(X_t).detach()

                    #sub1_s, sub2_s = Router(feat_s)
                    #sub1_t, sub2_t = Router(feat_t)

                    pred_s1 = F1(feat_s)
                    pred_s2 = F2(feat_s)
                    pred_t1 = F1(feat_t)
                    pred_t2 = F2(feat_t)

                    loss_s1 = F.mse_loss(pred_s1, Y_s)
                    loss_s2 = F.mse_loss(pred_s2, Y_s)
                    dis = discrepancy_loss(pred_t1, pred_t2)
                    vpen = variance_penalty(pred_t1, pred_t2)

                    # Minimizing a negative 'dis' term maximizes the discrepancy boundary
                    loss_stepB = (loss_s1 + loss_s2) + da_weight * (args.gamma * vpen - args.eta * dis)

                    opt_f.zero_grad()
                    loss_stepB.backward()
                    opt_f.step()  # only update Router/ Heads

                    # Step C: Train Backbone G (Minimize Target Discrepancy)
                    # Freeze Router and Heads
                    for p in G.parameters(): p.requires_grad = True
                    for p in list(F1.parameters()) + list(F2.parameters()):
                        p.requires_grad = False

                    feat_t = G(X_t)
                    #sub1_t, sub2_t = Router(feat_t)
                    pred_t1 = F1(feat_t)
                    pred_t2 = F2(feat_t)

                    dis = discrepancy_loss(pred_t1, pred_t2)
                    vpen = variance_penalty(pred_t1, pred_t2)

                    # Minimizing positive 'dis' aligns target domain inside the boundaries
                    loss_stepC = da_weight * (args.eta * dis + args.gamma * vpen)

                    opt_g.zero_grad()
                    loss_stepC.backward()
                    opt_g.step()  # Only update Backbone G


            scheduler_g.step()
            scheduler_f.step()

            n = stats["n"]
            print(f"Epoch {epoch:03d}/{args.epochs} " #DA:{da_str}  "
                  f"MSE1={stats['mse1'] / n:.4f}  MSE2={stats['mse2'] / n:.4f}  "
                  )

            mae, rmse, r2, corr = evaluate(G, F1, F2, val_loader, device, normalizer)
            print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f} Corr.={corr:.4f}" )

            mlflow.log_metrics(
                {
                    "MSE1": stats['mse1']/n,
                    "MSE2": stats['mse2']/n,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R²": r2,
                    "Corr.": corr,
                },
                step=epoch,
            )
    return r2, corr


def parse_args():
    p = argparse.ArgumentParser(description="DRT")
    p.add_argument("--data_dict_path", type=str, default="C:/Users/leona/PythonScripts/MasterThesis/Tierdaten/")
    p.add_argument("--target_x_dict", type=str, default="X_anim1.csv")
    p.add_argument("--target_y_dict", type=str, default="Y_anim1.csv")
    p.add_argument("--source_x_dicts", type=str, default="X_anim10.csv,X_anim3.csv,X_anim4.csv,X_anim8.csv")
    p.add_argument("--source_y_dicts", type=str, default="Y_anim10.csv,Y_anim3.csv,Y_anim4.csv,Y_anim8.csv")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--warmup_epochs", type=int, default=100, help="Epochs of source-only training")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001, help="backbone lr")
    p.add_argument("--lr_f", type=float, default=0.005, help="head lr")
    p.add_argument("--eta", type=float, default=0.1, help="discrepancy weight")
    p.add_argument("--gamma", type=float, default=0.001, help="variance penalty weight")
    p.add_argument("--input_dim", type=int, default=14)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--schedule", type=int, nargs="+", default=[110, 135, 145])
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--MCD_activated", type=bool, default=False, help="If False source only")
    p.add_argument("--DRT_activated", type=bool, default=False)
    p.add_argument("--ST_activated", type=bool, default=False)

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

    allowed_values = [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, False),
        (True, True, True)
    ]

    keys = ["MCD_activated", "DRT_activated", "ST_activated"]
    combinations = [dict(zip(keys, v)) for v in allowed_values]

    for idx, config in enumerate(combinations, 1):
        args.MCD_activated = config["MCD_activated"]
        args.DRT_activated = config["DRT_activated"]
        args.ST_activated = config["ST_activated"]

        final_summary = {}
        all_target_r2_means = []
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

                seed_r2 = []
                seed_corr = []

                for i in range(len(seeds)):
                    args.seed = seeds[i]
                    run_r2, run_corr = train(args)
                    seed_r2.append(run_r2)
                    seed_corr.append(run_corr)

                mean_r2 = np.mean(seed_r2)
                std_r2 = np.std(seed_r2, ddof=1) # sample std

                mean_corr =  np.mean(seed_corr)
                std_corr = np.std(seed_corr, ddof=1)

                final_summary[target_name] = (mean_r2, std_r2, mean_corr, std_corr)
                all_target_r2_means.append(mean_r2)
                all_target_corr_means.append(mean_corr)
                print(f"Finished Target {target_name}! Result: {mean_r2:.4f} ± {std_r2:.4f}")

            mlflow.log_metrics({"mean_R²": float(np.mean(all_target_r2_means)),
                                "mean_Corr": float(np.mean(all_target_corr_means))})

        output_file = "drt_training_results.txt"

        with open(output_file, "a", encoding="utf-8") as f:

            header = f"Results for configuration: {config}: \n" + "-" * 40 + "\n"
            print(header, end="")
            f.write(header)

            for target_name, (mean_r2, std_r2, mean_corr, std_corr) in final_summary.items():
                target_info = f"Target: {target_name}\n R²: {mean_r2:.4f} ± {std_r2:.4f}\n Correlation: {mean_corr:.4f} ± {std_corr}\n" + "-" * 40 + "\n"
                print(target_info, end="")
                f.write(target_info)

            average_all_r2 = np.mean(all_target_r2_means, dtype=np.float64)
            std_all_r2 = np.std(all_target_r2_means, ddof=1)

            average_all_corr = np.mean(all_target_corr_means, dtype=np.float64)
            std_all_corr = np.std(all_target_corr_means, ddof=1)

            footer = f"Average Across All Targets:\n R²: {average_all_r2:.4f} ± {std_all_r2:.4f}\n Correlation: {average_all_corr:.4f} ± {std_all_corr:.4f}\n" + "=" * 40 + "\n\n"
            print(footer, end="")
            f.write(footer)

if __name__ == "__main__":
    main()