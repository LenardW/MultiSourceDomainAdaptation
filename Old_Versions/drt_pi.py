"""Routing diagnostics and Pi dumps for the v1 DRT implementation.

Collects the per-sample routing vector Pi = {pi_i^l(x)} across all dynamic
layers, so the Fig. 4 style t-SNE (Sec. 4.5 of arXiv:2103.10583) can be
reproduced. Duck-types on `residual_weights` / `drt_activated` so it can live
in a separate file without importing DynamicLinear.

Requires one change in DynamicLinear.forward: store `self.last_pi`.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F


def drt_layers(G):
    """Every DynamicLinear in G that actually has a routing path."""
    return [m for m in G.modules()
            if hasattr(m, "residual_weights")
            and getattr(m, "drt_activated", False)]


@torch.no_grad()
def collect_pi(G, loader, device):
    """Concatenate pi over all dynamic layers, for every sample in `loader`.

    Returns [N, n_layers * K], or None in static mode. Order of the column
    blocks follows G.modules(), i.e. layer1 then layer2.
    """
    layers = drt_layers(G)
    if not layers:
        return None

    was_training = G.training
    G.eval()
    chunks = []
    for X_b, _ in loader:
        G(X_b.to(device))
        if any(l.last_pi is None for l in layers):
            G.train(was_training)
            return None
        chunks.append(torch.cat([l.last_pi for l in layers], dim=1).cpu())
    G.train(was_training)
    return torch.cat(chunks).numpy() if chunks else None


def routing_entropy(pi, K):
    """Mean normalised entropy of pi, per layer block, in [0, 1].

    1.0 = uniform blend for every sample, i.e. the router carries no
    sample-dependent information and the dynamic path is inert.
    0.0 = hard one-hot routing.
    """
    L = pi.shape[1] // K
    out = []
    for l in range(L):
        b = np.clip(pi[:, l * K:(l + 1) * K].astype(np.float64), 1e-8, 1.0)
        out.append(float(-(b * np.log(b)).sum(axis=1).mean() / np.log(K)))
    return out


@torch.no_grad()
def basis_similarity(layer):
    """Mean pairwise |cosine| between the K basis matrices Phi_i.

    Nothing in the loss forces the Phi_i apart. If they converge, sum_i pi_i
    Phi_i stops depending on pi and routing degrades to a static layer with K
    times the parameters. Near 1 means collapse; exactly 0 means the weights
    are still at their zero init.
    """
    W = layer.residual_weights            # [K, out_dim, in_dim]
    K = W.shape[0]
    if K < 2:
        return None
    Wn = F.normalize(W.flatten(1), dim=1)
    sim = (Wn @ Wn.t()).abs()
    off = sim[~torch.eye(K, dtype=torch.bool, device=sim.device)]
    return float(off.mean())


def dump_pi(G, loader_fn, domains, names, device, out_dir, K, target_name,
            tag_extra="", max_per_domain=2000, seed=0):
    """Save Pi for every animal, with domain and target labels.

    `domains` is a list of (X_normalised, Y_original_scale) in the same order
    as `names`; index 0 is expected to be the held-out target. `loader_fn`
    is a callable (X, Y) -> DataLoader with shuffle=False, drop_last=False,
    so the row order of Pi matches Y.
    """
    if not drt_layers(G):
        return None

    rng = np.random.default_rng(seed)
    pis, doms, ys = [], [], []

    for d, (Xd, Yd) in enumerate(domains):
        if len(Xd) > max_per_domain:
            sel = rng.choice(len(Xd), max_per_domain, replace=False)
            Xd, Yd = Xd[sel], Yd[sel]
        pi_d = collect_pi(G, loader_fn(Xd, Yd), device)
        if pi_d is None:
            continue
        n = len(pi_d)
        pis.append(pi_d)
        doms.append(np.full(n, d, dtype=np.int64))
        ys.append(np.asarray(Yd[:n], dtype=np.float32))

    if not pis:
        return None

    pi_all = np.concatenate(pis)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"pi_{target_name}{tag_extra}.npz")
    np.savez_compressed(path,
                        pi=pi_all,
                        domain=np.concatenate(doms),
                        y=np.concatenate(ys),
                        names=np.array(names),
                        K=K,
                        target=target_name)

    ents = routing_entropy(pi_all, K)
    sims = [basis_similarity(l) for l in drt_layers(G)]
    return path, ents, sims