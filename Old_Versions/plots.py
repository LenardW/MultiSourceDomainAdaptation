"""
plot_predictions.py

Residuals-over-time and actual-vs-predicted strokework plots for the MCD+DRT
pipeline. Call `plot_fold(...)` at the end of each LOO fold (after training /
self-training) with that fold's target animal.

Assumptions (adjust in `collect_predictions` if your interfaces differ):
  - Feature extractor G, two regressor heads F1, F2; final prediction = mean of heads.
  - Rows of X are in temporal order for each animal (or you pass an explicit `t`).
  - If Y was standardized, pass `y_mean` / `y_std` to plot in original units.
"""

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # headless on Bjarne-PC
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_predictions(G, F1, F2, X, device="cuda", batch_size=1024):
    """Returns (pred_F1, pred_F2) as 1-D numpy arrays, in the same row order as X."""
    # eval() matters here: otherwise BatchNorm running stats get updated by
    # target data again and predictions depend on batch composition.
    for m in (G, F1, F2):
        m.eval()

    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    p1, p2 = [], []
    for i in range(0, len(X), batch_size):
        xb = X[i : i + batch_size].to(device)
        feat = G(xb)
        p1.append(F1(feat).squeeze(-1).cpu())
        p2.append(F2(feat).squeeze(-1).cpu())
    return torch.cat(p1).numpy(), torch.cat(p2).numpy()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _rolling(x, window):
    """Centered rolling mean and std (NaN-padded at edges)."""
    if window < 2 or len(x) < window:
        return np.full_like(x, np.nan, dtype=float), np.full_like(x, np.nan, dtype=float)
    kernel = np.ones(window) / window
    mean = np.convolve(x, kernel, mode="same")
    sq = np.convolve(x**2, kernel, mode="same")
    std = np.sqrt(np.clip(sq - mean**2, 0, None))
    half = window // 2
    mean[:half] = mean[-half:] = np.nan
    std[:half] = std[-half:] = np.nan
    return mean, std


def _metrics(y, y_hat):
    res = y - y_hat
    rmse = float(np.sqrt(np.mean(res**2)))
    mae = float(np.mean(np.abs(res)))
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    bias = float(np.mean(res))
    return dict(rmse=rmse, mae=mae, r2=r2, bias=bias)


def _shade_states(ax, t, states, cmap="tab10", alpha=0.12):
    """Background shading for contiguous intervention-state segments (source animals only)."""
    states = np.asarray(states)
    uniq = list(dict.fromkeys(states.tolist()))
    colors = plt.get_cmap(cmap)
    change = np.flatnonzero(states[1:] != states[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(states)]
    seen = set()
    for s, e in zip(starts, ends):
        st = states[s]
        label = str(st) if st not in seen else None
        seen.add(st)
        ax.axvspan(t[s], t[e - 1], color=colors(uniq.index(st) % 10), alpha=alpha, lw=0, label=label)


# --------------------------------------------------------------------------- #
# Main plot
# --------------------------------------------------------------------------- #
def plot_fold(
    G, F1, F2, X, y, animal_id,
    t=None,                 # time vector; defaults to sample index
    states=None,            # optional intervention-state labels (same length as y)
    y_mean=0.0, y_std=1.0,  # undo target standardization
    device="cuda",
    rolling_window=50,
    out_path=None,
    log_mlflow=True,
):
    p1, p2 = collect_predictions(G, F1, F2, X, device=device)

    y = np.asarray(y, dtype=float).ravel() * y_std + y_mean
    p1 = p1 * y_std + y_mean
    p2 = p2 * y_std + y_mean
    y_hat = 0.5 * (p1 + p2)
    res = y - y_hat
    t = np.arange(len(y)) if t is None else np.asarray(t)
    xlabel = "sample index" if t is None else "time"

    m = _metrics(y, y_hat)
    r_mean, r_std = _rolling(res, rolling_window)

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_ts = fig.add_subplot(gs[0, :2])
    ax_res = fig.add_subplot(gs[1, :2], sharex=ax_ts)
    ax_par = fig.add_subplot(gs[0, 2])
    ax_hist = fig.add_subplot(gs[1, 2])

    # --- Actual vs predicted over time ---
    if states is not None:
        _shade_states(ax_ts, t, states)
    ax_ts.plot(t, y, lw=1.0, color="black", label="actual")
    ax_ts.plot(t, y_hat, lw=1.0, color="tab:blue", label="predicted (mean F1,F2)")
    ax_ts.fill_between(t, np.minimum(p1, p2), np.maximum(p1, p2),
                       color="tab:blue", alpha=0.2, lw=0, label="F1–F2 spread")
    ax_ts.set_ylabel("strokework")
    ax_ts.set_title(f"{animal_id} (target) — RMSE {m['rmse']:.3g}, "
                    f"MAE {m['mae']:.3g}, R² {m['r2']:.3f}, bias {m['bias']:.3g}")
    ax_ts.legend(loc="upper right", fontsize=8, ncol=2)

    # --- Residuals over time ---
    if states is not None:
        _shade_states(ax_res, t, states)
    ax_res.scatter(t, res, s=3, alpha=0.4, color="tab:gray", label="residual (actual − pred)")
    ax_res.plot(t, r_mean, color="tab:red", lw=1.5, label=f"rolling mean (w={rolling_window})")
    ax_res.fill_between(t, r_mean - r_std, r_mean + r_std, color="tab:red", alpha=0.15, lw=0,
                        label="± rolling std")
    ax_res.axhline(0, color="black", lw=0.8, ls="--")
    ax_res.set_xlabel(xlabel)
    ax_res.set_ylabel("residual")
    ax_res.legend(loc="upper right", fontsize=8)

    # --- Parity plot ---
    sc = ax_par.scatter(y, y_hat, c=t, s=4, cmap="viridis", alpha=0.6)
    lo, hi = np.nanmin([y.min(), y_hat.min()]), np.nanmax([y.max(), y_hat.max()])
    ax_par.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax_par.set_xlabel("actual")
    ax_par.set_ylabel("predicted")
    ax_par.set_title("parity (color = time)")
    fig.colorbar(sc, ax=ax_par, shrink=0.8)

    # --- Residual distribution ---
    ax_hist.hist(res, bins=50, color="tab:gray", alpha=0.8, orientation="horizontal")
    ax_hist.axhline(0, color="black", lw=0.8, ls="--")
    ax_hist.set_xlabel("count")
    ax_hist.set_ylabel("residual")
    ax_hist.set_title("residual distribution")

    if out_path is None:
        out_path = f"pred_residuals_{animal_id}.png"
    fig.savefig(out_path, dpi=150)

    if log_mlflow:
        try:
            import mlflow
            mlflow.log_figure(fig, f"plots/pred_residuals_{animal_id}.png")
            mlflow.log_metrics({f"target_{k}": v for k, v in m.items()})
        except Exception as e:
            print(f"[plot_fold] MLflow logging skipped: {e}")

    plt.close(fig)
    return dict(y=y, y_hat=y_hat, p1=p1, p2=p2, residuals=res, t=t, metrics=m)


# --------------------------------------------------------------------------- #
# Example hook inside the LOO loop
# --------------------------------------------------------------------------- #
# for target_id in ["anim1", "anim3", "anim4", "anim8", "anim10"]:
#     ...train G, F1, F2 for this fold...
#     X_t, y_t = data[target_id]
#     plot_fold(G, F1, F2, X_t, y_t, animal_id=target_id,
#               y_mean=y_scaler_mean, y_std=y_scaler_std)
#
# For a source animal you can also pass its intervention states:
#     plot_fold(G, F1, F2, X_s, y_s, animal_id=f"{src_id}_src", states=states_s)