#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load dataset
# -----------------------
nc = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/TGCN_T30_quantile_fixed_test2020_2024/pred_inputs_daily_maps.nc"
ds = xr.open_dataset(nc)

m = ds["gauge_mask"].astype(bool)

# Flatten gauge obs once
obs = ds["gauge"].where(m).values.ravel()

# -----------------------
# Helpers
# -----------------------
def _paired(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]

def rmse(obs, sim):
    o, s = _paired(obs, sim)
    return float(np.sqrt(np.mean((s - o) ** 2))) if o.size else np.nan

def cc(obs, sim):
    o, s = _paired(obs, sim)
    if o.size < 2:
        return np.nan
    so = np.std(o); ss = np.std(s)
    if so == 0 or ss == 0:
        return np.nan
    return float(np.corrcoef(o, s)[0, 1])

def kge(obs, sim):
    """
    Kling-Gupta Efficiency (2009):
    KGE = 1 - sqrt( (r-1)^2 + (alpha-1)^2 + (beta-1)^2 )
    """
    o, s = _paired(obs, sim)
    if o.size < 2:
        return np.nan

    mu_o = np.mean(o)
    mu_s = np.mean(s)
    sig_o = np.std(o, ddof=0)
    sig_s = np.std(s, ddof=0)

    r = cc(o, s)
    if not np.isfinite(r) or sig_o == 0 or mu_o == 0:
        return np.nan

    alpha = sig_s / sig_o
    beta  = mu_s / mu_o
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

def compute_metrics(name, sim):
    return {
        "Product": name,
        "RMSE": rmse(obs, sim),
        "CC": cc(obs, sim),
        "KGE": kge(obs, sim),
    }

# -----------------------
# Pull products (linear space only)
# -----------------------
pred_ev = ds["pred_expected_mean"].where(m).values.ravel()

# Quantiles we want
q75 = ds["pred_q"].sel(tau=0.75).where(m).values.ravel()
q95 = ds["pred_q"].sel(tau=0.95).where(m).values.ravel()

era5 = ds["era5"].where(m).values.ravel()
im   = ds["imerg"].where(m).values.ravel()

# -----------------------
# Compute metrics table (dict-of-dicts)
# -----------------------
rows = []
rows.append(compute_metrics("TGCN_expected_mean", pred_ev))
rows.append(compute_metrics("TGCN_q75", q75))
rows.append(compute_metrics("TGCN_q95", q95))
rows.append(compute_metrics("ERA5", era5))
rows.append(compute_metrics("IMERG", im))

# Keep product order fixed for plots
product_order = [r["Product"] for r in rows]

metrics = ["RMSE", "CC", "KGE"]
vals = {met: [r[met] for r in rows] for met in metrics}

print("N paired (shared mask):", _paired(obs, era5)[0].size)
for r in rows:
    print(r)

#%%
# -----------------------
# Plot: one barplot per metric
# -----------------------
for met in metrics:
    y = np.array(vals[met], dtype=float)

    plt.figure()
    plt.bar(product_order, y)
    plt.title(f"{met} vs GAUGE (linear space)")
    plt.ylabel(met)
    plt.xticks(rotation=30, ha="right")

    # annotate values on bars
    for i, v in enumerate(y):
        if np.isfinite(v):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()
# %%
