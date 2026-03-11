#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load dataset
# -----------------------
nc = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/TGCN_T30_IDW_added_test2020_2024/pred_inputs_daily_maps.nc"
# nc = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/TGCN_T30_quantile_fixed_test2020_2024/pred_inputs_daily_maps.nc"

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
    so = np.std(o)
    ss = np.std(s)
    if so == 0 or ss == 0:
        return np.nan
    return float(np.corrcoef(o, s)[0, 1])

def kge(obs, sim):
    """
    Kling-Gupta Efficiency (2009):
    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
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
    beta = mu_s / mu_o
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

def compute_metrics(name, sim):
    return {
        "Product": name,
        "RMSE": rmse(obs, sim),
        "CC": cc(obs, sim),
        "KGE": kge(obs, sim),
    }

def get_flat(varname):
    return ds[varname].where(m).values.ravel()

# -----------------------
# Collect available products automatically
# -----------------------
rows = []

# Baselines
if "era5" in ds:
    rows.append(compute_metrics("ERA5", get_flat("era5")))

if "imerg" in ds:
    rows.append(compute_metrics("IMERG", get_flat("imerg")))

# Deterministic model output (MSE / Huber)
if "pred_det" in ds:
    rows.append(compute_metrics("TGCN_det", get_flat("pred_det")))

# Quantile regression outputs
if "pred_expected_mean" in ds:
    rows.append(compute_metrics("TGCN_expected_mean", get_flat("pred_expected_mean")))

if "pred_median" in ds:
    rows.append(compute_metrics("TGCN_median", get_flat("pred_median")))

if "pred_q" in ds and "tau" in ds:
    tau_vals = ds["tau"].values.astype(float)

    requested_taus = [0.75, 0.95]   # change if you want more / different taus
    for t_req in requested_taus:
        idx = int(np.argmin(np.abs(tau_vals - t_req)))
        t_use = float(tau_vals[idx])
        q = ds["pred_q"].isel(tau=idx).where(m).values.ravel()
        rows.append(compute_metrics(f"TGCN_q{t_use:.2f}", q))

# -----------------------
# Print metrics
# -----------------------
print("Available variables in dataset:")
print(list(ds.data_vars))

print("\nN paired (using gauge mask):", _paired(obs, obs)[0].size)
print("\nMetrics:")
for r in rows:
    print(r)

# -----------------------
# Plot barplots
# -----------------------
product_order = [r["Product"] for r in rows]
metrics = ["RMSE", "CC", "KGE"]
vals = {met: [r[met] for r in rows] for met in metrics}

for met in metrics:
    y = np.array(vals[met], dtype=float)

    plt.figure(figsize=(8, 5))
    plt.bar(product_order, y)
    plt.title(f"{met} vs GAUGE")
    plt.ylabel(met)
    plt.xticks(rotation=30, ha="right")

    for i, v in enumerate(y):
        if np.isfinite(v):
            va = "bottom" if v >= 0 else "top"
            plt.text(i, v, f"{v:.3f}", ha="center", va=va, fontsize=9)

    plt.tight_layout()
    plt.show()

# %%