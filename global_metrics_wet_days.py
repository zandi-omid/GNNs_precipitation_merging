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

# -----------------------
# Figure out the time dimension (robust)
# -----------------------
def _infer_time_dim(da):
    for cand in ["time", "date", "day", "valid_time"]:
        if cand in da.dims:
            return cand
    return None

time_dim = _infer_time_dim(ds["gauge"])

# -----------------------
# Helpers
# -----------------------
def _paired(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]

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
    if (not np.isfinite(r)) or (sig_o == 0) or (mu_o == 0):
        return np.nan

    alpha = sig_s / sig_o
    beta  = mu_s / mu_o
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

def flatten_on_mask(var, base_mask, time_sel=None):
    """
    var: DataArray
    base_mask: boolean DataArray same shape as var (or broadcastable)
    time_sel: boolean DataArray over time_dim (or None)
    """
    v = var
    mm = base_mask

    if (time_dim is not None) and (time_sel is not None):
        v = v.sel({time_dim: time_sel})
        mm = mm.sel({time_dim: time_sel})

    return v.where(mm).values.ravel()

# -----------------------
# Build the 3 evaluation selections
# 1) All-day: just gauge_mask
# 2) Wet-day: mean_obs_over_gauges > 1 mm/day  (global KGE pooling all gauges across those days)
# 3) Wet-gauge: only gauge points where obs > 1 mm/day (all days)
# -----------------------
gauge = ds["gauge"]

# 1) All-day mask
mask_all = m

# 2) Wet-day selector (only if time_dim exists)
wet_day_sel = None
if time_dim is not None:
    gauge_masked = gauge.where(m)
    spatial_dims = [d for d in gauge_masked.dims if d != time_dim]
    mean_obs_day = gauge_masked.mean(dim=spatial_dims, skipna=True)
    wet_day_sel = mean_obs_day > 1.0

# 3) Wet-gauge mask (obs > 1 mm/day on gauge points)
mask_wet_gauge = m & (gauge > 1.0)

# -----------------------
# Pull products
# -----------------------
products = {
    "TGCN_expected_mean": ds["pred_expected_mean"],
    "TGCN_q75": ds["pred_q"].sel(tau=0.75),
    "TGCN_q95": ds["pred_q"].sel(tau=0.95),
    "ERA5": ds["era5"],
    "IMERG": ds["imerg"],
}

# -----------------------
# Compute 3 KGE metrics for each product (pooling pairs globally)
# -----------------------
results = []
for name, sim_da in products.items():
    # All-day
    obs_all = flatten_on_mask(gauge, mask_all)
    sim_all = flatten_on_mask(sim_da, mask_all)
    kge_all = kge(obs_all, sim_all)

    # Wet-day (mean_obs_day > 1 mm)
    if wet_day_sel is not None:
        obs_wd = flatten_on_mask(gauge, mask_all, time_sel=wet_day_sel)
        sim_wd = flatten_on_mask(sim_da, mask_all, time_sel=wet_day_sel)
        kge_wet_day = kge(obs_wd, sim_wd)
        n_wet_days = int(wet_day_sel.sum().values)
    else:
        kge_wet_day = np.nan
        n_wet_days = None

    # Wet-gauge (obs > 1 mm at gauge points)
    obs_wg = flatten_on_mask(gauge, mask_wet_gauge)
    sim_wg = flatten_on_mask(sim_da, mask_wet_gauge)
    kge_wet_gauge = kge(obs_wg, sim_wg)

    results.append({
        "Product": name,
        "KGE_all_days": kge_all,
        "KGE_wet_days_meanObs_gt1": kge_wet_day,
        "KGE_wet_gauges_obs_gt1": kge_wet_gauge,
        "N_pairs_all_days": _paired(obs_all, sim_all)[0].size,
        "N_pairs_wet_gauges": _paired(obs_wg, sim_wg)[0].size,
        "N_wet_days": n_wet_days,
    })

# Print nicely
print("\n=== Global KGE (3 variants) ===")
for r in results:
    print(
        f"{r['Product']:>18} | "
        f"KGE_all={r['KGE_all_days']:.4f} | "
        f"KGE_wetDay(mean>1)={r['KGE_wet_days_meanObs_gt1']:.4f} | "
        f"KGE_wetGauge(obs>1)={r['KGE_wet_gauges_obs_gt1']:.4f} | "
        f"N_all={r['N_pairs_all_days']} | "
        f"N_wetGauge={r['N_pairs_wet_gauges']} | "
        f"N_wetDays={r['N_wet_days']}"
    )

# -----------------------
# Optional quick plots: 3 barplots (one per KGE variant)
# -----------------------
product_order = [r["Product"] for r in results]

def _barplot(key, title):
    y = np.array([r[key] for r in results], dtype=float)
    plt.figure()
    plt.bar(product_order, y)
    plt.title(title)
    plt.ylabel(key)
    plt.xticks(rotation=30, ha="right")
    for i, v in enumerate(y):
        if np.isfinite(v):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()

_barplot("KGE_all_days", "KGE vs GAUGE (All days)")
_barplot("KGE_wet_days_meanObs_gt1", "KGE vs GAUGE (Wet days: mean_obs > 1 mm/day)")
_barplot("KGE_wet_gauges_obs_gt1", "KGE vs GAUGE (Wet gauges: obs > 1 mm/day)")
#%%