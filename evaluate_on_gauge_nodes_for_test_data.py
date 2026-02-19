import xarray as xr
import numpy as np

# -----------------------
# Load dataset (quantile NetCDF)
# -----------------------
nc = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/TGCN_T30_quantile_fixed_test2020_2024/pred_inputs_daily_maps.nc"
ds = xr.open_dataset(nc)

m = ds["gauge_mask"].astype(bool)

# Flatten gauge obs once
obs = ds["gauge"].where(m).values.ravel()

# -----------------------
# Metrics helpers
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

def summarize(name, sim, obs=obs):
    o, s = _paired(obs, sim)
    print(f"\n{name}")
    print(f"  N paired: {o.size}")
    print(f"  RMSE   : {rmse(obs, sim):.6f}")
    print(f"  CC     : {cc(obs, sim):.6f}")
    print(f"  KGE    : {kge(obs, sim):.6f}")

def summarize_log(name, sim, obs=obs):
    # log1p in case you want “skill on wet days” de-emphasize extremes
    o = np.log1p(np.clip(obs, 0, None))
    s = np.log1p(np.clip(sim, 0, None))
    o2, s2 = _paired(o, s)
    print(f"\n{name}  [LOG1P]")
    print(f"  N paired: {o2.size}")
    print(f"  CC     : {cc(o, s):.6f}")
    print(f"  KGE    : {kge(o, s):.6f}")

# -----------------------
# Evaluate expected value
# -----------------------
pred_ev = ds["pred_expected_mean"].where(m).values.ravel()
summarize("PRED_EXPECTED_MEAN vs GAUGE", pred_ev)
summarize_log("PRED_EXPECTED_MEAN vs GAUGE", pred_ev)

# -----------------------
# Evaluate 3 highest quantiles
# -----------------------
tau = ds["tau"].values.astype(float)  # e.g. [0.05 ... 0.95]
top3 = tau[-3:]                       # last three: 0.75, 0.90, 0.95
print("\nSelected top-3 taus:", top3)

for tval in top3:
    q = ds["pred_q"].sel(tau=float(tval)).where(m).values.ravel()
    summarize(f"PRED_Q(tau={tval:g}) vs GAUGE", q)
    summarize_log(f"PRED_Q(tau={tval:g}) vs GAUGE", q)

# -----------------------
# Baselines
# -----------------------
era5 = ds["era5"].where(m).values.ravel()
im   = ds["imerg"].where(m).values.ravel()

summarize("ERA5 vs GAUGE", era5)
summarize_log("ERA5 vs GAUGE", era5)

summarize("IMERG vs GAUGE", im)
summarize_log("IMERG vs GAUGE", im)

# -----------------------
# Quantile Coverage Check
# -----------------------
print("\n--- Quantile Coverage ---")

for tval in ds["tau"].values[-3:]:
    q = ds["pred_q"].sel(tau=float(tval)).where(m).values.ravel()
    o, qv = _paired(obs, q)
    coverage = np.mean(o <= qv)
    print(f"tau={tval:.2f}  empirical_coverage={coverage:.4f}")