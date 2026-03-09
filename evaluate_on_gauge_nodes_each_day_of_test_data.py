#%%
import xarray as xr
import numpy as np
import pandas as pd

nc = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/TGCN_T30_quantile_fixed_test2020_2024/pred_inputs_daily_maps.nc"
ds = xr.open_dataset(nc)

m = ds["gauge_mask"].astype(bool)

def _paired(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]

def rmse(obs, sim):
    o, s = _paired(obs, sim)
    if o.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((s - o) ** 2)))

def kge(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    ok = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[ok]
    sim = sim[ok]

    if obs.size < 2:
        return np.nan

    mu_o = np.mean(obs)
    mu_s = np.mean(sim)

    sig_o = np.std(obs)
    sig_s = np.std(sim)

    if sig_o == 0 or mu_o == 0:
        return np.nan

    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sig_s / sig_o
    beta = mu_s / mu_o

    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

# Variables (masked at gauge points)
obs   = ds["gauge"].where(m)
q75   = ds["pred_q"].sel(tau=0.75).where(m)
era5  = ds["era5"].where(m)
imerg = ds["imerg"].where(m)

time_vals = ds["time"].values

rows = []
for t in range(len(time_vals)):
    o  = obs.isel(time=t).values.ravel()
    mq = q75.isel(time=t).values.ravel()
    e  = era5.isel(time=t).values.ravel()
    i  = imerg.isel(time=t).values.ravel()

    # gauge-only finite for diagnostics
    ok_o = np.isfinite(o)
    o_ok = o[ok_o]

    N = int(ok_o.sum())
    mean_o = float(np.nanmean(o)) if N > 0 else np.nan
    std_o  = float(np.nanstd(o))  if N > 0 else np.nan
    wet_frac = float(np.mean(o_ok > 1.0)) if o_ok.size else np.nan  # 1 mm/day threshold

    rows.append({
        "date": pd.to_datetime(time_vals[t]),
        "N_gauges": N,
        "mean_obs": mean_o,
        "std_obs": std_o,
        "wet_frac_obs_gt1": wet_frac,

        # KGE (can be NaN on dry/constant days)
        "KGE_q75": kge(o, mq),
        "KGE_ERA5": kge(o, e),
        "KGE_IMERG": kge(o, i),

        # RMSE (stable even on dry days)
        "RMSE_q75": rmse(o, mq),
        "RMSE_ERA5": rmse(o, e),
        "RMSE_IMERG": rmse(o, i),
    })

df_daily = pd.DataFrame(rows)

# -----------------------------
# KGE-based best/worst (only valid KGE days)
# -----------------------------
df_kge_valid = df_daily.dropna(subset=["KGE_q75"]).copy()

df_kge_worst = df_kge_valid.nsmallest(20, "KGE_q75")
df_kge_best  = df_kge_valid.nlargest(20, "KGE_q75")

print("\n🔻 20 Worst Days by KGE_q75 (valid KGE only):\n")
print(df_kge_worst[[
    "date","KGE_q75","KGE_ERA5","KGE_IMERG",
    "RMSE_q75","RMSE_ERA5","RMSE_IMERG",
    "N_gauges","mean_obs","std_obs","wet_frac_obs_gt1"
]])

print("\n🔺 20 Best Days by KGE_q75 (valid KGE only):\n")
print(df_kge_best[[
    "date","KGE_q75","KGE_ERA5","KGE_IMERG",
    "RMSE_q75","RMSE_ERA5","RMSE_IMERG",
    "N_gauges","mean_obs","std_obs","wet_frac_obs_gt1"
]])

# Optional: how many NaN KGE days
n_nan = int(df_daily["KGE_q75"].isna().sum())
print(f"\nDays with NaN KGE_q75: {n_nan} / {len(df_daily)}")
print("Typical reasons: mean_obs==0 (all zero) or std_obs==0 (no spatial variability) or too few gauges.")

# -----------------------------
# RMSE-based best/worst (RMSE should be valid for almost all days)
# -----------------------------
df_rmse_valid = df_daily.dropna(subset=["RMSE_q75"]).copy()

df_rmse_best  = df_rmse_valid.nsmallest(20, "RMSE_q75")
df_rmse_worst = df_rmse_valid.nlargest(20, "RMSE_q75")

print("\n✅ 20 Best Days by RMSE_q75 (lowest RMSE):\n")
print(df_rmse_best[[
    "date","RMSE_q75","RMSE_ERA5","RMSE_IMERG",
    "KGE_q75","KGE_ERA5","KGE_IMERG",
    "N_gauges","mean_obs","std_obs","wet_frac_obs_gt1"
]])

print("\n❌ 20 Worst Days by RMSE_q75 (highest RMSE):\n")
print(df_rmse_worst[[
    "date","RMSE_q75","RMSE_ERA5","RMSE_IMERG",
    "KGE_q75","KGE_ERA5","KGE_IMERG",
    "N_gauges","mean_obs","std_obs","wet_frac_obs_gt1"
]])
#%%