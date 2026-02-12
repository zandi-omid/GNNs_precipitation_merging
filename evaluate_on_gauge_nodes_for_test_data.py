import xarray as xr
import numpy as np

ds = xr.open_dataset("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/TGCN_T30_log_normal_test2020_2024/pred_inputs_daily_maps.nc")

m = ds["gauge_mask"].astype(bool)

obs  = ds["gauge"].where(m).values.ravel()
pred = ds["pred"].where(m).values.ravel()
era5 = ds["era5"].where(m).values.ravel()
im   = ds["imerg"].where(m).values.ravel()

def _paired(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]

def rmse(a, b):
    a, b = _paired(a, b)
    return float(np.sqrt(np.mean((b - a) ** 2))) if a.size else np.nan

def cc(a, b):
    a, b = _paired(a, b)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def kge(a, b):
    """
    Kling-Gupta Efficiency (2009):
    KGE = 1 - sqrt( (r-1)^2 + (alpha-1)^2 + (beta-1)^2 )
      r     = correlation
      alpha = std(sim) / std(obs)
      beta  = mean(sim) / mean(obs)
    """
    o, s = _paired(a, b)
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

import numpy as np

def kge_parts(obs, sim):
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    o = obs[ok]; s = sim[ok]

    mu_o = o.mean(); mu_s = s.mean()
    sig_o = o.std(ddof=0); sig_s = s.std(ddof=0)

    r = np.corrcoef(o, s)[0, 1] if sig_o > 0 and sig_s > 0 else np.nan
    alpha = sig_s / sig_o if sig_o > 0 else np.nan
    beta  = mu_s / mu_o if mu_o != 0 else np.nan

    kge = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2) if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta) else np.nan
    return dict(N=o.size, r=r, alpha=alpha, beta=beta, mu_obs=mu_o, mu_sim=mu_s, std_obs=sig_o, std_sim=sig_s, KGE=kge)

print("PRED ", kge_parts(obs, pred))
print("ERA5 ", kge_parts(obs, era5))
print("IMERG", kge_parts(obs, im))

def summarize(name, sim):
    print(f"\n{name}")
    print("  N paired:", _paired(obs, sim)[0].size)
    print("  RMSE:", rmse(obs, sim))
    print("  CC  :", cc(obs, sim))
    print("  KGE :", kge(obs, sim))

summarize("PRED vs GAUGE", pred)
summarize("ERA5 vs GAUGE", era5)
summarize("IMERG vs GAUGE", im)

import numpy as np
obs_l  = np.log1p(obs)
pred_l = np.log1p(pred)
era5_l = np.log1p(era5)
im_l   = np.log1p(im)

print("CC log: pred/era5/imerg",
      np.corrcoef(obs_l[np.isfinite(obs_l)&np.isfinite(pred_l)],
                  pred_l[np.isfinite(obs_l)&np.isfinite(pred_l)])[0,1])

import numpy as np

# ---- helper functions (same style as yours) ----
def _paired(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]

def cc(a, b):
    a, b = _paired(a, b)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def kge(a, b):
    """
    Kling-Gupta Efficiency (2009):
    KGE = 1 - sqrt( (r-1)^2 + (alpha-1)^2 + (beta-1)^2 )
    """
    o, s = _paired(a, b)
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

# ---- LOG-SPACE ARRAYS ----
# Use log1p to match your training transform. Values should be >= 0 already,
# but we clip just in case.
obs_l  = np.log1p(np.clip(obs,  0, None))
pred_l = np.log1p(np.clip(pred, 0, None))
era5_l = np.log1p(np.clip(era5, 0, None))
im_l   = np.log1p(np.clip(im,   0, None))

print("LOG-SPACE metrics vs GAUGE (log1p(mm/day))")
print("PRED  CC:", cc(obs_l, pred_l), "KGE:", kge(obs_l, pred_l))
print("ERA5  CC:", cc(obs_l, era5_l), "KGE:", kge(obs_l, era5_l))
print("IMERG CC:", cc(obs_l, im_l),   "KGE:", kge(obs_l, im_l))