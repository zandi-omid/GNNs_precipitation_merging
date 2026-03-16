#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV = "/xdisk/behrangi/omidzandi/GNNs/evaluation/master_test_gauge_comparison.csv"

df = pd.read_csv(CSV, parse_dates=["date"])

print(df.head())
print(df.columns.tolist())
print("Rows:", len(df))


# Products to evaluate against gauge
products = [
    "idw_loo_p0",
    "idw_loo_p2",
    "tgcn",
    "era5",
    "imerg",
]

pretty_names = {
    "idw_loo_p0": "IDW p=0",
    "idw_loo_p2": "IDW p=2",
    "tgcn": "TGCN",
    "era5": "ERA5",
    "imerg": "IMERG",
}

def _paired(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    return obs[ok], sim[ok]

def rmse(obs, sim):
    o, s = _paired(obs, sim)
    return float(np.sqrt(np.mean((s - o) ** 2))) if len(o) else np.nan

def mae(obs, sim):
    o, s = _paired(obs, sim)
    return float(np.mean(np.abs(s - o))) if len(o) else np.nan

def bias(obs, sim):
    o, s = _paired(obs, sim)
    return float(np.mean(s - o)) if len(o) else np.nan

def cc(obs, sim):
    o, s = _paired(obs, sim)
    if len(o) < 2:
        return np.nan
    so = np.std(o)
    ss = np.std(s)
    if so == 0 or ss == 0:
        return np.nan
    return float(np.corrcoef(o, s)[0, 1])

def kge(obs, sim):
    o, s = _paired(obs, sim)
    if len(o) < 2:
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

def pbias_pct(obs, sim):
    o, s = _paired(obs, sim)
    denom = np.sum(o)
    if len(o) == 0 or denom == 0:
        return np.nan
    return float(100.0 * np.sum(s - o) / denom)

#%% global metrics
global_rows = []

for p in products:
    row = {
        "Product": pretty_names[p],
        "RMSE": rmse(df["gauge"], df[p]),
        "MAE": mae(df["gauge"], df[p]),
        "Bias": bias(df["gauge"], df[p]),
        "PBIAS_pct": pbias_pct(df["gauge"], df[p]),
        "CC": cc(df["gauge"], df[p]),
        "KGE": kge(df["gauge"], df[p]),
    }
    global_rows.append(row)

global_metrics = pd.DataFrame(global_rows)
print(global_metrics)

metrics_to_plot = ["RMSE", "MAE", "Bias", "PBIAS_pct", "CC", "KGE"]

for met in metrics_to_plot:
    vals = global_metrics[met].values
    labels = global_metrics["Product"].values

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, vals)
    plt.title(f"Global {met} vs Gauge")
    plt.ylabel(met)
    plt.xticks(rotation=25, ha="right")

    for i, v in enumerate(vals):
        if np.isfinite(v):
            va = "bottom" if v >= 0 else "top"
            plt.text(i, v, f"{v:.3f}", ha="center", va=va, fontsize=9)

    plt.tight_layout()
    plt.show()

#%% each gauge KGE
node_cols = ["y", "x", "lon", "lat"]

per_node_rows = []

for (y, x, lon, lat), g in df.groupby(node_cols):
    row_base = {
        "y": y,
        "x": x,
        "lon": lon,
        "lat": lat,
        "n": len(g),
    }

    for p in products:
        row = row_base.copy()
        row["Product"] = pretty_names[p]
        row["KGE"] = kge(g["gauge"], g[p])
        row["CC"] = cc(g["gauge"], g[p])
        row["RMSE"] = rmse(g["gauge"], g[p])
        row["Bias"] = bias(g["gauge"], g[p])
        per_node_rows.append(row)

per_node = pd.DataFrame(per_node_rows)
print(per_node.head())
print("Rows:", len(per_node))

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
axes = axes.flatten()

vmin = -1
vmax = 1

for ax, p in zip(axes, [pretty_names[p] for p in products]):
    sub = per_node[per_node["Product"] == p].copy()

    sc = ax.scatter(
        sub["lon"],
        sub["lat"],
        c=sub["KGE"],
        s=22,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"KGE map: {p}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

# Hide last empty subplot
axes[-1].axis("off")

cbar = fig.colorbar(sc, ax=axes[:-1], shrink=0.9)
cbar.set_label("KGE")

plt.show()

#%% boxplot of gauge's KGE
kge_box_data = []
kge_labels = []

for p in [pretty_names[p] for p in products]:
    vals = per_node.loc[per_node["Product"] == p, "KGE"].dropna().values
    kge_box_data.append(vals)
    kge_labels.append(p)

plt.figure(figsize=(9, 5))
plt.boxplot(kge_box_data, labels=kge_labels, showfliers=False)
plt.ylabel("KGE")
plt.title("Distribution of node-level KGE")
plt.xticks(rotation=20)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

#%% scatter plot
# Optional subsampling for faster plotting if needed
max_points = 80000

rng = np.random.default_rng(42)
if len(df) > max_points:
    idx = rng.choice(len(df), size=max_points, replace=False)
    df_scatter = df.iloc[idx].copy()
else:
    df_scatter = df.copy()

print("Scatter sample size:", len(df_scatter))

fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
axes = axes.flatten()

obs_all = df_scatter["gauge"].values
xy_max = np.nanpercentile(obs_all, 99.5)

for ax, p in zip(axes, products):
    sim = df_scatter[p].values
    obs = df_scatter["gauge"].values

    ok = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[ok]
    sim = sim[ok]

    ax.scatter(obs, sim, s=8, alpha=0.35)
    ax.plot([0, xy_max], [0, xy_max], linestyle="--")
    ax.set_xlim(0, xy_max)
    ax.set_ylim(0, xy_max)
    ax.set_xlabel("Gauge")
    ax.set_ylabel(pretty_names[p])
    ax.set_title(f"Gauge vs {pretty_names[p]}")

axes[-1].axis("off")
plt.show()
# %%
