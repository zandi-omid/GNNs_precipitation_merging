#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

OUTDIR = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/quant_and_cat_test_gauges/100pct")
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

CSV = "/xdisk/behrangi/omidzandi/GNNs/evaluation/master_test_gauge_comparison.csv"

df = pd.read_csv(CSV, parse_dates=["date"])

print(df.head())
print(df.columns.tolist())
print("Rows:", len(df))


# Products to evaluate against gauge
products = [
    "idw_loo_p2",
    "tgcn",
    "era5",
    "imerg",
    "prism",

]

pretty_names = {
    "idw_loo_p2": "IDW",
    "tgcn": "TGCN",
    "era5": "ERA5",
    "imerg": "IMERG",
    "prism": "PRISM",

}

product_colors = {
    "IDW": "#4C78A8",
    "TGCN": "#E45756",
    "ERA5": "#72B7B2",
    "IMERG": "#F2CF5B",
    "PRISM": "#B279A2",

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

lower_is_better = {"RMSE", "MAE", "Bias_abs", "PBIAS_abs"}
higher_is_better = {"CC", "KGE"}

global_metrics_plot = global_metrics.copy()
global_metrics_plot["Bias_abs"] = global_metrics_plot["Bias"].abs()
global_metrics_plot["PBIAS_abs"] = global_metrics_plot["PBIAS_pct"].abs()

metrics_to_plot = ["RMSE", "MAE", "Bias_abs", "PBIAS_abs", "CC", "KGE"]

pretty_metric_names = {
    "RMSE": "RMSE",
    "MAE": "MAE [mm/day]",
    "Bias_abs": "|Bias|",
    "PBIAS_abs": "|PBIAS| (%)",
    "CC": "Correlation",
    "KGE": "KGE",
}

for met in metrics_to_plot:
    vals = global_metrics_plot[met].values
    labels = global_metrics_plot["Product"].values
    colors = [product_colors[lbl] for lbl in labels]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=1.2)

    subtitle = "Lower is better" if met in lower_is_better else "Higher is better"
    ax.set_title(f"{pretty_metric_names[met]} ({subtitle})", pad=12, weight="bold")
    ax.set_ylabel(pretty_metric_names[met])
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")
        tick.set_fontweight("bold")

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    for bar, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * y_range,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold"
            )

    if met == "KGE":
        ax.set_ylim(min(-0.1, np.nanmin(vals) - 0.05), max(1.0, np.nanmax(vals) + 0.1))
    elif met == "CC":
        ax.set_ylim(0, max(1.0, np.nanmax(vals) + 0.05))
    else:
        ax.set_ylim(0, np.nanmax(vals) * 1.18)

    plt.tight_layout()
    plt.show()

    fig.savefig(
    OUTDIR / f"{met}.png",
    dpi=500,
    bbox_inches="tight"
    )
    plt.close(fig)

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


#%% Poster-style boxplot of node-level KGE
kge_box_data = []
kge_labels = []

for p in [pretty_names[p] for p in products]:
    vals = per_node.loc[per_node["Product"] == p, "KGE"].dropna().values
    kge_box_data.append(vals)
    kge_labels.append(p)

product_colors = {
    "IDW": "#4C78A8",
    "TGCN": "#E45756",
    "ERA5": "#72B7B2",
    "IMERG": "#F2CF5B",
    "PRISM": "#B279A2",

}

fig, ax = plt.subplots(figsize=(10, 5.8))

bp = ax.boxplot(
    kge_box_data,
    labels=kge_labels,
    patch_artist=True,
    showfliers=False,
    widths=0.6,
    medianprops=dict(color="black", linewidth=2.2),
    boxprops=dict(linewidth=1.6),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
)

# color each box
for patch, label in zip(bp["boxes"], kge_labels):
    patch.set_facecolor(product_colors.get(label, "lightgray"))
    patch.set_edgecolor("black")
    patch.set_alpha(0.9)

# optional: overlay mean as white dot
for i, vals in enumerate(kge_box_data, start=1):
    if len(vals) > 0:
        ax.scatter(
            i, np.mean(vals),
            s=55,
            color="white",
            edgecolor="black",
            zorder=3
        )

ax.set_ylabel("KGE", fontsize=14)
ax.set_title("Distribution of Node-Level KGE", fontsize=16, weight="bold", pad=10)
ax.grid(axis="y", alpha=0.25, linestyle="--")
ax.set_axisbelow(True)

for tick in ax.get_xticklabels():
    tick.set_rotation(15)
    tick.set_fontweight("bold")

ax.tick_params(axis="both", labelsize=12)

# good KGE plotting range
ax.set_ylim(-1.0, 1.0)

plt.tight_layout()
plt.show()

fig.savefig(
    OUTDIR / "each_gauge_KGE_boxplot.png",
    dpi=500,
    bbox_inches="tight"
)
plt.close(fig)

# %%
