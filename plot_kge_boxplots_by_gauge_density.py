#%%
#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

# ============================================================
# CONFIG
# ============================================================
EVAL_DIR = Path("/xdisk/behrangi/omidzandi/GNNs/evaluation")
OUTDIR = Path("/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/quant_and_cat_test_gauges/all_scenarios")
OUTDIR.mkdir(parents=True, exist_ok=True)

csv_map = {
    "025pct": EVAL_DIR / "master_test_gauge_comparison_025pct.csv",
    "050pct": EVAL_DIR / "master_test_gauge_comparison_050pct.csv",
    "075pct": EVAL_DIR / "master_test_gauge_comparison_075pct.csv",
    "100pct": EVAL_DIR / "master_test_gauge_comparison_100pct.csv",
}

scenario_labels = {
    "025pct": "25%",
    "050pct": "50%",
    "075pct": "75%",
    "100pct": "100%",
}

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

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# ============================================================
# METRICS
# ============================================================
def _paired(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    return obs[ok], sim[ok]

def cc(obs, sim):
    o, s = _paired(obs, sim)
    if len(o) < 2:
        return np.nan
    so = np.std(o)
    ss = np.std(s)
    if so == 0 or ss == 0:
        return np.nan
    return float(np.corrcoef(o, s)[0, 1])

def rmse(obs, sim):
    o, s = _paired(obs, sim)
    return float(np.sqrt(np.mean((s - o) ** 2))) if len(o) else np.nan

def bias(obs, sim):
    o, s = _paired(obs, sim)
    return float(np.mean(s - o)) if len(o) else np.nan

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

# ============================================================
# LOAD ALL CSVs + COMPUTE NODE-LEVEL KGE
# ============================================================
node_cols = ["y", "x", "lon", "lat"]
all_rows = []

for scen_key, csv_path in csv_map.items():
    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"  Rows: {len(df):,}")

    for (y, x, lon, lat), g in df.groupby(node_cols):
        n_here = len(g)

        for p in products:
            row = {
                "scenario": scen_key,
                "scenario_label": scenario_labels[scen_key],
                "y": y,
                "x": x,
                "lon": lon,
                "lat": lat,
                "n": n_here,
                "Product": pretty_names[p],
                "KGE": kge(g["gauge"], g[p]),
                "CC": cc(g["gauge"], g[p]),
                "RMSE": rmse(g["gauge"], g[p]),
                "Bias": bias(g["gauge"], g[p]),
            }
            all_rows.append(row)

per_node = pd.DataFrame(all_rows)
print("\nCombined per-node table:")
print(per_node.head())
print("Rows:", len(per_node))

# save long table too
per_node.to_csv(OUTDIR / "per_node_metrics_all_scenarios.csv", index=False)

# ============================================================
# BUILD GROUPED BOXPLOT
# ============================================================
scenario_order = ["025pct", "050pct", "075pct", "100pct"]
product_order = ["IDW", "TGCN", "ERA5", "IMERG", "PRISM"]

# spacing controls
group_centers = np.arange(len(scenario_order)) * 2.2
offsets = np.array([-0.56, -0.28, 0.0, 0.28, 0.56])
box_width = 0.22

fig, ax = plt.subplots(figsize=(13.5, 6.5))

legend_handles = []

for j, prod in enumerate(product_order):
    color = product_colors[prod]
    legend_handles.append(Patch(facecolor=color, edgecolor="black", label=prod))

    for i, scen in enumerate(scenario_order):
        vals = per_node.loc[
            (per_node["scenario"] == scen) &
            (per_node["Product"] == prod),
            "KGE"
        ].dropna().values

        pos = group_centers[i] + offsets[j]

        bp = ax.boxplot(
            [vals],
            positions=[pos],
            widths=box_width,
            patch_artist=True,
            showfliers=True,
            whis=1.5,
            medianprops=dict(color="black", linewidth=2.0),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.1),
            capprops=dict(linewidth=1.1),
            flierprops=dict(
                marker="o",
                markersize=3.5,
                markerfacecolor="none",
                markeredgecolor=color,
                alpha=0.9,
            ),
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_alpha(0.85)

# x-axis
ax.set_xticks(group_centers)
ax.set_xticklabels([scenario_labels[s] for s in scenario_order], fontweight="bold")
ax.set_xlabel("Gauge Density Scenario")
ax.set_ylabel("KGE")
# ax.set_title("Distribution of Node-Level KGE Across Gauge Density Scenarios", weight="bold", pad=12)

# y/grid
ax.set_ylim(-1.2, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

# optional vertical separators between groups
for x in (group_centers[:-1] + group_centers[1:]) / 2:
    ax.axvline(x, color="red", linestyle="--", alpha=0.5, linewidth=1.0)

# legend
ax.legend(
    handles=legend_handles,
    ncol=5,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12),
    frameon=True,
)

plt.tight_layout()

fig.savefig(
    OUTDIR / "KGE_boxplot_by_gauge_density.png",
    dpi=500,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)
# %%
