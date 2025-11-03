#!/usr/bin/env python3
# coding: utf-8
"""
Visualize labeled graph nodes on the DEM background.
----------------------------------------------------
• Loads DEM and graph (pickle)
• Selects nodes with gauge labels ('target')
• Splits them into Train/Val/Test (50/20/30)
• Plots nodes on DEM using Cartopy
"""

#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import torch
import rasterio
from rasterio.plot import show


#%%
# ============================================================
# 1️⃣ Load DEM and graph
# ============================================================
dem_path = Path("/xdisk/behrangi/omidzandi/GNNs/data/DEM/ASTER_DEM_0p1deg_AZ_buffer.tif")
graph_path = Path("/xdisk/behrangi/omidzandi/GNNs/data/graphs/graph_with_features_labels.pkl")

# DEM
dem = rxr.open_rasterio(dem_path).squeeze()
extent = [float(dem.x.min()), float(dem.x.max()), float(dem.y.min()), float(dem.y.max())]

# Graph dictionary
with open(graph_path, "rb") as f:
    data_dict = pickle.load(f)
G = data_dict["graph"]  # extract the NetworkX graph

print(f"✅ DEM shape: {dem.shape}, extent={extent}")
print(f"✅ Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges.")
print(f"🕒 Time axis: {len(data_dict['time_axis'])} days")
print(f"📈 Dynamic features: {data_dict['dynamic_feature_names']}")

#%%
# ============================================================
# 2️⃣ Extract node info and labeled nodes
# ============================================================
nodes = list(G.nodes())
lons = np.array([G.nodes[n]["lon"] for n in nodes])
lats = np.array([G.nodes[n]["lat"] for n in nodes])

# Identify labeled nodes (with gauge targets)
labeled_nodes = [n for n in nodes if "target" in G.nodes[n] and len(G.nodes[n]["target"]) > 0]
labeled_idx = np.array([nodes.index(n) for n in labeled_nodes])
print(f"💧 Found {len(labeled_nodes)} labeled nodes with gauge observations.")

#%%
# ============================================================
# 3️⃣ Create train/val/test masks for labeled nodes
# ============================================================
np.random.seed(42)
np.random.shuffle(labeled_idx)
n_lab = len(labeled_idx)
n_train = int(0.4 * n_lab)
n_val   = int(0.2 * n_lab)
n_test  = n_lab - n_train - n_val

train_mask = np.zeros(len(nodes), dtype=bool)
val_mask   = np.zeros(len(nodes), dtype=bool)
test_mask  = np.zeros(len(nodes), dtype=bool)

train_mask[labeled_idx[:n_train]] = True
val_mask[labeled_idx[n_train:n_train+n_val]] = True
test_mask[labeled_idx[n_train+n_val:]] = True

print(f"🎯 Train/Val/Test split among labeled nodes:")
print(f"   • Train: {train_mask.sum()} nodes")
print(f"   • Val:   {val_mask.sum()} nodes")
print(f"   • Test:  {test_mask.sum()} nodes")

#%%
# ============================================================
# 4️⃣ Plotting
# ============================================================

with rasterio.open(dem_path) as src:
    dem = src.read(1)  # first band
    nodata = src.nodata
    bounds = src.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

# Mask nodata values
if nodata is not None:
    dem = np.where(dem == nodata, np.nan, dem)

# Compute vmin/vmax for visualization (avoid NaNs)
vmin = np.nanpercentile(dem, 2)
vmax = np.nanpercentile(dem, 98)

print(f"DEM range (2–98th pct): {vmin:.1f} → {vmax:.1f} m")


fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=250)

# Plot DEM using imshow
img = ax.imshow(
    dem,
    extent=extent,
    origin="upper",
    cmap="terrain",
    vmin=vmin,
    vmax=vmax,
    transform=ccrs.PlateCarree(),
    alpha=0.5
)

# Colorbar
cbar = plt.colorbar(img, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
cbar.set_label("Elevation (m)", fontsize=11)

# Overlay nodes (reuse your masks)

# --- Node overlay: black markers, distinct shapes ---
marker_styles = {
    "Train": "o",   # circle
    "Validation": "s",  # square
    "Test": "^"     # triangle
}

ax.scatter(lons[train_mask], lats[train_mask],
           s=10, color="black", marker=marker_styles["Train"],
           label="Train", transform=ccrs.PlateCarree(), zorder=3)

ax.scatter(lons[val_mask], lats[val_mask],
           s=10, color="gold", marker=marker_styles["Validation"],
           label="Validation", transform=ccrs.PlateCarree(), zorder=3)

ax.scatter(lons[test_mask], lats[test_mask],
           s=10, color="crimson", marker=marker_styles["Test"],
           label="Test", transform=ccrs.PlateCarree(), zorder=3)

# --- Unlabeled nodes (gray background) ---
unlabeled_mask = ~(train_mask | val_mask | test_mask)
ax.scatter(lons[unlabeled_mask], lats[unlabeled_mask], s=6, color="gray", alpha=0.6,
           label="Unlabeled", transform=ccrs.PlateCarree())
# Map decoration
ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.4)
ax.add_feature(cfeature.STATES, linestyle=":", alpha=0.4)
ax.coastlines("10m", linewidth=0.8)
ax.set_extent(extent)
ax.set_title("Graph Nodes on DEM (Train / Val / Test)", fontsize=15)
ax.legend(loc="upper right", frameon=True)

plt.savefig(
    "train_val_test_nodes_on_dem.png",
    dpi=300,
    bbox_inches="tight",
)

plt.tight_layout()
plt.show()


# %%
