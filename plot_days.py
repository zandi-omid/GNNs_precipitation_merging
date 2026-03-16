#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_bilateral
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.titlesize": 16
})

def gaussian_nan(arr, sigma=1.0):
    mask = np.isnan(arr)

    arr_filled = np.where(mask, 0.0, arr)

    smooth = gaussian_filter(arr_filled, sigma=sigma)
    weight = gaussian_filter((~mask).astype(float), sigma=sigma)

    out = np.full(arr.shape, np.nan, dtype=np.float64)
    good = weight > 1e-8
    out[good] = smooth[good] / weight[good]
    out[mask] = np.nan

    return out


def bilateral_nan(arr, sigma_spatial=2, sigma_intensity=6):
    mask = np.isnan(arr)

    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)

    norm = (arr - arr_min) / (arr_max - arr_min + 1e-6)
    norm[mask] = 0.0

    filtered = denoise_bilateral(
        norm,
        sigma_color=sigma_intensity / (arr_max - arr_min + 1e-6),
        sigma_spatial=sigma_spatial,
        channel_axis=None,
    )

    filtered = filtered * (arr_max - arr_min) + arr_min
    filtered[mask] = np.nan
    return filtered


def daily_mean_over_valid(da, valid_mask):
    arr = da.values
    out = np.full(arr.shape[0], np.nan, dtype=np.float64)
    for t in range(arr.shape[0]):
        vals = arr[t][valid_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            out[t] = vals.mean()
    return out

def get_var(ds, preferred, fallback_list):
    if preferred in ds.data_vars:
        return ds[preferred]
    for v in fallback_list:
        if v in ds.data_vars:
            return ds[v]
    raise KeyError(f"Could not find variable '{preferred}' or any of {fallback_list} in {list(ds.data_vars)}")

# -----------------------
# User settings
# -----------------------
nc_tgcn = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/TGCN_T30_avg10_test2020_2024/pred_inputs_daily_maps.nc"
nc_idw  = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/IDW_test2020_2024/pred_inputs_daily_maps_IDW.nc"
nc_prism = "/xdisk/behrangi/omidzandi/GNNs/data/PRISM_daily_5km/PRISM_on_ERA5_DAILY_2005_2024.nc"

# mean_thresh = 5
mean_thresh = 0.1
# mean_thresh = 0.1


# seed = 520
seed = 10
# seed = 50

use_gaussian = True
gaussian_sigma = 1.2

use_bilateral = False
bilateral_sigma_spatial = 2
bilateral_sigma_intensity = 6

# -----------------------
# Load datasets
# -----------------------
ds = xr.open_dataset(nc_tgcn)
ds_idw = xr.open_dataset(nc_idw)
ds_prism = xr.open_dataset(nc_prism)

# grab PRISM variable robustly
prism_da = get_var(ds_prism, "prism_on_era5", ["precipitation", "prism_ppt", "ppt", "prism"])

if "pred_det" in ds:
    model_var = "pred_det"
    model_name = "TGCN"
elif "pred_expected_mean" in ds:
    model_var = "pred_expected_mean"
    model_name = "TGCN_expected_mean"
elif "pred_median" in ds:
    model_var = "pred_median"
    model_name = "TGCN_median"
else:
    raise ValueError("Could not find model prediction variable in dataset.")

print("Using model variable:", model_var)

valid_mask = ds["valid_pixel"].values.astype(bool)
lon2d = ds["lon"].values
lat2d = ds["lat"].values

# -----------------------
# Build 1D lon/lat axes
# -----------------------
lon1d = np.nanmedian(lon2d, axis=0)
lat1d = np.nanmedian(lat2d, axis=1)

print("lon1d shape:", lon1d.shape)
print("lat1d shape:", lat1d.shape)
print("lon range:", np.nanmin(lon1d), np.nanmax(lon1d))
print("lat range:", np.nanmin(lat1d), np.nanmax(lat1d))

# -----------------------
# Choose one wet random day
# -----------------------
model_mean = daily_mean_over_valid(ds[model_var], valid_mask)

candidate_idx = np.where(model_mean > mean_thresh)[0]
if len(candidate_idx) == 0:
    raise ValueError(f"No days found with {model_name} domain-mean precipitation > {mean_thresh} mm/day")

print(f"Number of candidate days with mean > {mean_thresh} mm/day: {len(candidate_idx)}")

rng = np.random.default_rng(seed)
t = rng.choice(candidate_idx)
day_str = str(ds["time"].values[t])[:10]

print("Selected day:", day_str)
print(f"{model_name} domain mean on selected day: {model_mean[t]:.3f} mm/day")

# -----------------------
# Pull maps
# -----------------------
pred_map = ds[model_var].isel(time=t).values
era5_map = ds["era5"].isel(time=t).values
imerg_map = ds["imerg"].isel(time=t).values
idw_map = ds_idw["idw_map_p2"].isel(time=t).values
prism_on_plot_grid = prism_da.sel(time=day_str).interp(
    x=xr.DataArray(lon1d, dims="x"),
    y=xr.DataArray(lat1d, dims="y"),
    method="nearest"
)
prism_map = prism_on_plot_grid.values

print("valid_mask shape:", valid_mask.shape)
print("pred_map shape:", pred_map.shape)
print("era5_map shape:", era5_map.shape)
print("imerg_map shape:", imerg_map.shape)
print("idw_map shape:", idw_map.shape)
print("prism_map shape:", prism_map.shape)

pred_map = np.where(valid_mask, pred_map, np.nan)
era5_map = np.where(valid_mask, era5_map, np.nan)
imerg_map = np.where(valid_mask, imerg_map, np.nan)
idw_map = np.where(valid_mask, idw_map, np.nan)
prism_map = np.where(valid_mask, prism_map, np.nan)

# -----------------------
# Smooth only GNN for display
# -----------------------
pred_map_smooth = pred_map.copy()

if use_gaussian:
    pred_map_smooth = gaussian_nan(pred_map_smooth, sigma=gaussian_sigma)

if use_bilateral:
    pred_map_smooth = bilateral_nan(
        pred_map_smooth,
        sigma_spatial=bilateral_sigma_spatial,
        sigma_intensity=bilateral_sigma_intensity,
    )

# -----------------------
# Common color scale (DISCRETIZED)
# -----------------------
all_vals = np.concatenate([
    pred_map_smooth[np.isfinite(pred_map_smooth)],
    era5_map[np.isfinite(era5_map)],
    imerg_map[np.isfinite(imerg_map)],
    idw_map[np.isfinite(idw_map)],
    prism_map[np.isfinite(prism_map)],
])

vmin = 0.0
vmax_raw = np.nanpercentile(all_vals, 99)

# round DOWN to nearest multiple of 5
vmax = np.floor(vmax_raw / 5) * 5

print(f"Raw vmax={vmax_raw:.2f} -> Rounded vmax={vmax:.0f}")

# discretization step
step = 0.5
levels = np.arange(vmin, vmax + step, step)

from matplotlib.colors import BoundaryNorm
cmap = plt.get_cmap("jet", len(levels) - 1)
norm = BoundaryNorm(levels, cmap.N)

plot_extent = [
    np.nanmin(lon1d),
    np.nanmax(lon1d),
    np.nanmin(lat1d),
    np.nanmax(lat1d),
]

# -----------------------
# Plot
# -----------------------
proj = ccrs.PlateCarree()

fig, axes = plt.subplots(
    2, 3,
    figsize=(18, 10),
    subplot_kw={"projection": proj},
    constrained_layout=True
)

axes = axes.flatten()

maps = [
    pred_map_smooth,
    era5_map,
    imerg_map,
    idw_map,
    prism_map,
]


titles = [
    f"{model_name}",
    "ERA5",
    "IMERG",
    "IDW map",
    "PRISM",
]

states = cfeature.NaturalEarthFeature(
    category="cultural",
    name="admin_1_states_provinces_lines",
    scale="10m",
    facecolor="none",
)

for ax, data, title in zip(axes, maps, titles):


    # light-blue ocean/background
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", zorder=1)
    ax.set_facecolor("lightblue")

    im = ax.imshow(
        data,
        origin="upper",
        extent=plot_extent,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        zorder=2,
    )

    # borders and coastlines
    ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor="white", zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="white", zorder=3)

    # state boundaries in white
    ax.add_feature(states, linewidth=0.9, edgecolor="white", zorder=3)



    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.4,
        color="gray",
        alpha=0.9,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    ax.set_extent(plot_extent, crs=ccrs.PlateCarree())
    ax.set_title(f"{title}")

for j in range(len(maps), len(axes)):
    axes[j].set_visible(False)

# colorbar with discrete ticks
cbar = fig.colorbar(
    im,
    ax=axes,
    shrink=0.85,
    boundaries=levels,
    ticks=levels,
)

cbar.set_label("Precipitation (mm/day) on " f"{day_str}", fontsize = 20)

fig.savefig(
    f"precip_comparison_{day_str}.png",
    dpi=500,
    bbox_inches="tight"
)
plt.show()
# %%
