#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# User settings
# -----------------------
nc = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/TGCN_T30_IDW_added_test2020_2024/pred_inputs_daily_maps.nc"

mean_thresh = 0   # mm/day; only consider days with domain-mean precip > this
seed = 52

# -----------------------
# Load dataset
# -----------------------
ds = xr.open_dataset(nc)

# -----------------------
# Detect model variable automatically
# -----------------------
if "pred_det" in ds:
    model_var = "pred_det"
    model_name = "TGCN_det"
elif "pred_expected_mean" in ds:
    model_var = "pred_expected_mean"
    model_name = "TGCN_expected_mean"
elif "pred_median" in ds:
    model_var = "pred_median"
    model_name = "TGCN_median"
else:
    raise ValueError("Could not find model prediction variable in dataset.")

print("Using model variable:", model_var)

# -----------------------
# Valid grid mask
# -----------------------
valid_mask = ds["valid_pixel"].values.astype(bool)

# -----------------------
# Compute daily domain means
# -----------------------
def daily_mean_over_valid(da, valid_mask):
    arr = da.values  # (time, y, x)
    out = np.full(arr.shape[0], np.nan, dtype=np.float64)

    for t in range(arr.shape[0]):
        vals = arr[t][valid_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            out[t] = vals.mean()

    return out

model_mean = daily_mean_over_valid(ds[model_var], valid_mask)

# -----------------------
# Filter days above threshold
# -----------------------
candidate_idx = np.where(model_mean > mean_thresh)[0]

if len(candidate_idx) == 0:
    raise ValueError(f"No days found with {model_name} domain-mean precipitation > {mean_thresh} mm/day")

print(f"Number of candidate days with mean > {mean_thresh} mm/day: {len(candidate_idx)}")

# -----------------------
# Pick one random day
# -----------------------
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

# Mask invalid pixels for display
pred_map = np.where(valid_mask, pred_map, np.nan)
era5_map = np.where(valid_mask, era5_map, np.nan)
imerg_map = np.where(valid_mask, imerg_map, np.nan)

# Common color scale across all 3 maps
all_vals = np.concatenate([
    pred_map[np.isfinite(pred_map)],
    era5_map[np.isfinite(era5_map)],
    imerg_map[np.isfinite(imerg_map)],
])

vmin = 0.0
vmax = np.nanpercentile(all_vals, 99)  # robust upper limit
print(f"Using color scale: vmin={vmin:.2f}, vmax={vmax:.2f}")

# -----------------------
# Plot 1x3 figure
# -----------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

im0 = axes[0].imshow(pred_map, origin="upper", vmin=vmin, vmax=vmax)
axes[0].set_title(f"{model_name}\n{day_str}")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

im1 = axes[1].imshow(era5_map, origin="upper", vmin=vmin, vmax=vmax)
axes[1].set_title(f"ERA5\n{day_str}")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

im2 = axes[2].imshow(imerg_map, origin="upper", vmin=vmin, vmax=vmax)
axes[2].set_title(f"IMERG\n{day_str}")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")

cbar = fig.colorbar(im2, ax=axes, shrink=0.85)
cbar.set_label("Precipitation (mm/day)")

plt.show()
# %%