#%%
import xarray as xr
import numpy as np

ds = xr.open_dataset(
    "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/inference/"
    "TGCN_T30_log_normal_test2020_2024/pred_inputs_daily_maps.nc"
)

# Mean over time (skip NaNs)
mean_pred  = ds["pred"].mean(dim="time", skipna=True)
mean_era5  = ds["era5"].mean(dim="time", skipna=True)
mean_imerg = ds["imerg"].mean(dim="time", skipna=True)

print("Mean daily precipitation (mm/day):")
print("PRED  :", float(mean_pred.mean()))
print("ERA5  :", float(mean_era5.mean()))
print("IMERG :", float(mean_imerg.mean()))

#%%

import matplotlib.pyplot as plt

vmax = np.nanpercentile(mean_era5.values, 99)  # shared color scale

fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

im0 = axes[0].imshow(mean_era5, origin="upper", vmin=0, vmax=vmax)
axes[0].set_title("ERA5 mean (mm/day)")

im1 = axes[1].imshow(mean_imerg, origin="upper", vmin=0, vmax=vmax)
axes[1].set_title("IMERG mean (mm/day)")

im2 = axes[2].imshow(mean_pred, origin="upper", vmin=0, vmax=vmax)
axes[2].set_title("TGCN mean (mm/day)")

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

cbar = fig.colorbar(im2, ax=axes, shrink=0.85)
cbar.set_label("mm/day")

plt.show()
# %%

import xarray as xr
import numpy as np

ds = xr.open_dataset(
    "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/"
    "inference/TGCN_T30_log_normal_test2020_2024/pred_inputs_daily_maps.nc"
)

lat = ds["lat"].values

print("lat at y=0 (min, max):",
      np.nanmin(lat[0, :]),
      np.nanmax(lat[0, :]))

print("lat at y=end (min, max):",
      np.nanmin(lat[-1, :]),
      np.nanmax(lat[-1, :]))

# %%

