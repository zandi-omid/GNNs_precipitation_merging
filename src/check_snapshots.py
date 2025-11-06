import torch
import glob

snapshot_dir = "/xdisk/behrangi/omidzandi/GNNs/gnn_precipitation_retrieval/data/pyg_snapshots/daily_snapshots_landmask"
files = sorted(glob.glob(f"{snapshot_dir}/day_*.pt"))
ERA5_idx = 3

first_valid = None
nan_mask_ref = None

for f in files:
    try:
        d = torch.load(f)
    except Exception as e:
        print(f"⚠️ Skipping corrupted file: {f} ({e})")
        continue

    nan_mask = torch.isnan(d.x[:, ERA5_idx])
    if first_valid is None:
        nan_mask_ref = nan_mask
        n_ref = nan_mask.sum().item()
        first_valid = f
        continue

    if not torch.equal(nan_mask, nan_mask_ref):
        print(f"❌ Different NaN pattern in {f.split('/')[-1]}")
        break
else:
    print(f"✅ All non-corrupted snapshots share identical NaN locations ({n_ref} NaN nodes).")