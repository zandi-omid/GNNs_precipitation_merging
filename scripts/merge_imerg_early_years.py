#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

root = Path(__file__).resolve().parent
files = [root / "yearly" / f"IMERG_EARLY_DAILY_{year}.nc" for year in range(2005, 2025)]
missing = [str(path) for path in files if not path.exists()]
if missing:
    raise FileNotFoundError("Missing annual files:\n" + "\n".join(missing))
datasets = [xr.open_dataset(path, engine="netcdf4") for path in files]
try:
    merged = xr.concat(datasets, dim="time").sortby("time")
    expected = pd.date_range("2005-01-01", "2024-12-31", freq="D").values.astype("datetime64[D]")
    actual = merged.time.values.astype("datetime64[D]")
    if not np.array_equal(actual, expected):
        raise RuntimeError("Merged time coverage is incomplete or contains duplicate dates.")
    output = root / "IMERG_EARLY_DAILY_2005_2024_MERGED.nc"
    merged.to_netcdf(output, engine="netcdf4", encoding={"precipitation": {
        "zlib": True, "complevel": 4, "dtype": "float32"}})
    print(f"Validated {len(actual):,} days and wrote {output}")
finally:
    for dataset in datasets:
        dataset.close()
