#!/usr/bin/env python3
"""Download native IMERG Early half-hourly granules and accumulate UTC days."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import earthaccess
import requests
from earthaccess.store import Store
from requests.adapters import HTTPAdapter
import xarray as xr
from netCDF4 import Dataset
from urllib3.util.retry import Retry

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
SHORT_NAME = "GPM_3IMERGHHE"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--xmin", type=float, default=-117.05)
    parser.add_argument("--xmax", type=float, default=-106.45)
    parser.add_argument("--ymin", type=float, default=29.25)
    parser.add_argument("--ymax", type=float, default=39.00)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--keep-half-hourly", action="store_true")
    return parser.parse_args()


def earthdata_store():
    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        raise RuntimeError("Earthdata .netrc authentication failed.")
    return Store(auth)


def configure_session(store, url):
    store.set_requests_session(url)
    session = store.get_requests_session()
    session.headers["User-Agent"] = "GSI-IMERG-Early-half-hourly/1.0"
    retries = Retry(total=8, backoff_factor=2,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=("GET",))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def search_day(day):
    response = requests.get(CMR, params={
        "short_name": SHORT_NAME,
        "version": "07",
        "temporal": f"{day:%Y-%m-%d}T00:00:00Z,{day:%Y-%m-%d}T23:59:59Z",
        "page_size": 100,
        "sort_key[]": "start_date",
    }, timeout=120)
    response.raise_for_status()
    records = []
    for entry in response.json()["feed"]["entry"]:
        links = [link["href"] for link in entry.get("links", [])
                 if link.get("rel", "").endswith("/data#")
                 and link.get("href", "").endswith(".HDF5")]
        if links:
            records.append((entry["time_start"], links[0]))
    return sorted(records)


def download(session, url, target, timeout):
    partial = target.with_suffix(target.suffix + ".part")
    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            shutil.copyfileobj(response.raw, stream)
    partial.replace(target)


def read_rate(path, bounds):
    xmin, xmax, ymin, ymax = bounds
    with Dataset(path) as root:
        grid = root.groups["Grid"]
        lon = np.asarray(grid.variables["lon"][:])
        lat = np.asarray(grid.variables["lat"][:])
        ix = np.flatnonzero((lon >= xmin) & (lon <= xmax))
        iy = np.flatnonzero((lat >= ymin) & (lat <= ymax))
        if not ix.size or not iy.size:
            raise RuntimeError(f"Bounds do not intersect {path.name}")
        rate = np.asarray(grid.variables["precipitation"][
            0, ix[0]:ix[-1] + 1, iy[0]:iy[-1] + 1], dtype=np.float32).T
        rate[rate < 0] = np.nan
        x, y = lon[ix], lat[iy]
        if y[-1] > y[0]:
            y, rate = y[::-1], rate[::-1]
    return x, y, rate


def main():
    args = parse_args()
    store = earthdata_store()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    bounds = args.xmin, args.xmax, args.ymin, args.ymax
    daily_dir = args.cache_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    totals, dates = [], []
    x = y = None
    for day in pd.date_range(args.start, args.end, freq="D"):
        daily_path = daily_dir / f"IMERG_EARLY_DAILY_{day:%Y%m%d}.nc"
        if daily_path.exists():
            with xr.open_dataset(daily_path, engine="netcdf4") as saved:
                saved_data = saved["precipitation"].load()
            x, y = saved_data.x.values, saved_data.y.values
            totals.append(saved_data.values[0])
            dates.append(day)
            continue
        records = search_day(day)
        if len(records) != 48:
            raise RuntimeError(f"{day:%Y-%m-%d}: found {len(records)} granules; expected 48")
        session = configure_session(store, records[0][1])
        pending = []
        for _, url in records:
            local = args.cache_dir / Path(urlparse(url).path).name
            if not local.exists():
                pending.append((session, url, local, args.timeout))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(lambda values: download(*values), pending))
        total = valid = None
        for _, url in records:
            local = args.cache_dir / Path(urlparse(url).path).name
            gx, gy, rate = read_rate(local, bounds)
            if total is None:
                x, y = gx, gy
                total = np.zeros(rate.shape, dtype=np.float32)
                valid = np.ones(rate.shape, dtype=bool)
            total += np.nan_to_num(rate) * 0.5
            valid &= np.isfinite(rate)
            if not args.keep_half_hourly:
                local.unlink(missing_ok=True)
        total[~valid] = np.nan
        checkpoint = xr.DataArray(
            total[None], dims=("time", "y", "x"),
            coords={"time": [day], "y": y, "x": x}, name="precipitation",
        ).astype(np.float32)
        checkpoint.to_netcdf(daily_path, engine="netcdf4", encoding={
            "precipitation": {"zlib": True, "complevel": 4, "dtype": "float32"}
        })
        totals.append(total)
        dates.append(day)
        print(f"Completed {day:%Y-%m-%d}: 48/48 granules", flush=True)

    data = xr.DataArray(np.stack(totals), dims=("time", "y", "x"),
                        coords={"time": dates, "y": y, "x": x},
                        name="precipitation").astype(np.float32)
    data.attrs.update(units="mm/day", source="GPM_3IMERGHHE.07",
                      aggregation="sum of 48 half-hourly precipitation rates multiplied by 0.5 hour")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(args.output, engine="netcdf4", encoding={"precipitation": {
        "zlib": True, "complevel": 4, "dtype": "float32"}})
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
