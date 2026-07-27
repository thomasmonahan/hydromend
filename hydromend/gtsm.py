"""
Reader for GTSM-ERA5(-E) monthly water-level NetCDFs (optional convenience IO).

The GTSM reanalysis is distributed as monthly files, each holding
``waterlevel(time, stations)`` plus ``station_x_coordinate`` /
``station_y_coordinate``. This module locates the station nearest a coordinate
and streams one station's hourly series out of the monthly files, so you never
load the full (time x 43k-station) array into memory.

``station_index`` throughout is the **positional** index along the ``stations``
dimension (0 .. n_stations-1) — the same index used to slice ``waterlevel`` —
which is what the ERA5-GTSM operator library stores as ``gtsm_station_idx``.

Requires ``xarray`` (install the ``[io]`` extra).
"""
from __future__ import annotations

import glob as _glob
import os
import re

import numpy as np
import pandas as pd

_MONTH_RE = re.compile(r"_(\d{4})_(\d{2})_v\d+")


def _expand(gtsm_glob) -> list:
    """Accept a glob pattern, a directory, or a list of file paths."""
    if isinstance(gtsm_glob, (list, tuple)):
        files = [f for g in gtsm_glob for f in _expand(g)]
    elif os.path.isdir(gtsm_glob):
        files = _glob.glob(os.path.join(gtsm_glob, "**", "*.nc"), recursive=True)
    else:
        files = _glob.glob(gtsm_glob)
    return sorted(set(files), key=_month_key)


def _month_key(path) -> tuple:
    m = _MONTH_RE.search(os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def station_coordinates(gtsm_glob) -> pd.DataFrame:
    """Return a DataFrame(station_index, lon, lat) from the first matching file."""
    import xarray as xr
    files = _expand(gtsm_glob)
    if not files:
        raise FileNotFoundError(f"No GTSM NetCDF files matched {gtsm_glob!r}.")
    with xr.open_dataset(files[0]) as ds:
        lon = np.asarray(ds["station_x_coordinate"].values, dtype=float)
        lat = np.asarray(ds["station_y_coordinate"].values, dtype=float)
    return pd.DataFrame({"station_index": np.arange(len(lon)), "lon": lon, "lat": lat})


def nearest_station(lat, lon, gtsm_glob=None, *, coords=None):
    """Nearest GTSM station to (lat, lon). Returns ``(station_index, dist_km)``.

    Pass ``coords=station_coordinates(...)`` to avoid re-reading the header on
    repeated lookups.
    """
    if coords is None:
        coords = station_coordinates(gtsm_glob)
    d = _haversine_km(lat, lon, coords["lat"].to_numpy(float), coords["lon"].to_numpy(float))
    k = int(np.nanargmin(d))
    return int(coords["station_index"].iloc[k]), float(d[k])


def load_station(gtsm_glob, station_index, *, start=None, end=None, var="waterlevel") -> pd.Series:
    """Stream one station's hourly series out of the monthly files.

    ``start`` / ``end`` (year ints or datetime-like) restrict which monthly
    files are opened, so a single-site, single-decade pull is fast. Returns a
    Series named ``model`` indexed by time.
    """
    import xarray as xr
    files = _expand(gtsm_glob)
    if not files:
        raise FileNotFoundError(f"No GTSM NetCDF files matched {gtsm_glob!r}.")
    y0 = _to_year(start, default=-np.inf)
    y1 = _to_year(end, default=np.inf)
    files = [f for f in files if y0 <= _month_key(f)[0] <= y1]

    times, vals = [], []
    for f in files:
        with xr.open_dataset(f) as ds:
            vals.append(np.asarray(ds[var].values[:, int(station_index)], dtype=np.float32))
            times.append(np.asarray(ds["time"].values))
    if not times:
        return pd.Series(dtype="float32", name="model")
    idx = pd.DatetimeIndex(np.concatenate(times))
    s = pd.Series(np.concatenate(vals), index=idx, name="model").sort_index()
    s = s[~s.index.duplicated(keep="first")]
    if start is not None or end is not None:
        s = s.loc[_to_ts(start):_to_ts(end)]
    return s


def _to_year(x, default):
    if x is None:
        return default
    if isinstance(x, (int, np.integer)):
        return int(x)
    return pd.Timestamp(x).year


def _to_ts(x):
    return None if x is None else (pd.Timestamp(int(x), 1, 1) if isinstance(x, (int, np.integer)) else pd.Timestamp(x))


def _haversine_km(lat, lon, lat2, lon2):
    p = np.pi / 180.0
    a = (np.sin((lat2 - lat) * p / 2) ** 2
         + np.cos(lat * p) * np.cos(lat2 * p) * np.sin((lon2 - lon) * p / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))
