"""
Convenience glue for the GESLA-4.1 + GTSM-ERA5-E workflow.

:func:`load_pair` reads one tide-gauge record, finds (or takes) its nearest GTSM
station, streams that station over the gauge's period, and returns a single
hourly-aligned DataFrame with ``observations`` and ``model`` columns — exactly
the shape :func:`hydromend.build_feature_set` and :class:`hydromend.Operator`
expect. That makes "load the right GTSM point, load the gauge, apply the
operator, and compare" a two-liner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import gesla, gtsm


def load_pair(
    gesla_file,
    gtsm_glob,
    *,
    station_index=None,
    coords=None,
    demean: bool = False,
    pad_days: int = 3,
    start=None,
    end=None,
):
    """Load an hourly-aligned ``(observations, model)`` frame for one gauge.

    Parameters
    ----------
    gesla_file
        Path to a GESLA-4.1 record.
    gtsm_glob
        Glob/dir/list locating the GTSM monthly NetCDFs.
    station_index
        Positional GTSM station index to use. If ``None``, the nearest station
        to the gauge coordinates is chosen automatically.
    coords
        Optional cached :func:`hydromend.gtsm.station_coordinates` table to
        speed up repeated nearest-station lookups.
    demean
        If True, subtract each series' mean (removes datum offset for a quick
        visual compare). Off by default so absolute levels are preserved.
    pad_days
        Days of GTSM to load either side of the gauge period.
    start, end
        Optional bounds (year int or datetime-like) restricting both series —
        handy to keep an example fast by using only a few recent years.

    Returns
    -------
    (df, info) : (pandas.DataFrame, dict)
        ``df`` is indexed by time with columns ``observations`` and ``model``;
        ``info`` carries the gauge header, chosen ``station_index`` and the
        gauge-to-station distance in km.
    """
    obs, hdr = gesla.read_record(gesla_file)
    obs = obs.dropna()
    if start is not None or end is not None:
        lo = None if start is None else (pd.Timestamp(int(start), 1, 1) if isinstance(start, (int, np.integer)) else pd.Timestamp(start))
        hi = None if end is None else (pd.Timestamp(int(end), 1, 1) if isinstance(end, (int, np.integer)) else pd.Timestamp(end))
        obs = obs.loc[lo:hi]
    if obs.empty:
        raise ValueError(f"No usable observations in {gesla_file} for the requested period.")
    lat, lon = hdr.get("LATITUDE"), hdr.get("LONGITUDE")

    dist_km = np.nan
    if station_index is None:
        if lat is None or lon is None or not np.isfinite([lat, lon]).all():
            raise ValueError("Gauge header lacks coordinates; pass station_index=... explicitly.")
        station_index, dist_km = gtsm.nearest_station(lat, lon, gtsm_glob, coords=coords)

    pad = pd.Timedelta(days=pad_days)
    model = gtsm.load_station(
        gtsm_glob, station_index,
        start=obs.index.min() - pad, end=obs.index.max() + pad,
    )

    df = pd.concat([obs.rename("observations"), model.rename("model")], axis=1).dropna()
    if df.empty:
        raise ValueError("No overlapping timestamps between gauge and GTSM station.")
    if demean:
        df = df - df.mean()

    info = {
        "site": hdr.get("SITE NAME"), "file": hdr.get("file"),
        "lat": lat, "lon": lon,
        "station_index": int(station_index), "gtsm_dist_km": float(dist_km),
        "header": hdr,
    }
    return df, info
