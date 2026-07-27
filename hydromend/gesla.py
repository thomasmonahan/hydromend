"""
Reader for GESLA-4.1 tide-gauge flat files (optional convenience IO).

Each GESLA-4.1 record is a text file: a block of ``# KEY ... value`` header
lines followed by whitespace-delimited rows

    yyyy/mm/dd  hh:mm:ss  sea_level  qc_flag  use_flag

:func:`read_header` parses the metadata block, :func:`read_record` loads the
observations onto a UTC hourly grid (applying the null value, QC and use flags,
and the time-zone shift), and :func:`yearly_coverage` reports per-year
completeness. This module has no third-party dependencies beyond numpy/pandas.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Header keys of interest (longest-first so multi-word keys match before short
# prefixes). first occurrence of each wins.
_HEADER_KEYS = [
    "FORMAT VERSION", "SITE NAME", "SITE CODE", "COUNTRY",
    "LATITUDE", "LONGITUDE", "START DATE/TIME", "END DATE/TIME",
    "NUMBER OF YEARS", "TIME ZONE HOURS", "DATUM INFORMATION",
    "GAUGE TYPE", "OVERALL RECORD QUALITY", "NULL VALUE", "CONTRIBUTOR",
]


def read_header(path, max_lines: int = 60) -> dict:
    """Return a dict of header fields (first occurrence of each key)."""
    out = {"file": os.path.basename(path)}
    with open(path, "r", errors="replace") as fh:
        for _ in range(max_lines):
            line = fh.readline()
            if not line or not line.startswith("#"):
                break
            body = line[1:].strip()
            for key in _HEADER_KEYS:
                if body.startswith(key) and key not in out:
                    out[key] = body[len(key):].strip()
                    break
    for k in ("LATITUDE", "LONGITUDE", "NUMBER OF YEARS", "TIME ZONE HOURS", "NULL VALUE"):
        if k in out:
            try:
                out[k] = float(out[k])
            except ValueError:
                out[k] = np.nan
    return out


def read_record(path, *, use_flag_only: bool = True, to_hourly: bool = True):
    """Load observations as a Series indexed by (UTC-naive) datetime.

    Applies the null value, keeps QC in {0, 1} (and use-flag == 1 if requested),
    shifts by the file's ``TIME ZONE HOURS`` to UTC, and (default) averages onto
    a regular hourly grid. Returns ``(series, header_dict)``.
    """
    hdr = read_header(path)
    null = hdr.get("NULL VALUE", -99.9999)
    tzh = hdr.get("TIME ZONE HOURS", 0.0) or 0.0
    df = pd.read_csv(
        path, sep=r"\s+", comment="#", header=None,
        names=["date", "time", "sl", "qc", "use"],
        usecols=[0, 1, 2, 3, 4], engine="c", na_values=[null],
        dtype={"sl": "float64", "qc": "float64", "use": "float64"},
    )
    ts = pd.to_datetime(df["date"] + " " + df["time"],
                        format="%Y/%m/%d %H:%M:%S", errors="coerce")
    s = pd.Series(df["sl"].values, index=ts)
    good = df["qc"].isin([0, 1]).values
    if use_flag_only:
        good &= (df["use"] == 1).values
    s = s[good & s.index.notna() & np.isfinite(s.values)]
    if tzh:
        s.index = s.index - pd.to_timedelta(tzh, unit="h")   # -> UTC
    s = s[~s.index.duplicated(keep="first")].sort_index()
    if to_hourly:
        s = s.resample("1h").mean()
    return s, hdr


def yearly_coverage(series) -> pd.Series:
    """Fraction of each calendar year's hours present (non-NaN), indexed by year."""
    s = series.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    cnt = s.groupby(s.index.year).size()
    hrs = pd.Series({y: (pd.Timestamp(y + 1, 1, 1) - pd.Timestamp(y, 1, 1))
                     // pd.Timedelta("1h") for y in cnt.index})
    return (cnt / hrs).clip(upper=1.0)
