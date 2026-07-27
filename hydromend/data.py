"""Load co-located observation/model series from NetCDF into per-site frames."""
from __future__ import annotations
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import xarray as xr
from ._constants import BAD_QC_FLAGS


def _expand_paths(path_like: str | Path | list[str] | tuple[str, ...]) -> list[Path]:
    if isinstance(path_like, (str, Path)):
        path_like = str(path_like)
        if any(char in path_like for char in "*?[]"):
            return [Path(p) for p in sorted(glob.glob(path_like))]
        p = Path(path_like)
        if p.is_dir():
            return sorted(p.glob("*.nc"))
        return [p]
    paths: list[Path] = []
    for item in path_like:
        paths.extend(_expand_paths(item))
    return sorted(paths)



def _infer_dims(ds: xr.Dataset, data_var: str = "model", time_coord: str = "time") -> tuple[str, str]:
    dims = list(ds[data_var].dims)
    if time_coord in dims:
        time_dim = time_coord
    else:
        time_dim = next((d for d in dims if "time" in d.lower()), None)
        if time_dim is None:
            time_dim = next((d for d in dims if d.lower().endswith("t_dim")), None)
    if time_dim is None:
        raise ValueError(f"Could not infer a time dimension from {dims}.")

    site_dim = next((d for d in dims if d != time_dim), None)
    if site_dim is None:
        raise ValueError(f"Could not infer a site dimension from {dims}.")
    return site_dim, time_dim



def _apply_qc_mask(ds: xr.Dataset, qc_var: str = "qc_flags", bad_qc_flags: tuple[str, ...] = BAD_QC_FLAGS) -> xr.Dataset:
    if qc_var not in ds:
        return ds

    masked = ds
    for flag in bad_qc_flags:
        masked = masked.where(masked[qc_var] != flag, np.nan)
    return masked



def load_location_dict_from_netcdf(
    files: str | Path | list[str] | tuple[str, ...],
    *,
    obs_var: str = "obs",
    model_var: str = "model",
    qc_var: str = "qc_flags",
    time_coord: str = "time",
    site_name_var: str = "site_name",
    demean_observations: bool = True,
    bad_qc_flags: tuple[str, ...] = BAD_QC_FLAGS,
    target_column: str = "observations",
    predictor_column: str = "model",
) -> dict[str, pd.DataFrame]:
    all_files = _expand_paths(files)
    if not all_files:
        raise FileNotFoundError(f"No NetCDF files found for {files!r}.")

    location_frames: dict[str, list[pd.DataFrame]] = {}
    for file_path in all_files:
        ds = xr.open_dataset(file_path)
        try:
            ds = _apply_qc_mask(ds, qc_var=qc_var, bad_qc_flags=bad_qc_flags)
            site_dim, time_dim = _infer_dims(ds, data_var=model_var, time_coord=time_coord)

            obs = ds[obs_var]
            if demean_observations:
                obs = obs - obs.mean(dim=time_dim)
            model = ds[model_var]

            if time_coord in ds:
                time_values = pd.to_datetime(ds[time_coord].values, utc=True)
            else:
                time_values = pd.to_datetime(ds[time_dim].values, utc=True)

            if site_name_var in ds:
                site_names = [str(x) for x in ds[site_name_var].values]
            elif site_dim in ds.coords:
                site_names = [str(x) for x in ds.coords[site_dim].values]
            else:
                site_names = [f"site_{i}" for i in range(model.sizes[site_dim])]

            model_values = np.asarray(model.values)
            obs_values = np.asarray(obs.values)
            if model_values.ndim != 2 or obs_values.ndim != 2:
                raise ValueError(
                    f"Expected 2D site-time arrays in {file_path}, got shapes {model_values.shape} and {obs_values.shape}."
                )

            if list(model.dims) != [site_dim, time_dim]:
                model_values = np.asarray(model.transpose(site_dim, time_dim).values)
            if list(obs.dims) != [site_dim, time_dim]:
                obs_values = np.asarray(obs.transpose(site_dim, time_dim).values)

            for site_idx, site in enumerate(site_names):
                frame = pd.DataFrame(
                    {
                        target_column: obs_values[site_idx],
                        predictor_column: model_values[site_idx],
                    },
                    index=time_values,
                )
                frame.index.name = "time"
                location_frames.setdefault(site, []).append(frame)
        finally:
            ds.close()

    merged: dict[str, pd.DataFrame] = {}
    for site, frames in location_frames.items():
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        merged[site] = df
    return merged



def load_model_groups(
    data_sources: dict[str, str | Path | list[str] | tuple[str, ...]],
    **loader_kwargs,
) -> dict[str, dict[str, pd.DataFrame]]:
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for name, source in data_sources.items():
        out[name] = load_location_dict_from_netcdf(source, **loader_kwargs)
    return out



def available_locations(model_groups: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
    locations: set[str] = set()
    for group in model_groups.values():
        locations.update(group.keys())
    return sorted(locations)
