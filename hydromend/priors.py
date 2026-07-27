"""GP priors over lag / bilinear weights and feature-name metadata parsing."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ._constants import LAG_FEATURE_RE


def _parse_single_feature_name(name: str) -> dict:
    name = str(name)
    if "_x_" in name:
        left, right = name.split("_x_", 1)
        left_meta = _parse_single_feature_name(left)
        right_meta = _parse_single_feature_name(right)
        base1 = left_meta.get("base")
        base2 = right_meta.get("base")
        lag1 = float(left_meta.get("lag_hours", np.nan)) if pd.notna(left_meta.get("lag_hours", np.nan)) else np.nan
        lag2 = float(right_meta.get("lag_hours", np.nan)) if pd.notna(right_meta.get("lag_hours", np.nan)) else np.nan
        base_pair = None
        if base1 is not None and base2 is not None:
            base_pair = tuple(sorted([str(base1), str(base2)]))
        same_base = base1 is not None and base1 == base2
        bilinear_lag_like = bool(left_meta.get("is_lag_like", False) and right_meta.get("is_lag_like", False))
        lag_min = np.nan if not np.isfinite(lag1) or not np.isfinite(lag2) else min(lag1, lag2)
        lag_max = np.nan if not np.isfinite(lag1) or not np.isfinite(lag2) else max(lag1, lag2)
        return {
            "feature": name,
            "feature_type": "interaction",
            "base": None,
            "group": None,
            "lag_hours": np.nan,
            "is_lag_like": False,
            "component1": left,
            "component2": right,
            "base1": base1,
            "base2": base2,
            "lag1_hours": lag1,
            "lag2_hours": lag2,
            "lag_min_hours": lag_min,
            "lag_max_hours": lag_max,
            "interaction_group": None if base_pair is None else "|".join(base_pair),
            "interaction_same_base": same_base,
            "base_pair": base_pair,
            "is_bilinear_lag_like": bilinear_lag_like,
        }

    match = LAG_FEATURE_RE.match(name)
    if match:
        base = match.group("base")
        lag_hours = float(match.group("lag"))
        feature_type = "lag"
    else:
        base = name
        lag_hours = 0.0
        feature_type = "current"

    return {
        "feature": name,
        "feature_type": feature_type,
        "base": base,
        "group": base,
        "lag_hours": lag_hours,
        "is_lag_like": True,
        "component1": None,
        "component2": None,
        "base1": None,
        "base2": None,
        "lag1_hours": np.nan,
        "lag2_hours": np.nan,
        "lag_min_hours": np.nan,
        "lag_max_hours": np.nan,
        "interaction_group": None,
        "interaction_same_base": False,
        "base_pair": None,
        "is_bilinear_lag_like": False,
    }


def parse_feature_metadata(feature_names: list[str] | tuple[str, ...] | pd.Index) -> pd.DataFrame:
    records: list[dict] = []
    for idx, name in enumerate([str(x) for x in feature_names]):
        meta = _parse_single_feature_name(name)
        meta["feature_index"] = idx
        records.append(meta)
    return pd.DataFrame.from_records(records)


def _kernel_from_distances(
    distance_matrix: np.ndarray,
    *,
    kernel: str,
    lengthscale: float,
    variance: float,
    rq_alpha: float = 1.0,
    period_hours: float | None = None,
) -> np.ndarray:
    kernel_key = kernel.lower().strip()
    r = np.asarray(distance_matrix, dtype=float) / max(lengthscale, 1e-8)

    if kernel_key == "rbf":
        base = np.exp(-0.5 * r ** 2)
    elif kernel_key == "matern32":
        base = (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)
    elif kernel_key == "matern52":
        base = (1.0 + np.sqrt(5.0) * r + (5.0 / 3.0) * r ** 2) * np.exp(-np.sqrt(5.0) * r)
    elif kernel_key == "rq":
        base = (1.0 + 0.5 * r ** 2 / max(rq_alpha, 1e-8)) ** (-rq_alpha)
    elif kernel_key == "periodic":
        if period_hours is None or period_hours <= 0:
            raise ValueError("period_hours must be positive for a periodic kernel.")
        sine_term = np.sin(np.pi * distance_matrix / period_hours)
        base = np.exp(-2.0 * sine_term ** 2 / max(lengthscale, 1e-8) ** 2)
    else:
        raise ValueError(
            f"Unknown GP kernel {kernel!r}. Use 'rbf', 'matern32', 'matern52', 'rq', or 'periodic'."
        )
    return variance * base



def build_lag_weight_prior(
    feature_names: list[str] | tuple[str, ...] | pd.Index,
    *,
    kernel: str = "matern32",
    lengthscale_hours: float = 6.0,
    prior_variance: float = 1.0,
    iid_variance: float | None = None,
    rq_alpha: float = 1.0,
    period_hours: float | None = None,
    jitter: float = 1e-6,
) -> tuple[np.ndarray, pd.DataFrame]:
    metadata = parse_feature_metadata(feature_names)
    d = len(metadata)
    iid_variance = float(prior_variance if iid_variance is None else iid_variance)
    K = np.zeros((d, d), dtype=float)

    for group_name, group_df in metadata.dropna(subset=["group"]).groupby("group", sort=False):
        idx = group_df["feature_index"].to_numpy(dtype=int)
        lags = group_df["lag_hours"].to_numpy(dtype=float)
        distances = np.abs(lags[:, None] - lags[None, :])
        block = _kernel_from_distances(
            distances,
            kernel=kernel,
            lengthscale=lengthscale_hours,
            variance=prior_variance,
            rq_alpha=rq_alpha,
            period_hours=period_hours,
        )
        K[np.ix_(idx, idx)] = block

    uncoupled_mask = metadata["group"].isna().to_numpy()
    if np.any(uncoupled_mask):
        diag_idx = metadata.loc[uncoupled_mask, "feature_index"].to_numpy(dtype=int)
        K[diag_idx, diag_idx] = iid_variance

    zero_diag = np.where(np.diag(K) == 0)[0]
    if len(zero_diag):
        K[zero_diag, zero_diag] = iid_variance
    K = K + float(jitter) * np.eye(d)
    return K, metadata


def _bivariate_kernel_from_lag_pairs(
    lag1: np.ndarray,
    lag2: np.ndarray,
    *,
    kernel: str,
    lengthscale_hours: float,
    prior_variance: float,
    rq_alpha: float = 1.0,
    period_hours: float | None = None,
    symmetric: bool = True,
) -> np.ndarray:
    lag1 = np.asarray(lag1, dtype=float)
    lag2 = np.asarray(lag2, dtype=float)
    base11 = _kernel_from_distances(
        np.abs(lag1[:, None] - lag1[None, :]),
        kernel=kernel,
        lengthscale=lengthscale_hours,
        variance=1.0,
        rq_alpha=rq_alpha,
        period_hours=period_hours,
    )
    base22 = _kernel_from_distances(
        np.abs(lag2[:, None] - lag2[None, :]),
        kernel=kernel,
        lengthscale=lengthscale_hours,
        variance=1.0,
        rq_alpha=rq_alpha,
        period_hours=period_hours,
    )
    block = prior_variance * base11 * base22
    if symmetric:
        base12 = _kernel_from_distances(
            np.abs(lag1[:, None] - lag2[None, :]),
            kernel=kernel,
            lengthscale=lengthscale_hours,
            variance=1.0,
            rq_alpha=rq_alpha,
            period_hours=period_hours,
        )
        base21 = _kernel_from_distances(
            np.abs(lag2[:, None] - lag1[None, :]),
            kernel=kernel,
            lengthscale=lengthscale_hours,
            variance=1.0,
            rq_alpha=rq_alpha,
            period_hours=period_hours,
        )
        block = 0.5 * (block + prior_variance * base12 * base21)
    return block


def build_bilinear_weight_prior(
    feature_names: list[str] | tuple[str, ...] | pd.Index,
    *,
    kernel: str = "matern32",
    lengthscale_hours: float = 6.0,
    prior_variance: float = 1.0,
    iid_variance: float | None = None,
    rq_alpha: float = 1.0,
    period_hours: float | None = None,
    symmetric_same_base: bool = True,
    jitter: float = 1e-6,
) -> tuple[np.ndarray, pd.DataFrame]:
    metadata = parse_feature_metadata(feature_names)
    d = len(metadata)
    iid_variance = float(prior_variance if iid_variance is None else iid_variance)
    K = np.zeros((d, d), dtype=float)

    mask = metadata["feature_type"].eq("interaction") & metadata["is_bilinear_lag_like"]
    interaction_df = metadata.loc[mask].copy()

    for _, group_df in interaction_df.groupby("interaction_group", sort=False):
        idx = group_df["feature_index"].to_numpy(dtype=int)
        lag1 = group_df["lag1_hours"].to_numpy(dtype=float)
        lag2 = group_df["lag2_hours"].to_numpy(dtype=float)
        symmetric = bool(group_df["interaction_same_base"].iloc[0] and symmetric_same_base)
        block = _bivariate_kernel_from_lag_pairs(
            lag1,
            lag2,
            kernel=kernel,
            lengthscale_hours=lengthscale_hours,
            prior_variance=prior_variance,
            rq_alpha=rq_alpha,
            period_hours=period_hours,
            symmetric=symmetric,
        )
        K[np.ix_(idx, idx)] = block

    uncoupled_mask = metadata["feature_type"].eq("interaction") & ~metadata["is_bilinear_lag_like"]
    if np.any(uncoupled_mask):
        diag_idx = metadata.loc[uncoupled_mask, "feature_index"].to_numpy(dtype=int)
        K[diag_idx, diag_idx] = iid_variance

    zero_diag = np.where(np.diag(K) == 0)[0]
    if len(zero_diag):
        K[zero_diag, zero_diag] = iid_variance
    K = K + float(jitter) * np.eye(d)
    return K, metadata


def build_volterra_weight_prior(
    feature_names: list[str] | tuple[str, ...] | pd.Index,
    *,
    linear_kernel: str = "matern32",
    linear_lengthscale_hours: float = 6.0,
    linear_prior_variance: float = 1.0,
    linear_iid_variance: float | None = None,
    bilinear_kernel: str = "matern32",
    bilinear_lengthscale_hours: float = 6.0,
    bilinear_prior_variance: float = 1.0,
    bilinear_iid_variance: float | None = None,
    linear_rq_alpha: float = 1.0,
    bilinear_rq_alpha: float = 1.0,
    linear_period_hours: float | None = None,
    bilinear_period_hours: float | None = None,
    symmetric_same_base: bool = True,
    jitter: float = 1e-6,
) -> tuple[np.ndarray, pd.DataFrame]:
    metadata = parse_feature_metadata(feature_names)
    d = len(metadata)
    K = np.zeros((d, d), dtype=float)

    linear_mask = metadata["is_lag_like"]
    if np.any(linear_mask):
        linear_features = metadata.loc[linear_mask, "feature"].tolist()
        K_linear, _ = build_lag_weight_prior(
            linear_features,
            kernel=linear_kernel,
            lengthscale_hours=linear_lengthscale_hours,
            prior_variance=linear_prior_variance,
            iid_variance=linear_iid_variance,
            rq_alpha=linear_rq_alpha,
            period_hours=linear_period_hours,
            jitter=0.0,
        )
        idx = metadata.loc[linear_mask, "feature_index"].to_numpy(dtype=int)
        K[np.ix_(idx, idx)] = K_linear

    bilinear_mask = metadata["feature_type"].eq("interaction")
    if np.any(bilinear_mask):
        bilinear_features = metadata.loc[bilinear_mask, "feature"].tolist()
        K_bilinear, _ = build_bilinear_weight_prior(
            bilinear_features,
            kernel=bilinear_kernel,
            lengthscale_hours=bilinear_lengthscale_hours,
            prior_variance=bilinear_prior_variance,
            iid_variance=bilinear_iid_variance,
            rq_alpha=bilinear_rq_alpha,
            period_hours=bilinear_period_hours,
            symmetric_same_base=symmetric_same_base,
            jitter=0.0,
        )
        idx = metadata.loc[bilinear_mask, "feature_index"].to_numpy(dtype=int)
        K[np.ix_(idx, idx)] = K_bilinear

    zero_diag = np.where(np.diag(K) == 0)[0]
    if len(zero_diag):
        fallback = float(linear_prior_variance) if linear_prior_variance > 0 else 1.0
        K[zero_diag, zero_diag] = fallback
    K = K + float(jitter) * np.eye(d)
    return K, metadata
