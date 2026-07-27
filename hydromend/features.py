"""Feature engineering: lagged predictors and bilinear (Volterra) interactions."""
from __future__ import annotations
from itertools import combinations, combinations_with_replacement
import numpy as np
import pandas as pd


def infer_sampling_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    diffs = index.to_series().diff().dropna().dt.total_seconds().to_numpy() / 3600.0
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 1.0
    return float(np.median(diffs))



def hours_to_samples(hours: float | None, sampling_hours: float) -> int | None:
    if hours is None:
        return None
    if sampling_hours <= 0:
        raise ValueError("sampling_hours must be positive.")
    return max(1, int(round(hours / sampling_hours)))



def build_lagged_feature_dataframe(
    df: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    lags_hours: list[int | float] | tuple[int | float, ...],
    *,
    include_current: bool = False,
    sampling_hours: float | None = None,
) -> pd.DataFrame:
    sampling_hours = sampling_hours or infer_sampling_hours(df.index)
    cols = {}

    for col in columns:
        if include_current:
            cols[col] = df[col]
        for lag in lags_hours:
            step_count = int(round(lag / sampling_hours))
            cols[f"{col}_lag{lag}"] = df[col].shift(step_count)

    return pd.DataFrame(cols, index=df.index)



def pairwise_interactions_df_OLD(df: pd.DataFrame, cols: list[str] | None = None, prefix_sep: str = "_x_") -> pd.DataFrame:
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    out = df.copy()
    for a, b in combinations(cols, 2):
        out[f"{a}{prefix_sep}{b}"] = df[a] * df[b]
    return out


def pairwise_interactions_df(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    prefix_sep: str = "_x_",
    *,
    include_squares: bool = False,
) -> pd.DataFrame:
    """Append pairwise products of ``cols`` to ``df``.

    By default only distinct pairs (τ₁ < τ₂) are formed. Set
    ``include_squares=True`` to also add each column's self-product
    (``col_x_col``) — the τ₁ = τ₂ diagonal of a second-order Volterra kernel,
    i.e. the pure quadratic / overtide terms.
    """
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()

    pairs = combinations_with_replacement(cols, 2) if include_squares else combinations(cols, 2)
    interactions = {f"{a}{prefix_sep}{b}": df[a] * df[b] for a, b in pairs}
    interactions_df = pd.DataFrame(interactions, index=df.index)
    return pd.concat([df, interactions_df], axis=1)



def build_feature_set(
    data: pd.DataFrame,
    *,
    predictor_columns: list[str] | tuple[str, ...] = ("model",),
    lags_hours: list[int | float] | tuple[int | float, ...] = (1, 2, 3),
    feature_set: str = "linear",
    include_current: bool = False,
    include_squares: bool = False,
) -> pd.DataFrame:
    """Build the design matrix of lagged (and optionally bilinear) features.

    ``feature_set="bilinear"`` appends pairwise products of the lagged columns.
    With ``include_squares=True`` those products also include each lag's
    self-product (the τ₁ = τ₂ Volterra diagonal / overtide terms); the released
    ERA5-GTSM operators are trained this way.
    """
    linear = build_lagged_feature_dataframe(
        data,
        columns=list(predictor_columns),
        lags_hours=list(lags_hours),
        include_current=include_current,
    )
    if feature_set == "linear":
        return linear
    if feature_set == "bilinear":
        return pairwise_interactions_df(
            linear, cols=linear.columns.tolist(), include_squares=include_squares
        )
    raise ValueError(f"Unknown feature_set={feature_set!r}. Use 'linear' or 'bilinear'.")
