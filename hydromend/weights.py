"""Extract learned linear/bilinear lag kernels and plot them."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .priors import parse_feature_metadata


def extract_linear_lag_weights(regressor, feature_names: list[str] | tuple[str, ...] | pd.Index | None = None) -> pd.DataFrame:
    if feature_names is None:
        if not hasattr(regressor, "feature_names_in_"):
            raise ValueError("feature_names must be provided when the regressor does not expose feature_names_in_.")
        feature_names = regressor.feature_names_in_

    metadata = parse_feature_metadata(feature_names)
    if not hasattr(regressor, "coef_"):
        raise ValueError(f"Regressor {type(regressor).__name__} does not expose linear coefficients.")

    coef = np.asarray(regressor.coef_, dtype=float).reshape(-1)
    if coef.shape[0] != len(metadata):
        raise ValueError(f"Coefficient length {coef.shape[0]} does not match number of features {len(metadata)}.")

    coef_sd = getattr(regressor, "coef_sd_", np.full(len(coef), np.nan))
    coef_sd = np.asarray(coef_sd, dtype=float).reshape(-1)
    if coef_sd.shape[0] != len(metadata):
        coef_sd = np.full(len(coef), np.nan)

    out = metadata.copy()
    out["weight_mean"] = coef
    out["weight_sd"] = coef_sd
    out["weight_lo"] = out["weight_mean"] - 1.96 * out["weight_sd"]
    out["weight_hi"] = out["weight_mean"] + 1.96 * out["weight_sd"]
    return out.sort_values(["base", "lag_hours", "feature"], kind="stable").reset_index(drop=True)



def plot_lag_weight_kernel(
    weights_df: pd.DataFrame,
    *,
    bases: list[str] | tuple[str, ...] | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 5),
):
    data = weights_df.copy()
    data = data[data["is_lag_like"]].copy()
    if bases is not None:
        data = data[data["base"].isin(list(bases))]
    if data.empty:
        raise ValueError("No lag-like features were found for plotting.")

    plt.figure(figsize=figsize)
    for base, group in data.groupby("base", sort=False):
        group = group.sort_values("lag_hours")
        plt.plot(group["lag_hours"], group["weight_mean"], marker="o", label=base)
        if np.isfinite(group["weight_sd"]).any():
            lo = group["weight_lo"].to_numpy(dtype=float)
            hi = group["weight_hi"].to_numpy(dtype=float)
            mask = np.isfinite(lo) & np.isfinite(hi)
            if np.any(mask):
                plt.fill_between(
                    group.loc[mask, "lag_hours"].to_numpy(dtype=float),
                    lo[mask],
                    hi[mask],
                    alpha=0.2,
                )
    plt.axhline(0.0, linewidth=1.0, linestyle="--")
    plt.xlabel("Lag (hours)")
    plt.ylabel("Weight")
    plt.title(title or "Learned lag-weight kernel")
    plt.legend()
    plt.tight_layout()



def plot_weight_heatmap(
    weights_df: pd.DataFrame,
    *,
    value_column: str = "weight_mean",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 4),
):
    data = weights_df.copy()
    data = data[data["is_lag_like"]].copy()
    if data.empty:
        raise ValueError("No lag-like features were found for heatmap plotting.")
    pivot = data.pivot_table(index="base", columns="lag_hours", values=value_column, aggfunc="first")
    plt.figure(figsize=figsize)
    sns.heatmap(pivot, annot=False, cmap="coolwarm", center=0.0)
    plt.title(title or f"Lag-weight heatmap ({value_column})")
    plt.xlabel("Lag (hours)")
    plt.ylabel("Predictor")
    plt.tight_layout()



def compare_lag_weight_models(
    fitted_cases: dict[str, dict],
    *,
    figsize: tuple[int, int] = (10, 5),
):
    plt.figure(figsize=figsize)
    for label, case in fitted_cases.items():
        weights = extract_linear_lag_weights(case["regressor"], case["X_train"].columns)
        weights = weights[weights["is_lag_like"]].sort_values("lag_hours")
        plt.plot(weights["lag_hours"], weights["weight_mean"], marker="o", label=label)
    plt.axhline(0.0, linewidth=1.0, linestyle="--")
    plt.xlabel("Lag (hours)")
    plt.ylabel("Weight")
    plt.title("Lag-kernel comparison")
    plt.legend()
    plt.tight_layout()


def _normalise_base_pair(base_pair):
    if base_pair is None:
        return None
    if isinstance(base_pair, str):
        return tuple(sorted(part.strip() for part in base_pair.split("|")))
    return tuple(sorted(str(x) for x in base_pair))


def extract_bilinear_lag_surface(
    regressor,
    feature_names: list[str] | tuple[str, ...] | pd.Index | None = None,
    *,
    base_pair: tuple[str, str] | list[str] | str | None = None,
) -> pd.DataFrame:
    if feature_names is None:
        if not hasattr(regressor, "feature_names_in_"):
            raise ValueError("feature_names must be provided when the regressor does not expose feature_names_in_.")
        feature_names = regressor.feature_names_in_

    metadata = parse_feature_metadata(feature_names)
    if not hasattr(regressor, "coef_"):
        raise ValueError(f"Regressor {type(regressor).__name__} does not expose linear coefficients.")

    coef = np.asarray(regressor.coef_, dtype=float).reshape(-1)
    if coef.shape[0] != len(metadata):
        raise ValueError(f"Coefficient length {coef.shape[0]} does not match number of features {len(metadata)}.")

    coef_sd = getattr(regressor, "coef_sd_", np.full(len(coef), np.nan))
    coef_sd = np.asarray(coef_sd, dtype=float).reshape(-1)
    if coef_sd.shape[0] != len(metadata):
        coef_sd = np.full(len(coef), np.nan)

    out = metadata.copy()
    out["weight_mean"] = coef
    out["weight_sd"] = coef_sd
    out["weight_lo"] = out["weight_mean"] - 1.96 * out["weight_sd"]
    out["weight_hi"] = out["weight_mean"] + 1.96 * out["weight_sd"]
    out = out[out["feature_type"].eq("interaction") & out["is_bilinear_lag_like"]].copy()
    if out.empty:
        raise ValueError("No bilinear lag-interaction features were found for plotting.")

    out["base_pair_label"] = out.apply(lambda row: "|".join(sorted([str(row["base1"]), str(row["base2"])])), axis=1)
    out["term_label"] = out.apply(lambda row: f"{row['base1']}@{row['lag1_hours']:g} x {row['base2']}@{row['lag2_hours']:g}", axis=1)

    target_base_pair = _normalise_base_pair(base_pair)
    if target_base_pair is not None:
        out = out[out["base_pair_label"] == "|".join(target_base_pair)].copy()
        if out.empty:
            raise ValueError(f"No bilinear features found for base_pair={target_base_pair}.")

    return out.sort_values(["base_pair_label", "lag1_hours", "lag2_hours", "feature"], kind="stable").reset_index(drop=True)


def plot_bilinear_kernel_surface(
    surface_df: pd.DataFrame,
    *,
    value_column: str = "weight_mean",
    base_pair: tuple[str, str] | list[str] | str | None = None,
    fill_symmetric: bool = True,
    title: str | None = None,
    figsize: tuple[int, int] = (7, 6),
    ax=None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    data = surface_df.copy()
    target_base_pair = _normalise_base_pair(base_pair)
    if target_base_pair is not None:
        data = data[data["base_pair_label"] == "|".join(target_base_pair)].copy()
    if data.empty:
        raise ValueError("No bilinear surface data available for plotting.")

    unique_pairs = data["base_pair_label"].dropna().unique().tolist()
    if len(unique_pairs) > 1 and target_base_pair is None:
        raise ValueError(f"Multiple base pairs are present {unique_pairs}. Pass base_pair=... to choose one.")

    lag1_values = sorted(data["lag1_hours"].dropna().unique().tolist())
    lag2_values = sorted(data["lag2_hours"].dropna().unique().tolist())
    pivot = data.pivot_table(index="lag1_hours", columns="lag2_hours", values=value_column, aggfunc="first")
    pivot = pivot.reindex(index=lag1_values, columns=lag2_values)

    if fill_symmetric and bool(data["interaction_same_base"].iloc[0]):
        reflected = pivot.T.reindex(index=pivot.index, columns=pivot.columns)
        pivot = pivot.combine_first(reflected)

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    sns.heatmap(pivot, annot=False, cmap="coolwarm", center=0.0, ax=ax, vmin=vmin, vmax=vmax)
    ax.set_xlabel(f"Lag 2 (hours) | {data['base2'].iloc[0]}")
    ax.set_ylabel(f"Lag 1 (hours) | {data['base1'].iloc[0]}")
    ax.set_title(title or f"Bilinear lag surface ({data['base_pair_label'].iloc[0]})")
    return ax


def plot_top_bilinear_terms(
    surface_df: pd.DataFrame,
    *,
    top_n: int = 20,
    sort_by: str = "abs_weight",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
):
    data = surface_df.copy()
    if data.empty:
        raise ValueError("No bilinear surface data available for plotting.")
    data["abs_weight"] = data["weight_mean"].abs()
    data = data.sort_values(sort_by, ascending=False).head(top_n).iloc[::-1]
    plt.figure(figsize=figsize)
    plt.barh(data["term_label"], data["weight_mean"])
    plt.axvline(0.0, linewidth=1.0, linestyle="--")
    plt.xlabel("Weight")
    plt.ylabel("Lag pair")
    plt.title(title or f"Top {min(top_n, len(data))} bilinear terms")
    plt.tight_layout()


def compare_bilinear_surfaces(
    fitted_cases: dict[str, dict],
    *,
    base_pair: tuple[str, str] | list[str] | str | None = None,
    value_column: str = "weight_mean",
    fill_symmetric: bool = True,
    figsize_per_panel: tuple[float, float] = (5.5, 4.5),
):
    surfaces = {}
    finite_values = []
    for label, case in fitted_cases.items():
        try:
            surface = extract_bilinear_lag_surface(case["regressor"], case["X_train"].columns, base_pair=base_pair)
        except Exception:
            continue
        surfaces[label] = surface
        vals = surface[value_column].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            finite_values.append(vals)

    if not surfaces:
        raise ValueError("No bilinear surfaces could be extracted from the supplied cases.")

    all_values = np.concatenate(finite_values) if finite_values else np.array([0.0])
    vmax = float(np.nanmax(np.abs(all_values))) if all_values.size else 1.0
    vmax = max(vmax, 1e-12)
    vmin = -vmax

    n = len(surfaces)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]), squeeze=False)
    axes = axes[0]
    for ax, (label, surface) in zip(axes, surfaces.items()):
        plot_bilinear_kernel_surface(
            surface,
            value_column=value_column,
            fill_symmetric=fill_symmetric,
            title=label,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )
    plt.tight_layout()
    return fig, axes
