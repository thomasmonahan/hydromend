"""Diagnostic plots: raw series, residual KDE, learning curves, predictions."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .data import available_locations


def plot_raw_timeseries(
    model_groups: dict[str, dict[str, pd.DataFrame]],
    *,
    location: str,
    model_names: list[str] | tuple[str, ...] | None = None,
    start: str | None = None,
    end: str | None = None,
    target_column: str = "observations",
    predictor_column: str = "model",
    figsize: tuple[int, int] = (12, 5),
):
    model_names = list(model_names or model_groups.keys())
    plt.figure(figsize=figsize)
    obs_plotted = False
    for model_name in model_names:
        if location not in model_groups[model_name]:
            continue
        df = model_groups[model_name][location].sort_index()
        if start or end:
            df = df.loc[start:end]
        plt.plot(df.index, df[predictor_column], label=model_name)
        if not obs_plotted:
            plt.plot(df.index, df[target_column], label="obs", linewidth=1.5)
            obs_plotted = True
    plt.title(f"Raw series at {location}")
    plt.legend()
    plt.tight_layout()



def plot_residual_kde(
    model_groups: dict[str, dict[str, pd.DataFrame]],
    *,
    locations: list[str] | tuple[str, ...] | None = None,
    model_names: list[str] | tuple[str, ...] | None = None,
    target_column: str = "observations",
    predictor_column: str = "model",
    figsize: tuple[int, int] = (12, 6),
):
    locations = list(locations or available_locations(model_groups))
    model_names = list(model_names or model_groups.keys())

    plt.figure(figsize=figsize)
    for model_name in model_names:
        residuals = []
        for location in locations:
            if location not in model_groups[model_name]:
                continue
            df = model_groups[model_name][location]
            residual = (df[predictor_column] - df[target_column]).dropna().to_numpy()
            if residual.size:
                residuals.append(residual)
        if residuals:
            residuals_arr = np.concatenate(residuals)
            mae = np.mean(np.abs(residuals_arr))
            sns.kdeplot(residuals_arr, label=f"{model_name}: MAE={mae:.3f}")
    plt.title("Residual density: raw model minus observations")
    plt.legend()
    plt.tight_layout()



def plot_learning_curve(
    results_df: pd.DataFrame,
    *,
    metric: str = "MAE_Test",
    location: str | None = None,
    model_name: str | None = None,
    feature_set: str | None = None,
    figsize: tuple[int, int] = (10, 6),
):
    data = results_df.query("Status == 'ok'").copy()
    if location is not None:
        data = data[data["Location"] == location]
    if model_name is not None:
        data = data[data["Model"] == model_name]
    if feature_set is not None:
        data = data[data["Features"] == feature_set]

    plt.figure(figsize=figsize)
    sns.lineplot(data=data, x="TrainHours", y=metric, hue="Features", style="Regressor", marker="o")
    plt.title(f"Learning curve ({metric})")
    plt.tight_layout()



def plot_location_bars(
    results_df: pd.DataFrame,
    *,
    train_hours: float | None = None,
    metric: str = "Improvement_Ratio",
    model_name: str | None = None,
    regressor: str | None = None,
    figsize: tuple[int, int] = (14, 5),
):
    data = results_df.query("Status == 'ok'").copy()
    if train_hours is not None:
        data = data[np.isclose(data["TrainHours"], train_hours)]
    if model_name is not None:
        data = data[data["Model"] == model_name]
    if regressor is not None:
        data = data[data["Regressor"] == regressor]

    plt.figure(figsize=figsize)
    sns.barplot(data=data, x="Location", y=metric, hue="Features")
    plt.xticks(rotation=90)
    plt.title(f"Per-location comparison ({metric})")
    plt.tight_layout()



def plot_prediction_example(
    predictions: dict[tuple, pd.DataFrame],
    *,
    location: str,
    model_name: str,
    feature_set: str,
    regressor: str,
    train_hours: float,
    start: str | None = None,
    end: str | None = None,
    figsize: tuple[int, int] = (12, 5),
):
    key = (location, model_name, feature_set, regressor, float(train_hours))
    if key not in predictions:
        raise KeyError(f"Prediction key {key} not found. Available keys begin with: {list(predictions)[:5]}")

    df = predictions[key].copy()
    if start or end:
        df = df.loc[start:end]

    plt.figure(figsize=figsize)
    plt.plot(df.index, df["observed"], label="observed")
    plt.plot(df.index, df["baseline"], label="baseline")
    plt.plot(df.index, df["predicted"], label=f"{regressor} ({feature_set})")
    plt.title(f"Prediction example: {location} | {model_name} | {feature_set} | {regressor}")
    plt.legend()
    plt.tight_layout()
