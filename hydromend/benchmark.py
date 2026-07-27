"""Train/test splitting, single-case fitting, and the multi-site benchmark loop."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .features import infer_sampling_hours, hours_to_samples, build_feature_set
from .models import make_regressor, _default_min_train_samples, _regressor_label


@dataclass
class SplitConfig:
    test_size_hours: float | None = 4 * 8760
    test_start: str | pd.Timestamp | None = None
    train_sizes_hours: list[float] | None = None
    train_selection: str = "first"
def _resolve_train_test_slices(
    feature_df: pd.DataFrame,
    *,
    split_config: SplitConfig,
) -> tuple[slice, slice, float]:
    if feature_df.empty:
        raise ValueError("feature_df is empty.")

    sampling_hours = infer_sampling_hours(feature_df.index)

    if split_config.test_start is not None:
        test_start = pd.Timestamp(split_config.test_start)
        if test_start.tz is None:
            test_start = test_start.tz_localize("UTC")
        else:
            test_start = test_start.tz_convert("UTC")
        split_idx = int(feature_df.index.searchsorted(test_start))
    else:
        if split_config.test_size_hours is None:
            raise ValueError("Provide either split_config.test_start or split_config.test_size_hours.")
        n_test = hours_to_samples(split_config.test_size_hours, sampling_hours)
        if n_test is None:
            raise ValueError("Could not infer the number of test samples.")
        split_idx = len(feature_df) - n_test

    split_idx = max(1, min(split_idx, len(feature_df) - 1))
    return slice(0, split_idx), slice(split_idx, len(feature_df)), sampling_hours



def _select_training_window(X_pool: pd.DataFrame, y_pool: pd.Series, n_samples: int, selection: str = "first"):
    n_samples = min(n_samples, len(X_pool))
    if selection == "last":
        return X_pool.iloc[-n_samples:], y_pool.iloc[-n_samples:]
    return X_pool.iloc[:n_samples], y_pool.iloc[:n_samples]



def _prepare_site_case(
    site_df: pd.DataFrame,
    *,
    target_column: str,
    predictor_columns: list[str] | tuple[str, ...],
    lags_hours: list[int | float] | tuple[int | float, ...],
    feature_set: str,
    split_config: SplitConfig,
    include_current: bool,
):
    feature_df = build_feature_set(
        site_df,
        predictor_columns=predictor_columns,
        lags_hours=lags_hours,
        feature_set=feature_set,
        include_current=include_current,
    ).dropna()
    if feature_df.empty:
        raise ValueError("No non-missing lagged features after construction.")

    cols_to_add = [target_column]
    if predictor_columns[0] not in feature_df.columns:
        cols_to_add.append(predictor_columns[0])
    aligned = pd.concat([feature_df, site_df[cols_to_add]], axis=1).dropna()
    feature_df = aligned[feature_df.columns]
    y = aligned[target_column]
    baseline_series = aligned[predictor_columns[0]]
    train_pool_slice, test_slice, sampling_hours = _resolve_train_test_slices(feature_df, split_config=split_config)

    return {
        "feature_df": feature_df,
        "y": y,
        "baseline": baseline_series,
        "train_pool_slice": train_pool_slice,
        "test_slice": test_slice,
        "sampling_hours": sampling_hours,
    }



def fit_single_case(
    model_groups: dict[str, dict[str, pd.DataFrame]],
    *,
    location: str,
    model_name: str,
    target_column: str = "observations",
    predictor_columns: list[str] | tuple[str, ...] = ("model",),
    lags_hours: list[int | float] | tuple[int | float, ...] = (1, 2, 3),
    feature_set: str = "linear",
    regressor="vb_ard",
    split_config: SplitConfig | None = None,
    include_current: bool = False,
    train_hours: float | None = None,
    min_train_samples: int | None = None,
):
    if split_config is None:
        split_config = SplitConfig()
    site_df = model_groups[model_name][location].sort_index().copy()
    case = _prepare_site_case(
        site_df,
        target_column=target_column,
        predictor_columns=predictor_columns,
        lags_hours=lags_hours,
        feature_set=feature_set,
        split_config=split_config,
        include_current=include_current,
    )

    X_pool = case["feature_df"].iloc[case["train_pool_slice"]]
    y_pool = case["y"].iloc[case["train_pool_slice"]]
    X_test = case["feature_df"].iloc[case["test_slice"]]
    y_test = case["y"].iloc[case["test_slice"]]
    baseline_test = case["baseline"].iloc[case["test_slice"]]

    if train_hours is None:
        train_hours = float(len(X_pool) * case["sampling_hours"])
    n_train = hours_to_samples(train_hours, case["sampling_hours"]) or len(X_pool)
    min_required = min_train_samples if min_train_samples is not None else _default_min_train_samples(regressor, X_pool.shape[1])
    if n_train < min_required:
        raise ValueError(f"Requested train window is too small ({n_train} samples < {min_required}).")

    X_train, y_train = _select_training_window(X_pool, y_pool, n_train, selection=split_config.train_selection)
    reg = make_regressor(regressor)
    reg.fit(X_train, y_train)
    y_pred = np.asarray(reg.predict(X_test)).reshape(-1)

    out = dict(case)
    out.update(
        {
            "location": location,
            "model_name": model_name,
            "feature_set": feature_set,
            "regressor_spec": regressor,
            "regressor_label": _regressor_label(regressor),
            "regressor": reg,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": pd.Series(y_pred, index=y_test.index, name="predicted"),
            "baseline_test": baseline_test,
        }
    )
    return out



def run_benchmark(
    model_groups: dict[str, dict[str, pd.DataFrame]],
    *,
    target_column: str = "observations",
    predictor_columns: list[str] | tuple[str, ...] = ("model",),
    lags_hours: list[int | float] | tuple[int | float, ...] = (1, 2, 3),
    feature_sets: list[str] | tuple[str, ...] = ("linear", "bilinear"),
    regressors: list | tuple = ("vb_ard",),
    split_config: SplitConfig | None = None,
    include_current: bool = False,
    min_train_samples: int | None = None,
    store_predictions: bool = True,
) -> tuple[pd.DataFrame, dict[tuple, pd.DataFrame]]:
    if split_config is None:
        split_config = SplitConfig()

    results: list[dict] = []
    predictions: dict[tuple, pd.DataFrame] = {}

    for model_name, location_dict in model_groups.items():
        for location, site_df in location_dict.items():
            site_df = site_df.sort_index().copy()
            for feature_set in feature_sets:
                try:
                    case = _prepare_site_case(
                        site_df,
                        target_column=target_column,
                        predictor_columns=predictor_columns,
                        lags_hours=lags_hours,
                        feature_set=feature_set,
                        split_config=split_config,
                        include_current=include_current,
                    )
                except ValueError:
                    continue

                X_pool = case["feature_df"].iloc[case["train_pool_slice"]]
                y_pool = case["y"].iloc[case["train_pool_slice"]]
                X_test = case["feature_df"].iloc[case["test_slice"]]
                y_test = case["y"].iloc[case["test_slice"]]
                baseline_test = case["baseline"].iloc[case["test_slice"]]

                if len(X_pool) == 0 or len(X_test) == 0:
                    continue

                baseline_mae = mean_absolute_error(y_test, baseline_test)
                train_sizes_hours = split_config.train_sizes_hours or [len(X_pool) * case["sampling_hours"]]

                for train_hours in train_sizes_hours:
                    n_train = hours_to_samples(train_hours, case["sampling_hours"]) if train_hours is not None else len(X_pool)
                    if n_train is None:
                        n_train = len(X_pool)

                    X_train, y_train = _select_training_window(
                        X_pool,
                        y_pool,
                        n_samples=n_train,
                        selection=split_config.train_selection,
                    )

                    for regressor_spec in regressors:
                        min_required = min_train_samples if min_train_samples is not None else _default_min_train_samples(regressor_spec, case["feature_df"].shape[1])
                        if n_train < min_required:
                            continue
                        regressor_label = _regressor_label(regressor_spec)
                        try:
                            reg = make_regressor(regressor_spec)
                            reg.fit(X_train, y_train)
                            y_pred = np.asarray(reg.predict(X_test)).reshape(-1)
                            mae_test = mean_absolute_error(y_test, y_pred)
                            mse_test = mean_squared_error(y_test, y_pred)
                            status = "ok"
                            error_message = ""
                        except Exception as exc:
                            y_pred = np.full(len(y_test), np.nan)
                            mae_test = np.nan
                            mse_test = np.nan
                            status = "error"
                            error_message = str(exc)

                        results.append(
                            {
                                "Location": location,
                                "Model": model_name,
                                "Features": feature_set.title(),
                                "Regressor": regressor_label,
                                "TrainHours": float(train_hours) if train_hours is not None else np.nan,
                                "TrainSamples": int(len(X_train)),
                                "TestHours": float(len(X_test) * case["sampling_hours"]),
                                "TestSamples": int(len(X_test)),
                                "MAE_Test": mae_test,
                                "MSE_Test": mse_test,
                                "Baseline_MAE": baseline_mae,
                                "Improvement_Ratio": mae_test / baseline_mae if baseline_mae and np.isfinite(mae_test) else np.nan,
                                "Status": status,
                                "Error": error_message,
                            }
                        )

                        if store_predictions and status == "ok":
                            key = (
                                location,
                                model_name,
                                feature_set.title(),
                                regressor_label,
                                float(train_hours) if train_hours is not None else np.nan,
                            )
                            predictions[key] = pd.DataFrame(
                                {
                                    "observed": y_test.to_numpy(),
                                    "predicted": y_pred,
                                    "baseline": baseline_test.to_numpy(),
                                },
                                index=y_test.index,
                            )

    return pd.DataFrame(results), predictions



def summarise_results(results_df: pd.DataFrame, metric: str = "Improvement_Ratio") -> pd.DataFrame:
    summary = (
        results_df
        .query("Status == 'ok'")
        .groupby(["Model", "Regressor", "Features"])[metric]
        .mean()
        .sort_values()
        .reset_index()
    )
    return summary
