import numpy as np
import pandas as pd
import pytest

import hydromend as hm


@pytest.fixture
def site():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-01", periods=4000, freq="1h")
    m = np.sin(np.arange(4000) * 2 * np.pi / 12.42) + 0.3 * rng.standard_normal(4000)
    df = pd.DataFrame({"model": m}, index=idx)
    feats = hm.build_feature_set(df, lags_hours=(1, 2), feature_set="bilinear", include_current=True)
    df["observations"] = (
        0.8 * df["model"] + 0.2 * feats["model_lag1"]
        + 0.1 * feats["model_x_model_lag2"] + 0.02 * rng.standard_normal(4000)
    )
    return df.dropna()


def test_linear_feature_names_and_shape(site):
    feats = hm.build_feature_set(site, lags_hours=(1, 2, 3), feature_set="linear", include_current=True)
    assert list(feats.columns) == ["model", "model_lag1", "model_lag2", "model_lag3"]


def test_bilinear_adds_upper_triangle(site):
    lin = hm.build_feature_set(site, lags_hours=(1, 2), feature_set="linear", include_current=True)
    bil = hm.build_feature_set(site, lags_hours=(1, 2), feature_set="bilinear", include_current=True)
    n = lin.shape[1]
    assert bil.shape[1] == n + n * (n - 1) // 2
    assert "model_lag1_x_model_lag2" in bil.columns


def test_integer_lag_naming_no_float_suffix(site):
    feats = hm.build_feature_set(site, lags_hours=[1, 2], feature_set="linear")
    assert "model_lag1" in feats.columns and "model_lag1.0" not in feats.columns


def test_sampling_inference():
    idx = pd.date_range("2020-01-01", periods=10, freq="3h")
    assert hm.infer_sampling_hours(idx) == pytest.approx(3.0)
