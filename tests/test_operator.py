import numpy as np
import pandas as pd
import pytest

import hydromend as hm


@pytest.fixture
def fitted():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2001-01-01", periods=5000, freq="1h")
    m = np.sin(np.arange(5000) * 2 * np.pi / 12.42) + 0.3 * rng.standard_normal(5000)
    df = pd.DataFrame({"model": m}, index=idx)
    recipe = dict(lags_hours=(1, 2, 3), feature_set="bilinear", include_current=True)
    feats = hm.build_feature_set(df, **recipe)
    df["observations"] = (
        0.8 * df["model"] + 0.2 * feats["model_lag1"]
        + 0.1 * feats["model_x_model_lag2"] + 0.02 * rng.standard_normal(5000)
    )
    df = df.dropna()
    X = hm.build_feature_set(df, **recipe).dropna()
    y = df.loc[X.index, "observations"]
    reg = hm.make_regressor("ols").fit(X, y)
    op = hm.Operator.from_regressor(
        reg, feature_names=list(X.columns), **recipe,
        metadata={"site": "syn", "lat": 50.0, "lon": 1.0, "tidal_range_m": 2.0},
    )
    return df, X, reg, op


def test_operator_matches_regressor(fitted):
    df, X, reg, op = fitted
    pred = op.predict(df[["model"]])
    direct = pd.Series(reg.predict(X), index=X.index)
    assert float((pred.reindex(direct.index) - direct).abs().max()) < 1e-10


def test_operator_reduces_error(fitted):
    df, X, reg, op = fitted
    pred = op.predict(df[["model"]]).reindex(df.index)
    raw = (df["observations"] - df["model"]).abs().mean()
    corr = (df["observations"] - pred).abs().mean()
    assert corr < raw


def test_predict_accepts_series(fitted):
    df, X, reg, op = fitted
    a = op.predict(df["model"])
    b = op.predict(df[["model"]])
    assert float((a - b).abs().max()) == 0.0


def test_library_parquet_roundtrip(fitted, tmp_path):
    df, X, reg, op = fitted
    lib = hm.OperatorLibrary.from_operators([op])
    p = tmp_path / "lib.parquet"
    lib.to_parquet(p)
    lib2 = hm.OperatorLibrary.from_parquet(p)
    assert lib2.sites == ["syn"]
    op2 = lib2.get("syn")
    pred1 = op.predict(df[["model"]])
    pred2 = op2.predict(df[["model"]])
    assert float((pred1 - pred2).abs().max()) == 0.0


def test_nearest_lookup(fitted):
    df, X, reg, op = fitted
    lib = hm.OperatorLibrary.from_operators([op])
    near = lib.nearest(50.01, 1.01)
    assert near.metadata["site"] == "syn"
    assert near.metadata["query_dist_km"] < 5.0


def test_missing_predictor_raises(fitted):
    df, X, reg, op = fitted
    # A multi-column frame with no "model" column is ambiguous -> must raise.
    bad = df.rename(columns={"model": "sealevel"})
    with pytest.raises(ValueError):
        op.predict(bad)
