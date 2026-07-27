import numpy as np
import pandas as pd
import pytest

import hydromend as hm


@pytest.fixture
def fitted():
    rng = np.random.default_rng(3)
    n = 4000
    t = np.arange(n)
    m = np.cos(2 * np.pi * t / 12.42) + 0.4 * np.cos(2 * np.pi * t / 12.0) + 0.3 * rng.standard_normal(n)
    idx = pd.date_range("2020", periods=n, freq="1h")
    df = pd.DataFrame({"model": m}, index=idx)
    recipe = dict(lags_hours=list(range(1, 13)), feature_set="bilinear",
                  include_current=True, include_squares=True)
    X = hm.build_feature_set(df, **recipe)
    df["observations"] = (0.9 * X["model"] + 0.1 * X["model_x_model"]
                          + 0.05 * X["model_lag2_x_model_lag3"] + 0.02 * rng.standard_normal(n))
    d = pd.concat([X, df["observations"]], axis=1).dropna()
    reg = hm.make_regressor("ols").fit(d[X.columns], d["observations"])
    return df, X, reg, recipe


def test_admittance_shape_and_dc(fitted):
    df, X, reg, recipe = fitted
    f, H = hm.linear_admittance(reg, list(X.columns))
    assert len(f) == len(H) and np.iscomplexobj(H)
    assert f[0] == 0.0                              # DC bin present


def test_qtf_symmetric(fitted):
    df, X, reg, recipe = fitted
    f, H2 = hm.quadratic_transfer_function(reg, list(X.columns))
    assert H2.shape == (len(f), len(f))
    # kernel is symmetric -> QTF is symmetric in (f1, f2)
    assert np.allclose(H2, H2.T, atol=1e-8)


def test_weighted_qtf_nonnegative(fitted):
    df, X, reg, recipe = fitted
    f, W = hm.weighted_quadratic_transfer(reg, df["model"], list(X.columns))
    assert W.shape == (len(f), len(f))
    assert np.all(W >= 0.0)


def test_works_via_operator(fitted):
    df, X, reg, recipe = fitted
    op = hm.Operator.from_regressor(reg, feature_names=list(X.columns), **recipe)
    f1, W1 = hm.weighted_quadratic_transfer(reg, df["model"], list(X.columns))
    f2, W2 = hm.weighted_quadratic_transfer(op, df["model"])      # feature names inferred
    assert np.allclose(np.nan_to_num(W1), np.nan_to_num(W2))


def test_plot_smoke(fitted):
    import matplotlib
    matplotlib.use("Agg")
    df, X, reg, recipe = fitted
    f, W = hm.weighted_quadratic_transfer(reg, df["model"], list(X.columns))
    ax = hm.plot_quadratic_transfer(f, W)
    assert ax is not None
