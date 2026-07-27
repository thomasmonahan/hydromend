# hydromend — concepts guide

`hydromend` fits and applies **lag operators** that post-process a hydrodynamic
model against observations. This page explains the model, the feature
construction, and the objects you work with. For the callable reference see
[`api.md`](api.md); for a runnable walkthrough see
[`../examples/quickstart.ipynb`](../examples/quickstart.ipynb).

## The model

Given an hourly model series *m(t)* (e.g. GTSM total water level) and an observed
series *y(t)* (e.g. a tide gauge), a hydromend operator predicts

- **linear:**  ŷ(t) = c + Σ_τ w(τ) · m(t−τ)
- **bilinear:** ŷ(t) = c + Σ_τ w₁(τ)·m(t−τ) + Σ_{τ₁≤τ₂} w₂(τ₁,τ₂)·m(t−τ₁)·m(t−τ₂)

The linear part is a finite-impulse-response *lag kernel*; the bilinear part is a
second-order **Volterra kernel** that represents the quadratic tide–surge and
shallow-water interactions a purely additive correction cannot. Lags run over a
chosen memory window (e.g. 0–24 h at 1 h spacing → 25 linear + 325 bilinear
terms).

## Feature construction

`build_feature_set(df, predictor_columns, lags_hours, feature_set, include_current)`
turns a frame into the design matrix:

1. `build_lagged_feature_dataframe` shifts each predictor by every lag
   (columns `model_lag1`, `model_lag2`, …; `include_current=True` keeps `model`
   itself as the τ = 0 term). Lags are given in **hours** and converted to sample
   steps from the index's inferred sampling interval.
2. For `feature_set="bilinear"`, `pairwise_interactions_df` appends every
   pairwise product (`model_lag1_x_model_lag2`, …).

The column names encode the lag structure, and every downstream tool
(`parse_feature_metadata`, the kernel extractors, the GP priors) reads that
structure back out — so an operator is fully described by its **feature names +
coefficients + intercept**.

## Regressors

`make_regressor(spec="vb_ard")` builds the default VB-ARD estimator, or any of:

| spec | model |
|---|---|
| `"vb_ard"` *(default)* | variational-Bayes ARD — per-feature relevance, coefficient SDs |
| `"ols"` | least squares (`sklearn.LinearRegression`) |
| `"gam"` | additive splines (`statsmodels`/`sklearn` backend) |
| `"gp_lag"` | linear lag model, GP prior couples nearby lags |
| `"gp_volterra"` | linear + bilinear weights, each with a GP prior |

`spec` may be a name or a dict of constructor kwargs, e.g.
`{"name": "gp_volterra", "optimize": True, "label": "GP-Volterra (opt)"}`.
The GP-prior models keep the regression linear in the features but make the
learned kernel smooth and directly interpretable, with uncertainty bands.

## Splitting and benchmarking

`SplitConfig` defines the temporal hold-out (a trailing `test_size_hours`, or an
explicit `test_start`, plus optional `train_sizes_hours` for learning curves).
`fit_single_case` runs one (site, model, regressor) combination and returns the
fitted regressor, the test predictions, and the raw-model baseline;
`run_benchmark` sweeps sites × models × feature sets × regressors × train sizes
and returns a tidy results table plus stored predictions.

## Inspecting the kernel

- `extract_linear_lag_weights(reg, feature_names)` → tidy `w(τ)` with 95 %
  bands; plot with `plot_lag_weight_kernel` / `plot_weight_heatmap`.
- `extract_bilinear_lag_surface(reg, feature_names)` → the `w(τ₁,τ₂)` surface;
  plot with `plot_bilinear_kernel_surface` / `plot_top_bilinear_terms`.
- `compare_lag_weight_models` / `compare_bilinear_surfaces` overlay several
  fitted cases.

In the frequency domain (`hydromend.spectral`): `linear_admittance` gives the
operator's *H(f)*, `quadratic_transfer_function` the quadratic transfer
*H₂(f₁,f₂)*, and `weighted_quadratic_transfer` the input-weighted QTF
`|H₂|·|X(f₁)||X(f₂)|` — plotted with `plot_admittance` / `plot_quadratic_transfer`.
These accept a fitted regressor or a saved `Operator`.

## Saving, shipping, applying

A fitted regressor becomes a portable `Operator`:

```python
op = hm.Operator.from_regressor(reg, feature_names=list(X.columns),
                                lags_hours=range(1, 25), feature_set="bilinear",
                                include_current=True, metadata={...})
```

`op.predict(model_series_or_frame)` rebuilds the features with the stored recipe
and applies the weights. Many operators collect into an `OperatorLibrary`, which
round-trips to a single Parquet file and supports `get(site)` and
`nearest(lat, lon)`. This is the format a released weight dataset (such as the
ERA5-GTSM tide-gauge library) ships in.

## The GESLA + GTSM path

For the common tide-gauge case the optional readers remove all boilerplate:

- `hydromend.gesla.read_record(path)` → hourly UTC gauge series + header.
- `hydromend.gtsm.nearest_station(lat, lon, glob)` and
  `hydromend.gtsm.load_station(glob, idx, start, end)` stream one station from
  the monthly GTSM NetCDFs without loading the full grid.
- `hydromend.datasets.load_pair(gesla_file, gtsm_glob)` does all of the above and
  returns one hourly-aligned `observations`/`model` frame ready for
  `build_feature_set` or `Operator.predict`.
