# hydromend — API reference

Everything below is importable from the top-level `hydromend` namespace (the IO
readers live in their submodules). Signatures show the arguments that matter;
see the docstrings (`help(hm.<name>)`) for the full set.

## Features — `hydromend.features`

- `build_feature_set(data, *, predictor_columns=("model",), lags_hours=(1,2,3), feature_set="linear", include_current=False)` → design-matrix `DataFrame`.
- `build_lagged_feature_dataframe(df, columns, lags_hours, *, include_current=False, sampling_hours=None)`.
- `pairwise_interactions_df(df, cols=None, prefix_sep="_x_")` — append all pairwise products.
- `infer_sampling_hours(index)`, `hours_to_samples(hours, sampling_hours)`.

## Data — `hydromend.data`

- `load_location_dict_from_netcdf(files, *, obs_var="obs", model_var="model", qc_var="qc_flags", ...)` → `{site: DataFrame}` with `observations`/`model` columns.
- `load_model_groups({name: files, ...}, **kwargs)` → `{model_name: {site: DataFrame}}`.
- `available_locations(model_groups)` → sorted site list.

## Regressors — `hydromend.models`

- `make_regressor(spec="vb_ard")` — default `"vb_ard"`; `spec` is `"vb_ard" | "ols" | "gam" | "gp_lag" | "gp_volterra"` or a dict of kwargs (with optional `name`, `label`).
- `VBARDRegressor(*, bias=True, normalize=True)`.
- `GAMRegressor(*, spline_df=6, degree=3)` — needs `hydromend[gam]`.
- `GPLagWeightRegressor(*, kernel="matern32", optimize=False, lengthscale_hours=None, prior_variance=1.0, noise_variance=0.25, ...)`.
- `GPVolterraRegressor(*, optimize=False, linear_kernel="matern32", bilinear_kernel="matern32", bilinear_prior_variance=0.5, noise_variance=0.25, ...)`.

All expose `.fit(X, y)`, `.predict(X)`, `.coef_`, `.intercept_`; the Bayesian
ones also expose `.coef_sd_`.

## Priors — `hydromend.priors`

- `parse_feature_metadata(feature_names)` → per-feature table (base, lag(s), interaction group).
- `build_lag_weight_prior(feature_names, *, kernel="matern32", lengthscale_hours=6.0, prior_variance=1.0, ...)`.
- `build_bilinear_weight_prior(...)`, `build_volterra_weight_prior(...)`.

## Benchmarking — `hydromend.benchmark`

- `SplitConfig(test_size_hours=4*8760, test_start=None, train_sizes_hours=None, train_selection="first")`.
- `fit_single_case(model_groups, *, location, model_name, lags_hours, feature_set, regressor, split_config=None, ...)` → dict with the fitted regressor, `y_pred`, `baseline_test`, …
- `run_benchmark(model_groups, *, lags_hours, feature_sets, regressors, split_config=None, ...)` → `(results_df, predictions)`.
- `summarise_results(results_df, metric="Improvement_Ratio")`.

## Kernel extraction & plots — `hydromend.weights`

- `extract_linear_lag_weights(regressor, feature_names=None)` → tidy `w(τ)` with `weight_mean/sd/lo/hi`.
- `extract_bilinear_lag_surface(regressor, feature_names=None, *, base_pair=None)` → `w(τ₁,τ₂)` rows.
- `plot_lag_weight_kernel`, `plot_weight_heatmap`, `plot_bilinear_kernel_surface`, `plot_top_bilinear_terms`, `compare_lag_weight_models`, `compare_bilinear_surfaces`.

## Frequency-domain views — `hydromend.spectral`

All accept a fitted regressor or an `Operator` (feature names inferred), and infer
the lag spacing from the features (override with `dt_hours=`).

- `linear_admittance(reg, feature_names=None, *, dt_hours=None, n_freq=512)` → `(freqs_cph, H)` — complex linear frequency response *H(f)*.
- `quadratic_transfer_function(reg, feature_names=None, *, dt_hours=None, n_freq=128)` → `(freqs_cph, H2)` — complex quadratic transfer *H₂(f₁,f₂)* (2-D DFT of the Volterra kernel).
- `weighted_quadratic_transfer(reg, input_series, feature_names=None, *, dt_hours=None, n_freq=128)` → `(freqs_cph, W)` — input-weighted QTF `W = |H₂(f₁,f₂)|·|X(f₁)||X(f₂)|`.
- `plot_admittance(reg=..., | freqs=, H=, ax=None, fmax=0.35, tidal_lines=True)` and `plot_quadratic_transfer(freqs, M, *, ax=None, log=True, fmax=0.35, tidal_lines=True, cbar_label=...)`.
- Primitives: `fir_frequency_response(weights, dt_hours, n_freq)`, `volterra_qtf(K, dt_hours, n_freq)`, `amp_spectrum(x, dt_hours)`.

## Diagnostic plots — `hydromend.plotting`

- `plot_raw_timeseries`, `plot_residual_kde`, `plot_learning_curve`, `plot_location_bars`, `plot_prediction_example`.

## Pretrained operators — `hydromend.pretrained`

**`Operator`**
- `Operator(coef, intercept, feature_names, *, predictor_columns=("model",), lags_hours, feature_set="bilinear", include_current=True, metadata=None)`.
- `Operator.from_regressor(regressor, feature_names=None, **recipe)` — build from a fitted regressor.
- `op.predict(data, *, predictor_column=None)` → corrected `Series`.
- `op.to_row()` / `Operator.from_row(row)` — dict (de)serialisation.
- `op.metadata` — provenance/quality dict.

**`OperatorLibrary`**
- `OperatorLibrary.from_parquet(path, *, primary_only=False, drop_questionable=False)`.
- `OperatorLibrary.from_operators(iterable)`, `lib.to_parquet(path)`.
- `lib.get(site)`, `lib.nearest(lat, lon, *, max_km=None)`, `lib.sites`, `len(lib)`.

## IO readers

**`hydromend.gesla`** — `read_header(path)`, `read_record(path, *, use_flag_only=True, to_hourly=True)`, `yearly_coverage(series)`.

**`hydromend.gtsm`** — `station_coordinates(glob)`, `nearest_station(lat, lon, glob, *, coords=None)`, `load_station(glob, station_index, *, start=None, end=None)`.

**`hydromend.datasets`** — `load_pair(gesla_file, gtsm_glob, *, station_index=None, coords=None, demean=False, pad_days=3)` → `(df, info)`.
