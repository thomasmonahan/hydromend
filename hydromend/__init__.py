"""
hydromend — learnable lag operators for hydrodynamic model post-processing.

`hydromend` turns a co-located pair of a hydrodynamic-model series (e.g. a GTSM /
tide-surge reanalysis point) and an observed series (e.g. a tide gauge) into a
small, interpretable *lag operator* that maps a window of recent model values to
the observation. Operators can be **linear** (a lag kernel) or **bilinear**
(a second-order Volterra kernel that captures tide-surge interaction), and are
fit with ordinary least squares, variational-Bayes ARD, a GAM, or GP-prior
regressors that couple neighbouring lags for smooth, physically readable kernels.

Two entry points:

* **Fit your own** from any NetCDF pair collection —
  :func:`load_location_dict_from_netcdf` / :func:`load_model_groups`,
  :func:`build_feature_set`, :func:`make_regressor`, and :func:`run_benchmark`.
* **Apply a released operator** — :class:`Operator` / :class:`OperatorLibrary`
  load pre-fit weights (e.g. the ERA5-GTSM tide-gauge release on Zenodo) and
  correct a new model series with one call.

The optional :mod:`hydromend.gesla` and :mod:`hydromend.gtsm` readers make the
GESLA-4.1 + GTSM-ERA5-E workflow a two-liner (see :func:`load_pair`).
"""
from __future__ import annotations

__version__ = "0.1.0"

# --- feature engineering ----------------------------------------------------
from .features import (
    infer_sampling_hours,
    hours_to_samples,
    build_lagged_feature_dataframe,
    pairwise_interactions_df,
    build_feature_set,
)

# --- data loading -----------------------------------------------------------
from .data import (
    load_location_dict_from_netcdf,
    load_model_groups,
    available_locations,
)

# --- regressors -------------------------------------------------------------
from .models import (
    make_regressor,
    VBARDRegressor,
    GAMRegressor,
    GPLagWeightRegressor,
    GPVolterraRegressor,
    bayes_linear_fit_ard,
)

# --- priors -----------------------------------------------------------------
from .priors import (
    parse_feature_metadata,
    build_lag_weight_prior,
    build_bilinear_weight_prior,
    build_volterra_weight_prior,
)

# --- benchmarking -----------------------------------------------------------
from .benchmark import (
    SplitConfig,
    fit_single_case,
    run_benchmark,
    summarise_results,
)

# --- weight extraction & kernel plots ---------------------------------------
from .weights import (
    extract_linear_lag_weights,
    extract_bilinear_lag_surface,
    plot_lag_weight_kernel,
    plot_weight_heatmap,
    plot_bilinear_kernel_surface,
    plot_top_bilinear_terms,
    compare_lag_weight_models,
    compare_bilinear_surfaces,
)

# --- diagnostic plots -------------------------------------------------------
from .plotting import (
    plot_raw_timeseries,
    plot_residual_kde,
    plot_learning_curve,
    plot_location_bars,
    plot_prediction_example,
)

# --- frequency-domain views -------------------------------------------------
from .spectral import (
    linear_admittance,
    quadratic_transfer_function,
    weighted_quadratic_transfer,
    fir_frequency_response,
    volterra_qtf,
    amp_spectrum,
    plot_admittance,
    plot_quadratic_transfer,
)

# --- pretrained operators ---------------------------------------------------
from .pretrained import Operator, OperatorLibrary

__all__ = [
    "__version__",
    # features
    "infer_sampling_hours", "hours_to_samples", "build_lagged_feature_dataframe",
    "pairwise_interactions_df", "build_feature_set",
    # data
    "load_location_dict_from_netcdf", "load_model_groups", "available_locations",
    # models
    "make_regressor", "VBARDRegressor", "GAMRegressor", "GPLagWeightRegressor",
    "GPVolterraRegressor", "bayes_linear_fit_ard",
    # priors
    "parse_feature_metadata", "build_lag_weight_prior",
    "build_bilinear_weight_prior", "build_volterra_weight_prior",
    # benchmark
    "SplitConfig", "fit_single_case", "run_benchmark", "summarise_results",
    # weights
    "extract_linear_lag_weights", "extract_bilinear_lag_surface",
    "plot_lag_weight_kernel", "plot_weight_heatmap", "plot_bilinear_kernel_surface",
    "plot_top_bilinear_terms", "compare_lag_weight_models", "compare_bilinear_surfaces",
    # plotting
    "plot_raw_timeseries", "plot_residual_kde", "plot_learning_curve",
    "plot_location_bars", "plot_prediction_example",
    # spectral / QTF
    "linear_admittance", "quadratic_transfer_function", "weighted_quadratic_transfer",
    "fir_frequency_response", "volterra_qtf", "amp_spectrum",
    "plot_admittance", "plot_quadratic_transfer",
    # pretrained
    "Operator", "OperatorLibrary",
]
