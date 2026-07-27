"""Regressors: OLS, variational-Bayes ARD, GAM, and GP-prior lag/Volterra models."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import optimize, special
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import LinearRegression

try:
    from statsmodels.gam.api import BSplines, GLMGam
    from statsmodels.genmod.families import Gaussian
    _HAS_STATSMODELS_GAM = True
except Exception:
    BSplines = None
    GLMGam = None
    Gaussian = None
    _HAS_STATSMODELS_GAM = False

try:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer
    _HAS_SKLEARN_SPLINE = True
except Exception:
    make_pipeline = None
    SplineTransformer = None
    _HAS_SKLEARN_SPLINE = False

from .priors import build_lag_weight_prior, build_volterra_weight_prior, parse_feature_metadata
from .weights import extract_linear_lag_weights, extract_bilinear_lag_surface


def _meanvar(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(array, dtype=float)
    mean = np.mean(arr, axis=0)
    if arr.ndim == 1 or 1 in arr.shape:
        var = np.var(arr)
        if np.isscalar(var):
            var = np.array([1.0 if var == 0 else float(var)])
    else:
        var = np.diag(np.cov(arr, rowvar=False)).copy()
        var[var == 0] = 1.0
    return mean, var



def _normalise(array: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mean, var = _meanvar(ref)
    return (array - mean) / np.sqrt(var)



def _unnormalise(array: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mean, var = _meanvar(ref)
    return array * np.sqrt(var) + mean



def _logdet(matrix: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.min(eigenvalues) <= 0:
        raise np.linalg.LinAlgError(f"Matrix is not positive definite. Min eigenvalue={np.min(eigenvalues):.6e}")
    chol = np.linalg.cholesky(matrix)
    return float(2.0 * np.sum(np.log(np.diag(chol.T))))



def _safe_std(array: np.ndarray) -> np.ndarray:
    std = np.std(array, axis=0, ddof=0)
    std = np.asarray(std, dtype=float)
    std[~np.isfinite(std)] = 1.0
    std[std == 0] = 1.0
    return std



def _regressor_name(spec) -> str:
    if isinstance(spec, dict):
        return str(spec.get("name", "custom")).lower().strip()
    return str(spec).lower().strip()


def _regressor_label(spec) -> str:
    if isinstance(spec, dict):
        return str(spec.get("label", spec.get("name", "custom")))
    return str(spec)


def _default_min_train_samples(spec, n_features: int) -> int:
    name = _regressor_name(spec)
    if name == "ols":
        return max(2, int(n_features) + 1)
    return 2
def bayes_linear_fit_ard(X: np.ndarray, y: np.ndarray):
    X = np.matrix(X)
    y = np.matrix(y)
    a0 = 1e-2
    b0 = 1e-4
    c0 = 1e-2
    d0 = 1e-4

    N, D = np.shape(X)
    X_corr = X.T * X
    Xy_corr = X.T * y
    an = a0 + N / 2.0
    gammaln_an = special.gammaln(an)
    cn = c0 + 0.5
    D_gammaln_cn = D * special.gammaln(cn)

    lower_bound_last = -np.inf
    max_iter = 500
    E_a = np.matrix(np.ones(D) * c0 / d0).T

    for _ in range(max_iter):
        invV = np.matrix(np.diag(np.array(E_a)[:, 0])) + X_corr
        V = np.matrix(np.linalg.inv(invV))
        logdetV = -_logdet(invV)
        w = np.dot(V, Xy_corr)[:, 0]

        sse = np.sum(np.power(X * w - y, 2), axis=0)
        sse = np.real(sse)[0]
        bn = b0 + 0.5 * (sse + np.sum((np.array(w)[:, 0] ** 2) * np.array(E_a)[:, 0], axis=0))
        E_t = an / bn

        dn = d0 + 0.5 * (E_t * (np.array(w)[:, 0] ** 2) + np.diag(V))
        E_a = np.matrix(cn / dn).T

        lower_bound = (
            -0.5 * (E_t * sse + np.sum(np.multiply(X, X * V)))
            + 0.5 * logdetV
            - b0 * E_t
            + gammaln_an
            - an * np.log(bn)
            + an
            + D_gammaln_cn
            - cn * np.sum(np.log(dn))
        )

        if lower_bound < lower_bound_last:
            raise RuntimeError(
                f"Variational bound decreased from {float(lower_bound_last):.6f} to {float(lower_bound):.6f}."
            )
        if abs(lower_bound_last - lower_bound) < abs(1e-5 * lower_bound):
            break
        lower_bound_last = lower_bound
    else:
        warnings.warn("VB ARD reached the maximum number of iterations.")

    lower_bound = (
        lower_bound
        - 0.5 * (N * np.log(2 * np.pi) - D)
        - special.gammaln(a0)
        + a0 * np.log(b0)
        + D * (-special.gammaln(c0) + c0 * np.log(d0))
    )
    return w, V, invV, logdetV, an, bn, E_a, lower_bound


class VBARDRegressor:
    def __init__(self, *, bias: bool = True, normalize: bool = True):
        self.bias = bias
        self.normalize = normalize

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)

        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else [f"x{i}" for i in range(X_arr.shape[1])]
        self.x_ref_ = X_arr.copy()
        self.y_ref_ = y_arr.copy()

        X_train = _normalise(X_arr, self.x_ref_) if self.normalize else X_arr
        y_train = _normalise(y_arr, self.y_ref_) if self.normalize else y_arr

        if self.bias:
            X_train = np.concatenate([X_train, np.ones((X_train.shape[0], 1))], axis=1)

        self.w_, self.V_, self.invV_, self.logdetV_, self.an_, self.bn_, self.E_a_, self.lower_bound_ = bayes_linear_fit_ard(
            X_train,
            y_train,
        )

        weights_scaled = np.asarray(self.w_).reshape(-1)
        feature_weights_scaled = weights_scaled[:-1] if self.bias else weights_scaled
        bias_scaled = float(weights_scaled[-1]) if self.bias else 0.0

        x_mean = np.mean(self.x_ref_, axis=0)
        x_std = _safe_std(self.x_ref_)
        y_mean = float(np.mean(self.y_ref_))
        y_std = float(_safe_std(self.y_ref_.reshape(-1, 1))[0])

        coef_scale = y_std / x_std
        self.coef_ = feature_weights_scaled * coef_scale
        self.intercept_ = y_mean + y_std * bias_scaled - float(np.dot(x_mean, self.coef_))

        V_arr = np.asarray(self.V_, dtype=float)
        if V_arr.ndim == 2 and V_arr.shape[0] >= len(self.coef_):
            coef_cov_scaled = V_arr[: len(self.coef_), : len(self.coef_)]
            scale_outer = np.outer(coef_scale, coef_scale)
            self.coef_cov_ = coef_cov_scaled * scale_outer
            self.coef_sd_ = np.sqrt(np.clip(np.diag(self.coef_cov_), 0.0, None))
        else:
            self.coef_cov_ = None
            self.coef_sd_ = np.full(len(self.coef_), np.nan)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.asarray(X_arr @ self.coef_ + self.intercept_).reshape(-1)


class GAMRegressor:
    """Additive spline model with a statsmodels backend when available."""

    def __init__(self, *, spline_df: int = 6, degree: int = 3):
        self.spline_df = spline_df
        self.degree = degree

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        X_df = pd.DataFrame(X).copy()
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        X_df.columns = [str(c) for c in X_df.columns]
        self.feature_names_in_ = X_df.columns.tolist()
        self.train_min_ = X_df.min()
        self.train_max_ = X_df.max()

        if _HAS_STATSMODELS_GAM:
            dfs = []
            for col in self.feature_names_in_:
                unique_count = int(X_df[col].nunique(dropna=True))
                dfs.append(max(self.degree + 1, min(self.spline_df, max(self.degree + 1, unique_count - 1))))
            try:
                self.smoother_ = BSplines(X_df[self.feature_names_in_], df=dfs, degree=[self.degree] * len(dfs))
                self.model_ = GLMGam(
                    y_arr,
                    exog=np.ones((len(X_df), 1)),
                    smoother=self.smoother_,
                    family=Gaussian(),
                )
                self.result_ = self.model_.fit()
                self.backend_ = "statsmodels"
                return self
            except Exception as exc:
                warnings.warn(f"statsmodels GAM fit failed ({exc}). Falling back to spline linear model.")

        if not _HAS_SKLEARN_SPLINE:
            raise ImportError("Neither statsmodels GAM nor sklearn SplineTransformer is available.")

        n_knots = max(self.degree + 1, self.spline_df)
        self.pipeline_ = make_pipeline(
            SplineTransformer(n_knots=n_knots, degree=self.degree, include_bias=False),
            LinearRegression(),
        )
        self.pipeline_.fit(X_df[self.feature_names_in_], y_arr)
        self.backend_ = "sklearn_spline"
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(X).copy()
        X_df.columns = [str(c) for c in X_df.columns]
        X_df = X_df[self.feature_names_in_]

        if getattr(self, "backend_", None) == "statsmodels":
            clipped = X_df[self.feature_names_in_].clip(lower=self.train_min_, upper=self.train_max_, axis=1)
            return np.asarray(
                self.result_.predict(
                    exog=np.ones((len(clipped), 1)),
                    exog_smooth=clipped,
                )
            ).reshape(-1)
        return np.asarray(self.pipeline_.predict(X_df[self.feature_names_in_])).reshape(-1)
def _posterior_for_gp_weight_model(X: np.ndarray, y: np.ndarray, K: np.ndarray, noise_variance: float):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    K = np.asarray(K, dtype=float)
    noise_variance = float(max(noise_variance, 1e-10))

    XtX = X.T @ X
    Xty = X.T @ y

    chol_K = cho_factor(K, lower=True, check_finite=False)
    K_inv = cho_solve(chol_K, np.eye(K.shape[0]), check_finite=False)

    precision = K_inv + XtX / noise_variance
    chol_precision = cho_factor(precision, lower=True, check_finite=False)
    posterior_cov = cho_solve(chol_precision, np.eye(precision.shape[0]), check_finite=False)
    posterior_mean = posterior_cov @ (Xty / noise_variance)

    logdet_K = 2.0 * np.sum(np.log(np.diag(chol_K[0])))
    logdet_precision = 2.0 * np.sum(np.log(np.diag(chol_precision[0])))
    quad = (y @ y) / noise_variance - (Xty @ posterior_mean) / noise_variance ** 2
    n_samples = X.shape[0]
    logdet_y = n_samples * np.log(noise_variance) + logdet_K + logdet_precision
    negative_log_marginal_likelihood = 0.5 * (quad + logdet_y + n_samples * np.log(2.0 * np.pi))
    return posterior_mean, posterior_cov, negative_log_marginal_likelihood


class GPLagWeightRegressor:
    """
    Bayesian linear lag model with a GP prior over lag coefficients.

    The regression remains linear in the lagged predictors, but the prior couples
    nearby lags so the learned kernel is smooth and directly interpretable.
    """

    def __init__(
        self,
        *,
        kernel: str = "matern32",
        optimize: bool = False,
        lengthscale_hours: float | None = None,
        prior_variance: float = 1.0,
        noise_variance: float = 0.25,
        iid_variance: float | None = None,
        rq_alpha: float = 1.0,
        period_hours: float | None = None,
        jitter: float = 1e-6,
        maxiter: int = 200,
    ):
        self.kernel = kernel
        self.optimize = optimize
        self.lengthscale_hours = lengthscale_hours
        self.prior_variance = prior_variance
        self.noise_variance = noise_variance
        self.iid_variance = iid_variance
        self.rq_alpha = rq_alpha
        self.period_hours = period_hours
        self.jitter = jitter
        self.maxiter = maxiter

    def _initial_lengthscale(self, metadata: pd.DataFrame) -> float:
        lag_values = metadata.loc[metadata["is_lag_like"], "lag_hours"].dropna().to_numpy(dtype=float)
        if lag_values.size <= 1:
            return 1.0
        unique_lags = np.unique(np.sort(lag_values))
        if unique_lags.size <= 1:
            return max(abs(unique_lags[0]), 1.0)
        diffs = np.diff(unique_lags)
        median_gap = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
        span = float(unique_lags.max() - unique_lags.min())
        return max(median_gap, span / 4.0, 1.0)

    def _build_prior(self, feature_names, lengthscale_hours: float, prior_variance: float):
        K, metadata = build_lag_weight_prior(
            feature_names,
            kernel=self.kernel,
            lengthscale_hours=lengthscale_hours,
            prior_variance=prior_variance,
            iid_variance=self.iid_variance,
            rq_alpha=self.rq_alpha,
            period_hours=self.period_hours,
            jitter=self.jitter,
        )
        return K, metadata

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        X_df = pd.DataFrame(X).copy()
        X_df.columns = [str(c) for c in X_df.columns]
        y_arr = np.asarray(y, dtype=float).reshape(-1)

        self.feature_names_in_ = X_df.columns.tolist()
        self.metadata_ = parse_feature_metadata(self.feature_names_in_)

        X_arr = X_df.to_numpy(dtype=float)
        self.x_mean_ = np.mean(X_arr, axis=0)
        self.x_std_ = _safe_std(X_arr)
        self.y_mean_ = float(np.mean(y_arr))
        self.y_std_ = float(_safe_std(y_arr.reshape(-1, 1))[0])

        X_scaled = (X_arr - self.x_mean_) / self.x_std_
        y_scaled = (y_arr - self.y_mean_) / self.y_std_

        init_lengthscale = self.lengthscale_hours or self._initial_lengthscale(self.metadata_)
        init_prior_variance = float(max(self.prior_variance, 1e-8))
        init_noise_variance = float(max(self.noise_variance, 1e-8))

        if self.optimize:
            lag_values = self.metadata_.loc[self.metadata_["is_lag_like"], "lag_hours"].dropna().to_numpy(dtype=float)
            lag_span = float(np.ptp(lag_values)) if lag_values.size else 1.0
            lower_ls = np.log(max(min(abs(init_lengthscale), lag_span + 1.0) * 0.1, 1e-3))
            upper_ls = np.log(max(lag_span * 5.0, init_lengthscale * 5.0, 1.0))
            x0 = np.log([init_lengthscale, init_prior_variance, init_noise_variance])

            def objective(theta: np.ndarray) -> float:
                lengthscale = float(np.exp(theta[0]))
                prior_variance = float(np.exp(theta[1]))
                noise_variance = float(np.exp(theta[2]))
                K, _ = self._build_prior(self.feature_names_in_, lengthscale, prior_variance)
                try:
                    _, _, nll = _posterior_for_gp_weight_model(X_scaled, y_scaled, K, noise_variance)
                except np.linalg.LinAlgError:
                    return np.inf
                return float(nll)

            result = optimize.minimize(
                objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=[(lower_ls, upper_ls), (np.log(1e-6), np.log(1e3)), (np.log(1e-6), np.log(1e2))],
                options={"maxiter": int(self.maxiter)},
            )
            theta_opt = result.x if result.success else x0
            self.optimization_result_ = result
            self.lengthscale_hours_ = float(np.exp(theta_opt[0]))
            self.prior_variance_ = float(np.exp(theta_opt[1]))
            self.noise_variance_ = float(np.exp(theta_opt[2]))
        else:
            self.optimization_result_ = None
            self.lengthscale_hours_ = init_lengthscale
            self.prior_variance_ = init_prior_variance
            self.noise_variance_ = init_noise_variance

        self.K_prior_scaled_, self.metadata_ = self._build_prior(
            self.feature_names_in_, self.lengthscale_hours_, self.prior_variance_
        )
        self.posterior_mean_scaled_, self.posterior_cov_scaled_, self.negative_log_marginal_likelihood_ = _posterior_for_gp_weight_model(
            X_scaled,
            y_scaled,
            self.K_prior_scaled_,
            self.noise_variance_,
        )

        coef_scale = self.y_std_ / self.x_std_
        self.coef_ = self.posterior_mean_scaled_ * coef_scale
        self.coef_cov_ = self.posterior_cov_scaled_ * np.outer(coef_scale, coef_scale)
        self.coef_sd_ = np.sqrt(np.clip(np.diag(self.coef_cov_), 0.0, None))
        self.intercept_ = self.y_mean_ - float(np.dot(self.x_mean_, self.coef_))
        self.noise_std_ = float(np.sqrt(self.noise_variance_) * self.y_std_)
        self.fit_summary_ = {
            "kernel": self.kernel,
            "lengthscale_hours": self.lengthscale_hours_,
            "prior_variance": self.prior_variance_,
            "noise_variance_scaled": self.noise_variance_,
            "noise_std_original_units": self.noise_std_,
            "negative_log_marginal_likelihood": self.negative_log_marginal_likelihood_,
        }
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(X).copy()
        X_df.columns = [str(c) for c in X_df.columns]
        X_arr = X_df[self.feature_names_in_].to_numpy(dtype=float)
        return np.asarray(X_arr @ self.coef_ + self.intercept_).reshape(-1)

    def weight_summary(self) -> pd.DataFrame:
        return extract_linear_lag_weights(self, self.feature_names_in_)




class GPVolterraRegressor:
    """Bayesian linear lag model with GP priors on linear and bilinear weights."""

    def __init__(
        self,
        *,
        optimize: bool = False,
        linear_kernel: str = "matern32",
        linear_lengthscale_hours: float | None = None,
        linear_prior_variance: float = 1.0,
        linear_iid_variance: float | None = None,
        bilinear_kernel: str = "matern32",
        bilinear_lengthscale_hours: float | None = None,
        bilinear_prior_variance: float = 0.5,
        bilinear_iid_variance: float | None = None,
        noise_variance: float = 0.25,
        linear_rq_alpha: float = 1.0,
        bilinear_rq_alpha: float = 1.0,
        linear_period_hours: float | None = None,
        bilinear_period_hours: float | None = None,
        symmetric_same_base: bool = True,
        jitter: float = 1e-6,
        maxiter: int = 250,
    ):
        self.optimize = optimize
        self.linear_kernel = linear_kernel
        self.linear_lengthscale_hours = linear_lengthscale_hours
        self.linear_prior_variance = linear_prior_variance
        self.linear_iid_variance = linear_iid_variance
        self.bilinear_kernel = bilinear_kernel
        self.bilinear_lengthscale_hours = bilinear_lengthscale_hours
        self.bilinear_prior_variance = bilinear_prior_variance
        self.bilinear_iid_variance = bilinear_iid_variance
        self.noise_variance = noise_variance
        self.linear_rq_alpha = linear_rq_alpha
        self.bilinear_rq_alpha = bilinear_rq_alpha
        self.linear_period_hours = linear_period_hours
        self.bilinear_period_hours = bilinear_period_hours
        self.symmetric_same_base = symmetric_same_base
        self.jitter = jitter
        self.maxiter = maxiter

    def _initial_lengthscale(self, metadata: pd.DataFrame, lag_columns: tuple[str, ...]) -> float:
        arrays = []
        for col in lag_columns:
            if col in metadata:
                values = metadata[col].dropna().to_numpy(dtype=float)
                if values.size:
                    arrays.append(values)
        if not arrays:
            return 1.0
        unique_lags = np.unique(np.sort(np.concatenate(arrays)))
        if unique_lags.size <= 1:
            return max(abs(unique_lags[0]) if unique_lags.size else 1.0, 1.0)
        diffs = np.diff(unique_lags)
        median_gap = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
        span = float(unique_lags.max() - unique_lags.min())
        return max(median_gap, span / 4.0, 1.0)

    def _build_prior(self, feature_names, linear_ls: float, linear_var: float, bilinear_ls: float, bilinear_var: float):
        return build_volterra_weight_prior(
            feature_names,
            linear_kernel=self.linear_kernel,
            linear_lengthscale_hours=linear_ls,
            linear_prior_variance=linear_var,
            linear_iid_variance=self.linear_iid_variance,
            bilinear_kernel=self.bilinear_kernel,
            bilinear_lengthscale_hours=bilinear_ls,
            bilinear_prior_variance=bilinear_var,
            bilinear_iid_variance=self.bilinear_iid_variance,
            linear_rq_alpha=self.linear_rq_alpha,
            bilinear_rq_alpha=self.bilinear_rq_alpha,
            linear_period_hours=self.linear_period_hours,
            bilinear_period_hours=self.bilinear_period_hours,
            symmetric_same_base=self.symmetric_same_base,
            jitter=self.jitter,
        )

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        X_df = pd.DataFrame(X).copy()
        X_df.columns = [str(c) for c in X_df.columns]
        y_arr = np.asarray(y, dtype=float).reshape(-1)

        self.feature_names_in_ = X_df.columns.tolist()
        self.metadata_ = parse_feature_metadata(self.feature_names_in_)

        X_arr = X_df.to_numpy(dtype=float)
        self.x_mean_ = np.mean(X_arr, axis=0)
        self.x_std_ = _safe_std(X_arr)
        self.y_mean_ = float(np.mean(y_arr))
        self.y_std_ = float(_safe_std(y_arr.reshape(-1, 1))[0])

        X_scaled = (X_arr - self.x_mean_) / self.x_std_
        y_scaled = (y_arr - self.y_mean_) / self.y_std_

        init_linear_ls = self.linear_lengthscale_hours or self._initial_lengthscale(self.metadata_, ("lag_hours",))
        init_bilinear_ls = self.bilinear_lengthscale_hours or self._initial_lengthscale(self.metadata_, ("lag1_hours", "lag2_hours"))
        init_linear_var = float(max(self.linear_prior_variance, 1e-8))
        init_bilinear_var = float(max(self.bilinear_prior_variance, 1e-8))
        init_noise_variance = float(max(self.noise_variance, 1e-8))

        if self.optimize:
            linear_lags = self.metadata_.loc[self.metadata_["is_lag_like"], "lag_hours"].dropna().to_numpy(dtype=float)
            bilinear_lags = self.metadata_.loc[self.metadata_["feature_type"].eq("interaction"), ["lag1_hours", "lag2_hours"]].to_numpy(dtype=float).reshape(-1)
            bilinear_lags = bilinear_lags[np.isfinite(bilinear_lags)]
            linear_span = float(np.ptp(linear_lags)) if linear_lags.size else 1.0
            bilinear_span = float(np.ptp(bilinear_lags)) if bilinear_lags.size else max(linear_span, 1.0)
            x0 = np.log([init_linear_ls, init_linear_var, init_bilinear_ls, init_bilinear_var, init_noise_variance])

            def objective(theta: np.ndarray) -> float:
                linear_ls, linear_var, bilinear_ls, bilinear_var, noise_variance = np.exp(theta)
                K, _ = self._build_prior(self.feature_names_in_, float(linear_ls), float(linear_var), float(bilinear_ls), float(bilinear_var))
                try:
                    _, _, nll = _posterior_for_gp_weight_model(X_scaled, y_scaled, K, float(noise_variance))
                except np.linalg.LinAlgError:
                    return np.inf
                return float(nll)

            result = optimize.minimize(
                objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=[
                    (np.log(max(min(init_linear_ls, linear_span + 1.0) * 0.1, 1e-3)), np.log(max(linear_span * 5.0, init_linear_ls * 5.0, 1.0))),
                    (np.log(1e-6), np.log(1e3)),
                    (np.log(max(min(init_bilinear_ls, bilinear_span + 1.0) * 0.1, 1e-3)), np.log(max(bilinear_span * 5.0, init_bilinear_ls * 5.0, 1.0))),
                    (np.log(1e-6), np.log(1e3)),
                    (np.log(1e-6), np.log(1e2)),
                ],
                options={"maxiter": int(self.maxiter)},
            )
            theta_opt = result.x if result.success else x0
            self.optimization_result_ = result
            self.linear_lengthscale_hours_, self.linear_prior_variance_, self.bilinear_lengthscale_hours_, self.bilinear_prior_variance_, self.noise_variance_ = [float(np.exp(v)) for v in theta_opt]
        else:
            self.optimization_result_ = None
            self.linear_lengthscale_hours_ = init_linear_ls
            self.linear_prior_variance_ = init_linear_var
            self.bilinear_lengthscale_hours_ = init_bilinear_ls
            self.bilinear_prior_variance_ = init_bilinear_var
            self.noise_variance_ = init_noise_variance

        self.K_prior_scaled_, self.metadata_ = self._build_prior(
            self.feature_names_in_,
            self.linear_lengthscale_hours_,
            self.linear_prior_variance_,
            self.bilinear_lengthscale_hours_,
            self.bilinear_prior_variance_,
        )
        self.posterior_mean_scaled_, self.posterior_cov_scaled_, self.negative_log_marginal_likelihood_ = _posterior_for_gp_weight_model(
            X_scaled,
            y_scaled,
            self.K_prior_scaled_,
            self.noise_variance_,
        )

        coef_scale = self.y_std_ / self.x_std_
        self.coef_ = self.posterior_mean_scaled_ * coef_scale
        self.coef_cov_ = self.posterior_cov_scaled_ * np.outer(coef_scale, coef_scale)
        self.coef_sd_ = np.sqrt(np.clip(np.diag(self.coef_cov_), 0.0, None))
        self.intercept_ = self.y_mean_ - float(np.dot(self.x_mean_, self.coef_))
        self.noise_std_ = float(np.sqrt(self.noise_variance_) * self.y_std_)
        self.fit_summary_ = {
            "linear_kernel": self.linear_kernel,
            "linear_lengthscale_hours": self.linear_lengthscale_hours_,
            "linear_prior_variance": self.linear_prior_variance_,
            "bilinear_kernel": self.bilinear_kernel,
            "bilinear_lengthscale_hours": self.bilinear_lengthscale_hours_,
            "bilinear_prior_variance": self.bilinear_prior_variance_,
            "noise_variance_scaled": self.noise_variance_,
            "noise_std_original_units": self.noise_std_,
            "negative_log_marginal_likelihood": self.negative_log_marginal_likelihood_,
        }
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(X).copy()
        X_df.columns = [str(c) for c in X_df.columns]
        X_arr = X_df[self.feature_names_in_].to_numpy(dtype=float)
        return np.asarray(X_arr @ self.coef_ + self.intercept_).reshape(-1)

    def linear_weight_summary(self) -> pd.DataFrame:
        return extract_linear_lag_weights(self, self.feature_names_in_)

    def bilinear_weight_summary(self) -> pd.DataFrame:
        return extract_bilinear_lag_surface(self, self.feature_names_in_)

def make_regressor(spec="vb_ard"):
    """Build a regressor from a name or dict spec.

    Defaults to ``"vb_ard"`` — the variational-Bayes ARD estimator, hydromend's
    recommended operator: it sparsifies irrelevant lags and returns coefficient
    uncertainties. Other options (``"ols"``, ``"gam"``, ``"gp_lag"``,
    ``"gp_volterra"``) are available by passing their name or a kwargs dict.
    """
    params = dict(spec) if isinstance(spec, dict) else {}
    params.pop("label", None)
    name = _regressor_name(params.pop("name", spec))
    if name == "ols":
        return LinearRegression(**params)
    if name in {"vb_ard", "vb ard", "vb-ard"}:
        return VBARDRegressor(**params)
    if name == "gam":
        return GAMRegressor(**params)
    if name in {"gp_lag", "gp lag", "gp-lag", "gp_weight", "gp-weight"}:
        return GPLagWeightRegressor(**params)
    if name in {"gp_volterra", "gp volterra", "gp-volterra", "gp_bilinear", "gp-bilinear", "gp_second_order", "gp second order"}:
        return GPVolterraRegressor(**params)
    raise ValueError(f"Unknown regressor {spec!r}. Use 'ols', 'vb_ard', 'gam', 'gp_lag', or 'gp_volterra'.")
