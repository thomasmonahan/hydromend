"""
Frequency-domain views of a learned lag operator.

A linear lag kernel *w(τ)* has a **frequency response** (admittance) *H(f)*; a
bilinear Volterra kernel *K(τ₁,τ₂)* has a **quadratic transfer function**
*H₂(f₁,f₂)* — the 2-D DFT of the kernel — which says how a pair of input
frequencies *(f₁,f₂)* is mapped to their sum/difference. Weighting it by the
input spectrum,

    W(f₁,f₂) = |H₂(f₁,f₂)| · |X(f₁)| · |X(f₂)|,

gives the **input-weighted QTF**: the interactions that are actually *active*
given the forcing, as plotted in the manuscript.

These operate on any fitted operator that exposes ``coef_`` + feature names
(hydromend regressors, sklearn linear models, or a :class:`hydromend.Operator`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .weights import extract_linear_lag_weights, extract_bilinear_lag_surface

# reference tidal / overtide frequencies, cycles per hour
TIDAL_LINES_CPH = {"M2": 1 / 12.42, "S2": 1 / 12.00, "M4": 2 / 12.42, "M6": 3 / 12.42}


# -----------------------------------------------------------------------------
# Primitive spectral transforms (kept identical to the manuscript definitions)
# -----------------------------------------------------------------------------
def amp_spectrum(x, dt_hours):
    """One-sided amplitude spectrum of ``x``; returns ``(freqs_cph, amplitude)``."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    spec = np.abs(np.fft.rfft(x - np.nanmean(x))) * 2.0 / n
    freq = np.fft.rfftfreq(n, d=dt_hours)
    return freq, spec


def fir_frequency_response(weights, dt_hours, n_freq=512):
    """Complex frequency response of a FIR lag filter ``w`` → ``(freqs_cph, H)``.

    ``H(f) = Σ_k w_k exp(-2πi f k Δt)`` with ``k`` the lag index and ``Δt`` the
    lag spacing in hours.
    """
    weights = np.asarray(weights, dtype=float)
    n = len(weights)
    freqs = np.fft.rfftfreq(max(n_freq, n), d=dt_hours)
    k = np.arange(n)
    H = weights @ np.exp(-2j * np.pi * np.outer(k, freqs) * dt_hours)
    return freqs, H


def volterra_qtf(K, dt_hours, n_freq=128):
    """Quadratic transfer function ``H₂(f₁,f₂)`` = 2-D DFT of Volterra kernel ``K``.

    Returns ``(freqs_cph, H2)`` where ``H2[i, j]`` corresponds to
    ``(freqs[i], freqs[j])``.
    """
    K = np.nan_to_num(np.asarray(K, dtype=float))
    n_lag = K.shape[0]
    freqs = np.fft.rfftfreq(max(n_freq, n_lag), d=dt_hours)
    lags = np.arange(n_lag)
    E = np.exp(-2j * np.pi * np.outer(freqs, lags) * dt_hours)   # (nf, n_lag)
    # H2[fi, fj] = sum_{i,j} K[i,j] E[fi,i] E[fj,j] = E @ K @ E^T
    H2 = E @ K @ E.T
    return freqs, H2


# -----------------------------------------------------------------------------
# Operator -> kernels (reuse the package's own extraction)
# -----------------------------------------------------------------------------
def _lag_axis(reg, feature_names):
    lw = extract_linear_lag_weights(reg, feature_names)
    lags = np.sort(lw.loc[lw["is_lag_like"], "lag_hours"].dropna().unique())
    dt = float(np.min(np.diff(lags))) if len(lags) > 1 else 1.0
    return lags, dt


def _linear_weight_vector(reg, feature_names):
    lw = extract_linear_lag_weights(reg, feature_names)
    lw = lw[lw["is_lag_like"]].sort_values("lag_hours")
    return lw["weight_mean"].to_numpy(dtype=float)


def _kernel_matrix(reg, feature_names, lags):
    surf = extract_bilinear_lag_surface(reg, feature_names)
    pos = {round(float(l), 6): k for k, l in enumerate(lags)}
    K = np.zeros((len(lags), len(lags)))
    for _, r in surf.iterrows():
        i = pos.get(round(float(r["lag1_hours"]), 6))
        j = pos.get(round(float(r["lag2_hours"]), 6))
        if i is not None and j is not None:
            K[i, j] = K[j, i] = float(r["weight_mean"])
    return K


# -----------------------------------------------------------------------------
# Public: operator -> frequency-domain objects
# -----------------------------------------------------------------------------
def linear_admittance(reg, feature_names=None, *, dt_hours=None, n_freq=512):
    """Linear admittance ``H(f)`` of a fitted operator → ``(freqs_cph, H)``."""
    lags, dt = _lag_axis(reg, feature_names)
    dt = dt_hours or dt
    return fir_frequency_response(_linear_weight_vector(reg, feature_names), dt, n_freq)


def quadratic_transfer_function(reg, feature_names=None, *, dt_hours=None, n_freq=128):
    """Quadratic transfer function ``H₂(f₁,f₂)`` → ``(freqs_cph, H2)`` (complex)."""
    lags, dt = _lag_axis(reg, feature_names)
    dt = dt_hours or dt
    K = _kernel_matrix(reg, feature_names, lags)
    return volterra_qtf(K, dt, n_freq)


def weighted_quadratic_transfer(reg, input_series, feature_names=None, *,
                                dt_hours=None, n_freq=128):
    """Input-weighted QTF ``W = |H₂(f₁,f₂)|·|X(f₁)||X(f₂)|`` → ``(freqs_cph, W)``.

    ``input_series`` is the model/baseline series the operator acts on (array or
    pandas Series); its amplitude spectrum ``X`` weights the transfer function so
    only interactions the forcing actually excites light up.
    """
    lags, dt = _lag_axis(reg, feature_names)
    dt = dt_hours or dt
    freqs, H2 = quadratic_transfer_function(reg, feature_names, dt_hours=dt, n_freq=n_freq)
    x = np.asarray(getattr(input_series, "values", input_series), dtype=float)
    fx, Xamp = amp_spectrum(x, dt)
    Xi = np.interp(freqs, fx, Xamp, left=0.0, right=0.0)
    return freqs, np.abs(H2) * np.outer(Xi, Xi)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_admittance(reg=None, feature_names=None, *, freqs=None, H=None, ax=None,
                    dt_hours=None, n_freq=512, fmax=0.35, tidal_lines=True,
                    label=None, title=None):
    """Plot |H(f)| of the linear operator. Pass a fitted ``reg`` or precomputed
    ``freqs, H``."""
    import matplotlib.pyplot as plt
    if freqs is None or H is None:
        freqs, H = linear_admittance(reg, feature_names, dt_hours=dt_hours, n_freq=n_freq)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freqs, np.abs(H), lw=2.0, label=label)
    if tidal_lines:
        _add_tidal_lines(ax, fmax, axis="x")
    ax.set_xlim(0, fmax)
    ax.set_xlabel("frequency (cph)"); ax.set_ylabel(r"$|H(f)|$")
    if title:
        ax.set_title(title)
    if label:
        ax.legend(fontsize=8)
    return ax


def plot_quadratic_transfer(freqs, M, *, ax=None, log=True, fmax=0.35,
                            tidal_lines=True, cmap="inferno", cbar=True,
                            cbar_label=r"$|H_2(f_1,f_2)|$", title=None):
    """Heatmap of a (weighted or bare) quadratic transfer function.

    ``M`` is the matrix returned by :func:`quadratic_transfer_function` (use
    ``np.abs``) or :func:`weighted_quadratic_transfer`. Frequencies in cph.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    M = np.asarray(M, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    norm = None
    if log:
        pos = M[M > 0]
        vmax = float(pos.max()) if pos.size else 1.0
        norm = LogNorm(vmin=vmax * 1e-3, vmax=vmax)
        M = np.where(M > 0, M, np.nan)
    im = ax.pcolormesh(freqs, freqs, M.T, cmap=cmap, shading="auto", norm=norm,
                       vmin=None if log else 0.0)
    ax.set_xlim(0, fmax); ax.set_ylim(0, fmax)
    ax.set_aspect("equal", adjustable="box")
    if tidal_lines:
        _add_tidal_lines(ax, fmax, axis="both")
    ax.set_xlabel(r"$f_1$ (cph)"); ax.set_ylabel(r"$f_2$ (cph)")
    if title:
        ax.set_title(title)
    if cbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    return ax


def _add_tidal_lines(ax, fmax, axis="both"):
    for name, fr in TIDAL_LINES_CPH.items():
        if fr > fmax:
            continue
        if axis in ("x", "both"):
            ax.axvline(fr, color="0.7", ls=":", lw=0.6)
        if axis in ("y", "both"):
            ax.axhline(fr, color="0.7", ls=":", lw=0.6)
