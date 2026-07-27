"""
Serialisable, pre-fit lag operators.

An :class:`Operator` is everything needed to reproduce a fitted model's
predictions without the training data: the predictor column(s), the lag set,
the feature set (``linear`` / ``bilinear``), the ordered feature names, the
linear coefficients and the intercept. Applying it to a new site rebuilds the
exact same features with :func:`hydromend.build_feature_set` and evaluates
``X @ coef + intercept`` — so a released weights file (e.g. the ERA5-GTSM
tide-gauge library) works on any model series with a single ``.predict`` call.

An :class:`OperatorLibrary` is a table of such operators (one row per site)
with ``get`` by name and ``nearest`` by lat/lon lookup. It round-trips to a
single Parquet file, which is the recommended distribution format.
"""
from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd

from .features import build_feature_set

# Columns of the library table that describe the model itself; everything else
# is treated as free-form site metadata (tidal_range_m, is_primary, ...).
_MODEL_COLUMNS = (
    "site", "lat", "lon", "predictor_columns", "lags_hours", "feature_set",
    "include_current", "include_squares", "feature_names", "coef", "intercept",
)


def _as_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):  # tolerate JSON-encoded cells from CSV round-trips
        return json.loads(x)
    return list(x)


def _coerce_number(v):
    """Normalise numpy scalars to plain int/float, keeping integers integral so
    lag feature names round-trip (``1`` -> ``model_lag1``, not ``model_lag1.0``)."""
    f = float(v)
    return int(f) if f.is_integer() else f


class Operator:
    """A fitted lag operator for one site.

    Parameters
    ----------
    coef, intercept
        Linear coefficients (one per feature) and scalar intercept, in the
        *raw* feature space produced by :func:`hydromend.build_feature_set`.
    feature_names
        Ordered feature names matching ``coef`` (the columns of the training
        design matrix). Used to re-order rebuilt features defensively.
    predictor_columns, lags_hours, feature_set, include_current
        The feature-construction recipe. Must match how the operator was fit.
    metadata
        Optional dict of provenance/quality fields (e.g. ``tidal_range_m``,
        ``mae_raw``, ``is_primary``). Exposed via ``.metadata`` and ``.__repr__``.
    """

    def __init__(
        self,
        coef,
        intercept,
        feature_names,
        *,
        predictor_columns=("model",),
        lags_hours,
        feature_set="bilinear",
        include_current=True,
        include_squares=False,
        metadata=None,
    ):
        self.coef = np.asarray(_as_list(coef), dtype=float)
        self.intercept = float(intercept)
        self.feature_names = [str(c) for c in _as_list(feature_names)]
        if len(self.coef) != len(self.feature_names):
            raise ValueError(
                f"coef length {len(self.coef)} != n feature_names {len(self.feature_names)}."
            )
        self.predictor_columns = tuple(_as_list(predictor_columns))
        # Preserve the exact lag values/types used at fit time: build_feature_set
        # names columns f"{col}_lag{lag}", so an int 1 -> "model_lag1" but a float
        # 1.0 -> "model_lag1.0". Coercing here would break the feature-name match.
        self.lags_hours = [_coerce_number(v) for v in _as_list(lags_hours)]
        self.feature_set = str(feature_set)
        self.include_current = bool(include_current)
        self.include_squares = bool(include_squares)
        self.metadata = dict(metadata or {})

    # -- construction --------------------------------------------------------
    @classmethod
    def from_regressor(cls, regressor, feature_names=None, **recipe):
        """Build an operator from a fitted hydromend/sklearn regressor.

        ``regressor`` must expose ``coef_`` and ``intercept_`` (OLS, VB-ARD,
        GP-lag/Volterra, sklearn linear models all do). ``recipe`` carries the
        feature-construction settings (``lags_hours=``, ``feature_set=``,
        ``predictor_columns=``, ``include_current=``, ``metadata=``).
        """
        if feature_names is None:
            feature_names = getattr(regressor, "feature_names_in_", None)
            if feature_names is None:
                raise ValueError("feature_names must be given when the regressor lacks feature_names_in_.")
        return cls(
            coef=np.asarray(regressor.coef_, dtype=float).reshape(-1),
            intercept=float(getattr(regressor, "intercept_", 0.0)),
            feature_names=feature_names,
            **recipe,
        )

    # -- application ---------------------------------------------------------
    def predict(self, data, *, predictor_column=None) -> pd.Series:
        """Post-process a model series/frame -> corrected series.

        ``data`` may be a Series of the model variable, or a DataFrame that
        contains the predictor column. Features are rebuilt with the stored
        recipe; the result is indexed by the timestamps where every lagged
        feature is available (leading/gappy rows are dropped).
        """
        df = self._as_predictor_frame(data, predictor_column)
        feats = build_feature_set(
            df,
            predictor_columns=self.predictor_columns,
            lags_hours=self.lags_hours,
            feature_set=self.feature_set,
            include_current=self.include_current,
            include_squares=self.include_squares,
        )
        missing = [c for c in self.feature_names if c not in feats.columns]
        if missing:
            raise ValueError(f"Rebuilt features are missing {len(missing)} expected columns, e.g. {missing[:3]}.")
        X = feats[self.feature_names].dropna()
        yhat = X.to_numpy() @ self.coef + self.intercept
        return pd.Series(yhat, index=X.index, name="corrected")

    def _as_predictor_frame(self, data, predictor_column) -> pd.DataFrame:
        target = self.predictor_columns[0]
        if isinstance(data, pd.Series):
            return data.rename(target).to_frame()
        if isinstance(data, pd.DataFrame):
            if predictor_column is not None:
                return data.rename(columns={predictor_column: target})
            if target in data.columns:
                return data
            if data.shape[1] == 1:
                return data.rename(columns={data.columns[0]: target})
            raise ValueError(
                f"DataFrame has no '{target}' column; pass predictor_column=... to name it."
            )
        raise TypeError("data must be a pandas Series or DataFrame.")

    # sklearn-style attributes so an Operator works anywhere a fitted regressor
    # is expected (extract_*_weights, the spectral / QTF views, ...).
    @property
    def coef_(self):
        return self.coef

    @property
    def intercept_(self):
        return self.intercept

    @property
    def feature_names_in_(self):
        return self.feature_names

    # -- serialisation -------------------------------------------------------
    def to_row(self) -> dict:
        """One flat dict row (metadata columns included) for a library table."""
        row = {
            "site": self.metadata.get("site"),
            "lat": self.metadata.get("lat"),
            "lon": self.metadata.get("lon"),
            "predictor_columns": list(self.predictor_columns),
            "lags_hours": list(self.lags_hours),
            "feature_set": self.feature_set,
            "include_current": self.include_current,
            "include_squares": self.include_squares,
            "feature_names": list(self.feature_names),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
        }
        for k, v in self.metadata.items():
            row.setdefault(k, v)
        return row

    @classmethod
    def from_row(cls, row) -> "Operator":
        row = dict(row)
        meta = {k: v for k, v in row.items() if k not in _MODEL_COLUMNS}
        for k in ("site", "lat", "lon"):
            if k in row:
                meta[k] = row[k]
        return cls(
            coef=row["coef"],
            intercept=row["intercept"],
            feature_names=row["feature_names"],
            predictor_columns=row.get("predictor_columns", ("model",)),
            lags_hours=row["lags_hours"],
            feature_set=row.get("feature_set", "bilinear"),
            include_current=row.get("include_current", True),
            include_squares=row.get("include_squares", False),
            metadata=meta,
        )

    def __repr__(self) -> str:
        m = self.metadata
        site = m.get("site", "?")
        tr = f" TR={m['tidal_range_m']:.1f}m" if "tidal_range_m" in m else ""
        return (f"<Operator {site} {self.feature_set} "
                f"n_lags={len(self.lags_hours)} n_feat={len(self.coef)}{tr}>")


class OperatorLibrary:
    """A collection of :class:`Operator` objects backed by a table.

    Load a released library with :meth:`from_parquet`, then ``get`` an operator
    by site name or find the ``nearest`` one to a coordinate. Optional filters
    (``primary_only``, ``drop_questionable``) act on the standard flag columns
    if present.
    """

    def __init__(self, table: pd.DataFrame):
        self.table = table.reset_index(drop=True)

    # -- io ------------------------------------------------------------------
    @classmethod
    def from_parquet(cls, path, *, primary_only=False, drop_questionable=False) -> "OperatorLibrary":
        df = pd.read_parquet(path)
        if primary_only and "is_primary" in df.columns:
            df = df[df["is_primary"].astype(bool)]
        if drop_questionable and "flag_questionable" in df.columns:
            df = df[~df["flag_questionable"].astype(bool)]
        return cls(df)

    @classmethod
    def from_operators(cls, operators: Iterable[Operator]) -> "OperatorLibrary":
        return cls(pd.DataFrame([op.to_row() for op in operators]))

    def to_parquet(self, path) -> None:
        self.table.to_parquet(path)

    # -- lookup --------------------------------------------------------------
    @property
    def sites(self) -> list:
        return self.table["site"].tolist()

    def get(self, key) -> Operator:
        """Look up an operator by ``site`` name, falling back to the unique
        ``file`` id when a site name is ambiguous or not present.

        Site names are not guaranteed unique across contributors, so if ``key``
        matches several rows by ``site`` you must disambiguate with the ``file``
        id (or use :meth:`nearest`).
        """
        if "site" in self.table.columns:
            site_hit = self.table[self.table["site"] == key]
            if len(site_hit) == 1:
                return Operator.from_row(site_hit.iloc[0])
        else:
            site_hit = self.table.iloc[0:0]
        if "file" in self.table.columns:
            file_hit = self.table[self.table["file"] == key]
            if len(file_hit) == 1:
                return Operator.from_row(file_hit.iloc[0])
        if len(site_hit) > 1:
            examples = site_hit["file"].tolist()[:3] if "file" in self.table.columns else []
            raise KeyError(
                f"{key!r} matches {len(site_hit)} gauges by site; pass a unique 'file' id "
                f"(e.g. {examples}) or use nearest()."
            )
        raise KeyError(key)

    def nearest(self, lat, lon, *, max_km=None) -> Operator:
        d = self._haversine_km(lat, lon, self.table["lat"].to_numpy(float),
                               self.table["lon"].to_numpy(float))
        k = int(np.nanargmin(d))
        if max_km is not None and d[k] > max_km:
            raise ValueError(f"nearest site is {d[k]:.0f} km away (> {max_km}).")
        op = Operator.from_row(self.table.iloc[k])
        op.metadata["query_dist_km"] = float(d[k])
        return op

    @staticmethod
    def _haversine_km(lat, lon, lat2, lon2):
        p = np.pi / 180.0
        a = (np.sin((lat2 - lat) * p / 2) ** 2
             + np.cos(lat * p) * np.cos(lat2 * p) * np.sin((lon2 - lon) * p / 2) ** 2)
        return 2 * 6371.0088 * np.arcsin(np.sqrt(a))

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return f"<OperatorLibrary {len(self)} operators>"
