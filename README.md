# hydromend

**Learnable lag operators for post-processing hydrodynamic model output against observations.**

`hydromend` turns a co-located pair of a **model** series (e.g. a GTSM / tide–surge
reanalysis point) and an **observed** series (e.g. a tide gauge) into a small,
interpretable *lag operator* that maps a window of recent model values onto the
observation. Operators can be

- **linear** — a lag kernel *w(τ)*, or
- **bilinear** — a second-order Volterra kernel *w(τ₁,τ₂)* that captures tide–surge
  interaction and shallow-water non-linearity.

The default estimator is a **variational-Bayes ARD** regressor (`vb_ard`): it
sparsifies irrelevant lags and returns coefficient uncertainties. Ordinary least
squares, a GAM, and Gaussian-process-prior variants are available as options.

```bash
pip install hydromend            # fit + apply
pip install "hydromend[gam]"     # + statsmodels GAM backend
```

---

## Fit your own operator — train, then test

Give `hydromend` a per-site `DataFrame` indexed by time with an `observations`
column and a `model` column. Build the feature set, hold out a period, fit on the
train part with the default VB-ARD estimator, and evaluate on the unseen test part.

```python
import hydromend as hm
import pandas as pd

# --- feature recipe: 0–24 h memory, full bilinear (Volterra) kernel -----------
recipe = dict(lags_hours=range(1, 25), feature_set="bilinear",
              include_current=True, include_squares=True)

X = hm.build_feature_set(site_df, **recipe)
d = pd.concat([X, site_df["observations"]], axis=1).dropna()

# --- temporal train/test split ------------------------------------------------
split = int(len(d) * 0.7)
Xtr, ytr = d.iloc[:split][X.columns], d.iloc[:split]["observations"]
Xte, yte = d.iloc[split:][X.columns], d.iloc[split:]["observations"]

# --- fit (VB-ARD by default) and evaluate on the held-out period --------------
reg  = hm.make_regressor().fit(Xtr, ytr)              # make_regressor("ols"|"gp_volterra"|... to switch
pred = pd.Series(reg.predict(Xte), index=Xte.index)

mae_raw = (yte - site_df["model"].reindex(Xte.index)).abs().mean()
mae_op  = (yte - pred).abs().mean()
print(f"held-out MAE: {mae_raw:.3f} -> {mae_op:.3f} m")
```

### Reuse the fitted operator anywhere

Wrap the fitted regressor in a portable `Operator` and apply it to any future
model series for that site:

```python
op = hm.Operator.from_regressor(reg, feature_names=list(X.columns), **recipe,
                                metadata={"site": "MySite", "lat": 51.5, "lon": 1.4})
op.predict(new_model_series)          # -> corrected Series
```

### See what it learned

```python
w = hm.extract_linear_lag_weights(reg, X.columns)
hm.plot_lag_weight_kernel(w)                          # linear kernel w(τ), with 95% bands

surf = hm.extract_bilinear_lag_surface(reg, X.columns)
hm.plot_bilinear_kernel_surface(surf)                 # bilinear kernel w(τ1,τ2)
```

**Worked examples** ([`examples/`](examples/)):

- [`quickstart.ipynb`](examples/quickstart.ipynb) — end-to-end on **real GESLA +
  GTSM** data (load the pair → fit → inspect → evaluate → save/reload).
- [`synthetic_volterra.ipynb`](examples/synthetic_volterra.ipynb) — **arbitrary
  models on synthetic data**: impose a known Volterra operator, recover it,
  compare linear vs bilinear, see the overtides in the spectrum, and use several
  predictors at once.
- [`1-d_estuary.ipynb`](examples/1-d_estuary.ipynb) — recreate the paper's **1-D
  shallow-water mechanism figures**: a numba estuary solver generates baseline
  vs target series, and `hydromend` learns the linear impulse response and the
  bilinear Volterra kernel *K(τ₁,τ₂)* for each mechanism.

---

## Apply a released operator (ERA5-GTSM)

Once a weight library exists (e.g. the ERA5-GTSM tide-gauge release on Zenodo),
correcting a model series needs no fitting at all:

```python
import hydromend as hm

lib = hm.OperatorLibrary.from_parquet("era5gtsm_operators.parquet")

op = lib.nearest(lat=51.5, lon=1.4)      # or lib.get("Sheerness")
corrected = op.predict(gtsm_series)       # hourly model Series -> corrected Series
```

`op.predict` rebuilds the exact lagged/bilinear features the operator was trained
on and evaluates `X · coef + intercept` (numpy + pandas only).

### The whole GESLA + GTSM compare, in a few lines

```python
from hydromend.datasets import load_pair

# reads the gauge, finds its nearest GTSM point, streams that station, aligns hourly
df, info = load_pair("gesla4/Sheerness.txt", "GTSM_ERA5_E/water_level_*/*.nc")

op = lib.get(info["site"])
df["corrected"] = op.predict(df[["model"]])
print((df["observations"] - df["corrected"]).abs().mean())
```

---

## Benchmark many sites / regressors at once (optional)

To compare feature sets or estimators across a whole network:

```python
groups = hm.load_model_groups({"gtsm": "pairs/*.nc"})           # {model: {site: df}}
results, preds = hm.run_benchmark(
    groups,
    lags_hours=range(1, 25),
    feature_sets=("linear", "bilinear"),
    regressors=("vb_ard", "ols"),        # default is ("vb_ard",); add others to compare
    split_config=hm.SplitConfig(test_size_hours=4 * 8760),
)
hm.summarise_results(results)          # mean improvement ratio by model/regressor/features
```

---

## What's in the box

| Module | Purpose |
|---|---|
| `hydromend.features` | lagged + bilinear (Volterra) feature construction |
| `hydromend.models` | regressors — `vb_ard` (default), `ols`, `gam`, `gp_lag`, `gp_volterra` |
| `hydromend.benchmark` | train/test splitting, single-case fit, multi-site benchmark |
| `hydromend.weights` | extract & plot the learned linear / bilinear kernels |
| `hydromend.data` | load observation/model pairs from NetCDF into per-site frames |
| `hydromend.plotting` | diagnostic time-series / residual / learning-curve plots |
| `hydromend.priors` | GP priors that couple neighbouring lag weights |
| `hydromend.pretrained` | `Operator` / `OperatorLibrary` — save, ship, and apply weights |
| `hydromend.gesla` | GESLA-4.1 tide-gauge reader (numpy/pandas only) |
| `hydromend.gtsm` | stream a single GTSM-ERA5 station out of the monthly NetCDFs |
| `hydromend.datasets` | `load_pair` — one gauge + its GTSM point, aligned hourly |

See [`docs/`](docs/) for the concepts guide and API reference.

## Estimators

`make_regressor()` returns the **default VB-ARD** operator; pass a name or a kwargs
dict to switch:

| name | class | what it is |
|---|---|---|
| **`vb_ard`** (default) | `VBARDRegressor` | variational-Bayes automatic relevance determination — sparsifies lags, gives coefficient uncertainties |
| `ols` | `LinearRegression` | least squares — fast baseline |
| `gam` | `GAMRegressor` | additive splines (needs `hydromend[gam]`) |
| `gp_lag` | `GPLagWeightRegressor` | linear lag model with a GP prior over `w(τ)` |
| `gp_volterra` | `GPVolterraRegressor` | linear **and** bilinear weights, each with a GP prior |

## Caveats

- An operator is only as good as the model↔observation pairing it was fit on;
  apply it at (or very near) the station it was trained for.
- Released `mae_*` fields describe *in-sample* fit quality unless stated
  otherwise — see the accompanying dataset / paper for cross-validated skill.
- `predict` returns values referenced to whatever datum the operator was trained
  against (released ERA5-GTSM operators use the most-recent gauge datum segment).

## Citation

If you use `hydromend` or the ERA5-GTSM operator library, please cite the
accompanying paper and dataset:
```
@article{monahan2026learning,
  title={Learning unresolved coastal dynamics in hydrodynamic models},
  author={Monahan, Thomas Carey and Polton, Jeff and Innocenti, Silvia and Matte, Pascal and Ayyad, Mahmoud and Saman, Krijn and Adcock, Thomas AA},
  year={2026},
  publisher={EarthArXiv}
}
```
