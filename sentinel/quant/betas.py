"""Factor sensitivities (engine A inputs) via ridge regression on weekly data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinel import config
from sentinel.data.store import MarketData
from sentinel.data.universe import FACTORS


@dataclass
class BetaResult:
    asof: pd.Timestamp
    beta: pd.DataFrame  # tickers x factors, in original units (return per unit factor change)
    r2: pd.Series
    n_obs: pd.Series
    backfilled_share: pd.Series  # share of the estimation window built from proxy data
    alpha: pd.Series | None = None  # ridge penalty used per ticker (chosen by CV when enabled)


def weekly_changes(md: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
    px = md.prices.resample("W-FRI").last()
    fx = md.factors.resample("W-FRI").last()
    asset_ret = px.pct_change(fill_method=None)
    fac = {}
    for k, f in FACTORS.items():
        if k not in fx:
            continue
        fac[k] = np.log(fx[k] / fx[k].shift(1)) if f.transform == "logret" else fx[k].diff()
    return asset_ret.iloc[1:], pd.DataFrame(fac).iloc[1:]


def _standardize(X: np.ndarray, y: np.ndarray):
    mx, sx = X.mean(0), X.std(0)
    sx[sx == 0] = 1.0
    return (X - mx) / sx, sx, y - y.mean()


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Ridge on standardized X; returns coefficients in original units and in-sample R^2."""
    Z, sx, yc = _standardize(X, y)
    n, p = Z.shape
    coef_z = np.linalg.solve(Z.T @ Z / n + alpha * np.eye(p), Z.T @ yc / n)
    resid = yc - Z @ coef_z
    ss = float(yc @ yc)
    r2 = 1.0 - float(resid @ resid) / ss if ss > 0 else 0.0
    return coef_z / sx, r2


def select_alpha(X: np.ndarray, y: np.ndarray, alphas=None) -> float:
    """Penalty with the lowest leave-one-out squared error (same criterion as sklearn RidgeCV).

    Uses the closed-form LOO residual e_i / (1 - h_ii) from one SVD, so a full grid costs
    about as much as a single fit. The hat matrix includes the intercept (1/n term).
    """
    alphas = config.RIDGE_ALPHAS if alphas is None else alphas
    Z, _, yc = _standardize(X, y)
    n = len(yc)
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    uty = U.T @ yc
    best, best_err = float(alphas[0]), np.inf
    for a in alphas:
        shrink = s**2 / (s**2 + a * n)  # penalty alpha on Z'Z/n equals alpha*n on Z'Z
        fitted = U @ (shrink * uty)
        h = 1.0 / n + (U**2) @ shrink
        err = float(np.mean(((yc - fitted) / (1.0 - h)) ** 2))
        if err < best_err:
            best, best_err = float(a), err
    return best


def estimate_betas(md: MarketData, asof: pd.Timestamp, tickers: list[str]) -> BetaResult:
    """Estimate betas using only weeks that ended on or before ``asof``."""
    asof = pd.Timestamp(asof)
    ar, fc = weekly_changes(md.truncate(asof))
    ar, fc = ar.iloc[-config.BETA_WINDOW_WEEKS :], fc.iloc[-config.BETA_WINDOW_WEEKS :]
    rows, r2s, ns, bf, alphas = {}, {}, {}, {}, {}
    for t in tickers:
        df = pd.concat([ar[t].rename("_y"), fc], axis=1).dropna() if t in ar else pd.DataFrame()
        if len(df) < config.BETA_MIN_WEEKS:
            rows[t] = pd.Series(np.nan, index=fc.columns)
            r2s[t], ns[t], bf[t], alphas[t] = np.nan, len(df), np.nan, np.nan
            continue
        X, y = df[fc.columns].to_numpy(), df["_y"].to_numpy()
        alphas[t] = select_alpha(X, y) if config.RIDGE_ALPHA == "cv" else float(config.RIDGE_ALPHA)
        coef, r2 = _ridge(X, y, alphas[t])
        rows[t] = pd.Series(coef, index=fc.columns)
        r2s[t], ns[t] = r2, len(df)
        inc = md.inception.get(t)
        bf[t] = float((df.index < inc).mean()) if inc is not None else 0.0
    return BetaResult(
        asof=asof,
        beta=pd.DataFrame(rows).T,
        r2=pd.Series(r2s),
        n_obs=pd.Series(ns),
        backfilled_share=pd.Series(bf),
        alpha=pd.Series(alphas, dtype=float),
    )
