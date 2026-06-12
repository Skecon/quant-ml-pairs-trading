import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def compute_hurst_exponent(series, max_lag=20):
    """
    Estimate the Hurst exponent via rescaled range (R/S) analysis.
    H < 0.5  → mean-reverting (good for pairs trading)
    H = 0.5  → random walk
    H > 0.5  → trending
    Returns NaN if series is too short or has no variance.
    """
    series = series.dropna()
    if len(series) < max_lag * 2:
        return np.nan
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        chunks = [series.values[i:i + lag] for i in range(0, len(series) - lag, lag)]
        if len(chunks) < 2:
            continue
        rs_vals = []
        for chunk in chunks:
            if np.std(chunk) == 0:
                continue
            mean_adj = chunk - np.mean(chunk)
            cumsum = np.cumsum(mean_adj)
            rs = (cumsum.max() - cumsum.min()) / np.std(chunk)
            rs_vals.append(rs)
        if rs_vals:
            tau.append(np.mean(rs_vals))
    if len(tau) < 2:
        return np.nan
    lags_used = list(range(2, 2 + len(tau)))
    poly = np.polyfit(np.log(lags_used), np.log(tau), 1)
    return poly[0]


def compute_ou_half_life(spread):
    """
    Fit an Ornstein-Uhlenbeck process to the spread and return the
    half-life of mean reversion in the same time units as the spread index.

    Regresses: Δspread_t = α + β * spread_{t-1} + ε_t
    Half-life = -ln(2) / β  (valid only when β < 0)

    Returns NaN if β ≥ 0 (non-mean-reverting).
    """
    spread = spread.dropna()
    delta = spread.diff().dropna()
    lagged = spread.shift(1).dropna()
    common = delta.index.intersection(lagged.index)
    delta, lagged = delta.loc[common], lagged.loc[common]

    X = add_constant(lagged)
    model = OLS(delta, X).fit()
    beta = model.params.iloc[1]

    if beta >= 0:
        return np.nan
    return -np.log(2) / beta


def compute_pair_features(price_df, pair, window=30):
    """
    Compute a rich feature set for an (s1, s2) pair:
      - rolling_corr        : 30-day rolling return correlation
      - rolling_vol_1/2     : 30-day rolling return volatility per leg
      - hurst               : Hurst exponent of the spread (rolling window)
      - ou_half_life        : OU half-life of the spread (rolling window)
      - spread_zscore_abs   : |z-score| of the spread (how far from mean)
    """
    s1, s2 = pair
    returns1 = price_df[s1].pct_change()
    returns2 = price_df[s2].pct_change()

    # OLS hedge ratio for spread
    beta = np.polyfit(price_df[s2], price_df[s1], 1)[0]
    spread = price_df[s1] - beta * price_df[s2]

    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    zscore = (spread - rolling_mean) / rolling_std

    # Rolling Hurst and half-life — computed on expanding windows for speed
    hurst_vals = []
    half_life_vals = []
    for i in range(len(spread)):
        if i < window * 2:
            hurst_vals.append(np.nan)
            half_life_vals.append(np.nan)
        else:
            window_spread = spread.iloc[max(0, i - window * 2):i]
            hurst_vals.append(compute_hurst_exponent(window_spread))
            half_life_vals.append(compute_ou_half_life(window_spread))

    features = pd.DataFrame({
        'rolling_corr': returns1.rolling(window).corr(returns2),
        'rolling_vol_1': returns1.rolling(window).std(),
        'rolling_vol_2': returns2.rolling(window).std(),
        'hurst': pd.Series(hurst_vals, index=spread.index),
        'ou_half_life': pd.Series(half_life_vals, index=spread.index),
        'spread_zscore_abs': zscore.abs(),
    })
    return features.dropna()
