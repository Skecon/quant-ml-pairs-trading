import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def detect_regimes(price_df, window=30, n_regimes=2, random_state=42):
    """
    Cluster periods into volatility regimes using rolling volatility
    of each asset's returns as features.

    Returns a Series of regime labels (0..n_regimes-1) aligned to
    price_df.index, with NaNs (warm-up period) dropped. Labels are
    relabeled so that 0 = lowest average volatility, increasing
    with regime index.
    """
    returns = price_df.pct_change()
    vol = returns.rolling(window).std()
    features = vol.dropna()

    km = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)
    raw_labels = km.fit_predict(features)

    # Order regimes by average overall volatility (mean across assets), ascending
    avg_vol_per_cluster = (
        pd.DataFrame(features.values, index=features.index)
        .groupby(raw_labels)
        .mean()
        .mean(axis=1)
        .sort_values()
    )
    relabel_map = {old: new for new, old in enumerate(avg_vol_per_cluster.index)}
    relabeled = pd.Series(raw_labels, index=features.index).map(relabel_map)

    return relabeled


def performance_by_regime(equity, regimes):
    """
    Given a cumulative-return equity series and aligned regime labels,
    compute per-regime return statistics (mean daily return, total
    return contribution, number of periods).
    """
    common_idx = equity.index.intersection(regimes.index)
    eq = equity.loc[common_idx]
    reg = regimes.loc[common_idx]

    daily_returns = eq.diff().fillna(0)

    stats = {}
    for r in sorted(reg.unique()):
        mask = reg == r
        stats[r] = {
            'n_periods': int(mask.sum()),
            'mean_daily_return': daily_returns[mask].mean(),
            'total_return': daily_returns[mask].sum(),
        }
    return pd.DataFrame(stats).T
