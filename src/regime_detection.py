import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


def detect_regimes_kmeans(price_df, window=30, n_regimes=2, random_state=42):
    """
    Cluster market periods into volatility regimes using KMeans on rolling
    multi-asset volatility.

    Limitation: KMeans ignores temporal ordering — consecutive periods may
    receive different regime labels without a regime "transition". Prefer
    detect_regimes_hmm() when hmmlearn is available.

    Regime labels are ordered by ascending average volatility (0 = calmest).

    Returns
    -------
    Series of regime labels aligned to price_df.index (NaNs in warm-up period).
    """
    returns = price_df.pct_change()
    vol = returns.rolling(window).std()
    features = vol.dropna()

    km = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)
    raw_labels = km.fit_predict(features)

    avg_vol_per_cluster = (
        pd.DataFrame(features.values, index=features.index)
        .groupby(raw_labels)
        .mean()
        .mean(axis=1)
        .sort_values()
    )
    relabel_map = {old: new for new, old in enumerate(avg_vol_per_cluster.index)}
    return pd.Series(raw_labels, index=features.index).map(relabel_map)


def detect_regimes_hmm(price_df, window=30, n_regimes=2, random_state=42):
    """
    Fit a Gaussian Hidden Markov Model on rolling volatility features.

    Unlike KMeans, HMM respects temporal ordering and models regime
    persistence (transition probabilities), producing smoother, more
    economically meaningful regime sequences.

    Regime labels are ordered by ascending average volatility (0 = calmest).

    Falls back to KMeans if hmmlearn is not installed.

    Returns
    -------
    Series of regime labels aligned to price_df.index.
    """
    if not HMM_AVAILABLE:
        print("hmmlearn not available — falling back to KMeans regime detection.")
        return detect_regimes_kmeans(price_df, window, n_regimes, random_state)

    returns = price_df.pct_change()
    vol = returns.rolling(window).std()
    features = vol.dropna()

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
    )
    model.fit(features.values)
    raw_labels = model.predict(features.values)

    # Order regimes by ascending mean volatility
    avg_vol = (
        pd.DataFrame(features.values)
        .groupby(raw_labels)
        .mean()
        .mean(axis=1)
        .sort_values()
    )
    relabel_map = {old: new for new, old in enumerate(avg_vol.index)}
    relabeled = pd.Series(raw_labels, index=features.index).map(relabel_map)
    return relabeled


def detect_regimes(price_df, window=30, n_regimes=2, method="hmm", random_state=42):
    """
    Detect market regimes. method='hmm' (default) or 'kmeans'.
    """
    if method == "hmm":
        return detect_regimes_hmm(price_df, window, n_regimes, random_state)
    return detect_regimes_kmeans(price_df, window, n_regimes, random_state)


def performance_by_regime(equity, regimes):
    """
    Compute per-regime return statistics from a cumulative-return equity curve.

    Returns DataFrame indexed by regime label with columns:
      n_periods, mean_daily_return, total_return, annualized_sharpe
    """
    common_idx = equity.index.intersection(regimes.index)
    eq = equity.loc[common_idx]
    reg = regimes.loc[common_idx]
    daily_returns = eq.diff().fillna(0)

    stats = {}
    for r in sorted(reg.unique()):
        mask = reg == r
        dr = daily_returns[mask]
        stats[r] = {
            'n_periods': int(mask.sum()),
            'mean_daily_return': dr.mean(),
            'total_return': dr.sum(),
            'annualized_sharpe': (dr.mean() / dr.std() * np.sqrt(252))
                                  if dr.std() > 0 else np.nan,
        }
    return pd.DataFrame(stats).T
