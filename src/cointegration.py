import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def find_cointegrated_pairs(price_df, significance=0.05, correct_multiple_testing=True):
    """
    Identify cointegrated pairs using the Engle-Granger two-step test.

    Multiple testing correction: with N pairs tested at alpha=0.05, we expect
    ~N*0.05 false positives by chance. Benjamini-Hochberg (BH) controls the
    False Discovery Rate (FDR) to avoid spurious pairs.

    Parameters
    ----------
    price_df : DataFrame of price series (columns = tickers)
    significance : float, nominal FDR level (BH) or per-test alpha (uncorrected)
    correct_multiple_testing : bool, apply BH correction (recommended)

    Returns
    -------
    List of (s1, s2, raw_pvalue, bh_adjusted_pvalue) sorted by adjusted p-value.
    If correct_multiple_testing=False, adjusted_pvalue == raw_pvalue.
    """
    tickers = price_df.columns.tolist()
    n = len(tickers)
    raw_results = []

    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = tickers[i], tickers[j]
            _, pvalue, _ = coint(price_df[s1], price_df[s2])
            raw_results.append((s1, s2, pvalue))

    if not raw_results:
        return []

    if correct_multiple_testing:
        # Benjamini-Hochberg procedure
        m = len(raw_results)
        sorted_by_p = sorted(raw_results, key=lambda x: x[2])
        adjusted = []
        for rank, (s1, s2, p) in enumerate(sorted_by_p, start=1):
            bh_p = min(p * m / rank, 1.0)
            adjusted.append((s1, s2, p, bh_p))
        # Propagate monotonicity (step-down)
        for k in range(len(adjusted) - 2, -1, -1):
            adjusted[k] = adjusted[k][:3] + (min(adjusted[k][3], adjusted[k + 1][3]),)
        pairs = [(s1, s2, p, bh) for s1, s2, p, bh in adjusted if bh < significance]
        return sorted(pairs, key=lambda x: x[3])
    else:
        pairs = [(s1, s2, p, p) for s1, s2, p in raw_results if p < significance]
        return sorted(pairs, key=lambda x: x[3])


def estimate_hedge_ratio(price_df, s1, s2):
    """
    Estimate the OLS cointegrating hedge ratio beta such that
        spread = price[s1] - beta * price[s2]
    is most stationary. Returns beta (scalar).
    """
    beta = np.polyfit(price_df[s2], price_df[s1], 1)[0]
    return beta


def compute_spread(price_df, s1, s2, beta=None):
    """
    Compute the spread for a pair, using the OLS hedge ratio if beta is None.
    Returns (spread Series, beta float).
    """
    if beta is None:
        beta = estimate_hedge_ratio(price_df, s1, s2)
    spread = price_df[s1] - beta * price_df[s2]
    return spread, beta
