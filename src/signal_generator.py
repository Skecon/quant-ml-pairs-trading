import numpy as np
import pandas as pd
from src.cointegration import compute_spread


def generate_signals(spread, entry_z=2.0, exit_z=0.0, window=30):
    """
    Generate trading signals from a spread series using a rolling z-score.

    Entry: open position when |z| > entry_z
    Exit:  close position when |z| crosses back through exit_z (toward zero)

    Uses position = +1 (long spread) or -1 (short spread).

    Returns
    -------
    z : Series of rolling z-scores
    signal : Series of position sizes {-1, 0, 1}
    """
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    z = (spread - mean) / std

    signal = np.zeros(len(z))
    position = 0
    for i in range(len(z)):
        zi = z.iloc[i]
        if np.isnan(zi):
            signal[i] = 0
            continue
        if position == 0:
            if zi > entry_z:
                position = -1   # spread too high → short
            elif zi < -entry_z:
                position = 1    # spread too low → long
        elif position == 1:
            if zi >= exit_z:    # mean reversion complete (from below)
                position = 0
        elif position == -1:
            if zi <= exit_z:    # mean reversion complete (from above)
                position = 0
        signal[i] = position

    return z, pd.Series(signal, index=z.index)


def generate_signals_for_pair(price_df, s1, s2, beta=None, **kwargs):
    """
    Convenience wrapper: computes the OLS-hedged spread first, then generates signals.

    Returns (spread, beta, z, signal).
    """
    spread, beta = compute_spread(price_df, s1, s2, beta)
    z, signal = generate_signals(spread, **kwargs)
    return spread, beta, z, signal
