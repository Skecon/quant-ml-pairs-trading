import pandas as pd
import numpy as np


def backtest(z, signal, spread, prices=None, cost=0.0005):
    """
    Backtest a pairs-trading signal on a spread series.

    P&L is normalized by capital deployed (price1 + price2) when `prices`
    is provided, giving percentage returns. Raw spread P&L otherwise.

    Realistic assumptions
    ---------------------
    - 1-bar execution lag: signal is shifted by 1 (no look-ahead bias)
    - Proportional round-trip transaction cost: `cost` per unit of capital
      turned over on each side change (default 5bps, realistic for NSE)
    - No slippage model (limit of a vectorised backtest)

    Parameters
    ----------
    z       : Series, rolling z-scores (unused here, kept for API compat)
    signal  : Series {-1, 0, 1} of trading signals
    spread  : Series, price spread (s1 - beta*s2)
    prices  : DataFrame with 2 columns [price_s1, price_s2], or None
    cost    : float, one-way transaction cost as fraction of capital

    Returns
    -------
    equity : Series of cumulative returns
    """
    positions = signal.shift(1).fillna(0)
    pnl = positions * spread.diff()

    if prices is not None:
        if hasattr(prices, 'iloc') and prices.ndim == 2:
            capital = prices.iloc[:, 0] + prices.iloc[:, 1]
        else:
            capital = prices[0] + prices[1]
        returns = pnl / capital.shift(1)
        turnover_cost = abs(positions.diff().fillna(0)) * cost
    else:
        returns = pnl
        turnover_cost = abs(positions.diff().fillna(0)) * cost

    returns = returns - turnover_cost
    return returns.cumsum()


def compute_metrics(equity):
    """
    Compute a summary of strategy performance from a cumulative-return equity curve.

    Returns a dict with:
      total_return      : final cumulative return (fraction)
      annualized_return : geometric annualized return (assuming 252 trading days)
      annualized_vol    : annualized daily-return standard deviation
      sharpe_ratio      : annualized Sharpe (risk-free rate = 0)
      max_drawdown      : worst peak-to-trough drawdown (fraction)
      calmar_ratio      : annualized_return / |max_drawdown|
    """
    daily = equity.diff().dropna()
    total = equity.iloc[-1]
    n_days = len(daily)
    ann_ret = (1 + total) ** (252 / n_days) - 1 if n_days > 0 else np.nan
    ann_vol = daily.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / (roll_max.abs() + 1e-9)
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    return {
        'total_return': total,
        'annualized_return': ann_ret,
        'annualized_vol': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
    }
