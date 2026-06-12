def backtest(z, signal, spread, prices=None, cost=0.0005):
    """
    Backtest a pairs-trading signal.

    If `prices` (tuple/list of the two price series, or DataFrame with 2 cols)
    is provided, returns are normalized by capital deployed (price1 + price2),
    giving percentage returns. Otherwise falls back to raw spread P&L.
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
