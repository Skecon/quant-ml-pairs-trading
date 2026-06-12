import itertools
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from src.signal_generator import generate_signals
from src.backtester import backtest
from src.cointegration import find_cointegrated_pairs
from src.regime_detection import detect_regimes, performance_by_regime
from src.ml_pair_selector import (
    build_pair_dataset,
    train_pair_classifier,
    rank_pairs_by_predicted_profitability,
)

df = pd.read_csv('../data/sample_prices.csv', index_col=0, parse_dates=True)

# --- Cointegration screening ---
coint_pairs = find_cointegrated_pairs(df)
print("Cointegrated pairs (sorted by p-value):")
for s1, s2, p in coint_pairs:
    print(f"  {s1} - {s2}: p={p:.4f}")

# All-pairs candidate set (used for ML training since coint set may be empty)
all_pairs = [
    (a, b, None) for a, b in itertools.combinations(df.columns, 2)
]

# --- ML pair selection ---
print("\nBuilding ML training dataset from all candidate pairs...")
dataset = build_pair_dataset(df, all_pairs)
print(f"Dataset size: {len(dataset)} rows, {dataset['pair'].nunique()} pairs")

model, report, auc = train_pair_classifier(dataset)
print(f"\nML pair classifier AUC: {auc:.3f}")
print(f"Test accuracy: {report['accuracy']:.3f}")

ranked = rank_pairs_by_predicted_profitability(model, df, all_pairs)
print("\nTop pairs by predicted profitability:")
for s1, s2, _, prob in ranked[:5]:
    print(f"  {s1} - {s2}: P(profitable)={prob:.3f}")

# --- Select pair for backtest: prefer cointegrated, else top ML-ranked ---
if coint_pairs:
    stock1, stock2, pval = coint_pairs[0]
    print(f"\nUsing top cointegrated pair: {stock1} - {stock2} (p={pval:.4f})")
else:
    stock1, stock2, _, prob = ranked[0]
    print(f"\nNo cointegrated pairs found. Using top ML-ranked pair: "
          f"{stock1} - {stock2} (P(profitable)={prob:.3f})")

spread = df[stock1] - df[stock2]
z, signal = generate_signals(spread)
equity = backtest(z, signal, spread, prices=df[[stock1, stock2]])

# --- Regime detection ---
regimes = detect_regimes(df)
regime_perf = performance_by_regime(equity, regimes)
print("\nPerformance by volatility regime (0=low vol ... high vol):")
print(regime_perf)

# --- Plots ---
fig, axes = plt.subplots(4, 1, figsize=(10, 13))
axes[0].plot(spread)
axes[0].set_title(f'Spread: {stock1} - {stock2}')

axes[1].plot(z)
axes[1].axhline(2, color='r', linestyle='--')
axes[1].axhline(-2, color='r', linestyle='--')
axes[1].set_title('Z-Score')

axes[2].plot(equity)
axes[2].set_title('Cumulative Strategy Returns (% of capital)')

axes[3].plot(regimes)
axes[3].set_title('Volatility Regime (0=low, higher=more volatile)')

plt.tight_layout()
plt.savefig('../results/equity_curve.png')
print("\nSaved results/equity_curve.png")
print(f"Final cumulative return: {equity.iloc[-1]*100:.2f}%")
