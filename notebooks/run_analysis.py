"""
run_analysis.py — End-to-end pairs trading pipeline (improved).

Improvements over baseline:
  1. Benjamini-Hochberg multiple-testing correction on cointegration tests
  2. OLS hedge ratio (not 1:1 spread)
  3. Richer ML features: Hurst exponent, OU half-life, spread z-score
  4. Time-based train/test split for the ML classifier (no leakage)
  5. TimeSeriesSplit CV + GridSearchCV for hyperparameter selection
  6. HMM regime detection (temporally coherent, falls back to KMeans)
  7. Extended backtest metrics: Sharpe, Calmar, max drawdown
"""

import itertools
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '..')

from src.signal_generator import generate_signals_for_pair
from src.backtester import backtest, compute_metrics
from src.cointegration import find_cointegrated_pairs
from src.regime_detection import detect_regimes, performance_by_regime
from src.ml_pair_selector import (
    build_pair_dataset,
    train_pair_classifier,
    evaluate_classifier,
    rank_pairs_by_predicted_profitability,
)

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv('../data/sample_prices.csv', index_col=0, parse_dates=True)
print(f"Loaded {len(df)} trading days × {len(df.columns)} stocks")
print(f"Stocks: {', '.join(df.columns)}\n")

# ── 1. Cointegration screening with BH correction ────────────────────────────
print("=" * 60)
print("1. COINTEGRATION SCREENING (Benjamini-Hochberg corrected)")
print("=" * 60)
coint_pairs = find_cointegrated_pairs(df, significance=0.05, correct_multiple_testing=True)

if coint_pairs:
    print(f"Found {len(coint_pairs)} cointegrated pairs after BH correction:")
    for s1, s2, raw_p, adj_p in coint_pairs:
        print(f"  {s1:20s} - {s2:20s}  raw_p={raw_p:.4f}  adj_p={adj_p:.4f}")
else:
    print("No pairs survived BH correction at α=0.05.")

# All candidate pairs for ML
all_pairs = [(a, b, None) for a, b in itertools.combinations(df.columns, 2)]
print(f"\nTotal candidate pairs: {len(all_pairs)}")

# ── 2. ML pair selection (time-based split, no leakage) ──────────────────────
print("\n" + "=" * 60)
print("2. ML PAIR SELECTION (time-based 70/30 train-test split)")
print("=" * 60)
print("Building dataset... (this may take a minute)")

train_df, test_df = build_pair_dataset(df, all_pairs, horizon=1, train_frac=0.70)
print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

if not train_df.empty and train_df['label'].nunique() > 1:
    print("\nTraining classifier (GridSearchCV + TimeSeriesSplit CV)...")
    model, best_params = train_pair_classifier(train_df)

    report, auc = evaluate_classifier(model, test_df)
    print(f"\n  Out-of-sample AUC:      {auc:.3f}")
    if report:
        print(f"  Out-of-sample accuracy: {report['accuracy']:.3f}")

    ranked = rank_pairs_by_predicted_profitability(model, df, all_pairs)
    print("\nTop pairs by predicted profitability:")
    for s1, s2, prob in ranked[:5]:
        print(f"  {s1:20s} - {s2:20s}  P(profitable)={prob:.3f}")
else:
    print("Insufficient data for ML classifier — skipping.")
    ranked = [(a, b, 0.0) for a, b, _ in all_pairs]

# ── 3. Select pair for backtest ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. PAIR SELECTION & BACKTEST")
print("=" * 60)

if coint_pairs:
    s1, s2, raw_p, adj_p = coint_pairs[0]
    print(f"Using top cointegrated pair: {s1} - {s2}  (adj_p={adj_p:.4f})")
elif ranked:
    s1, s2, prob = ranked[0]
    print(f"No cointegrated pairs. Using top ML pair: {s1} - {s2}  (P={prob:.3f})")
else:
    cols = df.columns.tolist()
    s1, s2 = cols[0], cols[1]
    print(f"Falling back to first pair: {s1} - {s2}")

spread, beta, z, signal = generate_signals_for_pair(df, s1, s2)
print(f"Hedge ratio (OLS): β = {beta:.4f}  →  spread = {s1} − {beta:.4f}×{s2}")

equity = backtest(z, signal, spread, prices=df[[s1, s2]])
metrics = compute_metrics(equity)

print("\nBacktest performance:")
for k, v in metrics.items():
    print(f"  {k:25s}: {v*100 if 'return' in k or 'drawdown' in k else v:.3f}"
          + ("%" if 'return' in k or 'drawdown' in k else ""))

# ── 4. Regime detection (HMM) ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. REGIME DETECTION (HMM)")
print("=" * 60)

regimes = detect_regimes(df, method="hmm", n_regimes=2)
regime_perf = performance_by_regime(equity, regimes)
print("Performance by volatility regime (0 = low vol):")
print(regime_perf.to_string())

# ── 5. Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(12, 16))

axes[0].plot(df[s1], label=s1, alpha=0.8)
axes[0].plot(df[s2] * beta, label=f"{s2} × β={beta:.3f}", alpha=0.8)
axes[0].legend(fontsize=8)
axes[0].set_title(f'Price series: {s1} vs β·{s2} (OLS-scaled)')

axes[1].plot(spread, color='steelblue')
axes[1].axhline(spread.mean(), color='k', linestyle='--', linewidth=0.8, label='mean')
axes[1].set_title(f'OLS-Hedged Spread (β={beta:.4f})')
axes[1].legend(fontsize=8)

axes[2].plot(z, color='darkorange')
axes[2].axhline(2, color='r', linestyle='--', linewidth=0.8)
axes[2].axhline(-2, color='r', linestyle='--', linewidth=0.8)
axes[2].axhline(0, color='k', linestyle='--', linewidth=0.5)
axes[2].set_title('Rolling Z-Score of Spread')

axes[3].plot(equity * 100, color='green')
axes[3].axhline(0, color='k', linestyle='--', linewidth=0.5)
axes[3].set_title(
    f'Cumulative Returns (%)  |  Sharpe={metrics["sharpe_ratio"]:.2f}  '
    f'|  MaxDD={metrics["max_drawdown"]*100:.1f}%'
)
axes[3].set_ylabel('%')

colors = {0: 'skyblue', 1: 'salmon'}
regime_colors = regimes.map(colors)
axes[4].scatter(regimes.index, regimes.values, c=regime_colors.values, s=4, alpha=0.7)
axes[4].set_yticks([0, 1])
axes[4].set_yticklabels(['Low vol', 'High vol'])
axes[4].set_title('HMM Volatility Regime (0=low, 1=high)')

plt.tight_layout()
plt.savefig('../results/equity_curve.png', dpi=150)
print("\nSaved results/equity_curve.png")
print(f"Final cumulative return: {equity.iloc[-1]*100:.2f}%")
