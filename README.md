# quant-ml-pairs-trading
Market neutral pairs trading strategy using cointegration testing and ML driven pair selection, with regime detection and backtesting on NIFTY bank stocks.
# Statistical Arbitrage Pairs Trading: A Cointegration and Machine Learning Approach to Market-Neutral Strategy Design

A quantitative finance project combining econometric pairs trading with machine learning-based enhancements for pair selection and regime detection — designed for BFSI research and data science applications.

## Overview

This project implements a market-neutral pairs trading strategy:
1. Identify statistically cointegrated stock pairs (Engle-Granger test)
2. Generate trading signals based on z-score deviations of the price spread
3. Backtest the strategy with realistic transaction costs
4. (In progress) Enhance pair selection using ML-based feature analysis and regime detection

## Methodology

- **Cointegration Testing:** Engle-Granger two-step method to identify pairs whose price spread is mean-reverting
- **Signal Generation:** Z-score of the spread; entry at ±2σ, exit at 0
- **Backtesting:** Vectorized backtest incorporating transaction costs
- **ML Enhancement (in progress):** Feature engineering (rolling correlation, volatility) for ML-based pair ranking and regime detection (KMeans/HMM)

## Project Structure

## Tech Stack

Python, pandas, numpy, statsmodels, scikit-learn, XGBoost, yfinance, matplotlib

## Roadmap

- [ ] Generate sample price dataset (NIFTY Bank stocks)
- [ ] Run cointegration tests and identify tradeable pairs
- [ ] Implement and backtest baseline z-score strategy
- [ ] Build ML feature pipeline for pair selection
- [ ] Train ML model for pair profitability prediction
- [ ] Implement regime detection and analyze performance across regimes
- [ ] Add EDA and results notebooks with visualizations

## Author

Shree — BS-MS Economics, IIT Roorkee
