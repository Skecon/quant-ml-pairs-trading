import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from src.feature_engineering import compute_pair_features
from src.signal_generator import generate_signals
from src.backtester import backtest


def build_pair_dataset(price_df, pairs, horizon=1):
    """
    For each (s1, s2, pval) pair, compute features and a forward-looking
    profitability label: 1 if the strategy's return over the next
    `horizon` periods (from the backtest on that pair's spread) is
    positive, else 0.

    Returns a single concatenated DataFrame of features + label,
    with a 'pair' column identifying which pair each row belongs to.
    """
    frames = []
    for s1, s2, _ in pairs:
        spread = price_df[s1] - price_df[s2]
        feats = compute_pair_features(price_df, (s1, s2))

        z, signal = generate_signals(spread)
        equity = backtest(z, signal, spread, prices=price_df[[s1, s2]])

        # Forward strategy return over `horizon` periods
        fwd_return = equity.shift(-horizon) - equity
        label = (fwd_return > 0).astype(int)

        df = feats.copy()
        df['label'] = label
        df['pair'] = f"{s1}_{s2}"
        df = df.dropna()
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


def train_pair_classifier(dataset, test_size=0.3, random_state=42):
    """
    Train a RandomForestClassifier to predict pair-period profitability
    from rolling correlation/volatility features.

    Returns (model, report_dict, auc).
    """
    feature_cols = ['rolling_corr', 'rolling_vol_1', 'rolling_vol_2']
    X = dataset[feature_cols]
    y = dataset['label']

    if y.nunique() < 2:
        raise ValueError("Dataset has only one class; cannot train classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, preds, output_dict=True)
    auc = roc_auc_score(y_test, proba)

    return model, report, auc


def rank_pairs_by_predicted_profitability(model, price_df, pairs):
    """
    For each pair, compute the latest feature row and use the trained
    model to predict probability of profitability. Returns pairs sorted
    by predicted probability, descending.
    """
    feature_cols = ['rolling_corr', 'rolling_vol_1', 'rolling_vol_2']
    scored = []
    for s1, s2, pval in pairs:
        feats = compute_pair_features(price_df, (s1, s2))
        if feats.empty:
            continue
        latest = feats[feature_cols].iloc[[-1]]
        prob = model.predict_proba(latest)[0, 1]
        scored.append((s1, s2, pval, prob))

    return sorted(scored, key=lambda x: x[3], reverse=True)
