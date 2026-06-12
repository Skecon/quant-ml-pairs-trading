import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import compute_pair_features
from src.signal_generator import generate_signals_for_pair
from src.backtester import backtest
from src.cointegration import compute_spread


FEATURE_COLS = ['rolling_corr', 'rolling_vol_1', 'rolling_vol_2',
                'hurst', 'ou_half_life', 'spread_zscore_abs']


def build_pair_dataset(price_df, pairs, horizon=1, train_frac=0.7):
    """
    Build a labelled ML dataset for pair profitability classification.

    IMPORTANT — time-based train/test split
    ----------------------------------------
    We split each pair's feature matrix by date (first `train_frac` of
    trading days = train, remainder = test) BEFORE returning, so the
    caller can train on train rows and evaluate on test rows without any
    future information leaking into training.

    Label: 1 if the strategy's `horizon`-step-ahead equity change > 0,
    else 0. The equity is computed on the OLS-hedged spread to be
    consistent with the signal generator.

    Returns
    -------
    train_df, test_df : DataFrames with feature columns + 'label' + 'pair'
    """
    train_frames, test_frames = [], []

    for item in pairs:
        s1, s2 = item[0], item[1]

        try:
            spread, beta = compute_spread(price_df, s1, s2)
            feats = compute_pair_features(price_df, (s1, s2))

            _, _, z, signal = generate_signals_for_pair(price_df, s1, s2, beta=beta)
            equity = backtest(z, signal, spread, prices=price_df[[s1, s2]])

            fwd_return = equity.shift(-horizon) - equity
            label = (fwd_return > 0).astype(int)

            df = feats[FEATURE_COLS].copy()
            df['label'] = label
            df['pair'] = f"{s1}_{s2}"
            df = df.dropna()

            if len(df) < 20:
                continue

            # TIME-BASED SPLIT — no shuffling
            cutoff = int(len(df) * train_frac)
            train_frames.append(df.iloc[:cutoff])
            test_frames.append(df.iloc[cutoff:])

        except Exception as e:
            print(f"  Skipping {s1}-{s2}: {e}")
            continue

    train_df = pd.concat(train_frames) if train_frames else pd.DataFrame()
    test_df = pd.concat(test_frames) if test_frames else pd.DataFrame()
    return train_df, test_df


def train_pair_classifier(train_df, cv_folds=5):
    """
    Train a RandomForestClassifier with hyperparameter search using
    TimeSeriesSplit cross-validation (respects temporal ordering).

    Returns (fitted_pipeline, hyperparams_found).
    """
    X = train_df[FEATURE_COLS]
    y = train_df['label']

    if y.nunique() < 2:
        raise ValueError("Training set has only one class — cannot train classifier.")

    # Pipeline: scale → forest (RF is scale-invariant but scaling helps GridSearch
    # interact cleanly with other estimators if swapped in later)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42, class_weight='balanced')),
    ])

    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 5, 10],
        'clf__min_samples_leaf': [1, 5],
    }

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    gs = GridSearchCV(pipe, param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1)
    gs.fit(X, y)

    print(f"  Best CV params: {gs.best_params_}")
    print(f"  Best CV AUC:    {gs.best_score_:.3f}")
    return gs.best_estimator_, gs.best_params_


def evaluate_classifier(model, test_df):
    """
    Evaluate the trained model on the held-out test set.
    Returns (report_dict, auc).
    """
    X_test = test_df[FEATURE_COLS]
    y_test = test_df['label']

    if y_test.nunique() < 2:
        print("  Warning: test set has only one class — AUC undefined.")
        return {}, np.nan

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds, output_dict=True)
    auc = roc_auc_score(y_test, proba)
    return report, auc


def rank_pairs_by_predicted_profitability(model, price_df, pairs):
    """
    Score each pair by P(profitable) using the model's latest feature row.
    Returns pairs sorted by predicted probability, descending.
    """
    scored = []
    for item in pairs:
        s1, s2 = item[0], item[1]
        try:
            feats = compute_pair_features(price_df, (s1, s2))
            if feats.empty:
                continue
            latest = feats[FEATURE_COLS].iloc[[-1]]
            prob = model.predict_proba(latest)[0, 1]
            scored.append((s1, s2, prob))
        except Exception:
            continue
    return sorted(scored, key=lambda x: x[2], reverse=True)
