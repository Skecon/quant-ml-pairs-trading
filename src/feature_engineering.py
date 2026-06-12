import pandas as pd

def compute_pair_features(price_df, pair):
    s1, s2 = pair
    returns1 = price_df[s1].pct_change()
    returns2 = price_df[s2].pct_change()
    
    features = pd.DataFrame({
        'rolling_corr': returns1.rolling(30).corr(returns2),
        'rolling_vol_1': returns1.rolling(30).std(),
        'rolling_vol_2': returns2.rolling(30).std(),
    })
    return features.dropna()