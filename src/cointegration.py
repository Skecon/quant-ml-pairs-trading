import numpy as np
from statsmodels.tsa.stattools import coint

def find_cointegrated_pairs(price_df, significance=0.05):
    n = price_df.shape[1]
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            score, pvalue, _ = coint(price_df.iloc[:, i], price_df.iloc[:, j])
            if pvalue < significance:
                pairs.append((price_df.columns[i], price_df.columns[j], pvalue))
    return sorted(pairs, key=lambda x: x[2])