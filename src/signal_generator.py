import numpy as np
import pandas as pd

def generate_signals(spread, entry_z=2.0, exit_z=0.0, window=30):
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
                position = -1
            elif zi < -entry_z:
                position = 1
        else:
            if position == 1 and zi >= -exit_z:
                position = 0
            elif position == -1 and zi <= exit_z:
                position = 0
        signal[i] = position

    return z, pd.Series(signal, index=z.index)
