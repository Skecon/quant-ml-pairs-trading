import yfinance as yf
import pandas as pd

def load_prices(tickers, start="2022-01-01", end="2024-01-01"):
    data = yf.download(tickers, start=start, end=end)['Close']
    return data.dropna()

if __name__ == "__main__":
    tickers = ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS", 
           "SBIN.NS", "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS"]
    df = load_prices(tickers)
    df.to_csv("data/sample_prices.csv")
    print("Saved data/sample_prices.csv")