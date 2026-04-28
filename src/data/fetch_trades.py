"""
Fetches high-frequency trade data from Coinbase.
"""
import requests
import pandas as pd
import argparse
import time
import os

URL = "https://api.exchange.coinbase.com/products/BTC-USD/trades"

def fetch_trades():
    response = requests.get(URL)
    if response.status_code != 200:
        print("Error:", response.text)
        return pd.DataFrame()

    data = response.json()
    if not isinstance(data, list):
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(float)
    df["time"] = pd.to_datetime(df["time"])
    df["side_num"] = df["side"].map({"buy": 1, "sell": -1})

    return df

def collect_trades(n_batches=5, sleep=0.2):
    dfs = []
    for _ in range(n_batches):
        df = fetch_trades()
        if not df.empty:
            dfs.append(df)
        time.sleep(sleep)
    
    # Drop duplicates in case of API overlap
    combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["trade_id"])
    return combined.sort_values("time").reset_index(drop=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--output", type=str, default="data/raw/trades.csv")
    args = parser.parse_args()

    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = collect_trades(n_batches=args.batches)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} unique trades to {args.output}")