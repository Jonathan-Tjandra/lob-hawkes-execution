"""
Tests the Hawkes Process MLE fit using raw Coinbase trades.
"""
import pandas as pd
from src.models.hawkes import HawkesEstimator

def load_and_prep_data(filepath="data/raw/trades.csv"):
    """
    Loads trade data and extracts a sorted array of timestamps in seconds.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}. Run the fetch_trades.py script first.")
        return None

    # Convert to datetime and sort
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Convert timestamps to float seconds relative to the first trade
    # This is critical: Hawkes math requires numerical time arrays, not datetime objects
    timestamps = (df["time"] - df["time"].iloc[0]).dt.total_seconds().values
    
    return timestamps

if __name__ == "__main__":
    print("Loading trade data...")
    t = load_and_prep_data()
    
    if t is not None:
        print(f"Loaded {len(t)} trades.")
        print(f"Time horizon: {t[-1]:.2f} seconds.")
        
        # Initialize and fit the Hawkes model
        estimator = HawkesEstimator()
        success = estimator.fit(t)
        
        if success:
            print("\nNext step: We will use these parameters to generate the dynamic intensity feature for the RL state space.")