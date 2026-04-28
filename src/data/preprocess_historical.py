from pathlib import Path
import pandas as pd
import numpy as np

# 1. FIND PROJECT ROOT (lob-hawkes-execution)
# This goes up 2 levels from src/data/ to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def preprocess_data():
    # Use the PROJECT_ROOT to create ABSOLUTE paths
    raw_path = PROJECT_ROOT / "data" / "raw" / "trades.csv"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_path = processed_dir / "historical_intensity.csv"

    if not raw_path.exists():
        print(f"Error: {raw_path} not found.")
        print(f"Make sure you are in the project folder and have run fetch_trades.py")
        return

    print(f"Loading raw trades from: {raw_path}")
    df = pd.read_csv(raw_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    # 1. Create 1-second buckets
    df = df.set_index("time")
    arrivals = df.resample("1S").size().reset_index(name="actual_arrivals")
    
    print(f"Aggregated into {len(arrivals)} one-second intervals.")

    # 2. Calculate Historical Hawkes Intensity
    mu = 4.7
    beta = 10.0
    alpha = 5.0 * 0.05  # Matching your trained agent's environment

    lambdas = []
    current_lambda = mu
    decay = np.exp(-beta * 1.0)

    for _, row in arrivals.iterrows():
        new_arrivals = row["actual_arrivals"]
        lambdas.append(current_lambda)
        
        # Update for next step
        new_lambda = mu + (current_lambda - mu) * decay + (alpha * new_arrivals)
        current_lambda = np.clip(new_lambda, 0.0, 200.0)

    arrivals["lambda"] = lambdas
    arrivals["step"] = range(len(arrivals))

    # 3. Save to processed (Using absolute path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    final_df = arrivals[["step", "actual_arrivals", "lambda"]]
    final_df.to_csv(processed_path, index=False)
    print(f"--- SUCCESS ---")
    print(f"Saved processed intensities to: {processed_path}")

if __name__ == "__main__":
    preprocess_data()