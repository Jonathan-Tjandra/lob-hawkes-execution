"""
Historical Replay Environment.
Inherits from the standard environment but pulls Hawkes intensity 
and arrivals from the pre-processed historical CSV.
"""
import pandas as pd
import numpy as np
from src.env.execution_env import OptimalExecutionEnv

class HistoricalExecutionEnv(OptimalExecutionEnv):
    def __init__(self, processed_path="data/processed/historical_intensity.csv", **kwargs):
        # Initialize the parent class
        super().__init__(**kwargs)
        
        # Load the pre-processed data
        self.historical_data = pd.read_csv(processed_path)
        
        # Ensure we don't try to run longer than the data we have
        self.max_steps = min(self.max_steps, len(self.historical_data) - 1)
        
    def reset(self, seed=None, options=None):
        # We don't use the seed for Hawkes arrivals anymore, 
        # but we call parent reset for inventory and step counters
        super().reset(seed=seed)
        
        # Always start at the beginning of the historical file
        self.current_lambda = self.historical_data.iloc[0]['lambda']
        return self._get_obs(), {}

    def _update_hawkes(self):
        """
        OVERRIDE: Instead of simulating, we just peek at the next 
        row of our real historical data.
        """
        if self.current_step < len(self.historical_data):
            row = self.historical_data.iloc[self.current_step]
            self.current_lambda = row['lambda']
        else:
            # Fallback if we somehow exceed the file
            self.current_lambda = self.mu