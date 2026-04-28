"""
PPO Agent vs. TWAP Baseline Comparison.

Proves "Alpha in Execution" by running both algorithms through the
exact same market conditions (identical random seeds) and plotting
their cumulative returns and inventory drawdowns.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Setup path so Python can find 'src'
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.env.execution_env import OptimalExecutionEnv

def run_ppo(env, model, seed):
    """Runs the trained PPO agent on a specific seeded environment."""
    obs, _ = env.reset(seed=seed)
    data = []
    terminated = truncated = False
    
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        # Convert action to native int if it's a numpy array
        action_val = int(action.item() if isinstance(action, np.ndarray) else action)
        
        obs, reward, terminated, truncated, _ = env.step(action_val)
        
        data.append({
            'step': env.current_step,
            'inventory': env.inventory,
            'reward': reward * env._reward_scale # Un-normalize reward for raw tracking
        })
        
    return pd.DataFrame(data)

def run_twap(env, seed):
    """
    Runs a Time-Weighted Average Price (TWAP) baseline.
    TWAP attempts to sell inventory evenly across the time horizon.
    """
    obs, _ = env.reset(seed=seed)
    data = []
    terminated = truncated = False
    
    while not (terminated or truncated):
        # Calculate the "ideal" inventory TWAP should have at this step
        steps_remaining = env.max_steps - env.current_step
        
        if steps_remaining > 0:
            ideal_inventory = env.initial_inventory * ((steps_remaining - 1) / env.max_steps)
        else:
            ideal_inventory = 0
            
        # The action is whatever size gets us closest to the ideal inventory
        target_sell = env.inventory - ideal_inventory
        action_val = int(np.clip(round(target_sell), 0, 10)) # Bound to valid action space (0-10)
        
        obs, reward, terminated, truncated, _ = env.step(action_val)
        
        data.append({
            'step': env.current_step,
            'inventory': env.inventory,
            'reward': reward * env._reward_scale
        })
        
    return pd.DataFrame(data)

def main():
    # Load environment and the trained "Brain"
    env = OptimalExecutionEnv()
    try:
        model = PPO.load("results/models/ppo_execution_agent")
    except FileNotFoundError:
        print("Error: Could not find trained model. Run train_ppo.py first.")
        return

    # Use a fixed seed so both algorithms face the exact same market volatility
    TEST_SEED = 1337
    print(f"Running PPO vs TWAP simulation (Seed: {TEST_SEED})...")
    
    df_ppo = run_ppo(env, model, seed=TEST_SEED)
    df_twap = run_twap(env, seed=TEST_SEED)
    
    # Calculate Cumulative Returns
    df_ppo['cum_reward'] = df_ppo['reward'].cumsum()
    df_twap['cum_reward'] = df_twap['reward'].cumsum()
    
    # --- Plotting the Comparison ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Alpha Execution: Reinforcement Learning vs. TWAP Baseline", fontsize=15, fontweight='bold')
    
    # Plot 1: Cumulative Reward
    ax1.set_title("Cumulative Execution Return (Higher is Better)", fontsize=12)
    ax1.plot(df_ppo['step'], df_ppo['cum_reward'], label=f'PPO Agent (Final: {df_ppo["cum_reward"].iloc[-1]:.2f})', color='#9467bd', linewidth=3)
    ax1.plot(df_twap['step'], df_twap['cum_reward'], label=f'TWAP Baseline (Final: {df_twap["cum_reward"].iloc[-1]:.2f})', color='#7f7f7f', linewidth=2, linestyle='--')
    ax1.set_ylabel("Raw Reward (Slippage Eq.)")
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.legend(loc='lower left')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Inventory Trajectory
    ax2.set_title("Inventory Drawdown Profile", fontsize=12)
    ax2.plot(df_ppo['step'], df_ppo['inventory'], label='PPO Trajectory', color='#1f77b4', linewidth=2)
    ax2.plot(df_twap['step'], df_twap['inventory'], label='TWAP Trajectory (Linear)', color='#7f7f7f', linewidth=2, linestyle='--')
    ax2.set_ylabel("Remaining Inventory")
    ax2.set_xlabel("Time Step")
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = "results/plots/baseline_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"Simulation complete. Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()