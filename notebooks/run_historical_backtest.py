import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

# Setup path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.env.historical_env import HistoricalExecutionEnv

def run_unified_backtest():
    model = PPO.load("results/models/ppo_execution_agent")
    env = HistoricalExecutionEnv()
    
    # --- 1. RUN PPO AGENT ---
    obs, _ = env.reset()
    agent_hist = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(int(action))
        agent_hist.append({
            'step': env.current_step,
            'inventory': env.inventory,
            'action': int(action),
            'reward': reward * env._reward_scale,
            'lambda': env.current_lambda
        })
        if term or trunc: break
    df_agent = pd.DataFrame(agent_hist)
    
    # --- 2. RUN TWAP BASELINE ---
    obs, _ = env.reset()
    twap_hist = []
    while True:
        steps_rem = env.max_steps - env.current_step
        target = env.inventory / steps_rem if steps_rem > 0 else env.inventory
        action = int(np.clip(round(target), 0, 10))
        obs, reward, term, trunc, _ = env.step(action)
        twap_hist.append({
            'step': env.current_step,
            'inventory': env.inventory,
            'reward': reward * env._reward_scale
        })
        if term or trunc: break
    df_twap = pd.DataFrame(twap_hist)

    # --- 3. METRICS ---
    agent_total = df_agent['reward'].sum()
    twap_total = df_twap['reward'].sum()
    alpha = agent_total - twap_total

    # --- 4. PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"BTC-USD Historical Backtest: Agent vs TWAP Baseline", fontsize=15, fontweight='bold')

    # Top: Inventory Trajectories + Toxicity
    ax1.plot(df_agent['step'], df_agent['inventory'], label='PPO Agent', color='#1f77b4', lw=2.5)
    ax1.plot(df_twap['step'], df_twap['inventory'], label='TWAP Baseline', color='#7f7f7f', ls='--', lw=2)
    ax1.set_ylabel("Inventory (Units)", fontweight='bold')
    
    ax1r = ax1.twinx()
    ax1r.plot(df_agent['step'], df_agent['lambda'], color='#d62728', alpha=0.3, label='Market Toxicity')
    ax1r.set_ylabel("Toxicity λ(t)", color='#d62728')
    ax1.legend(loc='upper right')

    # Bottom: Cumulative Reward Comparison
    ax2.plot(df_agent['step'], df_agent['reward'].cumsum(), label=f'PPO Agent ({agent_total:.1f})', color='#2ca02c', lw=2.5)
    ax2.plot(df_twap['step'], df_twap['reward'].cumsum(), label=f'TWAP Baseline ({twap_total:.1f})', color='#d62728', lw=2)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_ylabel("Cumulative Raw Reward", fontweight='bold')
    ax2.set_xlabel("Time (Seconds)")
    ax2.legend(loc='lower left')

    # Summary Box
    stats_text = (
        f"FINAL PERFORMANCE\n"
        f"-----------------\n"
        f"Agent:  +{agent_total:.1f}\n"
        f"TWAP:   {twap_total:.1f}\n"
        f"ALPHA:  +{alpha:.1f}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax2.text(0.98, 0.2, stats_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='bottom', horizontalalignment='right', bbox=props, family='monospace')

    plt.tight_layout()
    plt.savefig("results/plots/historical_comparison.png")
    print(f"Master Backtest Dashboard saved. Alpha: {alpha:.2f}")
    plt.show()

if __name__ == "__main__":
    run_unified_backtest()