import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO

# Setup path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.env.historical_env import HistoricalExecutionEnv

def run_historical_dashboard():
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
        
    df = pd.DataFrame(agent_hist)

    # --- 2. SUMMARY METRICS ---
    total_reward = df['reward'].sum()
    n_steps = len(df)
    pct_wait = (df['action'] == 0).mean() * 100
    avg_size = df[df['action'] > 0]['action'].mean()

    # --- 3. PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    fig.suptitle("Out-of-Sample BTC Execution: RL Agent vs. Historical Order Flow", fontsize=15, fontweight='bold')

    # Add light gridlines
    ax1.grid(True, alpha=0.25)
    ax2.grid(True, alpha=0.25, axis='y')

    # Top Panel: Inventory & Hawkes Toxicity
    color_inv = '#1f77b4'
    color_lam = '#d62728'

    lns1 = ax1.plot(df['step'], df['inventory'], label='Remaining Inventory', color=color_inv, lw=2.5)
    ax1.set_ylabel("Inventory (Units)", color=color_inv, fontweight='bold')
    
    ax1r = ax1.twinx()
    lns2 = ax1r.plot(df['step'], df['lambda'], label='Historical Hawkes λ(t)', color=color_lam, alpha=0.5, lw=1.5)
    ax1r.set_ylabel("Market Toxicity", color=color_lam)
    
    # Combine legends for top panel
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')

    # Bottom Panel: Execution Size (Bar Chart)
    color_bar = '#5cb85c'
    ax2.bar(df['step'], df['action'], color=color_bar, alpha=0.9, label='Units Sold per Step')
    ax2.set_ylabel("Execution Size", color='#2ca02c', fontweight='bold')
    ax2.set_xlabel("Time (Seconds)", fontweight='bold')
    
    # Set y-axis bounds for actions (0 to 12 as per your image)
    ax2.set_ylim(0, 13)
    ax2.legend(loc='upper right')

    # Annotation Box
    stats_text = (
        f"Steps: {n_steps} | Wait: {pct_wait:.0f}%\n"
        f"Avg Size: {avg_size:.1f} units\n"
        f"Raw Reward: +{total_reward:.2f}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
    ax2.text(0.98, 0.75, stats_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save and show
    plt.savefig("results/plots/historical_backtest_result.png", dpi=150, bbox_inches='tight')
    print("Dashboard saved to results/plots/historical_backtest_result.png")
    plt.show()

if __name__ == "__main__":
    run_historical_dashboard()