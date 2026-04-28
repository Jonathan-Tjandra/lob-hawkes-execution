"""
Analysis and plotting for the trained PPO execution agent.
Fixes vs original:
  - Y-axis labels corrected for the 0-10 action space (was hardcoded for old 3-action env)
  - Third panel added: per-step reward so you can see exactly where value is made/lost
  - Cumulative reward overlay so you can see the return curve shape
  - Lambda axis scaled correctly (was clipped by alpha=0.3 style)
  - Episode summary printed with correct raw reward (un-normalised)
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.env.execution_env import OptimalExecutionEnv


def run_analysis(n_episodes=1):
    model = PPO.load("results/models/ppo_execution_agent")
    env = OptimalExecutionEnv()

    all_episodes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        data = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            data.append({
                'step':      env.current_step,
                'inventory': env.inventory,
                'lambda':    env.current_lambda,
                'action':    int(action.item() if isinstance(action, np.ndarray) else action),
                'reward':    reward * env._reward_scale,   # un-normalise for display
            })

        df = pd.DataFrame(data)
        df['cum_reward'] = df['reward'].cumsum()
        all_episodes.append(df)

    # Use the last episode for the main plot
    df = all_episodes[-1]

    # ── Summary ──────────────────────────────────────────────────────────────
    final_inv  = df['inventory'].iloc[-1]
    total_raw  = df['reward'].sum()
    n_steps    = len(df)
    avg_action = df['action'].mean()
    pct_wait   = (df['action'] == 0).mean() * 100

    print("=" * 50)
    print("  EPISODE SUMMARY")
    print("=" * 50)
    print(f"  Steps taken         : {n_steps}")
    print(f"  Final inventory     : {final_inv} / {env.initial_inventory}")
    print(f"  Shares sold         : {env.initial_inventory - final_inv}")
    print(f"  Raw total reward    : {total_raw:.2f}")
    print(f"  Avg units/step      : {avg_action:.2f}")
    print(f"  Steps spent waiting : {pct_wait:.0f}%")
    print("=" * 50)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 11))
    fig.suptitle("PPO Execution Agent — Dashboard", fontsize=15, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

    steps = df['step']

    # ── Panel 1: Inventory + Lambda ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("Inventory Drawdown vs Market Toxicity", fontsize=11)

    color_inv = '#1f77b4'
    color_lam = '#d62728'

    lns1 = ax1.plot(steps, df['inventory'], color=color_inv,
                    label='Remaining inventory', linewidth=2.5)
    ax1.set_ylabel('Inventory (units)', color=color_inv)
    ax1.tick_params(axis='y', labelcolor=color_inv)
    ax1.set_ylim(-2, env.initial_inventory * 1.05)

    ax1r = ax1.twinx()
    lns2 = ax1r.plot(steps, df['lambda'], color=color_lam,
                     alpha=0.55, linewidth=1.5, label='Hawkes λ(t)')
    ax1r.set_ylabel('Market toxicity λ(t)', color=color_lam)
    ax1r.tick_params(axis='y', labelcolor=color_lam)
    ax1r.set_ylim(0, df['lambda'].max() * 1.2)

    # Combined legend
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right', fontsize=9)
    ax1.set_xlabel('')

    # ── Panel 2: Actions ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_title("Agent Action per Step  (0 = wait, 10 = sell max)", fontsize=11)

    ax2.step(steps, df['action'], where='post', color='#2ca02c',
             linewidth=1.8, label='Units sold')
    ax2.fill_between(steps, df['action'], step='post',
                     color='#2ca02c', alpha=0.15)

    # Correct y-ticks for 0-10 action space
    ax2.set_yticks(range(0, 11, 2))
    ax2.set_yticklabels([str(i) for i in range(0, 11, 2)])
    ax2.set_ylim(-0.5, 11)
    ax2.set_ylabel('Units sold')
    ax2.axhline(y=avg_action, color='#2ca02c', linestyle='--',
                alpha=0.5, linewidth=1, label=f'Mean = {avg_action:.1f}')
    ax2.legend(loc='upper right', fontsize=9)

    # ── Panel 3: Per-step reward + cumulative ────────────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_title("Per-step Reward & Cumulative Return", fontsize=11)

    color_bar  = '#ff7f0e'
    color_cum  = '#9467bd'

    # Bar chart for per-step (positive = green, negative = red)
    colors_bar = [color_bar if r >= 0 else '#d62728' for r in df['reward']]
    ax3.bar(steps, df['reward'], color=colors_bar, alpha=0.6,
            width=0.8, label='Step reward')
    ax3.axhline(0, color='black', linewidth=0.7, linestyle='-')
    ax3.set_ylabel('Step reward (raw)', color=color_bar)
    ax3.tick_params(axis='y', labelcolor=color_bar)

    ax3r = ax3.twinx()
    ax3r.plot(steps, df['cum_reward'], color=color_cum,
              linewidth=2, label='Cumulative return')
    ax3r.set_ylabel('Cumulative return', color=color_cum)
    ax3r.tick_params(axis='y', labelcolor=color_cum)

    # Combined legend for panel 3
    handles1, labels1 = ax3.get_legend_handles_labels()
    handles2, labels2 = ax3r.get_legend_handles_labels()
    ax3.legend(handles1 + handles2, labels1 + labels2,
               loc='lower right', fontsize=9)

    ax3.set_xlabel('Time step')

    plt.savefig("results/plots/execution_analysis.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to results/plots/execution_analysis.png")
    plt.show()


if __name__ == "__main__":
    run_analysis()