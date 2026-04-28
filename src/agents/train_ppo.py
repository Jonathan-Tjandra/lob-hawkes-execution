"""
Trains a PPO agent to optimally execute an order block
while navigating the Hawkes-simulated limit order book.

Run fresh:   python train.py
Resume:      python train.py --resume
"""
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.env.execution_env import OptimalExecutionEnv

MODEL_PATH = "results/models/ppo_execution_agent"


def train_agent(timesteps=500_000, resume=False):
    """
    Initializes the environment and trains the PPO agent.
    Pass resume=True to continue training from the last saved checkpoint.
    """
    env = OptimalExecutionEnv(initial_inventory=100, max_steps=200)
    check_env(env, warn=True)

    save_dir = "results/models"
    os.makedirs(save_dir, exist_ok=True)

    zip_path = MODEL_PATH + ".zip"

    if resume and os.path.exists(zip_path):
        print(f"Resuming training from {zip_path} ...")
        model = PPO.load(MODEL_PATH, env=env)
        # Restore optimizer and other train state that .load() drops by default
        model.set_env(env)
    else:
        if resume:
            print(f"No saved model found at {zip_path}, starting fresh.")
        else:
            print("Starting fresh training run.")

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            # Larger rollout buffer — episodes are ~15-20 steps, so this holds
            # ~200 full episodes per update, giving a much richer gradient signal.
            n_steps=4096,
            batch_size=256,
            learning_rate=3e-4,
            # Longer discount horizon — the terminal state (inventory cleared)
            # is what matters, and it can be 15-200 steps away.
            gamma=0.995,
            gae_lambda=0.95,
            # Explicit entropy bonus prevents the policy collapsing to
            # "always sell max" before it learns the toxicity signal.
            ent_coef=0.01,
            # Slightly wider network than SB3 default [64, 64].
            # The obs is only 3-dim but the reward surface is non-linear.
            policy_kwargs=dict(net_arch=[128, 128]),
        )

    print(f"Training for {timesteps:,} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        reset_num_timesteps=not resume,   # keeps global step counter when resuming
    )

    model.save(MODEL_PATH)
    print(f"\nTraining complete. Model saved to {zip_path}")
    return model, env


def evaluate_agent(model, env):
    """
    Runs one deterministic episode and prints step-by-step decisions.
    """
    print("\n--- Running Evaluation Episode ---")
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()

    # Undo reward normalisation for display: multiply back by scale
    display_reward = total_reward * env._reward_scale
    print(f"\nEpisode finished.")
    print(f"  Normalised reward : {total_reward:.4f}")
    print(f"  Raw reward (slippage equivalent): {display_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO execution agent.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last saved checkpoint instead of starting fresh.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of environment steps to train for (default: 500,000).",
    )
    args = parser.parse_args()

    trained_model, eval_env = train_agent(
        timesteps=args.timesteps,
        resume=args.resume,
    )
    evaluate_agent(trained_model, eval_env)