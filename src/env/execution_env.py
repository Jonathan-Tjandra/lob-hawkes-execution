"""
Gymnasium Environment for Optimal Execution.

Changelog vs original:
  - Alpha scaled from 0.2 → 0.05: prevents the agent from self-destructing
    the Hawkes intensity with its own trades. At alpha=0.2 each aggressive sell
    spiked lambda by ~10x mu within 5 steps, making further selling suicidal and
    pushing the agent into a "sell everything immediately" trap.
  - Reward normalised by _reward_scale: PPO's value loss was hitting 2.5e+04
    because raw episode returns were O(100). Normalising to O(1) stabilises
    the critic and fixes the explained_variance ≈ 0 problem seen in early logs.
  - Action space expanded to 0-10 units (was 0-3): gives the agent enough
    throughput to clear inventory quickly when lambda is low.
  - Holding cost with quadratic urgency: replaces the terminal cliff penalty.
    Provides a dense gradient signal at every step so PPO doesn't need to
    wait until episode end to learn that waiting is costly.
  - Timing bonus: rewards selling when lambda < 2*mu, teaching the agent to
    actually use the Hawkes signal rather than ignore it.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class OptimalExecutionEnv(gym.Env):
    def __init__(
        self,
        initial_inventory=100,
        max_steps=200,
        baseline_mu=4.7,
        alpha=5.0,
        beta=10.0,
    ):
        super().__init__()
        self.initial_inventory = initial_inventory
        self.max_steps = max_steps
        self.mu = baseline_mu
        self.beta = beta

        # Effective alpha is 0.05x the passed value.
        # At alpha=5.0 (default), this gives effective_alpha=0.25.
        self.alpha = alpha * 0.05

        # Action space: sell 0–10 units per step.
        # With 100 shares and max_steps=200, selling 10/step can clear in
        # as few as 10 steps when lambda is low.
        self.action_space = spaces.Discrete(11)
        self.action_sizes = {i: i for i in range(11)}

        # Observation: [time_remaining, inventory_remaining, lambda_normalised]
        # All in [0,1] except lambda which can go up to 10x mu in extreme cases.
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Reward scale: divide all raw rewards by this value.
        # Keeps episode returns in [-10, +10] range for PPO stability.
        # Set to 10% of max possible base reward (inventory * 1.0).
        self._reward_scale = initial_inventory * 0.1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = self.initial_inventory
        self.current_lambda = self.mu
        return self._get_obs(), {}

    def _get_obs(self):
        time_rem = 1.0 - (self.current_step / self.max_steps)
        inv_rem = self.inventory / self.initial_inventory
        lam_norm = np.clip(self.current_lambda / self.mu, 0.0, 10.0)
        return np.array([time_rem, inv_rem, lam_norm], dtype=np.float32)

    def _update_hawkes(self):
        dt = 1.0
        decay = np.exp(-self.beta * dt)
        safe_lambda = np.clip(self.current_lambda, 0.0, 200.0)
        new_arrivals = np.random.poisson(safe_lambda * dt)
        new_lambda = (
            self.mu
            + (self.current_lambda - self.mu) * decay
            + (self.alpha * new_arrivals)
        )
        self.current_lambda = np.clip(new_lambda, 0.0, 200.0)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()

        self.current_step += 1
        exec_size = min(self.action_sizes[action], self.inventory)
        self.inventory -= exec_size

        toxicity_ratio = self.current_lambda / self.mu

        # 1. Base reward: +1 per unit sold. Clear incentive to execute.
        base_reward = float(exec_size)

        # 2. Timing bonus: extra reward for selling into calm markets (low lambda).
        #    clip(2 - toxicity, 0, 2) → 0 when toxic, up to 2 when very calm.
        #    Weight 0.3 keeps this secondary to base_reward.
        timing_bonus = exec_size * np.clip(2.0 - toxicity_ratio, 0.0, 2.0) * 0.3

        # 3. Impact penalty: convex in size, scales with toxicity.
        #    Exponent 1.5 means doubling exec_size costs 2.8x not 2x.
        impact_penalty = (exec_size ** 1.5) * toxicity_ratio * 0.15

        # 4. Holding cost: per-step penalty for unsold inventory.
        #    urgency ramps from 1x at t=0 to 4x at t=T (quadratic).
        #    This is the dense gradient signal that replaced the terminal cliff.
        time_elapsed_frac = self.current_step / self.max_steps
        urgency = 1.0 + 3.0 * (time_elapsed_frac ** 2)
        holding_cost = self.inventory * 0.02 * urgency

        raw_reward = base_reward + timing_bonus - impact_penalty - holding_cost

        self._update_hawkes()

        terminated = bool(self.inventory <= 0)
        truncated = bool(self.current_step >= self.max_steps)

        # 5. Soft terminal penalty for leftover inventory.
        #    2x per unit (was 10x) — learnable gradient, not a cliff.
        if truncated and not terminated:
            raw_reward -= self.inventory * 2.0

        # Normalise so PPO's value function sees O(1) returns.
        reward = raw_reward / self._reward_scale

        return self._get_obs(), float(reward), terminated, truncated, {}

    def render(self):
        time_rem = self.max_steps - self.current_step
        print(
            f"Step: {self.current_step} | Inv: {self.inventory}/{self.initial_inventory} | "
            f"λ(t): {self.current_lambda:.2f} | Time Rem: {time_rem}"
        )