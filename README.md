# High-Frequency Optimal Execution via Reinforcement Learning

An end-to-end quantitative trading framework that optimizes the execution of large BTC-USD orders using **Proximal Policy Optimization (PPO)**. This project integrates structural microstructure models (**Almgren-Chriss**) with self-exciting point processes (**Hawkes Processes**) to navigate market toxicity and minimize implementation shortfall.

## 🚀 Overview

Institutional traders face a fundamental tradeoff: execute quickly and suffer high **market impact**, or execute slowly and risk **price drift/volatility**. This repository provides a deep reinforcement learning solution that learns a dynamic liquidation policy by "sensing" market heat through a Hawkes Intensity signal.

### Key Features
* **Stochastic Market Environment**: A custom OpenAI Gym-style environment implementing Almgren-Chriss permanent and temporary impact.
* **Hawkes Process Integration**: Real-time modeling of trade-arrival clustering ($\lambda$) to signal market toxicity.
* **PPO Agent**: A Stable Baselines3 implementation using a Multi-Layer Perceptron (MLP) policy.
* **Historical Backtester**: A robust evaluation suite that replays real Coinbase BTC-USD tick data to calculate "Alpha" over TWAP benchmarks.

---

## 📉 Methodology

### 1. Market Impact (Almgren-Chriss)
The environment simulates price dynamics where your actions move the market.
* **Temporary Impact**: Non-linear slippage calculated as $\eta \cdot q_t^k$.
* **Permanent Impact**: Long-term price "scars" following $\gamma \cdot q_t$.
* **Inventory Risk**: A quadratic penalty $\phi I_t^2$ to discourage holding large positions in volatile regimes.

### 2. Market Toxicity (Hawkes Process)
We model the "heat" of the market using a self-exciting process:
$$\lambda(t) = \mu + \alpha \sum_{t_i < t} e^{-\beta(t-t_i)}$$
The agent uses $\lambda(t)$ as a leading indicator to pause execution during toxic bursts and accelerate during quiet liquidity windows.

---

## 📊 Results: Agent vs. TWAP

The PPO agent consistently outperforms the **Time-Weighted Average Price (TWAP)** benchmark by capturing opportunistic liquidity.

| Metric | TWAP Baseline | PPO Agent | Alpha (Gain) |
| :--- | :--- | :--- | :--- |
| **Raw Reward (Slippage Eq.)** | -197.7 | +68.9 | **+266.6** |
| **Execution Time** | 200s (Fixed) | ~25s (Dynamic) | -175s |
| **Inventory Risk** | High (Linear) | Low (Front-loaded) | Reduced |

### Historical Backtest (Out-of-Sample)
The agent, trained on synthetic Hawkes data, successfully generalized to real-world Coinbase BTC-USD order flow, identifying and avoiding historical toxicity spikes.

---

## 🛠️ Installation & Usage

1. **Clone the Repo**
   ```bash
   git clone [https://github.com/Jonathan-Tjandra/lob-hawkes-execution.git](https://github.com/Jonathan-Tjandra/lob-hawkes-execution.git)
   cd lob-hawkes-execution

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

**1. Run the Historical Backtest**
Evaluate the pre-trained agent against historical Coinbase data to generate the comparative alpha dashboard.
    ```bash
    python src/analysis/unified_backtest.py
    ```

**2. (Optional) Re-train the Agent**
Initialize a new PPO training loop on the stochastic Almgren-Chriss environment.
    ```bash
    python src/train_ppo.py
    ```

---

## 📦 Pre-Trained Weights

The fully trained agent weights have been pushed to this repository. You do not need to re-train the model from scratch to evaluate its performance.

- **Weights Location:** `results/models/ppo_execution_agent.zip`

---

## 📁 Repository Structure

```
├── data/
│   ├── raw/                # Historical BTC-USD trades from Coinbase
│   └── processed/          # Pre-calculated historical Hawkes intensities
├── results/
│   ├── models/             # Pre-trained PPO weights (.zip)
│   └── plots/              # Execution dashboards and backtest charts
├── src/
│   ├── env/
│   │   ├── execution_env.py   # Synthetic AC + Hawkes Environment
│   │   └── historical_env.py  # Historical Replay Environment
│   ├── data/
│   │   └── preprocess.py      # Hawkes intensity estimation logic
│   └── train_ppo.py           # RL Training script
└── README.md
```