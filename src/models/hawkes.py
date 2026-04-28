"""
Recursive O(N) Hawkes Process Estimator.
Optimizes parameters (mu, alpha, beta) using Maximum Likelihood Estimation (MLE).
"""
import numpy as np
from scipy.optimize import minimize

class HawkesEstimator:
    def __init__(self):
        self.mu = None
        self.alpha = None
        self.beta = None

    def _neg_log_likelihood(self, params, t, T):
        """
        Computes the negative log-likelihood in O(N) time using recursion.
        """
        mu, alpha, beta = params
        
        # Stability and bounds checks: alpha < beta is required for stationarity
        if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
            return 1e9 

        N = len(t)
        
        # 1. Compute the integral term
        # \int \lambda(s) ds = \mu T + \frac{\alpha}{\beta} \sum (1 - e^{-\beta(T - t_i)})
        integral = mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - t)))

        # 2. Compute the recursive sum of log intensities
        log_lambda_sum = 0.0
        R = 0.0
        
        for i in range(1, N):
            dt = t[i] - t[i-1]
            # Recursive update for the exponential decay
            R = np.exp(-beta * dt) * (1 + R)
            
            # Current intensity
            lam_i = mu + alpha * R
            log_lambda_sum += np.log(lam_i)

        return -(log_lambda_sum - integral)

    def fit(self, timestamps):
        """
        Fits the Hawkes process to an array of event timestamps (in seconds).
        """
        # Normalize timestamps to start at 0
        t = np.sort(timestamps)
        t = t - t[0]
        T = t[-1]

        # Initial guess: mu=0.1, alpha=0.5, beta=1.0
        initial_guess = np.array([0.1, 0.5, 1.0])
        
        # Bounds: all parameters strictly positive, stationary condition handled in loss
        bounds = ((1e-5, None), (1e-5, None), (1e-5, None))

        print("Fitting Hawkes parameters via MLE...")
        result = minimize(
            self._neg_log_likelihood, 
            initial_guess, 
            args=(t, T), 
            method='L-BFGS-B', 
            bounds=bounds
        )

        if result.success:
            self.mu, self.alpha, self.beta = result.x
            print(f"Fit successful! μ: {self.mu:.4f}, α: {self.alpha:.4f}, β: {self.beta:.4f}")
            print(f"Branching Ratio (α/β): {self.alpha / self.beta:.4f}")
        else:
            print("Optimization failed:", result.message)
            
        return result.success