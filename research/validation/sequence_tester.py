import numpy as np
import pandas as pd

class SequenceTester:
    def __init__(self, trades: list, n_shuffles: int = 1000):
        self.trades = trades
        self.n_shuffles = n_shuffles

    def run(self) -> dict:
        if not self.trades:
            return {"max_drawdown_95th": 0.0, "recovery_time_95th": 0.0}

        # 1. Extract raw dollar PnL per trade
        pnl_array = np.array([t['pnl'] for t in self.trades])
        initial_capital = 10000.0

        # 2. Vectorized Shuffling (Optimization)
        # Create a matrix of [n_shuffles, n_trades]
        # Each row is a different random sequence of your same trades
        shuffled_indices = np.array([np.random.permutation(len(pnl_array)) for _ in range(self.n_shuffles)])
        shuffled_pnls = pnl_array[shuffled_indices]
        
        # 3. Calculate Equity Curves for all shuffles at once
        # Result: [n_shuffles, n_trades + 1]
        equity_curves = initial_capital + np.hstack([
            np.zeros((self.n_shuffles, 1)), 
            np.cumsum(shuffled_pnls, axis=1)
        ])

        # 4. Vectorized Drawdown Calculation
        # Running max for each curve
        running_max = np.maximum.accumulate(equity_curves, axis=1)
        drawdowns = (running_max - equity_curves) / running_max
        
        # Peak drawdown per shuffle
        max_drawdowns = np.max(drawdowns, axis=1)
        
        # 5. Extract the 95th Percentile (The "Bad Luck" case)
        # We want to know: "In a bad sequence, how deep is the hole?"
        max_dd_95 = np.percentile(max_drawdowns, 95)

        return {
            "max_drawdown_avg": float(np.mean(max_drawdowns)),
            "max_drawdown_95th": float(max_dd_95),
            "passed_sequence_risk": bool(max_dd_95 < 0.20) # Pass if 95% of sequences stay under 20% DD
        }