import numpy as np
import pandas as pd

class ExecutionTester:
    def __init__(self, trades: list, pip_penalty: float = 3.0):
        """
        pip_penalty: The total combined friction of Spread + Slippage + Commission.
        For GBP/USD, 3.0 pips is a standard institutional stress test.
        """
        self.trades = trades
        self.pip_penalty = pip_penalty
        # Standard Forex Pip multiplier for pairs like GBP/USD, EUR/USD
        self.pip_multiplier = 0.0001 

    def run(self) -> dict:
        if not self.trades or len(self.trades) < 50:
            return {"execution_sharpe": 0.0, "passed_execution": False}

        penalty_value = self.pip_penalty * self.pip_multiplier
        
        degraded_returns = []
        gross_profit = 0.0
        gross_loss = 0.0

        for t in self.trades:
            # 1. Calculate the Raw Return
            if t['direction_val'] == 1:
                # LONG: We buy higher (ask) and sell lower (bid)
                simulated_entry = t['entry_price'] + (penalty_value / 2)
                simulated_exit = t['close_price'] - (penalty_value / 2)
                ret = (simulated_exit - simulated_entry) / simulated_entry
            else:
                # SHORT: We sell lower (bid) and buy higher (ask)
                simulated_entry = t['entry_price'] - (penalty_value / 2)
                simulated_exit = t['close_price'] + (penalty_value / 2)
                ret = (simulated_entry - simulated_exit) / simulated_entry
            
            degraded_returns.append(ret)

            # 2. Track degraded PnL for Profit Factor
            # Assuming a standard $10,000 base sizing for simple PF calculation
            simulated_pnl = ret * 10000 
            if simulated_pnl > 0:
                gross_profit += simulated_pnl
            else:
                gross_loss += abs(simulated_pnl)

        # 3. Calculate Degraded Metrics
        ret_arr = np.array(degraded_returns)
        mean_ret = np.mean(ret_arr)
        std_ret = np.std(ret_arr) if np.std(ret_arr) > 0 else 1e-9
        
        execution_sharpe = float((mean_ret / std_ret) * np.sqrt(200))
        execution_pf = (gross_profit / gross_loss) if gross_loss > 0 else 99.0

        # The Gauntlet Rule: Strategy must maintain at least a 1.0 Sharpe and 1.1 PF 
        # AFTER heavy institutional slippage is applied.
        passed_execution = bool(execution_sharpe >= 1.0 and execution_pf >= 1.1)

        return {
            "execution_sharpe": round(execution_sharpe, 2),
            "execution_profit_factor": round(execution_pf, 2),
            "passed_execution": passed_execution
        }