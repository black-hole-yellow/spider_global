import copy
import pandas as pd
import numpy as np
from trading.backtest.event_driven_backtester import EventDrivenBacktester
from trading.execution.paper_broker import PaperBroker
from trading.portfolio.portfolio_manager import PortfolioManager
from trading.execution.execution_engine import ExecutionEngine
from strategies.library.factory import StrategyFactory

class RobustnessTester:
    def __init__(self, df: pd.DataFrame, base_config: dict):
        self.df = df
        self.base_config = base_config
        # We test a 10% and 20% shift up and down for numeric parameters
        self.shift_matrix = [-0.20, -0.10, 0.10, 0.20] 

    def _evaluate_variation(self, mutated_config: dict) -> float:
        """Abstraction: Handles the complete execution of a single parameter variation."""
        # 1. Isolate the setup
        strategy = StrategyFactory.load_from_config(mutated_config)
        broker = PaperBroker(initial_cash=10000.0)
        pm = PortfolioManager(mutated_config)
        engine = ExecutionEngine(broker)
        
        # 2. Run the backtest
        # Note: Since the strategy parameters changed, we MUST recalculate signals
        temp_df = self.df.copy()
        temp_df['signals'] = strategy.generate_signals(temp_df)
        
        backtester = EventDrivenBacktester(temp_df, strategy, broker, pm, engine)
        backtester.run()
        
        # 3. Quickly calculate the Sharpe ratio for this variation
        trades = backtester.completed_trades
        if len(trades) < 50: return 0.0 # Punish variations that destroy trade frequency
        
        returns = [(t['close_price'] - t['entry_price']) / t['entry_price'] if t['direction_val'] == 1 else (t['entry_price'] - t['close_price']) / t['entry_price'] for t in trades]
        returns_arr = np.array(returns)
        mean_ret = np.mean(returns_arr)
        std_ret = np.std(returns_arr) if np.std(returns_arr) > 0 else 1e-9
        
        return float((mean_ret / std_ret) * np.sqrt(200))

    def run(self) -> dict:
        print("   Running parameter perturbation matrix...")
        params = self.base_config.get("parameters", {})
        variation_sharpes = []
        
        # Loop over every parameter that is a number (int or float)
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float)) and not isinstance(param_value, bool):
                
                for shift in self.shift_matrix:
                    # SAFELY create a completely separate config for this test
                    test_config = copy.deepcopy(self.base_config)
                    
                    # Apply the shift (e.g., 1.5 * 1.10 = 1.65)
                    new_val = param_value * (1 + shift)
                    # Keep integers as integers (e.g., period lengths)
                    if isinstance(param_value, int): new_val = int(round(new_val))
                        
                    test_config["parameters"][param_name] = new_val
                    
                    # Evaluate this specific shift
                    sharpe = self._evaluate_variation(test_config)
                    variation_sharpes.append(sharpe)
                    
        if not variation_sharpes:
            return {"robustness_score": 0.0, "passed_robustness": False}
            
        # Calculate how much the Sharpe ratio degrades on average
        avg_variation_sharpe = np.mean(variation_sharpes)
        baseline_sharpe = self.base_config.get("metrics", {}).get("baseline_sharpe", 1.0)
        
        # Robustness Score: Ratio of mutated performance vs original performance
        robustness_score = avg_variation_sharpe / baseline_sharpe if baseline_sharpe > 0 else 0
        
        return {
            "robustness_score": float(robustness_score),
            "avg_variation_sharpe": float(avg_variation_sharpe),
            # Pass if the strategy retains at least 70% of its original performance when shifted
            "passed_robustness": bool(robustness_score >= 0.70) 
        }