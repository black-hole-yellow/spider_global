import numpy as np
import pandas as pd
from trading.backtest.event_driven_backtester import EventDrivenBacktester
from trading.execution.paper_broker import PaperBroker
from trading.portfolio.portfolio_manager import PortfolioManager
from trading.execution.execution_engine import ExecutionEngine

class OOSTester:
    def __init__(self, df: pd.DataFrame, strategy, config: dict, train_split: float = 0.8):
        self.df = df
        self.strategy = strategy
        self.config = config
        self.train_split = train_split

    @property
    def oos_data(self) -> pd.DataFrame:
        """Abstraction: Cleanly isolate the Out-of-Sample data portion."""
        if self.df.empty:
            return pd.DataFrame()
        split_idx = int(len(self.df) * self.train_split)
        return self.df.iloc[split_idx:].copy()

    def run(self) -> dict:
        oos_df = self.oos_data
        
        # Fail gracefully if there isn't enough OOS data to test
        if oos_df.empty or len(oos_df) < 100:
            return {"oos_sharpe": 0.0, "oos_win_rate": 0.0, "oos_profit_factor": 0.0, "oos_trades": 0}

        # 1. Initialize clean components strictly for the OOS period
        broker = PaperBroker(initial_cash=10000.0)
        pm = PortfolioManager(self.config)
        engine = ExecutionEngine(broker)
        
        # 2. Run the event-driven simulation ONLY on the unseen data
        backtester = EventDrivenBacktester(oos_df, self.strategy, broker, pm, engine)
        backtester.run()
        
        trades = backtester.completed_trades
        if not trades:
            return {"oos_sharpe": 0.0, "oos_win_rate": 0.0, "oos_profit_factor": 0.0, "oos_trades": 0}

        # 3. Vectorized Metric Calculation (Optimized)
        returns = []
        gross_profit = 0.0
        gross_loss = 0.0
        wins = 0
        
        for t in trades:
            ret = (t['close_price'] - t['entry_price']) / t['entry_price'] if t['direction_val'] == 1 else (t['entry_price'] - t['close_price']) / t['entry_price']
            returns.append(ret)
            
            if t['pnl'] > 0:
                wins += 1
                gross_profit += t['pnl']
            else:
                gross_loss += abs(t['pnl'])

        returns_arr = np.array(returns)
        mean_ret = np.mean(returns_arr)
        std_ret = np.std(returns_arr) if np.std(returns_arr) > 0 else 1e-9
        
        oos_sharpe = (mean_ret / std_ret) * np.sqrt(200) # Assuming ~200 trades/year scaling
        oos_win_rate = (wins / len(trades)) * 100
        oos_profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 99.0

        return {
            "oos_sharpe": float(oos_sharpe),
            "oos_win_rate": float(oos_win_rate),
            "oos_profit_factor": float(oos_profit_factor),
            "oos_trades": len(trades)
        }