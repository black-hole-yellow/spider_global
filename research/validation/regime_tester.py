import numpy as np
import pandas as pd

class RegimeTester:
    # CHANGED: Accept baseline trades and the fully featured DataFrame
    def __init__(self, trades: list, df: pd.DataFrame, config: dict):
        self.trades = trades
        self.df = df
        self.config = config

    def run(self) -> dict:
        if not self.trades or self.df.empty:
            return {"passed_regime": False, "regime_metrics": {}}

        # Ensure we have a regime column. If the strategy didn't request a complex 
        # Markov regime, we fallback to a simple Volatility Regime calculation.
        regime_col = 'Regime' if 'Regime' in self.df.columns else 'volatility_regime'
        if regime_col not in self.df.columns:
            # Simple vectorized High/Low Volatility calculation based on ATR or StdDev
            returns_std = self.df['close'].pct_change().rolling(100).std()
            long_std = self.df['close'].pct_change().rolling(1000).std()
            self.df['volatility_regime'] = np.where(returns_std > long_std, 'High Vol', 'Low Vol')

        # 1. Map each trade to its market regime
        regime_returns = {}
        
        for t in self.trades:
            entry_time = t['entry_time']
            
            try:
                # Find the exact row in the dataframe for the trade entry time
                idx = self.df.index.get_indexer([entry_time], method='pad')[0]
                regime = str(self.df[regime_col].iloc[idx])
            except Exception:
                regime = "Unknown"

            if regime not in regime_returns:
                regime_returns[regime] = []
            
            # Calculate standard percentage return
            ret = (t['close_price'] - t['entry_price']) / t['entry_price'] if t['direction_val'] == 1 else (t['entry_price'] - t['close_price']) / t['entry_price']
            regime_returns[regime].append(ret)

        # 2. Calculate metrics per regime group (Vectorized)
        regime_metrics = {}
        passed_all = True
        
        for regime, returns in regime_returns.items():
            if len(returns) < 20: 
                # Ignore regimes with too few trades to be statistically valid
                continue
                
            ret_arr = np.array(returns)
            wins = np.sum(ret_arr > 0)
            win_rate = (wins / len(returns)) * 100
            
            mean_ret = np.mean(ret_arr)
            std_ret = np.std(ret_arr) if np.std(ret_arr) > 0 else 1e-9
            sharpe = float((mean_ret / std_ret) * np.sqrt(200))
            
            regime_metrics[regime] = {
                "trades": len(returns),
                "win_rate": float(win_rate),
                "sharpe": sharpe
            }
            
            # The Gauntlet Rule: Strategy fails if it has a sharply negative Sharpe (< -0.5) 
            # in ANY major market regime. It must survive all environments.
            if sharpe < -0.5:
                passed_all = False

        return {
            "passed_regime": passed_all,
            "regime_metrics": regime_metrics
        }