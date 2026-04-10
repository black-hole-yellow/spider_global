import numpy as np
import pandas as pd

class CapacityTester:
    def __init__(self, trades: list, df: pd.DataFrame, max_volume_participation: float = 0.05):
        """
        max_volume_participation: Institutional rule. You should not be more than 5% 
        of the candle's total volume, otherwise you move the market against yourself.
        """
        self.trades = trades
        self.df = df
        self.max_volume_participation = max_volume_participation

    def run(self) -> dict:
        if not self.trades or len(self.trades) < 50:
            return {"optimal_kelly": 0.0, "max_capacity_lots": 0, "passed_capacity": False}

        # 1. VECTORIZED KELLY CRITERION CALCULATION
        # Kelly % = W - [(1 - W) / R]  where W = Win Rate, R = Reward/Risk Ratio
        pnls = np.array([t['pnl'] for t in self.trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        
        if len(losses) == 0 or len(wins) == 0:
             return {"optimal_kelly": 0.01, "max_capacity_lots": 0, "passed_capacity": False}

        win_rate = len(wins) / len(pnls)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        reward_risk_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        # We use "Half-Kelly" to be conservative and reduce drawdown volatility
        full_kelly = win_rate - ((1 - win_rate) / reward_risk_ratio)
        half_kelly = max(0.0, full_kelly / 2.0)

        # 2. LIQUIDITY & CAPACITY AUDIT
        # Check the volume of the specific candles where we entered trades
        trade_volumes = []
        volume_col = 'volume' if 'volume' in self.df.columns else 'Volume'
        
        if volume_col in self.df.columns:
            for t in self.trades:
                try:
                    idx = self.df.index.get_indexer([t['entry_time']], method='pad')[0]
                    vol = self.df[volume_col].iloc[idx]
                    trade_volumes.append(vol)
                except Exception:
                    continue
        
        # Calculate Max Capacity
        if trade_volumes:
            # We look at the 5th percentile of volume to find our "worst-case" liquidity
            bottleneck_volume = np.percentile(trade_volumes, 5)
            # Assuming volume is in standard lots (100,000 units)
            max_capacity_lots = bottleneck_volume * self.max_volume_participation
        else:
            # Fallback if no volume data is provided (assume standard retail limits)
            max_capacity_lots = 50.0 

        # The Gauntlet Rule: Strategy must have a positive Kelly edge and support 
        # at least 5 standard lots of liquidity at its worst entry points.
        passed_capacity = bool(half_kelly > 0.005 and max_capacity_lots >= 5.0)

        return {
            "optimal_kelly_fraction": round(half_kelly, 4),
            "max_capacity_lots": round(max_capacity_lots, 2),
            "passed_capacity": passed_capacity
        }