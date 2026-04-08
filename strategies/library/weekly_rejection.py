# File: strategies/library/weekly_rejection.py
import pandas as pd
import numpy as np
from strategies.library.base_strategy import BaseStrategy

class WeeklyRejectionStrategy(BaseStrategy):
    
    def get_required_features(self) -> list:
        # The Engine sees this and automatically runs the @provides functions
        # from shared/features/ to populate the DataFrame.
        return [
            'ATR', 
            'PWH', 'PWL', 
            'Confirmed_Fractal_High', 'Confirmed_Fractal_Low',
            'First_1W_Rej_Long', 'First_1W_Rej_Short'
        ]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Extract parameters from config
        atr_threshold = self.parameters.get('min_atr', 0.0010)
        
        # 1. Start with flat positions
        signals = pd.Series(0.0, index=df.index)
        
        # 2. Vectorized condition checks
        valid_volatility = df['ATR'] > atr_threshold
        
        long_condition = (df['First_1W_Rej_Long'] == 1) & valid_volatility
        short_condition = (df['First_1W_Rej_Short'] == 1) & valid_volatility
        
        # 3. Assign signals (1 for Long, -1 for Short)
        signals.loc[long_condition] = 1.0
        signals.loc[short_condition] = -1.0
        
        # Shift by 1 bar. If signal occurs at 10:00 close, we execute at 10:15 open.
        # This absolutely guarantees NO lookahead bias in Research or Production.
        return signals.shift(1).fillna(0.0)