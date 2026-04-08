import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseStrategy(ABC):
    """
    Unified Interface. 
    The Lab uses this to run 10-year backtests instantly.
    The Trading System uses this to generate live signals.
    """
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.universe = config.get("universe", {}).get("instruments", ["GBPUSD"])
        self.parameters = config.get("parameters", {})
        
        # We explicitly declare what @provides features this strategy needs.
        # The Engine will read this and build the DataFrame automatically.
        self.required_features = self.get_required_features()

    @abstractmethod
    def get_required_features(self) -> List[str]:
        """Return a list of feature strings (e.g., ['ATR', 'PWH', 'Log_Return'])"""
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        PURE VECTORIZED MATH ONLY. No iterrows. No risk management.
        Returns a Pandas Series of target positions (1.0 for Long, -1.0 for Short, 0.0 for Flat)
        """
        pass