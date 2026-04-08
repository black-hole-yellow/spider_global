import pandas as pd

def provides_features(*feature_names):
    """Decorator to register features for the dynamic research pipeline."""
    def decorator(func):
        func._provides_features = feature_names
        return func
    return decorator

def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures all column names are standard lowercase OHLCV."""
    # Handle multi-index if passed raw from yfinance
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    
    required = ['open', 'high', 'low', 'close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required OHLC columns: {missing}")
        
    return df