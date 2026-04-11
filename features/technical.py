import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import warnings

warnings.filterwarnings('ignore')

def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-lag logarithmic returns."""
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    return df

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Optimized: Vectorized True Range and Wilder's Smoothing in Percentage."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Transform True Range into a percentage of current price
    true_range_pct = (true_range / df['close']) * 100
    df['atr'] = true_range_pct.ewm(alpha=1/period, adjust=False).mean()
    return df

def add_volatility_zscore(df: pd.DataFrame, period: int = 100) -> pd.DataFrame:
    """Measures current ATR against its historical baseline."""
    atr_mean = df['atr'].rolling(window=period).mean()
    atr_std = df['atr'].rolling(window=period).std()
    df['volatility_zscore'] = (df['atr'] - atr_mean) / (atr_std + 1e-9)
    return df

def add_body_zscore(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """Measures the size of the candle body relative to recent history."""
    body_pct = (np.abs(df['close'] - df['open']) / df['open']) * 100
    rolling_mean = body_pct.rolling(window=lookback).mean()
    rolling_std = body_pct.rolling(window=lookback).std()
    
    df['body_zscore'] = (body_pct - rolling_mean) / (rolling_std + 1e-9)
    return df

def add_normalized_slope(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Calculates Linear Regression slope normalized by ATR."""
    x = np.arange(lookback)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    
    def calc_slope(y):
        y_mean = y.mean()
        covariance = ((x - x_mean) * (y - y_mean)).sum()
        return covariance / x_var

    raw_slope = df['close'].rolling(window=lookback).apply(calc_slope, raw=True)
    slope_pct = (raw_slope / df['close']) * 100 
    
    df['norm_slope'] = slope_pct / (df['atr'] + 1e-9)
    return df

def add_markov_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Simple heuristic regime identifier combining volatility and trend."""
    atr_baseline = df['atr'].rolling(window=50).mean()
    vol_state = np.where(df['atr'] > atr_baseline, 'High Vol', 'Low Vol')
    trend_state = np.where(df['norm_slope'] > 0, 'Bullish', 'Bearish')
    
    df['markov_regime'] = pd.Series(vol_state, index=df.index) + " / " + pd.Series(trend_state, index=df.index)
    df.loc[df['atr'].isna() | df['norm_slope'].isna(), 'markov_regime'] = np.nan
    df['markov_regime'] = df['markov_regime'].ffill()
    return df

def add_hmm_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Probabilistic Volatility Regime detection using Gaussian HMM."""
    train_data = df[['log_return']].dropna()
    if len(train_data) < 50:
        df['high_vol_prob'] = 0.0
        return df

    X = train_data.values
    
    try:
        model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(X)
        hidden_states_probs = model.predict_proba(X)
        
        var_state_0 = model.covars_[0][0]
        var_state_1 = model.covars_[1][0]
        high_vol_state = 0 if var_state_0 > var_state_1 else 1
        
        df['high_vol_prob'] = np.nan
        df.loc[train_data.index, 'high_vol_prob'] = hidden_states_probs[:, high_vol_state]
    except Exception:
        df['high_vol_prob'] = 0.0
        
    df['high_vol_prob'] = df['high_vol_prob'].ffill().fillna(0)
    return df

def add_algo_vol_crush(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies rapid volatility spikes followed by immediate weakness."""
    atr_spike = df['atr'] > df['atr'].rolling(50).mean() * 2
    close_weakness = df['close'] < df['open']
    
    df['algo_vol_crush_short'] = (atr_spike & close_weakness).astype(int)
    return df

def add_ifvg_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Inversion Fair Value Gaps (iFVG)."""
    bear_fvg_cond = df['low'].shift(2) > df['high']
    bull_fvg_cond = df['high'].shift(2) < df['low']

    df['bear_fvg_top'] = np.where(bear_fvg_cond, df['low'].shift(2), np.nan)
    df['bear_fvg_top'] = df['bear_fvg_top'].ffill()

    df['bull_fvg_bottom'] = np.where(bull_fvg_cond, df['high'].shift(2), np.nan)
    df['bull_fvg_bottom'] = df['bull_fvg_bottom'].ffill()

    prev_close = df['close'].shift(1)
    curr_close = df['close']

    df['ifvg_long'] = np.where((prev_close <= df['bear_fvg_top']) & (curr_close > df['bear_fvg_top']), 1, 0)
    df['ifvg_short'] = np.where((prev_close >= df['bull_fvg_bottom']) & (curr_close < df['bull_fvg_bottom']), 1, 0)

    df.drop(columns=['bear_fvg_top', 'bull_fvg_bottom'], inplace=True)
    return df

def add_m15_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Break of Structure (BOS) and structural stops via Williams Fractals."""
    swing_high_level = np.where(
        (df['high'].shift(2) > df['high'].shift(4)) &
        (df['high'].shift(2) > df['high'].shift(3)) &
        (df['high'].shift(2) > df['high'].shift(1)) &
        (df['high'].shift(2) > df['high']),
        df['high'].shift(2), np.nan
    )
    
    swing_low_level = np.where(
        (df['low'].shift(2) < df['low'].shift(4)) &
        (df['low'].shift(2) < df['low'].shift(3)) &
        (df['low'].shift(2) < df['low'].shift(1)) &
        (df['low'].shift(2) < df['low']),
        df['low'].shift(2), np.nan
    )

    df['last_swing_high'] = pd.Series(swing_high_level, index=df.index).ffill()
    df['last_swing_low'] = pd.Series(swing_low_level, index=df.index).ffill()

    df['bos_long'] = np.where((df['close'].shift(1) <= df['last_swing_high']) & (df['close'] > df['last_swing_high']), 1, 0)
    df['bos_short'] = np.where((df['close'].shift(1) >= df['last_swing_low']) & (df['close'] < df['last_swing_low']), 1, 0)

    df['structural_low'] = df['low'].rolling(window=24, min_periods=1).min()
    df['structural_high'] = df['high'].rolling(window=24, min_periods=1).max()

    df.drop(columns=['last_swing_high', 'last_swing_low'], inplace=True)
    return df

# ==========================================
# 🚀 MASTER WRAPPER (LAYER 1)
# ==========================================
def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point for Layer 1 features.
    Strict execution order prevents dependency errors.
    """
    if df.empty or len(df) < 5:
        return df

    # 1. Base Variables (Required for downstream math)
    df = add_log_returns(df)
    df = add_atr(df, period=14)
    
    # 2. Oscillators & Z-Scores
    df = add_volatility_zscore(df, period=100)
    df = add_body_zscore(df, lookback=50)
    df = add_normalized_slope(df, lookback=20)
    
    # 3. Market Regimes
    df = add_markov_regime(df)
    df = add_hmm_volatility_regime(df)
    df = add_algo_vol_crush(df)
    
    # 4. Structural Logic
    df = add_ifvg_signals(df)
    df = add_m15_structure(df)
    
    return df

