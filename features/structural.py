import pandas as pd
import numpy as np

# =========================================================================
# 🏗️ MASTER WRAPPER
# =========================================================================
def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master entry point for Structural Features (SMC/Order Flow).
    Ensures strict causal execution order without lookahead bias.
    """
    if df.empty or len(df) < 10:
        return df
        
    # 1. Base SMC Concepts
    df = _add_swings(df)
    df = _add_trend(df)
    df = _add_pd_arrays(df)
    df = _add_liquidity_sweeps(df)
    df = _add_fvg_size(df)
    
    # 2. Advanced Order Flow & Absorption
    df = _add_institutional_footprint(df)
    df = _add_liquidity_absorption(df)
    df = _add_conviction_decay(df)
    
    # 3. Cleanup initial NaNs safely
    df.bfill(inplace=True)
    
    return df

# =========================================================================
# 🧩 INTERNAL HELPERS (Base SMC)
# =========================================================================

def _add_swings(df: pd.DataFrame) -> pd.DataFrame:
    """1. СВИНГИ (SWING HIGHS / LOWS) - 5-свечной фрактал"""
    is_swing_high = (
        (df['high'].shift(2) > df['high'].shift(4)) &
        (df['high'].shift(2) > df['high'].shift(3)) &
        (df['high'].shift(2) > df['high'].shift(1)) &
        (df['high'].shift(2) > df['high'])
    )
    
    is_swing_low = (
        (df['low'].shift(2) < df['low'].shift(4)) &
        (df['low'].shift(2) < df['low'].shift(3)) &
        (df['low'].shift(2) < df['low'].shift(1)) &
        (df['low'].shift(2) < df['low'])
    )

    df['swing_high'] = np.where(is_swing_high, df['high'].shift(2), np.nan)
    df['swing_low'] = np.where(is_swing_low, df['low'].shift(2), np.nan)
    df['swing_high'] = df['swing_high'].ffill()
    df['swing_low'] = df['swing_low'].ffill()
    
    return df

def _add_trend(df: pd.DataFrame) -> pd.DataFrame:
    """2. РЫНОЧНАЯ СТРУКТУРА И ТРЕНД (BOS / CHoCH)"""
    bullish_bos = df['close'] > df['swing_high']
    bearish_bos = df['close'] < df['swing_low']
    
    trend = pd.Series(np.nan, index=df.index)
    trend[bullish_bos] = 1
    trend[bearish_bos] = -1
    df['struct_trend'] = trend.ffill().fillna(0)
    
    return df

def _add_pd_arrays(df: pd.DataFrame) -> pd.DataFrame:
    """3. PREMIUM / DISCOUNT ZONE (PD Array)"""
    range_size = df['swing_high'] - df['swing_low']
    range_size = range_size.replace(0, np.nan) 
    
    df['premium_discount'] = (df['close'] - df['swing_low']) / range_size
    df['premium_discount'] = df['premium_discount'].clip(0, 1) 
    
    return df

def _add_liquidity_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    """4 & 5. ДИСТАНЦИЯ ДО ЛИКВИДНОСТИ И СНЯТИЕ (SWEEPS)"""
    PIP = 0.0001
    df['dist_to_liq_high'] = ((df['swing_high'] - df['close']) / PIP).clip(lower=0)
    df['dist_to_liq_low'] = ((df['close'] - df['swing_low']) / PIP).clip(lower=0)

    df['sweep_bear'] = ((df['high'] > df['swing_high']) & (df['close'] < df['swing_high'])).astype(int)
    df['sweep_bull'] = ((df['low'] < df['swing_low']) & (df['close'] > df['swing_low'])).astype(int)
    
    return df

def _add_fvg_size(df: pd.DataFrame) -> pd.DataFrame:
    """6. IMBALANCE / FAIR VALUE GAPS (FVG)"""
    PIP = 0.0001
    bull_gap = df['low'] - df['high'].shift(2)
    df['fvg_bull_size'] = np.where(bull_gap > 0, bull_gap / PIP, 0)
    
    bear_gap = df['low'].shift(2) - df['high']
    df['fvg_bear_size'] = np.where(bear_gap > 0, bear_gap / PIP, 0)
    
    return df

# =========================================================================
# 🧩 INTERNAL HELPERS (Advanced Order Flow)
# =========================================================================

def _add_institutional_footprint(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Institutional Footprint Divergence (CVD proxy)"""
    if 'volume' not in df.columns: return df

    range_hl = (df['high'] - df['low']).replace(0, 1e-5)
    bar_delta_pct = (df['close'] - df['open']) / range_hl
    df['bar_volume_delta'] = bar_delta_pct * df['volume']
    
    cvd = df['bar_volume_delta'].rolling(window=lookback).sum()
    
    price_z = (df['close'] - df['close'].rolling(lookback).mean()) / df['close'].rolling(lookback).std()
    cvd_z = (cvd - cvd.rolling(lookback).mean()) / cvd.rolling(lookback).std()
    
    df['inst_footprint_divergence'] = price_z - cvd_z
    return df

def _add_liquidity_absorption(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """Liquidity Absorption Asymmetry"""
    if 'volume' not in df.columns: return df
    
    upper_wick = df['high'] - np.maximum(df['open'], df['close'])
    lower_wick = np.minimum(df['open'], df['close']) - df['low']
    
    avg_vol = df['volume'].rolling(window=lookback).mean()
    
    buy_absorption = (lower_wick / (df['high'] - df['low'] + 1e-5)) * avg_vol
    sell_absorption = (upper_wick / (df['high'] - df['low'] + 1e-5)) * avg_vol
    
    df['liquidity_absorption_ratio'] = np.log((buy_absorption + 1) / (sell_absorption + 1))
    return df

def _add_conviction_decay(df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    """Smart Money Conviction Decay"""
    if 'volume' not in df.columns: return df

    rolling_high = df['high'].shift(1).rolling(window=lookback).max()
    rolling_low = df['low'].shift(1).rolling(window=lookback).min()
    
    is_breakout_up = df['close'] > rolling_high
    is_breakout_down = df['close'] < rolling_low
    
    df['breakout_volume'] = np.where(is_breakout_up | is_breakout_down, df['volume'], np.nan)
    df['breakout_volume'] = df['breakout_volume'].ffill() 
    
    df['sm_conviction_decay'] = df['volume'] / (df['breakout_volume'] + 1e-5)
    
    bars_since_breakout = df.groupby((is_breakout_up | is_breakout_down).cumsum()).cumcount()
    df.loc[bars_since_breakout > lookback * 2, 'sm_conviction_decay'] = 1.0
    
    return df