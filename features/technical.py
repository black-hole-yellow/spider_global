import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from features.decorators import provides_features, validate_ohlcv

@provides_features('atr')
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Optimized: Vectorized True Range and Wilder's Smoothing in Percentage."""
    if df.empty: return df
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Перевод True Range в проценты от текущей цены
    true_range_pct = (true_range / df['close']) * 100
    
    df['atr'] = true_range_pct.ewm(alpha=1/period, adjust=False).mean()
    return df

@provides_features('rsi')
def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Optimized: Vectorized RSI calculation without loops."""
    if df.empty: return df
    
    delta = df['close'].diff()
    
    # Vectorized separation of gains and losses
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    # Vectorized rolling averages
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

@provides_features('volatility_zscore')
def calculate_volatility_zscore(df: pd.DataFrame, period: int = 100) -> pd.DataFrame:
    """Optimized: Used for Market Regime detection."""
    if df.empty or 'atr' not in df.columns: 
        return df
        
    atr_mean = df['atr'].rolling(window=period).mean()
    atr_std = df['atr'].rolling(window=period).std()
    
    df['volatility_zscore'] = (df['atr'] - atr_mean) / atr_std
    
    return df

@provides_features('Log_Return')
def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
    return df

@provides_features('ATR')
def add_atr(df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Перевод в проценты
    tr_pct = (tr / df['close']) * 100
    df['ATR'] = tr_pct.rolling(window=lookback).mean()
    return df

@provides_features('Body_ZScore')
def add_volatility_zscore(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    # Размер тела свечи в процентах
    body_pct = (abs(df['close'] - df['open']) / df['open']) * 100
    rolling_mean = body_pct.rolling(window=lookback).mean()
    rolling_std = body_pct.rolling(window=lookback).std()
    
    df['Body_ZScore'] = (body_pct - rolling_mean) / rolling_std
    return df

@provides_features('Norm_Slope')
def add_normalized_slope(df: pd.DataFrame, lookback: int = 20, atr_lookback: int = 14) -> pd.DataFrame:
    if 'ATR' not in df.columns: df = add_atr(df, atr_lookback)
    
    x = np.arange(lookback)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    
    def calc_slope(y):
        y_mean = y.mean()
        covariance = ((x - x_mean) * (y - y_mean)).sum()
        return covariance / x_var

    raw_slope = df['close'].rolling(window=lookback).apply(calc_slope, raw=True)
    
    # Так как ATR теперь в %, наклон тоже нужно перевести в %-изменение за бар
    slope_pct = (raw_slope / df['close']) * 100 
    
    df['Norm_Slope'] = slope_pct / df['ATR']
    return df

@provides_features('Markov_Regime')
def add_markov_regime(df: pd.DataFrame) -> pd.DataFrame:
    if 'ATR' not in df.columns or 'Norm_Slope' not in df.columns:
        raise ValueError("ATR and Norm_Slope must be generated before Markov Regime.")
    
    atr_baseline = df['ATR'].rolling(window=50).mean()
    vol_state = np.where(df['ATR'] > atr_baseline, 'High Vol', 'Low Vol')
    trend_state = np.where(df['Norm_Slope'] > 0, 'Bullish', 'Bearish')
    
    df['Markov_Regime'] = pd.Series(vol_state, index=df.index) + " / " + pd.Series(trend_state, index=df.index)
    df.loc[df['ATR'].isna() | df['Norm_Slope'].isna(), 'Markov_Regime'] = np.nan
    df['Markov_Regime'] = df['Markov_Regime'].ffill()
    return df

@provides_features('High_Vol_Prob')
def add_hmm_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    if 'Log_Return' not in df.columns:
        df = add_log_returns(df)
        
    train_data = df[['Log_Return']].dropna()
    X = train_data.values
    
    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
    model.fit(X)
    hidden_states_probs = model.predict_proba(X)
    
    var_state_0 = model.covars_[0][0]
    var_state_1 = model.covars_[1][0]
    high_vol_state = 0 if var_state_0 > var_state_1 else 1
    
    df['High_Vol_Prob'] = np.nan
    df.loc[train_data.index, 'High_Vol_Prob'] = hidden_states_probs[:, high_vol_state]
    df['High_Vol_Prob'] = df['High_Vol_Prob'].ffill().fillna(0)
    return df

@provides_features('Algo_Vol_Crush_Short')
def add_algo_vol_crush(df: pd.DataFrame) -> pd.DataFrame:
    # Assumes ATR is generated. Identifies when volatility spikes rapidly then dies.
    if 'ATR' not in df.columns:
        from features.technical import add_atr
        df = add_atr(df)
        
    atr_spike = df['ATR'] > df['ATR'].rolling(50).mean() * 2
    close_weakness = df['close'] < df['open']
    
    df['Algo_Vol_Crush_Short'] = (atr_spike & close_weakness).astype(int)
    return df

@provides_features('ifvg_long', 'ifvg_short')
def add_ifvg_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает Inversion Fair Value Gap (iFVG).
    Пробой телом (Close) за пределы противоположного FVG.
    """
    if df.empty or len(df) < 5: return df

    # 1. Поиск разрывов (FVG)
    bear_fvg_cond = df['low'].shift(2) > df['high']
    bull_fvg_cond = df['high'].shift(2) < df['low']

    # 2. Запоминаем уровни FVG (Верх медвежьего, Низ бычьего)
    df['bear_fvg_top'] = np.where(bear_fvg_cond, df['low'].shift(2), np.nan)
    df['bear_fvg_top'] = df['bear_fvg_top'].ffill()

    df['bull_fvg_bottom'] = np.where(bull_fvg_cond, df['high'].shift(2), np.nan)
    df['bull_fvg_bottom'] = df['bull_fvg_bottom'].ffill()

    # 3. iFVG: Закрытие текущей свечи за пределами актуального разрыва
    prev_close = df['close'].shift(1)
    curr_close = df['close']

    # iFVG Long: Была под FVG, а закрылась выше его верха
    df['ifvg_long'] = np.where((prev_close <= df['bear_fvg_top']) & (curr_close > df['bear_fvg_top']), 1, 0)
    
    # iFVG Short: Была над FVG, а закрылась ниже его низа
    df['ifvg_short'] = np.where((prev_close >= df['bull_fvg_bottom']) & (curr_close < df['bull_fvg_bottom']), 1, 0)

    # Уборка мусора из памяти
    df.drop(columns=['bear_fvg_top', 'bull_fvg_bottom'], inplace=True)
    return df

@provides_features('bos_long', 'bos_short', 'structural_low', 'structural_high')
def add_m15_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает Break of Structure (BOS) по фракталам Вильямса 
    и находит экстремумы для структурного стоп-лосса.
    """
    if df.empty or len(df) < 5: return df

    # 1. Фракталы Вильямса (без заглядывания в будущее!)
    # Фрактал формируется на свече i-2, но узнаем мы о нем только на текущей свече i
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

    # 2. BOS: Закрытие телом выше последнего локального максимума/минимума
    df['bos_long'] = np.where((df['close'].shift(1) <= df['last_swing_high']) & (df['close'] > df['last_swing_high']), 1, 0)
    df['bos_short'] = np.where((df['close'].shift(1) >= df['last_swing_low']) & (df['close'] < df['last_swing_low']), 1, 0)

    # 3. Dynamic Lookback для Стоп-лосса (Ищем экстремум за последние 6 часов / 24 свечи)
    df['structural_low'] = df['low'].rolling(window=24, min_periods=1).min()
    df['structural_high'] = df['high'].rolling(window=24, min_periods=1).max()

    df.drop(columns=['last_swing_high', 'last_swing_low'], inplace=True)
    return df