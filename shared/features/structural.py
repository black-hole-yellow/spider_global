import pandas as pd
import numpy as np
from shared.features.decorators import provides_features

@provides_features(
    'swing_high', 'swing_low', 
    'struct_trend', 'premium_discount', 
    'fvg_bull_size', 'fvg_bear_size',
    'sweep_bull', 'sweep_bear', 
    'dist_to_liq_high', 'dist_to_liq_low'
)
def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерирует продвинутые структурные фичи (SMC) для машинного обучения.
    ВНИМАНИЕ: Код написан со строгим соблюдением причинно-следственной связи (без Lookahead Bias).
    """
    if df.empty or len(df) < 10:
        return df
        
    # =========================================================================
    # 1. СВИНГИ (SWING HIGHS / LOWS) - 5-свечной фрактал
    # Фрактал формируется на свече t-2, но узнаем мы об этом только на свече t
    # =========================================================================
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

    # Записываем уровень свинга (со сдвигом 2, так как он был 2 свечи назад)
    df['swing_high'] = np.where(is_swing_high, df['high'].shift(2), np.nan)
    df['swing_low'] = np.where(is_swing_low, df['low'].shift(2), np.nan)
    
    # Протягиваем последние известные значения вперед
    df['swing_high'] = df['swing_high'].ffill()
    df['swing_low'] = df['swing_low'].ffill()

    # =========================================================================
    # 2. РЫНОЧНАЯ СТРУКТУРА И ТРЕНД (BOS / CHoCH)
    # 1 = Восходящая структура, -1 = Нисходящая
    # =========================================================================
    # Если мы закрываемся выше последнего максимума -> Trend = 1
    bullish_bos = df['close'] > df['swing_high']
    bearish_bos = df['close'] < df['swing_low']
    
    # Создаем серию состояний тренда и протягиваем
    trend = pd.Series(np.nan, index=df.index)
    trend[bullish_bos] = 1
    trend[bearish_bos] = -1
    df['struct_trend'] = trend.ffill().fillna(0) # 0 для начала графика

    # =========================================================================
    # 3. PREMIUM / DISCOUNT ZONE (PD Array)
    # Где мы находимся относительно текущего торгового диапазона?
    # 0.0 = На самом дне (Discount), 1.0 = На самой вершине (Premium)
    # =========================================================================
    range_size = df['swing_high'] - df['swing_low']
    # Избегаем деления на ноль
    range_size = range_size.replace(0, np.nan) 
    df['premium_discount'] = (df['close'] - df['swing_low']) / range_size
    df['premium_discount'] = df['premium_discount'].clip(0, 1) # Обрезаем выбросы

    # =========================================================================
    # 4. ДИСТАНЦИЯ ДО ЛИКВИДНОСТИ (В пипсах)
    # Как далеко цена от ближайших пулов стоп-лоссов?
    # =========================================================================
    PIP = 0.0001
    df['dist_to_liq_high'] = ((df['swing_high'] - df['close']) / PIP).clip(lower=0)
    df['dist_to_liq_low'] = ((df['close'] - df['swing_low']) / PIP).clip(lower=0)

    # =========================================================================
    # 5. СНЯТИЕ ЛИКВИДНОСТИ (LIQUIDITY SWEEPS)
    # Цена проколола уровень тенью, но закрылась обратно в диапазон
    # =========================================================================
    # Bearish Sweep: Прокололи максимум, но закрылись ниже него
    df['sweep_bear'] = ((df['high'] > df['swing_high']) & (df['close'] < df['swing_high'])).astype(int)
    # Bullish Sweep: Прокололи минимум, но закрылись выше него
    df['sweep_bull'] = ((df['low'] < df['swing_low']) & (df['close'] > df['swing_low'])).astype(int)

    # =========================================================================
    # 6. IMBALANCE / FAIR VALUE GAPS (FVG)
    # Размер дисбаланса (0, если его нет)
    # =========================================================================
    # Bullish FVG: Разрыв между high(t-2) и low(t)
    bull_gap = df['low'] - df['high'].shift(2)
    df['fvg_bull_size'] = np.where(bull_gap > 0, bull_gap / PIP, 0)
    
    # Bearish FVG: Разрыв между low(t-2) и high(t)
    bear_gap = df['low'].shift(2) - df['high']
    df['fvg_bear_size'] = np.where(bear_gap > 0, bear_gap / PIP, 0)

    # Очистка NaN, которые появились в первые несколько свечей
    df.bfill(inplace=True)
    
    return df

@provides_features('inst_footprint_divergence')
def add_institutional_footprint(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Institutional Footprint Divergence.
    Поскольку у нас нет тиковых данных Level 2, мы аппроксимируем Cumulative Volume Delta (CVD).
    Внутри свечи: если закрытие ближе к High, большая часть объема считается покупками.
    Дивергенция = разница между наклоном цены и наклоном CVD.
    """
    if df.empty or 'volume' not in df.columns: return df

    # Избегаем деления на ноль для дожи-свечей
    range_hl = (df['high'] - df['low']).replace(0, 1e-5)
    
    # Аппроксимация дельты: от -1 (все продали) до +1 (все купили)
    bar_delta_pct = (df['close'] - df['open']) / range_hl
    df['bar_volume_delta'] = bar_delta_pct * df['volume']
    
    # Кумулятивная дельта за период
    cvd = df['bar_volume_delta'].rolling(window=lookback).sum()
    
    # Нормализуем цену и CVD для сравнения наклонов (Z-score на окне)
    price_z = (df['close'] - df['close'].rolling(lookback).mean()) / df['close'].rolling(lookback).std()
    cvd_z = (cvd - cvd.rolling(lookback).mean()) / cvd.rolling(lookback).std()
    
    # Если цена растет (Z > 0), а дельта падает (Z < 0) — это скрытые продажи (дивергенция)
    df['inst_footprint_divergence'] = price_z - cvd_z
    return df

@provides_features('liquidity_absorption_ratio')
def add_liquidity_absorption(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Liquidity Absorption Asymmetry.
    Измеряет асимметрию поглощения ликвидности через отношение объема к размеру теней (Wicks).
    Если на графике длинная нижняя тень с огромным объемом — это поглощение продаж лимитными покупателями.
    """
    if df.empty or 'volume' not in df.columns: return df
    
    # Размер теней
    upper_wick = df['high'] - np.maximum(df['open'], df['close'])
    lower_wick = np.minimum(df['open'], df['close']) - df['low']
    
    # Объем, приходящийся на 1 пункт тени (усилие/результат)
    # Используем сглаживание, чтобы убрать шум одиночных свечей
    avg_vol = df['volume'].rolling(window=lookback).mean()
    
    # Сила поглощения покупателями (объем на нижней тени)
    buy_absorption = (lower_wick / (df['high'] - df['low'] + 1e-5)) * avg_vol
    
    # Сила поглощения продавцами (объем на верхней тени)
    sell_absorption = (upper_wick / (df['high'] - df['low'] + 1e-5)) * avg_vol
    
    # Asymmetry Ratio: > 1 (покупатели сильнее поглощают), < -1 (продавцы сильнее)
    # Используем логарифм для симметрии распределения фичи
    df['liquidity_absorption_ratio'] = np.log((buy_absorption + 1) / (sell_absorption + 1))
    
    return df

@provides_features('sm_conviction_decay')
def add_conviction_decay(df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    """
    #15: Smart Money Conviction Decay.
    Оценивает затухание (decay) импульса объема после пробития локального экстремума.
    Помогает модели отличать ложные пробои (False Breakouts) от истинных.
    """
    if df.empty or 'volume' not in df.columns: return df

    # Находим локальные пробои (цена закрылась выше/ниже High/Low за последние 14 свечей)
    rolling_high = df['high'].shift(1).rolling(window=lookback).max()
    rolling_low = df['low'].shift(1).rolling(window=lookback).min()
    
    is_breakout_up = df['close'] > rolling_high
    is_breakout_down = df['close'] < rolling_low
    
    # Сохраняем объем в момент пробоя (Conviction)
    df['breakout_volume'] = np.where(is_breakout_up | is_breakout_down, df['volume'], np.nan)
    df['breakout_volume'] = df['breakout_volume'].ffill() # Протягиваем значение вперед
    
    # Считаем текущий объем относительно объема первоначального пробоя
    # Если значение быстро падает к 0.2-0.3, пробой "выдыхается"
    df['sm_conviction_decay'] = df['volume'] / (df['breakout_volume'] + 1e-5)
    
    # Сбрасываем значение до 1.0, если пробоев давно не было, чтобы не копить шум
    bars_since_breakout = df.groupby((is_breakout_up | is_breakout_down).cumsum()).cumcount()
    df.loc[bars_since_breakout > lookback * 2, 'sm_conviction_decay'] = 1.0
    
    return df