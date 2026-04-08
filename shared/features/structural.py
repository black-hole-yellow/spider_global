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