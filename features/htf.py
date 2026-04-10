import pandas as pd
import numpy as np
from features.decorators import provides_features

@provides_features('pdh', 'pdl', 'dist_to_pdh', 'dist_to_pdl', 'pdh_sweep', 'pdl_sweep')
def add_daily_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает пулы ликвидности старшего таймфрейма (Previous Daily High / Low)
    с учетом закрытия Нью-Йорка (New York Close, 22:00 UTC).
    """
    if df.empty: return df

    # ХАК ДЛЯ FOREX: Сдвигаем время на +2 часа. 
    # Тогда 22:00 UTC становится 00:00 следующего дня, и стандартный .date работает идеально.
    shifted_idx = df.index + pd.Timedelta(hours=2)
    
    daily_highs = df.groupby(shifted_idx.date)['high'].max()
    daily_lows = df.groupby(shifted_idx.date)['low'].min()
    
    # Сдвигаем на 1 период для получения ПРЕДЫДУЩИХ экстремумов
    pdh = daily_highs.shift(1)
    pdl = daily_lows.shift(1)
    
    # Мапим обратно на 15-минутный таймфрейм
    df['pdh'] = pd.Series(shifted_idx.date, index=df.index).map(pdh).ffill()
    df['pdl'] = pd.Series(shifted_idx.date, index=df.index).map(pdl).ffill()
    
    # Расстояние до ликвидности в процентах
    df['dist_to_pdh'] = ((df['pdh'] - df['close']) / df['close']) * 100
    df['dist_to_pdl'] = ((df['close'] - df['pdl']) / df['close']) * 100
    
    # Жесткие флаги пробоя ликвидности для Стратегий
    df['pdh_sweep'] = np.where((df['high'] > df['pdh']) & df['pdh'].notna(), 1, 0)
    df['pdl_sweep'] = np.where((df['low'] < df['pdl']) & df['pdl'].notna(), 1, 0)
    
    return df

@provides_features('tap_4h_bull_fvg', 'tap_4h_bear_fvg', 'recent_tap_4h_bull_fvg', 'recent_tap_4h_bear_fvg')
def add_htf_fvg(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 100: return df

    htf_df = df.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()

    bull_cond = htf_df['high'].shift(2) < htf_df['low']
    bear_cond = htf_df['low'].shift(2) > htf_df['high']

    htf_df['bull_top'] = np.where(bull_cond, htf_df['low'], np.nan)
    htf_df['bull_bot'] = np.where(bull_cond, htf_df['high'].shift(2), np.nan)
    htf_df['bull_ce'] = (htf_df['bull_top'] + htf_df['bull_bot']) / 2

    htf_df['bear_top'] = np.where(bear_cond, htf_df['low'].shift(2), np.nan)
    htf_df['bear_bot'] = np.where(bear_cond, htf_df['high'], np.nan)
    htf_df['bear_ce'] = (htf_df['bear_top'] + htf_df['bear_bot']) / 2

    bull_fvg_id = bull_cond.cumsum()
    bear_fvg_id = bear_cond.cumsum()
    htf_df['bull_fvg_id'] = bull_fvg_id
    htf_df['bear_fvg_id'] = bear_fvg_id

    for col in ['bull_top', 'bull_bot', 'bull_ce', 'bear_top', 'bear_bot', 'bear_ce']:
        htf_df[col] = htf_df[col].ffill()

    bull_invalid = htf_df['close'] < htf_df['bull_ce']
    bear_invalid = htf_df['close'] > htf_df['bear_ce']

    bull_is_killed = bull_invalid.groupby(bull_fvg_id).cummax()
    bear_is_killed = bear_invalid.groupby(bear_fvg_id).cummax()

    htf_df['bull_top'] = np.where(bull_is_killed, np.nan, htf_df['bull_top'])
    htf_df['bear_bot'] = np.where(bear_is_killed, np.nan, htf_df['bear_bot'])

    htf_df = htf_df.shift(1)

    df['4h_bull_top'] = htf_df['bull_top'].reindex(df.index, method='ffill')
    df['4h_bear_bot'] = htf_df['bear_bot'].reindex(df.index, method='ffill')
    df['bull_fvg_id'] = htf_df['bull_fvg_id'].reindex(df.index, method='ffill')
    df['bear_fvg_id'] = htf_df['bear_fvg_id'].reindex(df.index, method='ffill')

    raw_bull_tap = (df['low'] <= df['4h_bull_top']) & df['4h_bull_top'].notna()
    raw_bear_tap = (df['high'] >= df['4h_bear_bot']) & df['4h_bear_bot'].notna()

    current_hour = pd.Series(df.index.floor('h'), index=df.index)
    
    bull_ffilled_hour = current_hour.where(raw_bull_tap).groupby(df['bull_fvg_id']).ffill()
    bear_ffilled_hour = current_hour.where(raw_bear_tap).groupby(df['bear_fvg_id']).ffill()

    df['tap_4h_bull_fvg'] = np.where((bull_ffilled_hour != bull_ffilled_hour.shift(1)) & raw_bull_tap, 1, 0)
    df['tap_4h_bear_fvg'] = np.where((bear_ffilled_hour != bear_ffilled_hour.shift(1)) & raw_bear_tap, 1, 0)

    df['recent_tap_4h_bull_fvg'] = df['tap_4h_bull_fvg'].rolling(window=12, min_periods=1).max().fillna(0)
    df['recent_tap_4h_bear_fvg'] = df['tap_4h_bear_fvg'].rolling(window=12, min_periods=1).max().fillna(0)

    df.drop(columns=['4h_bull_top', '4h_bear_bot', 'bull_fvg_id', 'bear_fvg_id'], inplace=True)
    
    return df

@provides_features('mtfa_score', 'trend_1d', 'trend_4h', 'trend_1h', 'linreg_score')
def add_mtfa_trend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 100: return df

    def _calc_tf_state(tf_string: str, bos_weight: int, fvg_weight: int, shift_hours: int = 0):
        temp_df = df.copy()
        
        # Смещение индекса для настройки кастомного начала суток (например, NY Close)
        if shift_hours != 0:
            temp_df.index = temp_df.index + pd.Timedelta(hours=shift_hours)

        tf_df = temp_df.resample(tf_string).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()

        # Возвращаем оригинальное время после группировки
        if shift_hours != 0:
            tf_df.index = tf_df.index - pd.Timedelta(hours=shift_hours)

        tr = np.maximum(
            tf_df['high'] - tf_df['low'],
            np.maximum(
                abs(tf_df['high'] - tf_df['close'].shift(1)),
                abs(tf_df['low'] - tf_df['close'].shift(1))
            )
        )
        tr_pct = (tr / tf_df['close']) * 100
        atr = tr_pct.rolling(10, min_periods=1).mean()

        swing_high = np.where(
            (tf_df['high'].shift(2) > tf_df['high'].shift(4)) &
            (tf_df['high'].shift(2) > tf_df['high'].shift(3)) &
            (tf_df['high'].shift(2) > tf_df['high'].shift(1)) &
            (tf_df['high'].shift(2) > tf_df['high']),
            tf_df['high'].shift(2), np.nan
        )
        swing_low = np.where(
            (tf_df['low'].shift(2) < tf_df['low'].shift(4)) &
            (tf_df['low'].shift(2) < tf_df['low'].shift(3)) &
            (tf_df['low'].shift(2) < tf_df['low'].shift(1)) &
            (tf_df['low'].shift(2) < tf_df['low']),
            tf_df['low'].shift(2), np.nan
        )
        
        tf_df['last_high'] = pd.Series(swing_high, index=tf_df.index).ffill()
        tf_df['last_low'] = pd.Series(swing_low, index=tf_df.index).ffill()

        candle_size_pct = ((tf_df['high'] - tf_df['low']) / tf_df['low']) * 100
        displacement_cond = candle_size_pct > (atr * 1.5)
        
        is_bos_long = (tf_df['close'] > tf_df['last_high']) & displacement_cond
        is_bos_short = (tf_df['close'] < tf_df['last_low']) & displacement_cond

        tf_df['bos_state'] = np.select(
            [is_bos_long, is_bos_short], 
            [bos_weight, -bos_weight], 
            default=np.nan
        )
        tf_df['bos_state'] = tf_df['bos_state'].ffill().fillna(0)

        is_bull_fvg = tf_df['low'] > tf_df['high'].shift(2)
        is_bear_fvg = tf_df['high'] < tf_df['low'].shift(2)

        tf_df['fvg_state'] = np.select(
            [is_bull_fvg, is_bear_fvg], 
            [fvg_weight, -fvg_weight], 
            default=np.nan
        )
        tf_df['fvg_state'] = tf_df['fvg_state'].ffill().fillna(0)

        tf_df['tf_score'] = tf_df['bos_state'] + tf_df['fvg_state']
        
        return tf_df['tf_score'].shift(1)

    # 1D: Max 35 (Сдвиг на +2 часа для точного выравнивания по NY Close)
    score_1d = _calc_tf_state('1D', bos_weight=25, fvg_weight=10, shift_hours=2)
    score_4h = _calc_tf_state('4h', bos_weight=15, fvg_weight=10)
    score_1h = _calc_tf_state('1h', bos_weight=10, fvg_weight=5)

    df['trend_1d'] = score_1d.reindex(df.index, method='ffill').fillna(0)
    df['trend_4h'] = score_4h.reindex(df.index, method='ffill').fillna(0)
    df['trend_1h'] = score_1h.reindex(df.index, method='ffill').fillna(0)

    # === TRUE LINEAR REGRESSION (16 bars = 4 hours) ===
    n = 16
    x = np.arange(n)
    x_mean = x.mean()
    x_var = ((x - x_mean)**2).sum()

    # Оптимизированный расчет наклона (Slope) 
    def calc_slope(y):
        return ((x - x_mean) * (y - y.mean())).sum() / x_var
        
    slope = df['close'].rolling(window=n).apply(calc_slope, raw=True)
    
    # Переводим наклон в % изменения на 1 бар
    pct_slope = slope / df['close']
    
    # Нормализуем: считаем моментум в ~0.0625% на бар максимальным (оценка в 25 баллов)
    normalized_linreg = (pct_slope / 0.000625) * 25
    df['linreg_score'] = normalized_linreg.clip(lower=-25, upper=25).fillna(0)

    # === ФИНАЛЬНЫЙ СКОРИНГ ===
    df['mtfa_score'] = df['trend_1d'] + df['trend_4h'] + df['trend_1h'] + df['linreg_score']
    df['mtfa_score'] = df['mtfa_score'].round(0)

    return df

@provides_features('pwh', 'pwl', 'dist_to_pwh', 'dist_to_pwl', 'is_eqh', 'is_eql')
def add_advanced_liquidity_and_eq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает недельные пулы ликвидности (Previous Weekly High/Low) 
    и детектирует компрессию ликвидности (Equal Highs / Equal Lows).
    """
    if df.empty or len(df) < 200: return df

    # === 1. PREVIOUS WEEKLY HIGH / LOW (PWH / PWL) ===
    # Сдвигаем на +2 часа для выравнивания по New York Close (как мы делали с дневками)
    shifted_idx = df.index + pd.Timedelta(hours=2)
    
    # Ресемплим по неделям (окончание недели - пятница)
    weekly_highs = df.groupby(pd.Grouper(freq='W-FRI'))['high'].max().shift(1)
    weekly_lows = df.groupby(pd.Grouper(freq='W-FRI'))['low'].min().shift(1)
    
    # Мапим недельные уровни обратно на 15m таймфрейм
    # Используем 'ffill', чтобы уровень держался всю следующую неделю
    df['pwh'] = df.index.to_series().apply(lambda x: weekly_highs.asof(x + pd.Timedelta(hours=2)))
    df['pwl'] = df.index.to_series().apply(lambda x: weekly_lows.asof(x + pd.Timedelta(hours=2)))
    
    df['dist_to_pwh'] = ((df['pwh'] - df['close']) / df['close']) * 100
    df['dist_to_pwl'] = ((df['close'] - df['pwl']) / df['close']) * 100

    # === 2. EQUAL HIGHS / EQUAL LOWS (EQH / EQL) DETECTOR ===
    # Находим локальные экстремумы (свинги) за последние 24 бара (6 часов)
    rolling_max = df['high'].rolling(window=24, min_periods=5).max()
    rolling_min = df['low'].rolling(window=24, min_periods=5).min()

    # Порог "равенства" уровней для GBPUSD (например, 2 пипса = 0.0002)
    pips_threshold = 0.0002

    # Условие EQH: Текущий High почти равен локальному максимуму (но не пробивает его сильно)
    # Это означает, что цена "ударилась" в ту же стену
    is_near_max = abs(df['high'] - rolling_max.shift(1)) <= pips_threshold
    # Дополнительное условие: это действительно локальный пик (свеча закрылась ниже)
    is_rejection_high = df['close'] < df['high'] - (df['high'] - df['low']) * 0.3

    df['is_eqh'] = np.where(is_near_max & is_rejection_high, 1, 0)

    # Аналогично для EQL
    is_near_min = abs(df['low'] - rolling_min.shift(1)) <= pips_threshold
    is_rejection_low = df['close'] > df['low'] + (df['high'] - df['low']) * 0.3

    df['is_eql'] = np.where(is_near_min & is_rejection_low, 1, 0)

    # Чтобы ИИ понимал, что EQH/EQL "активен" какое-то время, делаем затухающий сигнал (decay)
    # Сигнал держится 8 баров (2 часа), постепенно снижаясь
    df['eqh_active'] = df['is_eqh'].rolling(window=8, min_periods=1).max()
    df['eql_active'] = df['is_eql'].rolling(window=8, min_periods=1).max()

    return df

@provides_features('transfer_entropy_proxy')
def add_transfer_entropy(df: pd.DataFrame, htf_window: int = 4, lookback: int = 50) -> pd.DataFrame:
    """
    #6: Cross-Regime Information Flow (Transfer Entropy Proxy).
    Строгий расчет Transfer Entropy вычислительно очень тяжел. 
    Здесь мы используем прокси: насколько прошлый импульс старшего ТФ (например, 1H = 4 свечи) 
    предсказывает/коррелирует с текущей дисперсией младшего ТФ (15m).
    Высокая корреляция = младший таймфрейм жестко подчинен старшему.
    """
    if df.empty: return df
    
    # Доходность 15-минуток
    ltf_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    
    # Доходность старшего таймфрейма (например, 1H = сумма 4-х 15m доходностей)
    htf_returns = ltf_returns.rolling(window=htf_window).sum()
    
    # Сдвигаем HTF на 1 шаг назад, чтобы смотреть, как ПРОШЛЫЙ час влияет на ТЕКУЩИЕ 15 минут
    htf_lagged = htf_returns.shift(htf_window)
    
    # Скользящая корреляция (насколько 15m свечи следуют за вектором прошлого часа)
    # Используем формулу Пирсона для rolling окна
    rolling_cov = ltf_returns.rolling(window=lookback).cov(htf_lagged)
    rolling_std_ltf = ltf_returns.rolling(window=lookback).std()
    rolling_std_htf = htf_lagged.rolling(window=lookback).std()
    
    df['transfer_entropy_proxy'] = rolling_cov / (rolling_std_ltf * rolling_std_htf + 1e-9)
    return df

@provides_features('vol_term_structure')
def add_volatility_term_structure(df: pd.DataFrame, ltf_period: int = 14, htf_period: int = 56) -> pd.DataFrame:
    """
    #21: Volatility Term Structure Curvature Change.
    Сравнивает волатильность младшего ТФ (например, 14 баров = 3.5 часа) 
    с волатильностью старшего ТФ (56 баров = 14 часов).
    Если кривая инвертируется (LTF Vol > HTF Vol), это признак внезапного шока/смены режима.
    """
    if df.empty: return df
    
    def calc_atr(high, low, close, period):
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        return tr.rolling(window=period).mean()

    ltf_atr = calc_atr(df['high'], df['low'], df['close'], ltf_period)
    htf_atr = calc_atr(df['high'], df['low'], df['close'], htf_period)
    
    # Отношение: > 1 означает, что в моменте рынок аномально волатильнее, чем его средний фон
    df['vol_term_structure'] = ltf_atr / (htf_atr + 1e-9)
    return df

@provides_features('ob_migration_velocity')
def add_ob_migration_velocity(df: pd.DataFrame, window: int = 96) -> pd.DataFrame:
    """
    #20: Order Block Migration Velocity.
    Считает скорость (в пунктах за свечу), с которой смещаются экстремумы (зоны поддержки/сопротивления).
    Помогает поймать ускорение или замедление глобального макро-тренда.
    window = 96 свечей (1 сутки для 15m графика).
    """
    if df.empty: return df
    
    # Находим текущий локальный максимум и минимум за сутки
    rolling_max = df['high'].rolling(window=window).max()
    rolling_min = df['low'].rolling(window=window).min()
    
    # Находим прошлый экстремум (сутки назад)
    prev_max = rolling_max.shift(window)
    prev_min = rolling_min.shift(window)
    
    # Считаем смещение в % от цены
    max_migration = ((rolling_max - prev_max) / df['close']) * 100
    min_migration = ((rolling_min - prev_min) / df['close']) * 100
    
    # Velocity: берем то смещение, которое сильнее (если растем - max_migration будет > 0)
    # Если оба растут (например, бычий канал), берем среднюю скорость канала
    df['ob_migration_velocity'] = (max_migration + min_migration) / 2.0
    
    return df