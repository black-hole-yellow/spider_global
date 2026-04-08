import pandas as pd
import numpy as np
from shared.features.decorators import provides_features

@provides_features('log_return', 'volatility_z', 'trend_strength', 'cusum_signal', 'changepoint_prob', 'is_anomaly')
def add_regime_and_changepoint_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ортогональные фичи для ML и Гибридный детектор изломов (Variance Ratio + CUSUM).
    Полностью векторизовано и защищено от look-ahead bias.
    """
    if df.empty or len(df) < 100: return df

    # 1. Zero-lag Returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # 2. Orthogonal Features
    long_window = 96 # 1 день (для 15m)
    
    roll_std = df['log_return'].rolling(long_window).std()
    roll_mean = df['log_return'].rolling(long_window).mean()
    
    df['volatility_z'] = abs(df['log_return'] - roll_mean) / (roll_std + 1e-8)
    
    dir_move = abs(df['close'] - df['close'].shift(long_window))
    tot_move = abs(df['close'] - df['close'].shift(1)).rolling(long_window).sum()
    df['trend_strength'] = dir_move / (tot_move + 1e-8)

    # === 3. HYBRID CHANGEPOINT DETECTION ===
    
    # 3.1 Variance Ratio (Краткосрок против Долгосрока)
    short_window = 8 # 2 часа
    short_std = df['log_return'].rolling(short_window).std()
    variance_ratio = (short_std ** 2) / (roll_std ** 2 + 1e-8)
    vr_prob = 1 / (1 + np.exp(-2 * (variance_ratio - 2.5)))

    # 3.2 CUSUM (Cumulative Sum of Volatility Deviations)
    # Настраиваем параметры:
    # k (drift) - допустимое отклонение. Все, что ниже 0.5 Z-score, игнорируем.
    k_drift = 0.5 
    cusum_threshold = 3.0 # Порог срабатывания

    # Вычисляем отклонение текущей волатильности с учетом drift
    deviation = df['volatility_z'] - k_drift
    
    # Векторизованный CUSUM сброс (S_t = max(0, S_{t-1} + X_t) эквивалентно C_t - min(C_i))
    cumulative_deviation = deviation.cumsum()
    running_min = cumulative_deviation.cummin()
    
    # Нормализуем CUSUM, чтобы он не улетал в бесконечность (ограничиваем недавней историей через rolling)
    # Используем окно в 24 бара (6 часов) для поиска локального минимума
    local_min = cumulative_deviation.rolling(window=24, min_periods=1).min()
    cusum_score = cumulative_deviation - local_min

    # Переводим CUSUM скор в вероятность (0...1)
    df['cusum_signal'] = np.clip(cusum_score / cusum_threshold, 0, 1)

    # 3.3 Итоговая вероятность излома (Максимум из двух методов)
    df['changepoint_prob'] = np.maximum(vr_prob, df['cusum_signal'])

    # 4. Мгновенный Kill Switch
    df['is_anomaly'] = np.where((df['volatility_z'] > 4.0) | (df['changepoint_prob'] > 0.90), 1, 0)

    return df.fillna(0)