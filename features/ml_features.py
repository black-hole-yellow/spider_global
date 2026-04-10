import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.mixture import GaussianMixture
from features.decorators import provides_features

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

@provides_features('hurst_exponent', 'fractal_dimension')
def add_hurst_and_fractal(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """
    #7 & #17: Local Hurst Exponent & Fractal Dimension Trajectories.
    Оценивает персистентность рынка:
    H < 0.5 (Mean Reversion / Шум), H > 0.5 (Тренд / Momentum).
    Фрактальная размерность D = 2 - H.
    """
    if df.empty or len(df) < lookback: return df

    # Вспомогательная функция для расчета Херста (Аппроксимация через Variance)
    def get_hurst(prices):
        if len(prices) < 20: return np.nan
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    # Оптимизация: считаем каждую свечу, используя rolling().apply (будет долго на 25 годах, 
    # в проде можно переписать на numpy stride tricks, но pandas надежнее для тестов)
    df['hurst_exponent'] = df['close'].rolling(window=lookback).apply(get_hurst, raw=True)
    
    # Фрактальная размерность временного ряда (1D)
    df['fractal_dimension'] = 2.0 - df['hurst_exponent']
    
    return df

@provides_features('volatility_skewness')
def add_adaptive_volatility_skew(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    #1: Adaptive Fractal Volatility Skewness.
    Измеряет асимметрию кластеров волатильности. 
    Резкий рост skewness означает "панику" (ненормальное распределение диапазонов).
    """
    if df.empty: return df
    
    # Истинный диапазон (True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Скользящая асимметрия (Skewness)
    df['volatility_skewness'] = tr.rolling(window=lookback).apply(skew, raw=True)
    return df

@provides_features('market_efficiency_index')
def add_market_efficiency_index(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    #18: Adaptive Market Efficiency Index.
    Композитный индекс: чем он выше, тем сильнее рынок похож на случайное блуждание (Random Walk).
    Считается как отношение дисперсии за N баров к сумме дисперсий за 1 бар (Variance Ratio).
    """
    if df.empty: return df
    
    returns = np.log(df['close'] / df['close'].shift(1))
    
    # Дисперсия за N баров
    var_n = returns.rolling(window=lookback).sum().var() # Аппроксимация
    
    # Сумма N дисперсий за 1 бар
    var_1 = returns.rolling(window=lookback).var() * lookback
    
    # Variance Ratio (VR) - центрируем вокруг 1
    vr = var_n / (var_1 + 1e-9)
    df['market_efficiency_index'] = np.abs(vr - 1.0)
    return df

@provides_features('regime_trend_prob', 'regime_chop_prob', 'regime_break_prob')
def add_bayesian_regime_probabilities(df: pd.DataFrame, lookback: int = 500) -> pd.DataFrame:
    """
    #12: Bayesian Regime Probability Surface.
    Использует Gaussian Mixture Model (GMM) для кластеризации волатильности и импульса 
    в 3 скрытых режима рынка: Тренд (Trend), Флэт (Chop), и Прорыв (Breakout).
    """
    if df.empty or len(df) < lookback: 
        df['regime_trend_prob'] = 0.0
        df['regime_chop_prob'] = 0.0
        df['regime_break_prob'] = 0.0
        return df

    # Фичи для GMM: Лог-доходность и ATR в %
    ret = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    tr = (df['high'] - df['low']) / df['close'] * 100
    features = np.column_stack([ret.values, tr.values])
    
    prob_trend, prob_chop, prob_break = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
    
    # Оптимизация: пересчитываем GMM не каждую свечу, а раз в день (каждые 96 свечей для 15m),
    # чтобы код не работал вечность.
    step = 96 
    
    for i in range(lookback, len(df), step):
        window_data = features[i-lookback:i]
        
        # Обучаем GMM на 3 компоненты
        try:
            gmm = GaussianMixture(n_components=3, covariance_type='diag', random_state=42)
            gmm.fit(window_data)
            
            # Предсказываем вероятности для следующего блока (step)
            end_idx = min(i+step, len(df))
            test_data = features[i:end_idx]
            probs = gmm.predict_proba(test_data)
            
            # Сортируем режимы по волатильности (центрам кластеров по 2-й оси - TR)
            # 0 - Chop (Низкая вол-ть), 1 - Trend (Средняя), 2 - Breakout (Высокая)
            vol_centers = gmm.means_[:, 1]
            sorted_indices = np.argsort(vol_centers)
            
            chop_idx, trend_idx, break_idx = sorted_indices[0], sorted_indices[1], sorted_indices[2]
            
            prob_chop[i:end_idx] = probs[:, chop_idx]
            prob_trend[i:end_idx] = probs[:, trend_idx]
            prob_break[i:end_idx] = probs[:, break_idx]
        except:
            # Fallback если GMM не сошелся
            prob_chop[i:end_idx] = 0.33
            prob_trend[i:end_idx] = 0.33
            prob_break[i:end_idx] = 0.33

    df['regime_chop_prob'] = prob_chop
    df['regime_trend_prob'] = prob_trend
    df['regime_break_prob'] = prob_break
    
    # Заменяем нули в начале на NaN
    cols = ['regime_chop_prob', 'regime_trend_prob', 'regime_break_prob']
    df.loc[df.index[:lookback], cols] = np.nan
    
    return df