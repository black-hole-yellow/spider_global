import pandas as pd
import numpy as np

def _add_vector_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерирует непрерывные векторные представления (Embeddings) торговых сессий.
    Вместо жестких 1/0 используется плавная интенсивность ликвидности.
    Время ожидается в UTC.
    """
    if df.empty: return df

    # 1. CYCLICAL TIME ENCODING (Оставляем как было, это база)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)

    # Точное время в часах (например, 14:30 -> 14.5)
    time_float = df.index.hour + df.index.minute / 60.0

    # 2. SESSION INTENSITY EMBEDDINGS (Плавная ликвидность)
    # Используем формулу Гаусса: exp(-0.5 * ((x - mu) / sigma)^2)
    def gaussian_intensity(x, mu, sigma):
        # Обработка перехода через полночь для Азии (цикличность 24h)
        dist = np.minimum(abs(x - mu), 24 - abs(x - mu))
        return np.exp(-0.5 * (dist / sigma)**2)

    # Азия (Токио/Сидней): Пик активности около 02:00 UTC
    df['asia_intensity'] = gaussian_intensity(time_float, mu=2.0, sigma=3.0)

    # Лондон: Пик активности около 09:00 UTC (Франкфурт уже открыт, Лондон раскачался)
    df['london_intensity'] = gaussian_intensity(time_float, mu=9.0, sigma=2.5)

    # Нью-Йорк: Пик активности около 14:30 UTC (Открытие NYSE, макро-статистика)
    df['ny_intensity'] = gaussian_intensity(time_float, mu=14.5, sigma=2.5)

    # 3. OVERLAP SCORE (Метрика "Безумия")
    # Пересечение Лондона и Нью-Йорка дает самую высокую плотность ордеров в мире.
    df['session_overlap_score'] = df['london_intensity'] * df['ny_intensity']

    # 4. CATEGORICAL SESSION STATE (Для CatBoost Meta-Labeler)
    conditions = [
        (df['london_intensity'] > 0.3) & (df['ny_intensity'] > 0.3),
        (df['ny_intensity'] > 0.3),
        (df['london_intensity'] > 0.3)
    ]
    choices = ['London_NY_Overlap', 'NY', 'London']
    
    df['active_session_name'] = np.select(conditions, choices, default='Asian')

    return df

def _add_session_liquidity_transfer(df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """
    #9: Session Liquidity Transfer Strength.
    Сравнивает объем текущих торгов с фоновым объемом "тихой" сессии.
    Высокий трансфер означает сильный направленный институциональный импульс на открытии.
    """
    if df.empty or 'volume' not in df.columns: return df
    
    # 1. Считаем скользящее среднее объема за последние 8 свечей (2 часа)
    background_volume = df['volume'].shift(1).rolling(window=window).mean()
    
    # 2. Вычисляем аномальность текущего объема
    volume_surge = df['volume'] / (background_volume + 1e-9)
    
    # 3. Придаем силу направлению (Directional Transfer)
    range_hl = (df['high'] - df['low']).replace(0, 1e-5)
    bar_direction = (df['close'] - df['open']) / range_hl
    
    df['session_liquidity_transfer'] = volume_surge * bar_direction
    
    return df

# ==========================================
# 🚀 MASTER WRAPPER (Entry Point)
# ==========================================
def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Единая точка входа для слоя Session & Temporal Features.
    Рассчитывается в слое 4 (после Technical, Structural, HTF).
    """
    df = _add_vector_sessions(df)
    df = _add_session_liquidity_transfer(df)
    return df