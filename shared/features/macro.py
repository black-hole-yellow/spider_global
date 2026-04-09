import pandas as pd
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from shared.features.decorators import provides_features

# We pre-register the exact column names your macro hypotheses will ask for
MACRO_FEATURES = [
    'NFP_Day', 'NFP_Release_Bar', 'NFP_Surprise',
    'FOMC_Day', 'FOMC_Release_Bar', 'FOMC_Surprise',
    'US_CPI_Day', 'US_CPI_Release_Bar', 'US_CPI_Surprise',
    'UK_CPI_Day', 'UK_CPI_Release_Bar', 'UK_CPI_Surprise',
    'BoE_Day', 'BoE_Release_Bar'
]

@provides_features(*MACRO_FEATURES)
def add_macro_events(df: pd.DataFrame, events_path: str = "data/macro_events.json") -> pd.DataFrame:
    """
    Injects macroeconomic data into the price action DataFrame.
    Turns sparse JSON events into vectorized 1.0 / 0.0 flags and surprise deltas.
    """
    # 1. Initialize all macro columns to 0.0 (Neutral state)
    for col in MACRO_FEATURES:
        df[col] = 0.0

    path = Path(events_path)
    if not path.exists():
        print(f"⚠️ WARNING: Macro events file not found at {events_path}. Macro features will default to 0.")
        return df
        
    with open(path, 'r') as f:
        try:
            events = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ WARNING: Invalid JSON in macro_events.json")
            return df

    # 2. Convert events to a DataFrame for fast processing
    # --- THE SAFE PARSER FIX ---
    if isinstance(events, dict):
        # If the JSON is wrapped in a dictionary, find the actual array of data
        for key, val in events.items():
            if isinstance(val, list):
                events = val
                break
        # If it's still a dict (e.g., just one single event), wrap it in a list safely
        if isinstance(events, dict):
            events = [events]
            
    events_df = pd.DataFrame(events)
    if events_df.empty or 'date' not in events_df.columns:
        return df
        
    # Ensure timezone alignment with your price data (UTC)
    events_df['date'] = pd.to_datetime(events_df['date'])
    if events_df['date'].dt.tz is None:
        events_df['date'] = events_df['date'].dt.tz_localize('UTC')

    # 3. Map events to the main Price DataFrame
    for _, row in events_df.iterrows():
        event_time = row['date']
        event_type = str(row.get('event', '')).upper()
        # Surprise = Actual - Forecast (Used for trading divergence)
        surprise = float(row.get('surprise', 0.0)) 
        
        # Route to the correct prefix based on the JSON event string
        prefix = ""
        if "NFP" in event_type: prefix = "NFP"
        elif "FOMC" in event_type: prefix = "FOMC"
        elif "US CPI" in event_type or "US_CPI" in event_type: prefix = "US_CPI"
        elif "UK CPI" in event_type or "UK_CPI" in event_type: prefix = "UK_CPI"
        elif "BOE" in event_type: prefix = "BoE"
        
        if not prefix: continue
        
        # --- A. Flag the exact release bar ---
        # df.index.asof finds the closest past/current bar to the release time
        # Example: A 13:30 release falls exactly on the 13:30 15m candle.
        closest_bar = df.index.asof(event_time)
        if pd.notnull(closest_bar):
            df.loc[closest_bar, f"{prefix}_Release_Bar"] = 1.0
            if f"{prefix}_Surprise" in df.columns:
                df.loc[closest_bar, f"{prefix}_Surprise"] = surprise
                
        # --- B. Flag the entire day as a 'Macro Day' ---
        # Useful for filtering out regular trend-following systems on high-vol days
        event_date = event_time.date()
        day_mask = df.index.date == event_date
        df.loc[day_mask, f"{prefix}_Day"] = 1.0

    # 4. Forward fill the 'Surprise' delta for the rest of the day
    # This allows post-news momentum strategies to know the direction of the shock
    for col in MACRO_FEATURES:
        if "Surprise" in col:
            # Group by day so yesterday's NFP surprise doesn't leak into today
            df[col] = df.groupby(df.index.date)[col].ffill().fillna(0.0)
            
    return df

@provides_features(
    'CPI_Momentum_Long', 'CPI_Momentum_Short', 'Macro_CPI_Div_Long',
    'Retail_Div_Long', 'CB_Divergence_Long', 'CB_Divergence_Short',
    'BoE_Tone_Shift_Short', 'Unemp_Fakeout_Long',
    'UK_Shock_Cont_Long', 'UK_Shock_Cont_Short', 'Geo_Shock_Short',
    'NFP_Resumption_Long', 'NFP_Resumption_Short', 'NFP_Fade_Long', 'NFP_Fade_Short',
    'FOMC_Sell_News_Long', 'Macro_Inside_Bar_Short',
    'Election_Vol_Crush_Short', 'Sovereign_Risk_Short'
)
def add_macro_strategy_triggers(df: pd.DataFrame) -> pd.DataFrame:
    # --- DEPENDENCY INJECTION FIX ---
    # Force the engine to parse macro_events.json before calculating triggers
    if 'NFP_Release_Bar' not in df.columns:
        from shared.features.macro import add_macro_events
        df = add_macro_events(df)

    # Now we safely get the macro flags, knowing they were generated
    us_cpi = df.get('US_CPI_Release_Bar', pd.Series(0, index=df.index))
    nfp = df.get('NFP_Release_Bar', pd.Series(0, index=df.index))
    fomc = df.get('FOMC_Release_Bar', pd.Series(0, index=df.index))
    uk_cpi = df.get('UK_CPI_Release_Bar', pd.Series(0, index=df.index))
    boe = df.get('BoE_Release_Bar', pd.Series(0, index=df.index))
    
    # NFP Strategies
    df['NFP_Fade_Long'] = (nfp == 1) & (df['close'] < df['open'])
    df['NFP_Fade_Short'] = (nfp == 1) & (df['close'] > df['open'])
    df['NFP_Resumption_Long'] = (nfp.shift(4) == 1) & (df['close'] > df['close'].shift(4))
    df['NFP_Resumption_Short'] = (nfp.shift(4) == 1) & (df['close'] < df['close'].shift(4))
    
    # CPI Strategies
    df['CPI_Momentum_Long'] = (us_cpi == 1) & (df['close'] > df['open'])
    df['CPI_Momentum_Short'] = (us_cpi == 1) & (df['close'] < df['open'])
    df['Macro_CPI_Div_Long'] = (us_cpi == 1) & (df.get('US_CPI_Surprise', 0) < 0) 
    
    # FOMC & Central Bank
    df['FOMC_Sell_News_Long'] = (fomc == 1) & (df['close'] < df['open'])
    df['CB_Divergence_Long'] = (fomc == 1) & (df['close'] > df['open'])
    df['CB_Divergence_Short'] = (boe == 1) & (df['close'] < df['open'])
    df['BoE_Tone_Shift_Short'] = (boe == 1) & (df['close'] < df['open'])
    
    # UK Specific
    df['UK_Shock_Cont_Long'] = (uk_cpi == 1) & (df['close'] > df['open'])
    df['UK_Shock_Cont_Short'] = (uk_cpi == 1) & (df['close'] < df['open'])
    
    # Generic & Placeholders
    df['Unemp_Fakeout_Long'] = df['NFP_Fade_Long']
    df['Retail_Div_Long'] = df['CPI_Momentum_Long']
    df['Geo_Shock_Short'] = 0
    df['Election_Vol_Crush_Short'] = 0
    df['Sovereign_Risk_Short'] = 0
    df['Macro_Inside_Bar_Short'] = (nfp == 1) & (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    
    # Convert all booleans to 1.0/0.0
    cols = ['CPI_Momentum_Long', 'CPI_Momentum_Short', 'Macro_CPI_Div_Long', 'Retail_Div_Long', 'CB_Divergence_Long', 'CB_Divergence_Short', 'BoE_Tone_Shift_Short', 'Unemp_Fakeout_Long', 'UK_Shock_Cont_Long', 'UK_Shock_Cont_Short', 'Geo_Shock_Short', 'NFP_Resumption_Long', 'NFP_Resumption_Short', 'NFP_Fade_Long', 'NFP_Fade_Short', 'FOMC_Sell_News_Long', 'Macro_Inside_Bar_Short', 'Election_Vol_Crush_Short', 'Sovereign_Risk_Short']
    for c in cols:
        df[c] = df[c].astype(int)
        
    return df

@provides_features('llm_sentiment_score', 'llm_regime_prob', 'is_macro_alignment')
def add_llm_semantic_features(df: pd.DataFrame, llm_path: str = "data/llm_sentiment.json") -> pd.DataFrame:
    """
    Подтягивает семантическую оценку рынка от LLM.
    """
    df['llm_sentiment_score'] = 0.0
    df['llm_regime_prob'] = 0.0
    
    path = Path(llm_path)
    if not path.exists():
        return df
        
    with open(path, 'r') as f:
        try:
            # Читаем файл
            content = f.read()
            # Если файл пустой (0 байт), возвращаем пустой словарь
            llm_data = json.loads(content) if content.strip() else {}
        except json.JSONDecodeError:
            print("⚠️ WARNING: llm_sentiment.json is corrupted or empty. Defaulting to 0.0")
            return df
            
    # Конвертируем словарь LLM (где ключи - даты) в DataFrame
    llm_df = pd.DataFrame.from_dict(llm_data, orient='index')
    llm_df.index = pd.to_datetime(llm_df.index)
    
    # Мапим дневные оценки на 15-минутные бары нашего основного DataFrame
    df_dates = df.index.normalize() # Получаем даты без времени
    
    # Используем map для быстрого переноса значений
    df['llm_sentiment_score'] = df_dates.map(llm_df['llm_sentiment_score']).fillna(0.0)
    df['llm_regime_prob'] = df_dates.map(llm_df['regime_shift_prob']).fillna(0.0)
    
    # --- СИНЕРГИЯ ИИ и МАТЕМАТИКИ (The Alpha) ---
    # Макро-согласованность: если математический тренд (mtfa_score) и 
    # семантика LLM (llm_sentiment_score) смотрят в одну сторону, это супер-сигнал.
    
    if 'mtfa_score' in df.columns:
        # Умножение: + на + дает +, - на - дает +. 
        # Если знаки разные (дивергенция ИИ и цены), результат отрицательный.
        df['is_macro_alignment'] = df['mtfa_score'] * df['llm_sentiment_score']
    else:
        df['is_macro_alignment'] = 0.0

    return df

@provides_features('macro_divergence_score')
def add_macro_narrative_divergence(df: pd.DataFrame, forward_window: int = 4) -> pd.DataFrame:
    """
    #5: Macro Narrative Divergence Score.
    Сравнивает вектор макро-новостей с фактической реакцией цены.
    В идеале мы берем "тон" новости из LLM, но так как у нас в датасете макро-эмбеддинги, 
    мы аппроксимируем: реагирует ли рынок логично на всплеск макро-активности?
    """
    if df.empty or 'macro_emb_0' not in df.columns: return df
    
    # Ищем моменты выхода макро-новостей (когда вектор меняется)
    macro_change = df['macro_emb_0'].diff().abs() > 0
    
    # Направление цены после новости (например, за 1 час)
    future_return = (df['close'].shift(-forward_window) - df['close']) / df['close']
    
    # Интенсивность новости (сумма изменений всех 8 главных компонент)
    macro_cols = [c for c in df.columns if c.startswith('macro_emb_')]
    if not macro_cols: return df
    
    macro_intensity = df[macro_cols].diff().abs().sum(axis=1)
    
    # Дивергенция: если интенсивность новости высокая, но цена никуда не пошла (или откатилась), 
    # это сигнал "поглощения" (Absorption) или ложного нарратива.
    # (Здесь мы сохраняем историческую реакцию как фичу для будущих паттернов)
    df['macro_divergence_score'] = macro_intensity * np.abs(future_return)
    
    # Так как future_return заглядывает в будущее, мы сдвигаем эту фичу в прошлое,
    # чтобы модель видела: "Ага, 4 свечи назад была дивергенция, значит сейчас тренд иссяк"
    df['macro_divergence_score'] = df['macro_divergence_score'].shift(forward_window)
    df['macro_divergence_score'].fillna(0, inplace=True)
    
    return df

@provides_features('macro_regime_1', 'macro_regime_2')
def add_macro_hmm_regimes(df: pd.DataFrame, lookback: int = 500) -> pd.DataFrame:
    """
    #22: Probabilistic Macro Regime States (HMM).
    Обучает скрытую Марковскую модель (HMM) строго на макро-векторах (без цены),
    чтобы определить 2 глобальных макро-режима (например: Risk-On / Risk-Off).
    Выдает вероятность нахождения в каждом режиме.
    """
    if df.empty or 'macro_emb_0' not in df.columns or len(df) < lookback: 
        df['macro_regime_1'] = 0.5
        df['macro_regime_2'] = 0.5
        return df

    macro_cols = [c for c in df.columns if c.startswith('macro_emb_')]
    if not macro_cols: return df
    
    # Для HMM берем только первые 2 главные компоненты (PCA), чтобы не перегружать вычисления
    features = df[macro_cols[:2]].values
    
    prob_regime_1 = np.zeros(len(df))
    prob_regime_2 = np.zeros(len(df))
    
    # Пересчитываем HMM раз в неделю (480 свечей)
    step = 480 
    
    for i in range(lookback, len(df), step):
        window_data = features[i-lookback:i]
        
        try:
            model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
            model.fit(window_data)
            
            end_idx = min(i+step, len(df))
            test_data = features[i:end_idx]
            
            # Предсказание вероятностей (soft clustering)
            probs = model.predict_proba(test_data)
            
            prob_regime_1[i:end_idx] = probs[:, 0]
            prob_regime_2[i:end_idx] = probs[:, 1]
        except:
            # Fallback
            end_idx = min(i+step, len(df))
            prob_regime_1[i:end_idx] = 0.5
            prob_regime_2[i:end_idx] = 0.5

    df['macro_regime_1'] = prob_regime_1
    df['macro_regime_2'] = prob_regime_2
    
    return df