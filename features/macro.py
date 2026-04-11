import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM

# Pre-register the exact column names your macro hypotheses will ask for
macro_features = [
    'NFP_Day', 'NFP_Release_Bar', 'NFP_Surprise',
    'FOMC_Day', 'FOMC_Release_Bar', 'FOMC_Surprise',
    'US_CPI_Day', 'US_CPI_Release_Bar', 'US_CPI_Surprise',
    'UK_CPI_Day', 'UK_CPI_Release_Bar', 'UK_CPI_Surprise',
    'BoE_Day', 'BoE_Release_Bar'
]

# =========================================================================
# 1. SUB-FUNCTIONS (Internal Logic)
# =========================================================================

def add_macro_embeddings(df: pd.DataFrame, macro_path: str = "data/processed/sentiment_embeddings.parquet") -> pd.DataFrame:
    """Loads and injects PCA-compressed macro embeddings."""
    if os.path.exists(macro_path):
        try:
            df_macro = pd.read_parquet(macro_path)
            latest_macro = df_macro.iloc[-1:]
            for col in latest_macro.columns:
                df[col] = latest_macro[col].values[0]
            return df
        except Exception as e:
            print(f"⚠️ Ошибка чтения макро-данных: {e}")

    # Fallback: Zero padding to prevent Transformer crashes
    for i in range(384):
        df[f'macro_emb_{i}'] = 0.0
    return df

def add_macro_events(df: pd.DataFrame, events_path: str = "data/macro_events.json") -> pd.DataFrame:
    """Injects macroeconomic data into the price action DataFrame."""
    for col in macro_features:
        df[col] = 0.0

    path = Path(events_path)
    if not path.exists():
        return df
        
    with open(path, 'r') as f:
        try:
            events = json.load(f)
        except json.JSONDecodeError:
            return df

    # Safe Parser
    if isinstance(events, dict):
        for key, val in events.items():
            if isinstance(val, list):
                events = val
                break
        if isinstance(events, dict):
            events = [events]
            
    events_df = pd.DataFrame(events)
    if events_df.empty or 'date' not in events_df.columns:
        return df
        
    events_df['date'] = pd.to_datetime(events_df['date'])
    if events_df['date'].dt.tz is None:
        events_df['date'] = events_df['date'].dt.tz_localize('UTC')

    for _, row in events_df.iterrows():
        event_time = row['date']
        event_type = str(row.get('event', '')).upper()
        surprise = float(row.get('surprise', 0.0)) 
        
        prefix = ""
        if "NFP" in event_type: prefix = "NFP"
        elif "FOMC" in event_type: prefix = "FOMC"
        elif "US CPI" in event_type or "US_CPI" in event_type: prefix = "US_CPI"
        elif "UK CPI" in event_type or "UK_CPI" in event_type: prefix = "UK_CPI"
        elif "BOE" in event_type: prefix = "BoE"
        
        if not prefix: continue
        
        closest_bar = df.index.asof(event_time)
        if pd.notnull(closest_bar):
            df.loc[closest_bar, f"{prefix}_Release_Bar"] = 1.0
            if f"{prefix}_Surprise" in df.columns:
                df.loc[closest_bar, f"{prefix}_Surprise"] = surprise
                
        event_date = event_time.date()
        day_mask = df.index.date == event_date
        df.loc[day_mask, f"{prefix}_Day"] = 1.0

    for col in macro_features:
        if "Surprise" in col:
            df[col] = df.groupby(df.index.date)[col].ffill().fillna(0.0)
            
    return df

def add_macro_strategy_triggers(df: pd.DataFrame) -> pd.DataFrame:
    """Translates raw events into trading triggers (Divergences, Fade, Resumption)."""
    us_cpi = df.get('US_CPI_Release_Bar', pd.Series(0, index=df.index))
    nfp = df.get('NFP_Release_Bar', pd.Series(0, index=df.index))
    fomc = df.get('FOMC_Release_Bar', pd.Series(0, index=df.index))
    uk_cpi = df.get('UK_CPI_Release_Bar', pd.Series(0, index=df.index))
    boe = df.get('BoE_Release_Bar', pd.Series(0, index=df.index))
    
    df['NFP_Fade_Long'] = (nfp == 1) & (df['close'] < df['open'])
    df['NFP_Fade_Short'] = (nfp == 1) & (df['close'] > df['open'])
    df['NFP_Resumption_Long'] = (nfp.shift(4) == 1) & (df['close'] > df['close'].shift(4))
    df['NFP_Resumption_Short'] = (nfp.shift(4) == 1) & (df['close'] < df['close'].shift(4))
    
    df['CPI_Momentum_Long'] = (us_cpi == 1) & (df['close'] > df['open'])
    df['CPI_Momentum_Short'] = (us_cpi == 1) & (df['close'] < df['open'])
    df['Macro_CPI_Div_Long'] = (us_cpi == 1) & (df.get('US_CPI_Surprise', 0) < 0) 
    
    df['FOMC_Sell_News_Long'] = (fomc == 1) & (df['close'] < df['open'])
    df['CB_Divergence_Long'] = (fomc == 1) & (df['close'] > df['open'])
    df['CB_Divergence_Short'] = (boe == 1) & (df['close'] < df['open'])
    df['BoE_Tone_Shift_Short'] = (boe == 1) & (df['close'] < df['open'])
    
    df['UK_Shock_Cont_Long'] = (uk_cpi == 1) & (df['close'] > df['open'])
    df['UK_Shock_Cont_Short'] = (uk_cpi == 1) & (df['close'] < df['open'])
    
    df['Unemp_Fakeout_Long'] = df['NFP_Fade_Long']
    df['Retail_Div_Long'] = df['CPI_Momentum_Long']
    df['Geo_Shock_Short'] = 0
    df['Election_Vol_Crush_Short'] = 0
    df['Sovereign_Risk_Short'] = 0
    df['Macro_Inside_Bar_Short'] = (nfp == 1) & (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    
    cols = [
        'CPI_Momentum_Long', 'CPI_Momentum_Short', 'Macro_CPI_Div_Long', 'Retail_Div_Long', 
        'CB_Divergence_Long', 'CB_Divergence_Short', 'BoE_Tone_Shift_Short', 'Unemp_Fakeout_Long', 
        'UK_Shock_Cont_Long', 'UK_Shock_Cont_Short', 'Geo_Shock_Short', 'NFP_Resumption_Long', 
        'NFP_Resumption_Short', 'NFP_Fade_Long', 'NFP_Fade_Short', 'FOMC_Sell_News_Long', 
        'Macro_Inside_Bar_Short', 'Election_Vol_Crush_Short', 'Sovereign_Risk_Short'
    ]
    for c in cols:
        df[c] = df[c].astype(int)
        
    return df

def add_llm_semantic_features(df: pd.DataFrame, llm_path: str = "data/llm_sentiment.json") -> pd.DataFrame:
    df['llm_sentiment_score'] = 0.0
    df['llm_regime_prob'] = 0.0
    
    path = Path(llm_path)
    if not path.exists():
        return df
        
    with open(path, 'r') as f:
        try:
            content = f.read()
            llm_data = json.loads(content) if content.strip() else {}
        except json.JSONDecodeError:
            return df
            
    llm_df = pd.DataFrame.from_dict(llm_data, orient='index')
    llm_df.index = pd.to_datetime(llm_df.index)
    
    df_dates = df.index.normalize() 
    
    df['llm_sentiment_score'] = df_dates.map(llm_df['llm_sentiment_score']).fillna(0.0)
    df['llm_regime_prob'] = df_dates.map(llm_df.get('regime_shift_prob', 0.0)).fillna(0.0)
    
    if 'mtfa_score' in df.columns:
        df['is_macro_alignment'] = df['mtfa_score'] * df['llm_sentiment_score']
    else:
        df['is_macro_alignment'] = 0.0

    return df

def add_macro_narrative_divergence(df: pd.DataFrame, forward_window: int = 4) -> pd.DataFrame:
    if df.empty or 'macro_emb_0' not in df.columns: return df
    
    future_return = (df['close'].shift(-forward_window) - df['close']) / df['close']
    macro_cols = [c for c in df.columns if c.startswith('macro_emb_')]
    
    if not macro_cols: return df
    
    macro_intensity = df[macro_cols].diff().abs().sum(axis=1)
    df['macro_divergence_score'] = macro_intensity * np.abs(future_return)
    df['macro_divergence_score'] = df['macro_divergence_score'].shift(forward_window).fillna(0)
    
    return df

def add_macro_hmm_regimes(df: pd.DataFrame, lookback: int = 500) -> pd.DataFrame:
    if df.empty or 'macro_emb_0' not in df.columns or len(df) < lookback: 
        df['macro_regime_1'] = 0.5
        df['macro_regime_2'] = 0.5
        return df

    macro_cols = [c for c in df.columns if c.startswith('macro_emb_')]
    if not macro_cols: return df
    
    features = df[macro_cols[:2]].values
    prob_regime_1 = np.zeros(len(df))
    prob_regime_2 = np.zeros(len(df))
    step = 480 
    
    for i in range(lookback, len(df), step):
        window_data = features[i-lookback:i]
        try:
            model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
            model.fit(window_data)
            
            end_idx = min(i+step, len(df))
            test_data = features[i:end_idx]
            
            probs = model.predict_proba(test_data)
            prob_regime_1[i:end_idx] = probs[:, 0]
            prob_regime_2[i:end_idx] = probs[:, 1]
        except:
            end_idx = min(i+step, len(df))
            prob_regime_1[i:end_idx] = 0.5
            prob_regime_2[i:end_idx] = 0.5

    df['macro_regime_1'] = prob_regime_1
    df['macro_regime_2'] = prob_regime_2
    
    return df

# =========================================================================
# 2. MASTER WRAPPER (Layer 6 Entry Point)
# =========================================================================

def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master entry point for Macro Layer. 
    Strictly enforces dependency execution order.
    """
    df = add_macro_embeddings(df)              # Needs raw df
    df = add_macro_events(df)                  # Needs raw df
    df = add_macro_strategy_triggers(df)       # Needs Events
    df = add_llm_semantic_features(df)         # Needs Events
    df = add_macro_narrative_divergence(df)    # Needs Embeddings
    df = add_macro_hmm_regimes(df)             # Needs Embeddings
    
    return df