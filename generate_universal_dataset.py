import pandas as pd
import os

# Импортируем все твои функции из модулей
from shared.features.technical import (
    calculate_atr, calculate_rsi, calculate_volatility_zscore,
    add_log_returns, add_atr, add_volatility_zscore, add_normalized_slope,
    add_markov_regime, add_hmm_volatility_regime, add_algo_vol_crush,
    add_ifvg_signals, add_m15_structure
)
from shared.features.structural import add_structural_features
from shared.features.sessions import add_vector_sessions
from shared.features.htf import (
    add_daily_liquidity, add_htf_fvg, add_mtfa_trend, add_advanced_liquidity_and_eq
)
from shared.features.ml_features import add_regime_and_changepoint_features

def build_universal_dataset():
    input_path = "data/processed/gbpusd_15m.parquet"
    macro_path = "data/processed/sentiment_embeddings.parquet"
    output_path = "data/processed/gbpusd_with_all_features.parquet"

    print(f"1. Загрузка непрерывных данных из {input_path}...")
    if not os.path.exists(input_path):
        print(f"Ошибка: Файл {input_path} не найден!")
        return

    df = pd.read_parquet(input_path)

    # Сортировка индекса — КРИТИЧЕСКИ ВАЖНО для merge_asof
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Убедимся, что индекс - это время (timestamp)
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
    elif df.index.name != 'timestamp':
        df.index = pd.to_datetime(df.index)
        df.index.name = 'timestamp'

    # --- [НОВЫЙ БЛОК: ИНТЕГРАЦИЯ МАКРО] ---
    if os.path.exists(macro_path):
        print(f"-> Загрузка макро-эмбеддингов из {macro_path}...")
        df_macro = pd.read_parquet(macro_path)
        df_macro.index = pd.to_datetime(df_macro.index)
        df_macro.sort_index(inplace=True)

        # Merge AsOf: приклеиваем ближайшую ПРОШЛУЮ новость к текущей свече
        # Это гарантирует отсутствие заглядывания в будущее (Look-ahead bias)
        df = pd.merge_asof(
            df, 
            df_macro, 
            left_index=True, 
            right_index=True, 
            direction='backward'
        )
        
        # Заполняем пустоты нулями (если новостей не было, сентимент нейтральный)
        macro_cols = [c for c in df.columns if 'macro_emb' in c]
        df[macro_cols] = df[macro_cols].fillna(0.0)
        print(f"-> Макро успешно интегрировано. Колонки: {len(macro_cols)}")
    else:
        print("⚠️ ВНИМАНИЕ: sentiment_embeddings.parquet не найден. Продолжаем без макро.")

    print(f"Загружено {len(df)} свечей. Начинаем расчет фичей...\n")

    try:
        print("-> Расчет Технических индикаторов...")
        df = calculate_atr(df)
        df = calculate_rsi(df)
        df = calculate_volatility_zscore(df)
        df = add_log_returns(df)
        df = add_atr(df) # Требуется для Norm_Slope
        df = add_volatility_zscore(df)
        df = add_normalized_slope(df)
        df = add_markov_regime(df)
        df = add_hmm_volatility_regime(df)
        df = add_algo_vol_crush(df)
        df = add_ifvg_signals(df)
        df = add_m15_structure(df)

        print("-> Расчет Структуры рынка (Свинги, FVG, Ликвидность)...")
        df = add_structural_features(df)
        
        print("-> Расчет Сессий (Лондон, Нью-Йорк, Азия)...")
        df = add_vector_sessions(df)
        
        print("-> Расчет Старших таймфреймов (HTF)...")
        df = add_daily_liquidity(df)
        df = add_htf_fvg(df)
        df = add_mtfa_trend(df)
        df = add_advanced_liquidity_and_eq(df)
        
        print("-> Расчет ML и продвинутых фичей (Z-score волатильности, CUSUM)...")
        df = add_regime_and_changepoint_features(df)

    except Exception as e:
        print(f"\nОшибка при расчете фичей: {e}")
        return

    print("-> Очистка данных от пустых значений (из-за индикаторов с lookback)...")
    df.dropna(inplace=True)

    print(f"\n2. Расчет завершен! Итоговое количество фичей (колонок): {len(df.columns)}")

    print(f"3. Сохранение файла в {output_path}...")
    df.to_parquet(output_path)
    print("ГОТОВО! 🎉 Теперь у тебя есть универсальный DataFrame без предвзятости стратегий.")

if __name__ == "__main__":
    build_universal_dataset()