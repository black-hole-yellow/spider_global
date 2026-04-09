import pandas as pd
import os
import numpy as np

# Импортируем все функции из твоих модулей
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
    output_path = "data/processed/gbpusd_with_all_features.parquet"

    print(f"1. Загрузка чистых 15-минутных котировок из {input_path}...")
    if not os.path.exists(input_path):
        print(f"❌ Ошибка: Файл котировок не найден! Убедись, что базовые данные скачаны.")
        return
    
    df = pd.read_parquet(input_path)
    df.index = pd.to_datetime(df.index)
    
    # Избавляемся от таймзон для безопасности перед будущими склейками
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.sort_index(inplace=True)

    print(f"\n2. Начинаем расчет технических и структурных фичей...")
    try:
        # --- Блок 1: Базовая техника и волатильность ---
        print("   -> Расчет технических индикаторов...")
        df = calculate_atr(df)
        df = calculate_rsi(df)
        df = calculate_volatility_zscore(df)
        df = add_log_returns(df)
        df = add_atr(df)
        df = add_volatility_zscore(df)
        df = add_normalized_slope(df)
        
        # --- Блок 2: Режимы рынка ---
        print("   -> Определение режимов рынка...")
        df = add_markov_regime(df) 
        df = add_hmm_volatility_regime(df)
        df = add_algo_vol_crush(df)
        
        # --- Блок 3: Микроструктура ---
        print("   -> Анализ SMC и микроструктуры...")
        df = add_ifvg_signals(df)
        df = add_m15_structure(df)
        df = add_structural_features(df)
        df = add_vector_sessions(df)
        
        # --- Блок 4: Старшие таймфреймы (HTF) ---
        print("   -> Проекция старших таймфреймов (HTF)...")
        df = add_daily_liquidity(df)
        df = add_htf_fvg(df)
        df = add_mtfa_trend(df)
        df = add_advanced_liquidity_and_eq(df)
        
        # --- Блок 5: Продвинутые ML-фичи ---
        print("   -> Генерация ML-фичей...")
        df = add_regime_and_changepoint_features(df)

    except Exception as e:
        print(f"\n❌ Ошибка при расчете фичей: {e}")
        import traceback
        traceback.print_exc()
        return

    # Очистка
    print("\n3. Финальная очистка (удаление начальных строк с NaN из-за скользящих окон)...")
    # Удаляем только те строки, где нет базовых фичей (чтобы не убить весь датасет из-за одной редкой фичи)
    df.dropna(subset=['rsi', 'atr', 'log_return'], inplace=True)

    print(f"4. Сборка завершена. Получено колонок: {len(df.columns)}")
    print(f"5. Сохранение в {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine='pyarrow')
    print("✅ ГОТОВО! Технический Feature Store (gbpusd_with_all_features.parquet) успешно сгенерирован.")

if __name__ == "__main__":
    build_universal_dataset()