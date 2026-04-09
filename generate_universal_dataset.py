import pandas as pd
import os
import numpy as np

# Импортируем все функции из твоих модулей
from shared.features.technical import (
    calculate_atr, calculate_rsi, calculate_volatility_zscore,
    add_log_returns, add_atr, add_volatility_zscore, add_normalized_slope,
    add_mark_regime, add_hmm_volatility_regime, add_algo_vol_crush,
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

    print(f"1. Загрузка котировок из {input_path}...")
    if not os.path.exists(input_path):
        print(f"❌ Ошибка: Файл котировок не найден!")
        return
    
    df = pd.read_parquet(input_path)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    print(f"2. Загрузка макро-эмбеддингов из {macro_path}...")
    if os.path.exists(macro_path):
        df_macro = pd.read_parquet(macro_path)
        df_macro.index = pd.to_datetime(df_macro.index)
        df_macro.sort_index(inplace=True)

        # Безопасная склейка (Merge AsOf) - берем только прошлые события для каждой свечи
        df = pd.merge_asof(
            df, 
            df_macro, 
            left_index=True, 
            right_index=True, 
            direction='backward'
        )
        # Заполняем пустоты в макро-колонках (если новостей не было - сентимент нейтральный/0)
        macro_cols = [c for c in df.columns if 'macro_emb' in c]
        df[macro_cols] = df[macro_cols].fillna(0.0)
        print(f"✅ Макро-данные успешно интегрированы.")
    else:
        print(f"⚠️ ВНИМАНИЕ: Макро-эмбеддинги не найдены. Продолжаем без них.")

    print(f"\n3. Начинаем расчет технических и структурных фичей...")
    try:
        # Техника
        df = calculate_atr(df)
        df = calculate_rsi(df)
        df = calculate_volatility_zscore(df)
        df = add_log_returns(df)
        df = add_atr(df)
        df = add_volatility_zscore(df)
        df = add_normalized_slope(df)
        # df = add_markov_regime(df) # Проверь название функции в shared/features/technical.py
        df = add_hmm_volatility_regime(df)
        df = add_algo_vol_crush(df)
        df = add_ifvg_signals(df)
        df = add_m15_structure(df)

        # Структура и сессии
        df = add_structural_features(df)
        df = add_vector_sessions(df)
        
        # HTF (Старшие ТФ)
        df = add_daily_liquidity(df)
        df = add_htf_fvg(df)
        df = add_mtfa_trend(df)
        df = add_advanced_liquidity_and_eq(df)
        
        # ML фичи
        df = add_regime_and_changepoint_features(df)

    except Exception as e:
        print(f"\n❌ Ошибка при расчете фичей: {e}")
        import traceback
        traceback.print_exc()
        return

    # Очистка
    print("4. Финальная очистка (удаление начальных NaN)...")
    df.dropna(inplace=True)

    print(f"\n5. Сборка завершена. Колонки: {len(df.columns)}")
    print(f"6. Сохранение в {output_path}...")
    df.to_parquet(output_path, engine='pyarrow')
    print("ГОТОВО! 🎉 Универсальный датасет восстановлен.")

if __name__ == "__main__":
    build_universal_dataset()