import pandas as pd
import numpy as np
import os

def merge_tech_and_macro():
    # Пути к нашим подготовленным данным
    tech_path = "data/processed/gbpusd_with_all_features.parquet"
    macro_path = "data/processed/sentiment_embeddings.parquet"
    output_path = "data/processed/full_merged_dataset.parquet"

    print(f"1. Загрузка технических фичей ({tech_path})...")
    if not os.path.exists(tech_path):
        print("❌ Ошибка: Технический датасет не найден. Сначала запусти generate_universal_dataset.py")
        return
        
    df_tech = pd.read_parquet(tech_path)
    
    # Обязательно избавляемся от таймзон и сортируем индекс для безопасного слияния
    if df_tech.index.tz is not None:
        df_tech.index = df_tech.index.tz_localize(None)
    df_tech = df_tech.sort_index()

    print(f"2. Загрузка макро-эмбеддингов ({macro_path})...")
    if not os.path.exists(macro_path):
        print("❌ Ошибка: Макро датасет не найден. Сначала запусти llm_macro_parser.py")
        return
        
    df_macro = pd.read_parquet(macro_path)
    if df_macro.index.tz is not None:
        df_macro.index = df_macro.index.tz_localize(None)
    df_macro = df_macro.sort_index()

    print("3. Выполнение безопасного слияния (Merge AsOf: direction='backward')...")
    # МАГИЯ ЗДЕСЬ: direction='backward' гарантирует, что свеча 14:15 получит 
    # макро-вектор от 14:00 (или раньше), но НИКОГДА от 14:30. Утечка данных исключена.
    df_merged = pd.merge_asof(
        df_tech,
        df_macro,
        left_index=True,
        right_index=True,
        direction='backward'
    )

    print("4. Обработка пустот...")
    # До выхода первой новости в 2000 году макро-колонки будут NaN.
    # Заполняем их нулями (0.0), что для Трансформера означает "нейтральный фон".
    macro_cols = [col for col in df_merged.columns if 'macro_emb_' in col]
    df_merged[macro_cols] = df_merged[macro_cols].fillna(0.0)

    # Если в начале графика нет технических данных (например, из-за скользящих средних) - удаляем
    df_merged = df_merged.dropna()

    print(f"5. Сохранение итогового датасета...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_parquet(output_path, engine='pyarrow')
    
    print(f"✅ УСПЕХ! Финальный датасет готов.")
    print(f"Размерность: {df_merged.shape[0]} строк, {df_merged.shape[1]} колонок (Цена + Техника + 384 Макро).")

if __name__ == "__main__":
    merge_tech_and_macro()