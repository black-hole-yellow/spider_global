import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def process_macro_embeddings():
    input_path = "data/macro_events.json"
    output_path = "data/processed/sentiment_embeddings.parquet"

    print(f"1. Загрузка данных из {input_path}...")
    if not os.path.exists(input_path):
        print(f"❌ Ошибка: Файл {input_path} не найден! Выполни сначала generate_macro_events.py.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        events = json.load(f)

    df_events = pd.DataFrame(events)
    print(f"Найдено {len(df_events)} макро-событий.")

    # ЗАЩИТА ОТ КРАША: Если две новости вышли в одну секунду (например, инфляция и безработица),
    # мы объединяем их текст, чтобы индекс времени был строго уникальным.
    df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])
    df_grouped = df_events.groupby('timestamp').agg({
        'content': lambda x: " | ".join(x) # Склеиваем тексты через разделитель
    }).reset_index()

    print(f"После группировки одновременных событий осталось {len(df_grouped)} уникальных временных точек.")

    print("2. Загрузка модели SentenceTransformer ('all-MiniLM-L6-v2')...")
    # Используем легковесную финансово-совместимую модель (выдает 384 признака)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("3. Генерация 384-мерных векторов (Embeddings)... Это займет немного времени.")
    texts = df_grouped['content'].astype(str).tolist()
    
    # Модель возвращает numpy array размера (N_events, 384)
    embeddings = model.encode(texts, show_progress_bar=True)

    print("4. Сборка и форматирование датасета...")
    # Создаем 384 отдельные колонки (macro_emb_0 ... macro_emb_383)
    columns = [f'macro_emb_{i}' for i in range(embeddings.shape[1])]
    
    df_embeddings = pd.DataFrame(
        embeddings, 
        index=df_grouped['timestamp'], 
        columns=columns
    )
    
    # Гарантируем строгую хронологию для будущего Merge AsOf
    df_embeddings.sort_index(inplace=True)

    print("5. Сохранение в Parquet...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_embeddings.to_parquet(output_path, engine='pyarrow')
    
    print(f"✅ ГОТОВО! Эмбеддинги ({df_embeddings.shape[0]} строк, {df_embeddings.shape[1]} колонок) сохранены в {output_path}")

if __name__ == "__main__":
    process_macro_embeddings()