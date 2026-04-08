import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def process_macro_embeddings():
    """
    Читает сырые макро-события, пропускает их через LLM (Transformer) 
    для получения эмбеддингов и сохраняет в Parquet.
    """
    input_path = "data/macro_events.json"
    output_path = "data/processed/sentiment_embeddings.parquet"

    print(f"1. Загрузка данных из {input_path}...")
    if not os.path.exists(input_path):
        print(f"Ошибка: Файл {input_path} не найден! Убедись, что путь верный.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        events = json.load(f)

    # Преобразуем в DataFrame для удобства
    df_events = pd.DataFrame(events)
    
    # ВНИМАНИЕ: Проверь названия колонок в твоем JSON! 
    # Я предполагаю, что там есть колонки 'timestamp' (или 'date') и 'text' (или 'content', 'title')
    time_col = 'timestamp' if 'timestamp' in df_events.columns else 'date'
    text_col = 'text' if 'text' in df_events.columns else 'content'
    
    if text_col not in df_events.columns and 'title' in df_events.columns:
        text_col = 'title'

    print(f"Найдено {len(df_events)} макро-событий. Используем колонку '{text_col}' для текста.")

    # 2. Инициализация локальной LLM для эмбеддингов
    # Используем легковесную, но мощную финансовую/универсальную модель
    # 'all-MiniLM-L6-v2' генерирует вектор из 384 измерений
    print("2. Загрузка модели SentenceTransformer (при первом запуске скачается из интернета)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Генерация эмбеддингов
    print("3. Генерация многомерных векторов (Embeddings)... Это может занять пару минут.")
    # Превращаем все тексты в список
    texts = df_events[text_col].astype(str).tolist()
    
    # Модель возвращает numpy array размера (N_events, 384)
    embeddings = model.encode(texts, show_progress_bar=True)

    # 4. Сборка финального DataFrame
    print("4. Сборка и форматирование данных...")
    # Создаем DataFrame, где индекс - это время, а единственная колонка - это сам вектор (список чисел)
    df_embeddings = pd.DataFrame(index=pd.to_datetime(df_events[time_col]))
    
    # Сохраняем вектор как список float, чтобы parquet его корректно записал
    df_embeddings['sentiment_vector'] = embeddings.tolist()
    
    # Сортируем по времени на всякий случай
    df_embeddings.sort_index(inplace=True)

    # 5. Сохранение
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_embeddings.to_parquet(output_path, engine='pyarrow')
    
    print(f"ГОТОВО! 🎉 Эмбеддинги сохранены в {output_path}")
    print(f"Размерность одного вектора: {len(df_embeddings['sentiment_vector'].iloc[0])} признаков.")

if __name__ == "__main__":
    process_macro_embeddings()