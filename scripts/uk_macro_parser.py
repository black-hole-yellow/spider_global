import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

def load_and_clean_csv(filepath, value_name):
    """Вспомогательная функция для загрузки CSV и переименования колонок"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл {filepath} не найден!")
        
    df = pd.read_csv(filepath)
    
    # Пытаемся найти колонку с датой (Date, DATE, date, observation_date)
    date_col = [col for col in df.columns if 'date' in col.lower()][0]
    # Оставшаяся колонка - это наши значения
    val_col = [col for col in df.columns if col != date_col][0]
    
    df = df.rename(columns={date_col: 'date', val_col: value_name})
    df['date'] = pd.to_datetime(df['date'])
    return df[['date', value_name]]

def build_uk_macro_embeddings():
    print("1. Загрузка локальных CSV файлов Великобритании...")
    
    # Укажи пути к твоим скачанным файлам
    cpi_path = "data/raw/fred/UKCPI.csv"
    rate_path = "data/raw/fred/UKINRATE.csv"
    unemp_path = "data/raw/fred/UKUN.csv"
    
    df_cpi = load_and_clean_csv(cpi_path, 'uk_cpi')
    df_rate = load_and_clean_csv(rate_path, 'uk_rate')
    df_unemp = load_and_clean_csv(unemp_path, 'uk_unemp')

    print("2. Объединение и выравнивание (Forward Fill)...")
    # Создаем единый календарь дат (с 2000 года)
    all_dates = pd.date_range(start='2000-01-01', end=pd.Timestamp.today(), freq='D')
    df_macro = pd.DataFrame({'date': all_dates})
    
    # Джойним все данные к нашему календарю
    df_macro = pd.merge(df_macro, df_cpi, on='date', how='left')
    df_macro = pd.merge(df_macro, df_rate, on='date', how='left')
    df_macro = pd.merge(df_macro, df_unemp, on='date', how='left')
    
    # Заполняем пустоты (макро данные действуют, пока не выйдут новые)
    df_macro = df_macro.ffill().dropna()

    print(f"Подготовлено {len(df_macro)} макро-дней.")

    print("3. Синтез новостей Банка Англии и векторизация (SentenceTransformers)...")
    # Загружаем легковесную модель для эмбеддингов (выдает ровно 384 числа)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Чтобы не считать эмбеддинги для каждого дня (что долго), 
    # мы считаем их только для уникальных макро-состояний
    df_macro['macro_state_str'] = df_macro.apply(
        lambda row: f"Bank of England Official Bank Rate is {row['uk_rate']}%. UK Consumer Price Index (Inflation) is {row['uk_cpi']}%. UK Unemployment Rate is {row['uk_unemp']}%", 
        axis=1
    )
    
    unique_states = df_macro['macro_state_str'].unique()
    print(f"Найдено {len(unique_states)} уникальных макро-эпох. Считаем векторы...")
    
    # Генерируем эмбеддинги словарем для быстрого маппинга
    state_to_embedding = {}
    embeddings = model.encode(unique_states, show_progress_bar=True)
    
    for state, emb in zip(unique_states, embeddings):
        state_to_embedding[state] = emb
        
    print("4. Маппинг векторов обратно в календарь...")
    # Разворачиваем матрицу эмбеддингов в колонки
    emb_list = df_macro['macro_state_str'].map(state_to_embedding).tolist()
    emb_df = pd.DataFrame(emb_list, columns=[f'uk_macro_emb_{i}' for i in range(384)])
    
    # Собираем финальный датасет (Дата + 384 колонки)
    df_final = pd.concat([df_macro[['date']].reset_index(drop=True), emb_df], axis=1)
    
    # Сохраняем
    output_path = "data/processed/uk_macro_embeddings.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_parquet(output_path)
    
    print(f"✅ УСПЕХ! Британские эмбеддинги сохранены в {output_path}")
    print(f"Размерность: {df_final.shape} (Должно быть N строк и 385 колонок: 1 дата + 384 эмбеддинга)")

if __name__ == "__main__":
    build_uk_macro_embeddings()