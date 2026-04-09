import pandas as pd
import numpy as np
import os
import importlib
import inspect

def load_raw_data():
    """Загрузка сырых котировок (OHLCV)"""
    # Поддерживаем и CSV, и Parquet
    csv_path = "data/raw/GBPUSD_15m.csv"
    parquet_path = "data/processed/gbpusd_15m.parquet"
    
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Ищем колонку с датой (time, date, timestamp)
        date_col = next((c for c in df.columns if c.lower() in ['time', 'date', 'timestamp']), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    else:
        raise FileNotFoundError("❌ Не найден файл сырых данных в data/raw/")
        
    # Приводим названия колонок к нижнему регистру (open, high, low, close, volume)
    df.columns = [c.lower() for c in df.columns]
    df.sort_index(inplace=True)
    return df

def generate_dynamic_dataset():
    print("1. Загрузка сырых OHLCV данных...")
    df = load_raw_data()
    print(f"Загружено {len(df)} свечей.")

    print("\n2. Динамическая генерация фичей (Meta-Programming)...")
    features_dir = "shared/features"
    
    # Находим все .py файлы в папке фичей (кроме системных)
    modules = [f[:-3] for f in os.listdir(features_dir) 
               if f.endswith('.py') and f not in ['__init__.py', 'decorators.py']]
    
    for mod_name in modules:
        print(f"  📦 Модуль: {mod_name}.py")
        # Динамически импортируем модуль
        module = importlib.import_module(f"shared.features.{mod_name}")
        
        # Ищем все функции, которые начинаются с 'add_'
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if name.startswith('add_'):
                try:
                    df = func(df)
                    print(f"    ✅ Выполнено: {name}")
                except Exception as e:
                    print(f"    ❌ Ошибка в {name}: {e}")

    print("\n3. Базовая очистка...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Заполняем пропуски (современный синтаксис Pandas 2.0+)
    df.ffill(inplace=True)
    df.fillna(0, inplace=True) 

    output_path = "data/processed/gbpusd_with_all_features.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine='pyarrow')
    
    print(f"✅ УСПЕХ! Сгенерировано {len(df.columns)} колонок.")
    print(f"💾 Датасет сохранен в: {output_path}")

if __name__ == "__main__":
    generate_dynamic_dataset()