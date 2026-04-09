import pandas as pd
import numpy as np
import importlib
import inspect
import os
import warnings

warnings.filterwarnings('ignore')

class LivePipeline:
    def __init__(self, features_dir="shared/features", macro_path="data/processed/sentiment_embeddings.parquet"):
        self.features_dir = features_dir
        self.macro_path = macro_path
        print("⚡ Инициализация In-Memory Pipeline (Двухпроходный расчет)")

    def process_live_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()
        df.columns = [c.lower() for c in df.columns]

        # 1. Собираем все функции генерации из папки
        modules = [f[:-3] for f in os.listdir(self.features_dir) 
                   if f.endswith('.py') and f not in ['__init__.py', 'decorators.py']]
        
        funcs_to_run = []
        for mod_name in modules:
            module = importlib.import_module(f"shared.features.{mod_name}")
            for name, func in inspect.getmembers(module, inspect.isfunction):
                if name.startswith('add_'):
                    funcs_to_run.append((name, func))

        # 2. PASS 1: Пытаемся посчитать всё
        retry_queue = []
        for name, func in funcs_to_run:
            try:
                df = func(df)
            except Exception as e:
                # Откладываем функции, которым не хватило зависимостей
                retry_queue.append((name, func, str(e)))

        # 3. PASS 2: Повторный запуск карантинных функций
        for name, func, err_msg in retry_queue:
            try:
                df = func(df)
            except Exception as e:
                # Если функция все равно падает, пишем мягкий варнинг
                print(f"⚠️ Пропущена фича {name} (Возможно, макро-парсер без файлов): {e}")

        # 4. Подтягиваем последнее актуальное Макро
        if os.path.exists(self.macro_path):
            df_macro = pd.read_parquet(self.macro_path)
            latest_macro = df_macro.iloc[-1:] 
            for col in latest_macro.columns:
                df[col] = latest_macro[col].values[0]
        else:
            for i in range(384):
                df[f'macro_emb_{i}'] = 0.0

        # 5. Очистка датасета (Исправлен баг со строками)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Протягиваем последние валидные значения
        if hasattr(df, 'ffill'):
            df.ffill(inplace=True)
        else:
            df.fillna(method='ffill', inplace=True)
            
        # Находим ТОЛЬКО числовые колонки (float, int) и заполняем оставшиеся пустоты нулями
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

if __name__ == "__main__":
    print("Тестирование In-Memory Pipeline на сгенерированных данных...")
    dates = pd.date_range(end=pd.Timestamp.now(), periods=1000, freq='15min')
    dummy_df = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 1.3000,
        'high': np.random.randn(1000).cumsum() + 1.3020,
        'low': np.random.randn(1000).cumsum() + 1.2980,
        'close': np.random.randn(1000).cumsum() + 1.3010,
        'volume': np.random.randint(100, 1000, size=1000)
    }, index=dates)
    
    pipeline = LivePipeline()
    
    import time
    start_time = time.time()
    processed_df = pipeline.process_live_data(dummy_df)
    exec_time = time.time() - start_time
    
    print(f"✅ Успех! Расчет 1000 свечей занял {exec_time:.3f} секунд.")
    print(f"📊 Размерность данных на выходе: {processed_df.shape[0]} строк, {processed_df.shape[1]} колонок.")