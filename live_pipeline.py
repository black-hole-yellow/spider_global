import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 🧠 STATIC FEATURE GRAPH IMPORTS
# ---------------------------------------------------------
# We import the Master Wrappers from the central registry (features/__init__.py).
# This guarantees execution order and sub-millisecond latency.
from features import (
    add_technical_features,
    add_structural_features,
    add_htf_features,
    add_session_features,
    add_ml_features,
    add_macro_features
)

class LivePipeline:
    def __init__(self, macro_path="data/processed/sentiment_embeddings.parquet"):
        self.macro_path = macro_path
        print("⚡ Инициализация Low-Latency Pipeline (Статический Граф)")

    def process_live_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Главный метод генерации фичей для Live-торговли.
        Время выполнения должно быть < 50мс.
        """
        if raw_df.empty:
            return raw_df

        df = raw_df.copy()
        # Стандартизация базовых колонок (OHLCV)
        df.columns = [c.lower() for c in df.columns]

        # =========================================================
        # СТАТИЧЕСКИЙ ГРАФ ВЫЧИСЛЕНИЙ
        # Порядок СТРОГО фиксирован, так как слои зависят друг от друга.
        # =========================================================
        try:
            # Layer 1: Базовые индикаторы, ATR, Возвраты
            df = add_technical_features(df)
            
            # Layer 2: СМК, Ликвидность, Дивергенции объемов (зависит от ATR)
            df = add_structural_features(df)
            
            # Layer 3: Предыдущие дни/недели, MTFA (зависит от структуры)
            df = add_htf_features(df)
            
            # Layer 4: Время, Сессии, Перехват ликвидности
            df = add_session_features(df)
            
            # Layer 5: GMM Режимы, Херст, Изломы (зависит от сессий и технички)
            df = add_ml_features(df)
            
            # Layer 6: Эвенты, LLM Сентимент, Стратегические триггеры
            df = add_macro_features(df)
            
        except Exception as e:
            # В HFT лучше упасть сразу с четкой ошибкой, чем торговать на битых данных
            raise RuntimeError(f"❌ Критическая ошибка расчета графа фичей: {str(e)}")

        # =========================================================
        # ИНЪЕКЦИЯ ВНЕШНИХ ДАННЫХ И ОЧИСТКА
        # =========================================================
        # Подтягиваем Макро-контекст (Sentiment Embeddings)
        df = self._inject_macro_embeddings(df)

        # Санитаризация данных (Очистка от NaN и Inf перед подачей в Нейросеть)
        df = self._sanitize_data(df)

        return df

    def _inject_macro_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Инъекция последних макро-эмбеддингов (обновляются асинхронно парсером)"""
        if os.path.exists(self.macro_path):
            try:
                df_macro = pd.read_parquet(self.macro_path)
                if not df_macro.empty:
                    latest_macro = df_macro.iloc[-1:] 
                    for col in latest_macro.columns:
                        df[col] = latest_macro[col].values[0]
                    return df
            except Exception as e:
                print(f"⚠️ Ошибка чтения макро-данных: {e}")

        # Fallback: Если файла нет или он битый, забиваем нулями, чтобы размерность 
        # тензора (Quantformer) не сломалась и бот продолжил работу на технических данных.
        for i in range(384):
            df[f'macro_emb_{i}'] = 0.0
            
        return df

    def _sanitize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Финальная подготовка тензора для нейросети"""
        # Убиваем бесконечности (часто появляются при делении на нулевой объем)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Протягиваем последние валидные значения (защита от дыр в тиках)
        if hasattr(df, 'ffill'):
            df.ffill(inplace=True)
        else:
            df.fillna(method='ffill', inplace=True)
            
        # Находим ТОЛЬКО числовые колонки (float, int) и заполняем оставшиеся пустоты 
        # в начале графика (там где не хватило lookback окна) нулями
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

if __name__ == "__main__":
    print("Тестирование Low-Latency Pipeline...")
    dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='15min')
    dummy_df = pd.DataFrame({
        'open': np.random.randn(500).cumsum() + 1.3000,
        'high': np.random.randn(500).cumsum() + 1.3020,
        'low': np.random.randn(500).cumsum() + 1.2980,
        'close': np.random.randn(500).cumsum() + 1.3010,
        'volume': np.random.randint(100, 1000, size=500)
    }, index=dates)
    
    pipeline = LivePipeline()
    
    import time
    start = time.time()
    try:
        res = pipeline.process_live_data(dummy_df)
        print(f"✅ Успех! Расчет графа занял {(time.time() - start)*1000:.2f} мс.")
        print(f"📊 Выходная размерность: {res.shape}")
    except Exception as e:
        print(f"⚠️ Граф еще не собран. Ошибка: {e}")