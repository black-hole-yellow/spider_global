import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import json
import os

# 1. Архитектура Трансформера (Одинаковая для обоих агентов)
class Quantformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2):
        super(Quantformer, self).__init__()
        self.feature_projection = nn.Linear(num_features, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, 
            batch_first=True, dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.feature_projection(x)
        trans_out = self.transformer(x)
        last_step_out = trans_out[:, -1, :] 
        return self.fc_out(last_step_out)

# 2. Класс Агента Британии
class UKAgent:
    def __init__(self, model_path, scaler_path, pca_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[UK Agent] Инициализация на {self.device}...")
        
        # Те же самые технические индикаторы (техническая картина едина для пары GBP/USD)
        self.tech_features = [
            "rsi", "premium_discount", "dist_to_liq_high", "dist_to_liq_low", 
            "struct_trend", "Norm_Slope", "dist_to_pdh", "fvg_bear_size", 
            "fvg_bull_size", "dist_to_pdl", "trend_1h", "mtfa_score"
        ]
        
        # ЭТО ВАЖНО: Агент Британии будет искать СВОИ макро-эмбеддинги
        self.macro_cols = [f'uk_macro_emb_{i}' for i in range(384)]
        
        self.num_features = len(self.tech_features) + 8
        self.seq_len = 32
        
        # Загрузка артефактов БРИТАНСКОЙ модели
        self.model = Quantformer(num_features=self.num_features).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            print(f"⚠️ [UK Agent] ВНИМАНИЕ: Файл {model_path} не найден!")
            print("Пока модель не обучена, Агент будет работать в режиме MOCK (случайная выдача).")
            self.mock_mode = True
            return
            
        self.mock_mode = False
        self.scaler = joblib.load(scaler_path)
        self.pca = joblib.load(pca_path)
        print("[UK Agent] Готов к работе. Модель Банка Англии загружена.")

    def analyze(self, recent_data_df):
        """Анализ со стороны Британской экономики"""
        # Если модель еще не обучена, выдаем временный результат
        if self.mock_mode:
            return json.dumps({
                "agent": "UK_Macro_Quant",
                "direction": "LONG",
                "probability": 0.6200,
                "confidence": 0.6200,
                "reasoning": "[MOCK MODE] Модель не найдена. Симуляция: Банк Англии настроен агрессивно (hawkish)."
            }, indent=4)

        if len(recent_data_df) < self.seq_len:
            return json.dumps({"error": "Недостаточно свечей"})
        
        # Технические фичи: убираем inf и протягиваем последние значения, если есть пропуски
        tech_data = recent_data_df[self.tech_features].replace([np.inf, -np.inf], np.nan).ffill().bfill().values
        
        # Макро фичи: если новостей не было, заполняем нулями (нейтральный вектор)
        macro_data = recent_data_df[self.macro_cols].fillna(0.0).values

        macro_compressed = self.pca.transform(macro_data)
        combined_features = np.hstack([tech_data, macro_compressed])
        scaled_features = self.scaler.transform(combined_features)

        X_tensor = torch.tensor(scaled_features[-self.seq_len:], dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_logit = self.model(X_tensor)
            probability = torch.sigmoid(output_logit).item()

        direction = "LONG" if probability >= 0.5 else "SHORT"
        confidence = probability if direction == "LONG" else (1 - probability)

        response = {
            "agent": "UK_Macro_Quant",
            "direction": direction,
            "probability": round(probability, 4),
            "confidence": round(confidence, 4),
            "reasoning": f"Британский Трансформер оценивает вероятность роста как {probability:.1%}. Сигнал: {direction}."
        }
        
        return json.dumps(response, indent=4)

if __name__ == "__main__":
    # Укажи правильные пути к сохраненным файлам из твоего ноутбука
    model_file = "data/processed/uk_quantformer.pth"
    scaler_file = "data/processed/uk_scaler.pkl"
    pca_file = "data/processed/uk_pca.pkl"
    parquet_file = "data/processed/full_merged_dataset.parquet"
    
    try:
        agent = UKAgent(model_file, scaler_file, pca_file)
        
        print("\nЗагрузка последних данных с рынка...")
        df = pd.read_parquet(parquet_file)
        
        # 1. Извлекаем ТОЛЬКО нужные колонки (чтобы избежать конфликтов типов данных)
        cols_to_use = agent.tech_features + agent.macro_cols
        df_clean = df[cols_to_use].copy()
        
        # 2. Очищаем данные БЕЗ удаления важных свечей
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Для макро-колонок NaN - это нормально (нет новостей), ставим 0
        df_clean[agent.macro_cols] = df_clean[agent.macro_cols].fillna(0.0)
        
        # Для технических колонок делаем ffill (протягиваем предыдущее значение)
        df_clean[agent.tech_features] = df_clean[agent.tech_features].ffill()
        
        # Дропаем только если в самом начале графика вообще нет технических данных
        df_clean = df_clean.dropna()
        
        # 3. Берем последние 32 идеальные свечи
        recent_market_window = df_clean.tail(32)
        
        # 4. Агент выносит вердикт
        decision = agent.analyze(recent_market_window)
        print("\n🎯 ВЕРДИКТ АГЕНТА ВЕЛИКОБРИТАНИИ:")
        print(decision)
        
    except Exception as e:
        print(f"Ошибка тестирования: {e}")