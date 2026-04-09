import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. Точная копия архитектуры из Ноутбука 7 для загрузки весов
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
            nn.Linear(32, 1) # Бинарная классификация
        )

    def forward(self, x):
        x = self.feature_projection(x)
        trans_out = self.transformer(x)
        last_step_out = trans_out[:, -1, :] 
        return self.fc_out(last_step_out)


# 2. Боевой класс Агента
class GlobalAlphaAgent:
    def __init__(self, 
                 model_path="data/processed/core_quantformer.pth", 
                 scaler_path="data/processed/core_scaler.pkl", 
                 pca_path="data/processed/core_pca.pkl"):
        
        features_path = "data/processed/core_features.pkl"
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"❌ Не найден список обученных фичей: {features_path}")
        
        # joblib сам открывает файл в правильном бинарном режиме ('rb')
        self.tech_features = joblib.load(features_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🤖 Инициализация Alpha Agent на устройстве: {self.device}")
        
        # Загрузка белого списка фичей (Alpha-R1)
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"❌ Не найден белый список фичей: {features_path}")
        with open(features_path, "r", encoding="utf-8") as f:
            self.tech_features = json.load(f)
            
        self.macro_cols = [f'macro_emb_{i}' for i in range(384)]
        self.seq_len = 32
        
        # Вычисляем размерность входа: N тех. фичей + 8 макро-компонент
        self.num_features = len(self.tech_features) + 8
        
        # Загрузка артефактов
        self.scaler = joblib.load(scaler_path)
        self.pca = joblib.load(pca_path)
        
        self.model = Quantformer(num_features=self.num_features).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Строго режим инференса!
        
        print(f"✅ Модель успешно загружена. Ожидает {self.num_features} признаков на вход.")

    def analyze_market(self, recent_data: pd.DataFrame) -> dict:
        """
        Принимает последние N свечей (DataFrame), возвращает вероятность.
        """
        if len(recent_data) < self.seq_len:
            return {"status": "error", "message": f"Нужно минимум {self.seq_len} свечей, получено {len(recent_data)}"}

        # Проверка наличия всех нужных колонок
        missing_cols = [col for col in self.tech_features + self.macro_cols if col not in recent_data.columns]
        if missing_cols:
            return {"status": "error", "message": f"Отсутствуют колонки в датасете: {missing_cols[:5]}..."}

        # 1. Извлекаем последние 32 свечи
        window_df = recent_data.iloc[-self.seq_len:]
        
        # 2. Обработка Макро (PCA)
        macro_raw = window_df[self.macro_cols].values
        macro_compressed = self.pca.transform(macro_raw)
        
        # 3. Сборка и Масштабирование
        tech_raw = window_df[self.tech_features].values
        combined_features = np.hstack([tech_raw, macro_compressed])
        scaled_features = self.scaler.transform(combined_features)
        
        # 4. Создание Тензора: Формат (Batch, Seq_Len, Features) -> (1, 32, 38)
        tensor_input = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 5. Прогноз Трансформера
        with torch.no_grad():
            logit = self.model(tensor_input)
            prob = torch.sigmoid(logit).item() # Превращаем логит в вероятность от 0.0 до 1.0
            
        # 6. Интерпретация
        direction = "LONG" if prob >= 0.5 else "SHORT"
        
        # Confidence (Уверенность): чем дальше от 0.5, тем сильнее сигнал.
        # Масштабируем от 0% (полная неопределенность) до 100% (абсолютная уверенность)
        confidence = abs(prob - 0.5) * 2 * 100 
        
        return {
            "status": "success",
            "direction": direction,
            "raw_probability": round(prob, 4),
            "confidence_pct": round(confidence, 2),
            "timestamp": str(window_df.index[-1])
        }

# Для теста, если запустить файл напрямую
if __name__ == "__main__":
    print("Проверка загрузки агента...")
    agent = GlobalAlphaAgent()
    print("Агент готов к работе в пайплайне!")