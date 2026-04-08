import pandas as pd
import numpy as np
import json
import os

# Импортируем нашего настоящего Агента США
from us_agent import USAgent

class MockUKAgent:
    """Временная заглушка для Агента Британии, пока мы не обучим для него Трансформер"""
    def analyze(self):
        # Допустим, Банк Англии тоже настроен "по-медвежьи" для фунта
        return json.dumps({
            "agent": "UK_Macro_Quant",
            "direction": "SHORT",
            "probability": 0.3500,
            "confidence": 0.6500,
            "reasoning": "Mock: Слабые данные по ВВП Британии."
        })

class ChiefAgent:
    def __init__(self, us_agent):
        self.us_agent = us_agent
        self.uk_agent = MockUKAgent()
        
        # Настройки риск-менеджмента
        self.min_confidence_threshold = 0.60 # Минимальная средняя уверенность для входа
        
    def get_consensus(self, market_data):
        print("\n[CHIEF] 📞 Запрос аналитики у Агента США...")
        us_report = json.loads(self.us_agent.analyze(market_data))
        print(f"   -> US: {us_report['direction']} (Уверенность: {us_report['confidence']:.1%})")
        
        print("[CHIEF] 📞 Запрос аналитики у Агента Британии (Mock)...")
        uk_report = json.loads(self.uk_agent.analyze())
        print(f"   -> UK: {uk_report['direction']} (Уверенность: {uk_report['confidence']:.1%})")
        
        print("\n[CHIEF] ⚖️ Сведение консенсуса...")
        
        # 1. Проверка на конфликт
        if us_report['direction'] != uk_report['direction']:
            return self._format_decision("HOLD", 0, "Конфликт Агентов. US и UK смотрят в разные стороны. Сделка отменена.")
            
        # 2. Агенты согласны. Считаем среднюю уверенность
        avg_confidence = (us_report['confidence'] + uk_report['confidence']) / 2
        direction = us_report['direction']
        
        # 3. Проверка риск-фильтра
        if avg_confidence < self.min_confidence_threshold:
             return self._format_decision("HOLD", avg_confidence, f"Агенты согласны на {direction}, но общая уверенность ({avg_confidence:.1%}) ниже порога риска ({self.min_confidence_threshold:.1%}).")
             
        # 4. ФИНАЛЬНЫЙ СИГНАЛ (ЗЕЛЕНЫЙ СВЕТ)
        return self._format_decision(direction, avg_confidence, f"Строгий консенсус достигнут. Средняя уверенность: {avg_confidence:.1%}.")

    def _format_decision(self, action, confidence, reasoning):
        return json.dumps({
            "chief_action": action,
            "final_confidence": round(confidence, 4),
            "reasoning": reasoning,
            "ensure_ascii": False
        }, indent=4)

if __name__ == "__main__":
    # Укажи правильные пути к файлам
    model_file = "data/processed/us_quantformer.pth"
    scaler_file = "data/processed/us_scaler.pkl"
    pca_file = "data/processed/us_pca.pkl"
    parquet_file = "data/processed/full_merged_dataset.parquet"
    
    try:
        # Инициализируем реального Агента США
        real_us_agent = USAgent(model_file, scaler_file, pca_file)
        
        # Инициализируем Главного Агента
        chief = ChiefAgent(real_us_agent)
        
        # Грузим данные
        df = pd.read_parquet(parquet_file)
        cols_to_use = real_us_agent.tech_features + real_us_agent.macro_cols
        df_clean = df[cols_to_use].replace([np.inf, -np.inf], np.nan).dropna()
        recent_market_window = df_clean.tail(32)
        
        # Запускаем консенсус
        final_decision = chief.get_consensus(recent_market_window)
        
        print("\n==================================")
        print("🚀 ФИНАЛЬНОЕ РЕШЕНИЕ ОРКЕСТРАТОРА:")
        print("==================================")
        print(final_decision)
        
    except Exception as e:
        print(f"Ошибка: {e}")