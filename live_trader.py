import time
import pandas as pd
from agents.global_agent import GlobalAlphaAgent
from agents.chief_agent import ChiefRiskOfficer

def run_paper_trader():
    print("🚀 Запуск Live Trader (Режим Paper Trading)...")
    
    # 1. Инициализация Агентов
    try:
        alpha_agent = GlobalAlphaAgent()
        # Chief Agent разрешает сделки только с уверенностью > 3.0% и риском 2%
        chief_agent = ChiefRiskOfficer(max_risk_per_trade_pct=0.02, min_confidence_threshold=3.0)
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return

    # 2. Подключение к потоку данных
    # В реальном времени здесь будет API запрос к Oanda/Binance. 
    # Сейчас берем последний срез нашего идеального датасета.
    dataset_path = "data/processed/full_merged_dataset.parquet"
    print(f"\n📊 Подключение к потоку данных: {dataset_path}")
    
    try:
        df = pd.read_parquet(dataset_path)
    except FileNotFoundError:
        print("❌ Датасет не найден. Сначала выполни сборку данных.")
        return
    
    # Имитируем "Текущий момент" - берем последние 32 свечи
    current_market_data = df.iloc[-32:].copy()
    current_price = current_market_data['close'].iloc[-1]
    current_time = current_market_data.index[-1]
    
    print(f"🕒 Текущее время на графике: {current_time} | Цена GBP/USD: {current_price:.5f}")
    print("=" * 60)

    # 3. ФАЗА 1: Генерация Альфы (Мозг)
    print("🧠 [ALPHA AGENT]: Анализ микроструктуры, энтропии и макро-векторов...")
    signal = alpha_agent.analyze_market(current_market_data)
    
    if signal['status'] != 'success':
        print(f"⚠️ Ошибка генерации сигнала: {signal.get('message')}")
        return
        
    print(f"   ► Сигнал: {signal['direction']}")
    print(f"   ► Сырая вероятность: {signal['raw_probability']:.4f}")
    print(f"   ► Уверенность нейросети: {signal['confidence_pct']}%")
    print("-" * 60)

    # 4. ФАЗА 2: Риск-Менеджмент (Босс)
    print("👔 [CHIEF AGENT]: Оценка волатильности (ATR) и расчет лотности...")
    account_balance = 10000.0 # Виртуальный депозит $10,000
    
    decision = chief_agent.review_signal(signal, current_market_data, account_balance)
    
    # 5. ФАЗА 3: Исполнение (Execution)
    if decision['decision'] == "EXECUTE":
        print("\n✅ СДЕЛКА ОДОБРЕНА В ПРОДАКШЕН (EXECUTE) ✅")
        print(f"   🔸 Инструмент:   GBP/USD")
        print(f"   🔸 Направление:  {decision['action']}")
        print(f"   🔸 Объем (Лот):  {decision['size_lots']}")
        print(f"   🔸 Цена Входа:   {decision['entry_price']:.5f}")
        print(f"   🔸 Stop Loss:    {decision['sl_price']:.5f} (Дистанция: 1.5 ATR)")
        print(f"   🔸 Take Profit:  {decision['tp_price']:.5f} (Дистанция: 3.0 ATR)")
        print(f"   🔸 Текущий ATR:  {decision['atr_usd']:.5f}")
    else:
        print("\n🚫 СДЕЛКА ОТКЛОНЕНА (HOLD) 🚫")
        print(f"   Причина: {decision['reason']}")
        
    print("=" * 60)
    print("Ожидание закрытия следующей 15-минутной свечи...")

if __name__ == "__main__":
    run_paper_trader()