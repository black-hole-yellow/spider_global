import os
import time
import datetime
import pandas as pd  # <--- КРИТИЧЕСКИ ВАЖНО
from dotenv import load_dotenv
from live_pipeline import LivePipeline # Проверь путь к файлу
from agents.global_agent import GlobalAlphaAgent
from agents.chief_agent import ChiefRiskOfficer
from trading.execution.bybit_broker import BybitBroker
# Загружаем ключи из .env
load_dotenv()

class BybitTradFiBot:
    def __init__(self):
        print("🤖 Инициализация Квантовой Системы Bybit...")
        
        # Проверка: загрузились ли ключи?
        key = os.getenv("BYBIT_KEY")
        secret = os.getenv("BYBIT_SECRET")
        
        if not key or not secret:
            print("❌ ОШИБКА: Ключи не найдены! Проверь файл .env и его расположение.")
        else:
            print(f"🔑 Ключи найдены: {key[:5]}***")

        self.pipeline = LivePipeline()
        self.alpha_agent = GlobalAlphaAgent()
        # ... остальной код без изменений
        self.chief_agent = ChiefRiskOfficer(max_risk_per_trade_pct=0.02, min_confidence_threshold=3.0)
        
        # Инициализируем брокера. Тестнет=True для безопасности!
        self.broker = BybitBroker(
            api_key=os.getenv("BYBIT_KEY"),
            api_secret=os.getenv("BYBIT_SECRET"),
            testnet=True 
        )
        self.symbol = "BTCUSDT"

    def run_iteration(self):
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] --- Новый цикл анализа ---")
        
        # 1. Обновляем состояние кошелька
        self.broker.update_market_state()
        print(f"💰 Баланс: ${self.broker.equity:.2f}")

        # 2. Получаем ЖИВЫЕ котировки с биржи
        raw_data = self.broker.get_live_klines(self.symbol, limit=500)
        if raw_data.empty: return

        # 3. Считаем фичи (In-Memory)
        processed_data = self.pipeline.process_live_data(raw_data)

        # 4. Прогноз нейросети
        signal = self.alpha_agent.analyze_market(processed_data)
        print(f"🧠 Сигнал: {signal['direction']} (Уверенность: {signal['confidence_pct']}%)")

        # 5. Риск-менеджмент и исполнение
        decision = self.chief_agent.review_signal(signal, processed_data, self.broker.equity)
        
        if decision['decision'] == 'EXECUTE':
            print(f"🔥 ИСПОЛНЯЕМ ОРДЕР: {decision['action']} {decision['size_lots']} лотов")
            self.broker.execute_command(decision)
        else:
            print(f"🛡️ Пропуск: {decision['reason']}")

    def start(self, fast_mode=True):
        print(f"🚀 Бот запущен. Режим: {'БЫСТРЫЙ ТЕСТ' if fast_mode else 'LIVE'}")
        
        while True:
            if not fast_mode:
                # Стандартная синхронизация с 15-минутным баром
                now = datetime.datetime.now()
                wait_time = (15 - (now.minute % 15)) * 60 - now.second + 2
                print(f"💤 Ожидание следующей свечи: {wait_time} сек.")
            else:
                # В режиме теста просто ждем 5 секунд между попытками
                wait_time = 5
                print(f"🧪 ТЕСТОВЫЙ ЗАПУСК. Пауза {wait_time} сек...")
            
            time.sleep(wait_time)
            
            try:
                self.run_iteration()
                
                # Если мы в режиме теста и успешно совершили итерацию, 
                # можно даже выйти после первого раза, чтобы не спамить ордерами.
                # if fast_mode: break 
                
            except Exception as e:
                print(f"💥 Ошибка в цикле: {e}")
                time.sleep(10)

if __name__ == "__main__":
    bot = BybitTradFiBot()
    bot.start()