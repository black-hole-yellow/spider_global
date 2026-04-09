import time
import datetime
import pandas as pd
from live_pipeline import LivePipeline
from agents.global_agent import GlobalAlphaAgent
from agents.chief_agent import ChiefRiskOfficer
from trading.execution.paper_broker import PaperBroker
from trading.execution.capital_broker import OandaBroker

class LiveOrchestrator:
    def __init__(self):
        print("Запуск Global Orchestrator...")
        self.pipeline = LivePipeline()
        self.alpha_agent = GlobalAlphaAgent()
        self.chief_agent = ChiefRiskOfficer(max_risk_per_trade_pct=0.02, min_confidence_threshold=3.0)

        # ПОДКЛЮЧАЕМ БРОКЕРА
        self.broker = PaperBroker() 
        # В идеале баланс нужно запрашивать так: self.account_balance = self.broker.get_balance()
        self.account_balance = 10000.0

    def fetch_live_data(self) -> pd.DataFrame:
        return self.broker.get_historical_data(symbol="GBP/USD", timeframe="15m", limit=1000)

    def wait_for_next_candle(self, timeframe_minutes=15):
        """Синхронизация времени: бот спит до закрытия следующей 15-минутной свечи"""
        now = datetime.datetime.now()
        # Считаем, сколько минут осталось до следующего интервала (00, 15, 30, 45)
        minutes_to_wait = timeframe_minutes - (now.minute % timeframe_minutes)
        # Вычитаем секунды, чтобы проснуться ровно в 00 секунд
        seconds_to_wait = (minutes_to_wait * 60) - now.second + 2 # +2 сек запаса, чтобы брокер успел закрыть свечу
        
        next_run = now + datetime.timedelta(seconds=seconds_to_wait)
        print(f"\n💤 Ожидание закрытия свечи. Следующий запуск: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(seconds_to_wait)


    def execute_trade(self, decision: dict):
        """Отправка приказа Брокеру"""
        print("\n" + "="*50)
        print("⚡ ОТПРАВКА ОРДЕРА БРОКЕРУ ⚡")
        print(f"Инструмент: GBP/USD | Action: {decision['action']} | Size: {decision['size_lots']} Lots")
        print(f"Entry: Market | SL: {decision['sl_price']} | TP: {decision['tp_price']}")
        print("="*50 + "\n")
        
        # РЕАЛЬНЫЙ ВЫЗОВ
        try:
            # Названия аргументов зависят от твоего класса PaperBroker
            self.broker.place_order(
                symbol="GBP/USD",
                order_type="MARKET",
                side=decision['action'], # "LONG" / "SHORT"
                volume=decision['size_lots'],
                stop_loss=decision['sl_price'],
                take_profit=decision['tp_price']
            )
            print("✅ Ордер успешно передан брокеру!")
        except Exception as e:
            print(f"❌ Ошибка отправки ордера: {e}")

    def run(self):
        print("🚀 Оркестратор переведен в режим LIVE TRADING")
        
        while True:
            try:
                # 1. Ждем закрытия 15-минутной свечи
                self.wait_for_next_candle(timeframe_minutes=15)
                
                print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 🔄 Новая свеча! Начинаем цикл...")
                
                # 2. Скачиваем котировки
                raw_data = self.fetch_live_data()
                
                # 3. In-Memory генерация 500+ фичей
                start_time = time.time()
                processed_data = self.pipeline.process_live_data(raw_data)
                print(f"✅ Фичи рассчитаны за {time.time() - start_time:.2f} сек.")
                
                # 4. Мозг (Трансформер) генерирует Альфу
                signal = self.alpha_agent.analyze_market(processed_data)
                if signal['status'] != 'success':
                    print(f"⚠️ Ошибка Агента: {signal.get('message')}")
                    continue
                
                print(f"🧠 Сигнал: {signal['direction']} (Уверенность: {signal['confidence_pct']}%)")
                
                # 5. Босс (Chief Risk Officer) проверяет риски
                decision = self.chief_agent.review_signal(signal, processed_data, self.account_balance)
                
                # 6. Исполнение
                if decision['decision'] == 'EXECUTE':
                    self.execute_trade(decision)
                else:
                    print(f"🛡️ Chief Agent отклонил сделку. Причина: {decision['reason']}")

            except KeyboardInterrupt:
                print("\n🛑 Робот остановлен пользователем.")
                break
            except Exception as e:
                print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА В ГЛАВНОМ ЦИКЛЕ: {e}")
                print("Перезапуск цикла через 60 секунд...")
                time.sleep(60)

if __name__ == "__main__":
    orchestrator = LiveOrchestrator()
    orchestrator.run()