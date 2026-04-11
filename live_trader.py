import os
import logging
import pandas as pd
from dotenv import load_dotenv
from twisted.internet import reactor, task

from live_pipeline import LivePipeline
from agents.global_agent import GlobalAlphaAgent
from agents.chief_agent import ChiefRiskOfficer
from execution.ctrader_broker import CTraderBroker
# --- ДОБАВЬ ИМПОРТ СТРИМЕРА ---
from execution.ctrader_broker import CTraderStream

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class CTraderLiveBot:
    def __init__(self):
        logging.info("Запуск cTrader HFT Системы...")
        self.pipeline = LivePipeline()
        self.alpha_agent = GlobalAlphaAgent()
        self.chief_agent = ChiefRiskOfficer()
        self.symbol = "GBPUSD"

        # ... (Код загрузки ключей и создания self.broker остается прежним) ...

        # Инициализируем стример, передавая ему брокера
        self.streamer = CTraderStream(self.broker, symbol_name=self.symbol)

    def start(self):
        # 1. Сначала ждем готовности Брокера (авторизация TCP)
        self.broker.start_connection(on_ready_callback=self._on_broker_ready)
        
        logging.info("🚀 Запуск асинхронного Event Loop (Twisted)...")
        reactor.run() 

    def _on_broker_ready(self):
        # 2. Когда брокер готов, запускаем Стример (скачивание истории)
        logging.info("✅ Брокер готов. Подключаем дата-стрим...")
        self.streamer.start_stream(on_data_ready_callback=self.start_trading_loop)

    def start_trading_loop(self):
        # 3. Когда стример готов, запускаем таймер анализа
        logging.info("✅ Данные синхронизированы. Запуск цикла анализа...")
        
        # Вызываем цикл каждые 5 секунд (асинхронно)
        self.loop = task.LoopingCall(self.run_iteration)
        self.loop.start(5.0) 

    def run_iteration(self):
        logging.info("--- Новый такт анализа ---")
        
        # 1. Запрашиваем актуальный баланс
        self.broker.update_market_state()
        
        # 2. БЕРЕМ ЖИВЫЕ ДАННЫЕ ИЗ СТРИМЕРА (Больше никаких parquet!)
        raw_data = self.streamer.get_dataframe()
        
        if raw_data.empty or len(raw_data) < 100: 
            logging.warning("⚠️ Ожидание накопления свечей...")
            return

        try:
            # 3. In-Memory генерация (через новый статический граф)
            processed_data = self.pipeline.process_live_data(raw_data)

            # 4. Трансформер
            signal = self.alpha_agent.analyze_market(processed_data)
            logging.info(f"🧠 Сигнал: {signal.get('direction', 'NONE')} (Уверенность: {signal.get('confidence_pct', 0)}%)")

            # 5. Риск-менеджмент
            decision = self.chief_agent.review_signal(signal, processed_data, self.broker.equity)
            
            # 6. Исполнение
            if decision['decision'] == 'EXECUTE':
                self.broker.execute_command(decision, symbol_name=self.symbol)
            else:
                logging.info(f"🛡️ Пропуск: {decision['reason']}")

        except Exception as e:
            logging.error(f"❌ Ошибка в цикле расчета: {e}")

if __name__ == "__main__":
    bot = CTraderLiveBot()
    bot.start()