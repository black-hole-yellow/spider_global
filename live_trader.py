import os
import logging
import pandas as pd
from dotenv import load_dotenv
from twisted.internet import reactor, task

# Импорты нашего Квантового конвейера
from live_pipeline import LivePipeline
from agents.global_agent import GlobalAlphaAgent
from agents.chief_agent import ChiefRiskOfficer
from trading.execution.ctrader_broker import CTraderBroker

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class CTraderLiveBot:
    def __init__(self):
        logging.info("🤖 Запуск cTrader HFT Системы...")
        self.pipeline = LivePipeline()
        self.alpha_agent = GlobalAlphaAgent()
        self.chief_agent = ChiefRiskOfficer(max_risk_per_trade_pct=0.02, min_confidence_threshold=3.0)
        
        self.symbol = "GBPUSD"

        # Загрузка ключей cTrader
        client_id = os.getenv("CTRADER_CLIENT_ID")
        client_secret = os.getenv("CTRADER_CLIENT_SECRET")
        account_id = os.getenv("CTRADER_ACCOUNT_ID")
        access_token = os.getenv("CTRADER_ACCESS_TOKEN")

        if not all([client_id, client_secret, account_id, access_token]):
            logging.error("❌ ОШИБКА: Ключи cTrader не найдены в .env!")
            return

        self.broker = CTraderBroker(
            client_id=client_id,
            client_secret=client_secret,
            account_id=account_id,
            access_token=access_token,
            is_demo=True # Поставь False для реальных денег
        )

    def start(self):
        # Запускаем подключение к брокеру. 
        # Когда авторизация пройдет, вызовется self.start_trading_loop
        self.broker.start_connection(on_ready_callback=self.start_trading_loop)
        
        logging.info("🚀 Запуск асинхронного Event Loop (Twisted)...")
        reactor.run() # Этот метод блокирует консоль и держит TCP соединение вечно

    def start_trading_loop(self):
        """Вызывается только после того, как брокер сказал 'Я готов'"""
        logging.info("✅ Система синхронизирована. Запуск цикла анализа...")
        
        # Вместо time.sleep(5) мы используем таймер Twisted
        self.loop = task.LoopingCall(self.run_iteration)
        self.loop.start(5.0) # Для Fast Test Mode ставим 5 секунд.

    def get_latest_data(self):
        # В этой версии мы продолжаем брать свечи из твоего parquet-файла
        # Для загрузки истории напрямую из cTrader нужен отдельный стриминговый класс
        try:
            return pd.read_parquet("data/raw/gbpusd_15m.parquet").iloc[-500:]
        except:
            return pd.DataFrame()

    def run_iteration(self):
        logging.info("--- Новый такт анализа ---")
        
        # 1. Запрашиваем актуальный баланс
        self.broker.update_market_state()
        
        raw_data = self.get_latest_data()
        if raw_data.empty: return

        # 2. In-Memory генерация
        processed_data = self.pipeline.process_live_data(raw_data)

        # 3. Трансформер
        signal = self.alpha_agent.analyze_market(processed_data)
        logging.info(f"🧠 Сигнал: {signal['direction']} (Уверенность: {signal['confidence_pct']}%)")

        # 4. Риск-менеджмент
        decision = self.chief_agent.review_signal(signal, processed_data, self.broker.equity)
        
        # 5. Исполнение
        if decision['decision'] == 'EXECUTE':
            self.broker.execute_command(decision, symbol_name=self.symbol)
        else:
            logging.info(f"🛡️ Пропуск: {decision['reason']}")

if __name__ == "__main__":
    bot = CTraderLiveBot()
    bot.start()