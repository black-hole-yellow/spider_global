import logging
import pandas as pd
from pybit.unified_trading import HTTP

class BybitBroker:
    def __init__(self, api_key, api_secret, testnet=True):
        self.api_key = api_key
        self.api_secret = api_secret
        # Инициализируем официальный клиент Bybit
        self.session = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret,
        )
        self.equity = 0.0
        logging.info(f"🔌 Bybit V5 (pybit) Initialized")

    def update_market_state(self):
        """Официальный способ получения баланса через UTA"""
        try:
            res = self.session.get_wallet_balance(accountType="UNIFIED")
            if res['retCode'] == 0:
                self.equity = float(res['result']['list'][0]['totalEquity'])
                logging.info(f"💰 Актуальный баланс: ${self.equity}")
            else:
                logging.error(f"❌ Ошибка Bybit: {res['retMsg']}")
        except Exception as e:
            logging.error(f"❌ Ошибка авторизации: {e}")
            self.equity = 0.0

    def get_live_klines(self, symbol="BTCUSDT", limit=200):
        """Получение котировок без мучений с подписью"""
        try:
            res = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=15,
                limit=limit
            )
            if res['retCode'] == 0:
                data = res['result']['list']
                df = pd.DataFrame(data, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'to'])
                df = df.astype(float)
                df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                return df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
        except Exception as e:
            logging.error(f"❌ Ошибка данных: {e}")
        return pd.DataFrame()

    def execute_command(self, decision):
        """Выставление ордера через официальный метод"""
        try:
            side = "Buy" if decision['action'] == "LONG" else "Sell"
            res = self.session.place_order(
                category="linear",
                symbol="BTCUSDT",
                side=side,
                orderType="Market",
                qty="0.001",
                timeInForce="GTC",
            )
            if res['retCode'] == 0:
                logging.info(f"✅ ОРДЕР ВЫПОЛНЕН: {side}")
            else:
                logging.error(f"❌ Ошибка ордера: {res['retMsg']}")
        except Exception as e:
            logging.error(f"❌ Ошибка исполнения: {e}")