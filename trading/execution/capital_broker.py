import requests
import logging
import pandas as pd
import datetime

class CapitalComBroker:
    def __init__(self, api_key: str, identifier: str, password: str, is_demo: bool = True):
        # В Capital.com разные URL для демо и реальных счетов
        self.base_url = "https://demo-api-capital.backend-capital.com" if is_demo else "https://api-capital.backend-capital.com"
        self.api_key = api_key
        self.identifier = identifier
        self.password = password
        
        self.cst = None
        self.x_security_token = None
        self.equity = 0.0
        
        logging.info(f"🔌 Инициализация Capital.com Broker ({'DEMO' if is_demo else 'LIVE'})")
        self._authenticate()

    def _authenticate(self):
        """Получение сессионных токенов (CST и X-SECURITY-TOKEN)"""
        url = f"{self.base_url}/api/v1/session"
        headers = {
            "X-CAP-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "identifier": self.identifier,
            "password": self.password
        }
        
        res = requests.post(url, json=payload, headers=headers)
        
        if res.status_code == 200:
            self.cst = res.headers.get('CST')
            self.x_security_token = res.headers.get('X-SECURITY-TOKEN')
            logging.info("✅ Capital.com: Успешная авторизация сессии")
        else:
            logging.error(f"❌ Ошибка авторизации Capital.com: {res.text}")

    def _get_headers(self):
        """Генерация заголовков для защищенных запросов"""
        if not self.cst or not self.x_security_token:
            self._authenticate()
            
        return {
            "X-CAP-API-KEY": self.api_key,
            "CST": self.cst,
            "X-SECURITY-TOKEN": self.x_security_token,
            "Content-Type": "application/json"
        }

    def update_market_state(self):
        """Получение баланса аккаунта"""
        url = f"{self.base_url}/api/v1/clientaccount"
        res = requests.get(url, headers=self._get_headers())
        
        if res.status_code == 200:
            try:
                data = res.json()
                accounts = data.get('accounts', [])
                if accounts:
                    # Берем баланс (эквити) первого аккаунта
                    self.equity = float(accounts[0].get('balance', {}).get('balance', 0.0))
                    logging.info(f"💰 Баланс Capital.com: ${self.equity}")
            except Exception as e:
                logging.error(f"❌ Ошибка парсинга баланса: {e}")
        else:
            logging.error(f"❌ Не удалось получить баланс: {res.text}")

    def get_live_klines(self, epic="GBPUSD", limit=200):
        """
        Получение котировок (У Capital.com инструменты называются epics).
        Resolution: MINUTE_15
        """
        url = f"{self.base_url}/api/v1/prices/{epic}"
        params = {
            "resolution": "MINUTE_15",
            "max": limit
        }
        res = requests.get(url, headers=self._get_headers(), params=params)
        
        if res.status_code == 200:
            prices = res.json().get('prices', [])
            records = []
            for p in prices:
                # Capital.com отдает цены bid и ask отдельно. Берем среднее или bid.
                close_price = p['closePrice']['bid']
                open_price = p['openPrice']['bid']
                high_price = p['highPrice']['bid']
                low_price = p['lowPrice']['bid']
                volume = p['lastTradedVolume']
                
                records.append({
                    'timestamp': pd.to_datetime(p['snapshotTime']),
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
                
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            return df
        else:
            logging.error(f"❌ Ошибка получения котировок: {res.text}")
            return pd.DataFrame()

    def execute_command(self, decision: dict, epic="GBPUSD"):
        """Отправка маркет-ордера с прикрепленными SL и TP"""
        if self.equity <= 0: return
        
        url = f"{self.base_url}/api/v1/positions"
        direction = "BUY" if decision['action'] == "LONG" else "SELL"
        
        # Объем в контрактах (для CFD обычно размер задается иначе, чем в крипте)
        size = float(decision.get('size_lots', 1000)) 
        
        payload = {
            "epic": epic,
            "direction": direction,
            "size": size,
            "guaranteedStop": False
        }
        
        # Добавляем уровни SL и TP от риск-менеджера
        sl_price = decision.get('sl_price')
        tp_price = decision.get('tp_price')
        
        if sl_price: payload['stopLevel'] = round(sl_price, 5)
        if tp_price: payload['profitLevel'] = round(tp_price, 5)

        res = requests.post(url, json=payload, headers=self._get_headers())
        
        if res.status_code == 200:
            deal_ref = res.json().get('dealReference', 'Unknown')
            logging.info(f"✅ ОРДЕР ВЫПОЛНЕН: {direction} {size} {epic} | DealRef: {deal_ref}")
            if sl_price: logging.info(f"   -> Установлен SL: {sl_price}")
            if tp_price: logging.info(f"   -> Установлен TP: {tp_price}")
        else:
            logging.error(f"❌ Ошибка исполнения Capital.com: {res.text}")