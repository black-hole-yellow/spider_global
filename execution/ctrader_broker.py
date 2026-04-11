import logging
import pandas as pd
import datetime
from twisted.internet import reactor

# Protobuf Messages
from ctrader_open_api.messages import OpenApiMessages_pb2 as msg
from ctrader_open_api.messages import OpenApiModelMessages_pb2 as model_msg

class CTraderStream:
    def __init__(self, broker, symbol_name="GBPUSD", timeframe=model_msg.PROTO_OA_TRENDBAR_PERIOD_M15):
        """
        Управляет потоком данных (Trendbars) в реальном времени.
        :param broker: Экземпляр CTraderBroker (уже подключенный и авторизованный)
        """
        self.broker = broker
        self.symbol_name = symbol_name
        self.symbol_id = None
        self.timeframe = timeframe
        
        # In-memory хранилище свечей
        self.dataframe = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        self.dataframe.index.name = 'timestamp'
        
        # Флаг готовности данных
        self.is_ready = False

    def start_stream(self, on_data_ready_callback):
        """Запускает процесс подписки и загрузки истории"""
        self.symbol_id = self.broker.symbols_map.get(self.symbol_name)
        if not self.symbol_id:
            logging.error(f"❌ Стример не нашел Symbol ID для {self.symbol_name}")
            return

        self.on_data_ready = on_data_ready_callback
        logging.info(f"📡 Запуск потока данных для {self.symbol_name} (ID: {self.symbol_id})")

        # 1. Сначала скачиваем историю для 'прогрева' пайплайна (500 свечей)
        self._fetch_historical_data()

    def _fetch_historical_data(self):
        """Запрашивает последние 500 свечей M15"""
        req = msg.ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = self.broker.account_id
        req.period = self.timeframe
        req.symbolId = self.symbol_id
        
        # Время в cTrader API передается в миллисекундах
        now = datetime.datetime.utcnow()
        from_time = now - datetime.timedelta(days=7) # Берем с запасом
        
        req.fromTimestamp = int(from_time.timestamp() * 1000)
        req.toTimestamp = int(now.timestamp() * 1000)

        df = self.broker.client.send(req)
        
        def on_history(res):
            logging.info(f"✅ Получена история: {len(res.trendbar)} свечей.")
            self._process_history(res.trendbar)
            
            # 2. После получения истории подписываемся на LIVE обновления
            self._subscribe_live_trendbars()

        df.addCallbacks(on_history, self._on_error)

    def _process_history(self, trendbars):
        """Конвертирует Protobuf трендбары в Pandas DataFrame"""
        records = []
        for bar in trendbars:
            # cTrader отдает цены в 1/100000 (зависит от символа, обычно делим на 100000)
            # Базовая цена (low)
            low = bar.low / 100000.0
            
            records.append({
                'timestamp': pd.to_datetime(bar.utcTimestampInMinutes * 60, unit='s', utc=True),
                'open': low + (bar.deltaOpen / 100000.0),
                'high': low + (bar.deltaHigh / 100000.0),
                'low': low,
                'close': low + (bar.deltaClose / 100000.0),
                'volume': bar.volume
            })
            
        df = pd.DataFrame(records)
        df.set_index('timestamp', inplace=True)
        # Берем последние 500
        self.dataframe = df.iloc[-500:]
        
    def _subscribe_live_trendbars(self):
        """Подписка на живые события"""
        req = msg.ProtoOASubscribeLiveTrendbarReq()
        req.ctidTraderAccountId = self.broker.account_id
        req.period = self.timeframe
        req.symbolId = self.symbol_id

        df = self.broker.client.send(req)
        
        def on_subscribed(res):
            logging.info(f"✅ Подписка на LIVE Trendbars оформлена.")
            self.is_ready = True
            
            # Подключаем глобальный обработчик событий (SpotEvent) к клиенту
            self.broker.client.setSubscriber(msg.ProtoOASpotEvent().PayloadType, self._on_spot_event)
            
            # Сообщаем торговому боту, что данные готовы к анализу
            if self.on_data_ready:
                self.on_data_ready()

        df.addCallbacks(on_subscribed, self._on_error)

    def _on_spot_event(self, spot_event):
        """Обработчик входящих LIVE тиков/свечей"""
        if spot_event.symbolId != self.symbol_id: return
        
        # Если пришла новая/обновленная свеча
        for bar in spot_event.trendbar:
            if bar.period == self.timeframe:
                ts = pd.to_datetime(bar.utcTimestampInMinutes * 60, unit='s', utc=True)
                low = bar.low / 100000.0
                
                # Обновляем или добавляем свечу в DataFrame
                self.dataframe.loc[ts] = [
                    low + (bar.deltaOpen / 100000.0),
                    low + (bar.deltaHigh / 100000.0),
                    low,
                    low + (bar.deltaClose / 100000.0),
                    bar.volume
                ]
                
                # Держим размер в пределах 500 свечей, чтобы не переполнять память
                if len(self.dataframe) > 500:
                    self.dataframe = self.dataframe.iloc[-500:]

    def get_dataframe(self):
        """Возвращает актуальный слепок рынка для пайплайна"""
        return self.dataframe.copy()

    def _on_error(self, failure):
        logging.error(f"❌ Ошибка стримера данных: {failure}")