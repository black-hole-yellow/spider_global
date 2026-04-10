import logging
from ctrader_open_api import Client, EndPoints, TcpProtocol
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

class CTraderBroker:
    def __init__(self, client_id: str, client_secret: str, account_id: str, access_token: str, is_demo: bool = True):
        self.client_id = client_id
        self.client_secret = client_secret
        self.account_id = int(account_id)
        self.access_token = access_token
        
        # Выбор сервера (Demo или Live)
        host = EndPoints.PROTOBUF_DEMO_HOST if is_demo else EndPoints.PROTOBUF_LIVE_HOST
        self.client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        
        self.equity = 0.0
        self.symbols_map = {}  # cTrader использует ID вместо тикеров (например, 1 вместо 'GBPUSD')
        self.on_ready_callback = None
        
        # Настройка коллбеков (событий)
        self.client.setConnectedCallback(self._on_connected)
        self.client.setDisconnectedCallback(self._on_disconnected)
        
        logging.info(f"🔌 Инициализация cTrader TCP Протокола ({'DEMO' if is_demo else 'LIVE'})")

    def start_connection(self, on_ready_callback):
        """Запускает TCP сервис и передает коллбек для старта торгового цикла"""
        self.on_ready_callback = on_ready_callback
        self.client.startService()

    # --- ЦЕПОЧКА АВТОРИЗАЦИИ ---
    def _on_connected(self, client):
        logging.info("✅ TCP Подключен. Авторизация приложения...")
        req = ProtoOAApplicationAuthReq()
        req.clientId = self.client_id
        req.clientSecret = self.client_secret
        df = self.client.send(req)
        df.addCallbacks(self._on_app_auth, self._on_error)

    def _on_app_auth(self, res):
        logging.info("✅ Приложение авторизовано. Авторизация аккаунта...")
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = self.account_id
        req.accessToken = self.access_token
        df = self.client.send(req)
        df.addCallbacks(self._on_account_auth, self._on_error)

    def _on_account_auth(self, res):
        logging.info("✅ Аккаунт авторизован. Загрузка списка символов...")
        self._fetch_symbols()

    def _fetch_symbols(self):
        """Загружает маппинг всех тикеров брокера"""
        req = ProtoOASymbolsListReq()
        req.ctidTraderAccountId = self.account_id
        df = self.client.send(req)
        
        def on_symbols(res):
            for sym in res.symbol:
                self.symbols_map[sym.symbolName] = sym.symbolId
            logging.info(f"✅ База символов загружена. Готов к торговле.")
            self.update_market_state()
            if self.on_ready_callback:
                self.on_ready_callback()
                
        df.addCallbacks(on_symbols, self._on_error)

    # --- ТОРГОВЫЕ ОПЕРАЦИИ ---
    def update_market_state(self):
        """Асинхронный запрос баланса"""
        req = ProtoOATraderReq()
        req.ctidTraderAccountId = self.account_id
        df = self.client.send(req)
        
        def on_trader(res):
            self.equity = res.trader.equity / 100.0  # cTrader возвращает баланс в центах
            logging.info(f"💰 Эквити: ${self.equity}")
            
        df.addCallbacks(on_trader, self._on_error)

    def execute_command(self, decision: dict, symbol_name="GBPUSD"):
        symbol_id = self.symbols_map.get(symbol_name)
        if not symbol_id:
            logging.error(f"❌ Символ {symbol_name} не найден у брокера.")
            return

        side = PROTO_OA_TRADE_SIDE_BUY if decision['action'] == "LONG" else PROTO_OA_TRADE_SIDE_SELL
        
        # Объем в cTrader указывается в единицах базовой валюты (например, 1000 для 0.01 лота)
        volume_units = int(decision.get('size_lots', 0.01) * 100000)

        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        req.orderType = PROTO_OA_ORDER_TYPE_MARKET
        req.tradeSide = side
        req.volume = volume_units
        
        # Установка Стоп-Лосса и Тейк-Профита (в cTrader это делается прямо в ордере)
        sl = decision.get('sl_price')
        tp = decision.get('tp_price')
        if sl: req.stopLoss = float(sl)
        if tp: req.takeProfit = float(tp)

        df = self.client.send(req)
        
        def on_order(res):
            logging.info(f"🔥 ОРДЕР ИСПОЛНЕН: {decision['action']} | Объем: {volume_units} ед. | SL: {sl}")
            
        df.addCallbacks(on_order, self._on_error)

    def _on_disconnected(self, client, reason):
        logging.warning(f"❌ TCP Соединение разорвано: {reason}")

    def _on_error(self, failure):
        logging.error(f"❌ Ошибка cTrader API: {failure}")