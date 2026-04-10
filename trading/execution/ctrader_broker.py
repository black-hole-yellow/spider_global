import logging
from ctrader_open_api import Client, EndPoints, TcpProtocol

# Явный импорт модулей Protobuf для избежания ошибок "is not defined"
from ctrader_open_api.messages import OpenApiCommonMessages_pb2 as common_msg
from ctrader_open_api.messages import OpenApiMessages_pb2 as msg
from ctrader_open_api.messages import OpenApiModelMessages_pb2 as model_msg

class CTraderBroker:
    def __init__(self, client_id: str, client_secret: str, account_id: str, access_token: str, is_demo: bool = True):
        self.client_id = client_id
        self.client_secret = client_secret
        self.account_id = int(account_id)
        self.access_token = access_token
        
        host = EndPoints.PROTOBUF_DEMO_HOST if is_demo else EndPoints.PROTOBUF_LIVE_HOST
        self.client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        
        self.equity = 0.0
        self.symbols_map = {}  
        self.on_ready_callback = None
        
        self.client.setConnectedCallback(self._on_connected)
        self.client.setDisconnectedCallback(self._on_disconnected)
        
        logging.info(f"🔌 Инициализация cTrader TCP Протокола ({'DEMO' if is_demo else 'LIVE'})")

    def start_connection(self, on_ready_callback):
        self.on_ready_callback = on_ready_callback
        self.client.startService()

    def _on_connected(self, client):
        logging.info("✅ TCP Подключен. Авторизация приложения...")
        req = msg.ProtoOAApplicationAuthReq()
        req.clientId = self.client_id
        req.clientSecret = self.client_secret
        df = self.client.send(req)
        df.addCallbacks(self._on_app_auth, self._on_error)

    def _on_app_auth(self, res):
        logging.info("✅ Приложение авторизовано. Авторизация аккаунта...")
        req = msg.ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = self.account_id
        req.accessToken = self.access_token
        df = self.client.send(req)
        df.addCallbacks(self._on_account_auth, self._on_error)

    def _on_account_auth(self, res):
        logging.info("✅ Аккаунт авторизован. Загрузка списка символов...")
        self._fetch_symbols()

    def _fetch_symbols(self):
        req = msg.ProtoOASymbolsListReq()
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

    def update_market_state(self):
        req = msg.ProtoOATraderReq()
        req.ctidTraderAccountId = self.account_id
        df = self.client.send(req)
        
        def on_trader(res):
            self.equity = res.trader.equity / 100.0  
            logging.info(f"💰 Эквити: ${self.equity}")
            
        df.addCallbacks(on_trader, self._on_error)

    def execute_command(self, decision: dict, symbol_name="GBPUSD"):
        symbol_id = self.symbols_map.get(symbol_name)
        if not symbol_id:
            logging.error(f"❌ Символ {symbol_name} не найден у брокера.")
            return

        # Используем константы из model_msg
        side = model_msg.PROTO_OA_TRADE_SIDE_BUY if decision['action'] == "LONG" else model_msg.PROTO_OA_TRADE_SIDE_SELL
        
        volume_units = int(decision.get('size_lots', 0.01) * 100000)

        req = msg.ProtoOANewOrderReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        req.orderType = model_msg.PROTO_OA_ORDER_TYPE_MARKET
        req.tradeSide = side
        req.volume = volume_units
        
        sl = decision.get('sl_price')
        tp = decision.get('tp_price')
        if sl: req.stopLoss = float(sl)
        if tp: req.takeProfit = float(tp)

        df = self.client.send(req)
        
        def on_order(res):
            logging.info(f"🔥 ОРДЕР ИСПОЛНЕН: {decision['action']} | Объем: {volume_units} ед.")
            
        df.addCallbacks(on_order, self._on_error)

    def _on_disconnected(self, client, reason):
        logging.warning(f"❌ TCP Соединение разорвано: {reason}")

    def _on_error(self, failure):
        logging.error(f"❌ Ошибка cTrader API: {failure}")