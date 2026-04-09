import pandas as pd
import numpy as np

class ChiefRiskOfficer:
    def __init__(self, max_risk_per_trade_pct=0.02, min_confidence_threshold=5.0):
        """
        max_risk_per_trade_pct: 2% риска на одну сделку (институциональный стандарт)
        min_confidence_threshold: Если уверенность Агента ниже 5%, мы в сделку не входим
        """
        self.max_risk = max_risk_per_trade_pct
        self.min_confidence = min_confidence_threshold
        print(f"👔 Chief Agent инициализирован. Макс риск: {self.max_risk*100}%. Порог уверенности: {self.min_confidence}%")

    def calculate_atr(self, df: pd.DataFrame, period=14):
        """Считает текущую волатильность для постановки стоп-лосса"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean().iloc[-1]

    def review_signal(self, agent_signal: dict, current_market_data: pd.DataFrame, account_balance: float) -> dict:
        """
        Принимает сигнал от Global Agent и решает: входить ли, и каким объемом.
        """
        if agent_signal.get("status") != "success":
            return {"decision": "HOLD", "reason": "Отказ нижестоящего агента"}

        direction = agent_signal["direction"]
        confidence = agent_signal["confidence_pct"]
        current_price = current_market_data['close'].iloc[-1]

        # 1. Фильтр неуверенности (Chop filter)
        if confidence < self.min_confidence:
            return {
                "decision": "HOLD",
                "reason": f"Уверенность ({confidence}%) ниже порога ({self.min_confidence}%)"
            }

        # 2. Расчет волатильности (ATR)
        atr = self.calculate_atr(current_market_data)
        if pd.isna(atr) or atr == 0:
            return {"decision": "HOLD", "reason": "Невозможно рассчитать ATR"}

        # 3. Динамический Stop Loss и Take Profit (Risk/Reward = 1:2)
        # Ставим стоп за 1.5 ATR от текущей цены
        stop_loss_dist = atr * 1.5
        take_profit_dist = stop_loss_dist * 2.0

        if direction == "LONG":
            sl_price = current_price - stop_loss_dist
            tp_price = current_price + take_profit_dist
        else: # SHORT
            sl_price = current_price + stop_loss_dist
            tp_price = current_price - take_profit_dist

        # 4. Position Sizing (Размер позиции)
        # Рискуем строго % от депозита на расстояние до стоп-лосса
        risk_amount_usd = account_balance * self.max_risk
        
        # Для Форекса: 1 лот GBP/USD = 100,000 GBP. 
        # Упрощенная формула объема базовой валюты = (Риск в $) / (Дистанция SL в пунктах цены)
        position_size_units = risk_amount_usd / stop_loss_dist
        
        # Округляем до микролотов (0.01 лота = 1000 единиц)
        position_size_lots = round(position_size_units / 100000, 2)
        
        if position_size_lots <= 0.0:
            return {"decision": "HOLD", "reason": "Слишком маленький депозит для текущего ATR"}

        return {
            "decision": "EXECUTE",
            "action": direction,
            "size_lots": position_size_lots,
            "entry_price": current_price,
            "sl_price": round(sl_price, 5),
            "tp_price": round(tp_price, 5),
            "confidence": confidence,
            "atr_usd": round(atr, 5)
        }

if __name__ == "__main__":
    # Тест
    dummy_data = pd.DataFrame({'high': [1.2550]*20, 'low': [1.2500]*20, 'close': [1.2525]*20})
    signal = {"status": "success", "direction": "LONG", "confidence_pct": 12.5}
    
    chief = ChiefRiskOfficer()
    decision = chief.review_signal(signal, dummy_data, 10000)
    print("\nВердикт Chief Agent:")
    print(decision)