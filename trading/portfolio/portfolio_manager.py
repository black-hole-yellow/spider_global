import numpy as np

class PortfolioManager:
    def __init__(self, config: dict):
        """
        Инициализация риск-параметров. В идеале подтягивается из config.json.
        """
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02) # Макс риск 2%
        self.win_loss_ratio = config.get('win_loss_ratio', 1.5)          # Среднее отношение Прибыль/Убыток (параметр 'b')
        self.max_daily_drawdown = config.get('max_daily_drawdown', 0.05) # Дневной лимит потерь 5%
        
        self.current_daily_drawdown = 0.0
        
    def update_drawdown(self, current_dd: float):
        """Обновляется Брокером (Слой 4) каждый тик/бар"""
        self.current_daily_drawdown = current_dd

    def update_empirical_stats(self, trade_history: list):
        """
        Считает реальный Risk/Reward на основе последних закрытых сделок брокера.
        Если сделок мало, используется дефолтный config.
        """
        exits = [t for t in trade_history if t.get('type') == 'EXIT']
        
        # Нам нужно хотя бы 10 закрытых сделок, чтобы статистика имела смысл
        if len(exits) < 10:
            return 
            
        # Смотрим на последние 50 сделок, чтобы адаптироваться к текущему режиму рынка
        recent_exits = exits[-50:]
        
        wins = [t['pnl'] for t in recent_exits if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in recent_exits if t['pnl'] < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Если есть и прибыли, и убытки — обновляем динамический R/R
        if avg_loss > 0 and avg_win > 0:
            self.win_loss_ratio = avg_win / avg_loss

    def calculate_kelly_fraction(self, win_prob: float) -> float:
        """
        Расчет критерия Келли: f = (p * b - q) / b
        p = вероятность выигрыша (от ИИ)
        q = вероятность проигрыша (1 - p)
        b = win_loss_ratio
        """
        b = self.win_loss_ratio
        q = 1.0 - win_prob
        
        kelly_f = (win_prob * b - q) / b
        
        # Институциональный стандарт: используем Half-Kelly для сглаживания волатильности эквити
        half_kelly = max(0.0, kelly_f / 2.0)
        
        # Ограничиваем сверху нашим жестким риск-лимитом
        return min(half_kelly, self.max_risk_per_trade)

    def process_signal(self, base_signal: int, features: dict, ml_confidence: float) -> dict:
        """
        Главный метод принятия решений.
        base_signal: +1 (Long), -1 (Short), 0 (Hold) из Слоя 2
        features: срез последней строки из Слоя 1 (Data Pipeline)
        ml_confidence: вероятность успеха от 0.0 до 1.0 от LightGBM/CatBoost
        """
        # 1. PORTFOLIO CIRCUIT BREAKER (Лимит просадки)
        if self.current_daily_drawdown >= self.max_daily_drawdown:
            return {"action": "BLOCK", "reason": "Max Daily Drawdown Exceeded"}

        # 2. EMERGENCY PROTOCOL (Kill Switch из-за новостей/всплесков)
        if features.get('is_anomaly', 0) == 1:
            return {"action": "LIQUIDATE_ALL", "reason": "Market Anomaly Detected (Z-Score > 4)"}

        # 3. Если нет сетапа от стратегии
        if base_signal == 0:
            return {"action": "HOLD"}

        # 4. РАСЧЕТ БАЗОВОГО РИСКА ПО КЕЛЛИ
        # Если ИИ не уверен (например, prob = 0.3), Kelly вернет 0, и сделка отфильтруется математически
        base_risk_fraction = self.calculate_kelly_fraction(ml_confidence)

        if base_risk_fraction <= 0:
            return {"action": "SKIP", "reason": "Negative or Zero Kelly Edge"}

        # 5. СЕМАНТИЧЕСКОЕ И СТРУКТУРНОЕ ПОДАВЛЕНИЕ (Dampening)
        # Умножаем риск на (1 - вероятность излома). 
        # Если вероятность смены режима 90%, риск режется в 10 раз!
        cp_prob = features.get('changepoint_prob', 0.0)
        adjusted_risk = base_risk_fraction * (1.0 - cp_prob)

        # 6. ЖЕСТКИЙ ФИЛЬТР МИКРО-ПОЗИЦИЙ
        if adjusted_risk < 0.001: # Если сайз меньше 0.1%, не тратим деньги на спред/комиссии
            return {"action": "SKIP", "reason": "Risk too low after dampening"}

        # ИТОГОВЫЙ ПРИКАЗ ДЛЯ БРОКЕРА
        return {
            "action": "ENTER",
            "direction": base_signal,
            "risk_fraction": adjusted_risk,
            "ml_confidence": ml_confidence,
            "metadata": {
                "base_kelly": base_risk_fraction,
                "changepoint_penalty": cp_prob
            }
        }