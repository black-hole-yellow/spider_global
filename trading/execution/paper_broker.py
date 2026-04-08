import pandas as pd
import numpy as np

class PaperBroker:
    def __init__(self, initial_capital: float = 100000.0, config: dict = None):
        if config is None:
            config = {}
        
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.cash = initial_capital
        self.positions = 0.0          # Текущий размер позиции (в базовой валюте, напр. GBP)
        self.entry_price = 0.0
        
        # Институциональные настройки трения (Frictions)
        self.commission_per_100k = config.get('commission_per_100k', 2.50) # $2.50 за лот
        self.base_spread_pips = config.get('base_spread_pips', 0.5)        # Базовый спред 0.5 пипса
        
        # Для трекинга просадки (Drawdown)
        self.high_water_mark = initial_capital
        self.current_daily_dd = 0.0
        self.trade_history = []
        
        self.current_bar = None

    def update_market_state(self, current_bar: dict):
        """Обновляет состояние рынка на каждом тике/баре бэктеста."""
        self.current_bar = current_bar
        
        # Обновляем плавающий PnL и эквити
        if self.positions != 0:
            unrealized_pnl = self.positions * (current_bar['close'] - self.entry_price)
            self.equity = self.cash + unrealized_pnl
        else:
            self.equity = self.cash
            
        # Считаем просадку для передачи в Layer 3
        if self.equity > self.high_water_mark:
            self.high_water_mark = self.equity
        
        self.current_daily_dd = (self.high_water_mark - self.equity) / self.high_water_mark

    def _calculate_execution_price(self, order_size: float, direction: int, is_emergency: bool = False) -> float:
        """
        Рассчитывает реалистичную цену исполнения с учетом спреда, размера позиции и паники.
        """
        base_price = self.current_bar['close']
        pip_value = 0.0001
        
        # 1. Базовый спред (половина спреда на вход, половина на выход)
        spread_cost = (self.base_spread_pips / 2.0) * pip_value
        
        # 2. Market Impact (Проскальзывание от объема)
        # Предположим, каждые 1 млн (10 лотов) сдвигают цену еще на 0.2 пипса
        volume_impact = (abs(order_size) / 1_000_000) * (0.2 * pip_value)
        
        # 3. Emergency Penalty (Новостной сквиз)
        # Если это ликвидация по Kill Switch, спред раздвигается в зависимости от волатильности
        emergency_cost = 0.0
        if is_emergency:
            # Получаем ATR или volatility_z из контекста бара (передается из Слоя 1)
            vol_z = self.current_bar.get('volatility_z', 1.0)
            emergency_cost = (vol_z * 2.0) * pip_value # Пенальти до нескольких пипсов
            
        total_slippage = spread_cost + volume_impact + emergency_cost
        
        # Ухудшаем цену: для покупок (+1) цена выше, для продаж (-1) ниже
        return base_price + (direction * total_slippage)

    def execute_command(self, portfolio_command: dict):
        """
        Принимает приказ от PortfolioManager (Слой 3) и исполняет его.
        """
        if self.current_bar is None:
            return

        action = portfolio_command.get('action')
        
        # Экстренная ликвидация (Kill Switch)
        if action == 'LIQUIDATE_ALL':
            self._close_position(is_emergency=True, reason=portfolio_command.get('reason'))
            return
            
        # Блокировка или пропуск
        if action in ['BLOCK', 'SKIP', 'HOLD']:
            return
            
        # Исполнение входа (ENTER)
        if action == 'ENTER':
            direction = portfolio_command['direction']
            risk_fraction = portfolio_command['risk_fraction']
            
            # Если уже в позиции, в этой базовой версии мы не доливаемся, а игнорируем
            # (Для пирамидинга логика будет сложнее)
            if self.positions != 0:
                return
                
            # Переводим риск по Келли в номинальный объем (с плечом)
            # Допустим, стоп-лосс (в процентах от цены) равен 1x ATR
            atr_pct = self.current_bar.get('atr_pct', 0.002) # Заглушка, если нет ATR
            capital_to_risk = self.equity * risk_fraction
            notional_size = capital_to_risk / (atr_pct + 1e-6)
            
            # Исполнение
            exec_price = self._calculate_execution_price(notional_size, direction)
            commission = (notional_size / 100_000) * self.commission_per_100k
            
            self.positions = notional_size * direction
            self.entry_price = exec_price
            self.cash -= commission
            
            self.trade_history.append({
                'time': self.current_bar.get('timestamp'),
                'type': 'ENTER',
                'direction': direction,
                'size': notional_size,
                'price': exec_price,
                'commission': commission,
                'confidence': portfolio_command.get('ml_confidence'),
                'equity_at_exit': self.equity
            })

    def _close_position(self, is_emergency: bool = False, reason: str = "Strategy Exit"):
        """Закрывает текущую позицию с расчетом PnL."""
        if self.positions == 0:
            return
            
        direction_to_close = -1 if self.positions > 0 else 1
        close_size = abs(self.positions)
        
        exec_price = self._calculate_execution_price(close_size, direction_to_close, is_emergency)
        commission = (close_size / 100_000) * self.commission_per_100k
        
        # Расчет PnL
        pnl = self.positions * (exec_price - self.entry_price)
        self.cash += pnl - commission
        self.equity = self.cash
        
        self.trade_history.append({
            'time': self.current_bar.get('timestamp'),
            'type': 'EXIT',
            'reason': reason,
            'pnl': pnl,
            'price': exec_price,
            'commission': commission,
            'is_emergency': is_emergency
        })
        
        # Сброс позиции
        self.positions = 0.0
        self.entry_price = 0.0