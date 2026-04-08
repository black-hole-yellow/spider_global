import pandas as pd
import numpy as np

class EventDrivenBacktester:
    def __init__(self, df: pd.DataFrame, strategy, broker, portfolio_manager, execution_engine):
        self.df = df
        self.strategy = strategy
        self.broker = broker
        self.portfolio_manager = portfolio_manager
        self.execution_engine = execution_engine
        self.completed_trades = [] 

    def run(self):
        """Simulates trading and returns an Equity DataFrame for the Gauntlet."""
        if self.df.empty: return pd.DataFrame()

        if 'signals' not in self.df.columns:
            self.df['signals'] = self.strategy.generate_signals(self.df)

        equity_curve = []
        
        # --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ КОНТРОЛЯ СДЕЛОК ---
        last_trade_date = None
        daily_trades_count = 0
        last_trade_direction = 0
        
        # Память о сигнале на прошлом баре (защита от мгновенного спама)
        prev_bar_signal = 0 

        for row in self.df.itertuples():
            current_time = row.Index
            current_date = current_time.date()
            signal = row.signals
            
            # 1. Если начался новый день - сбрасываем счетчики
            if current_date != last_trade_date:
                daily_trades_count = 0
                last_trade_direction = 0
                last_trade_date = current_date

            # 2. Обновляем текущие позиции (проверка SL/TP)
            if self.broker.has_open_position():
                self.portfolio_manager.update_portfolio(current_time, row, self.broker)

            # 3. Логика входа в новые сделки
            if signal != 0 and not self.broker.has_open_position():
                
                # Проверяем, что это "свежий" триггер (на прошлом баре сигнала не было).
                # Это гарантирует, что мы входим именно после нового свипа (Re-sweep), 
                # а не просто потому, что старый сигнал еще "горит".
                is_fresh_signal = (signal != prev_bar_signal)
                
                can_enter = False
                
                # ПРАВИЛО 1: Первая сделка за день
                if daily_trades_count == 0:
                    if is_fresh_signal:
                        can_enter = True
                        
                # ПРАВИЛО 2: Re-entry (Вторая сделка)
                elif daily_trades_count == 1:
                    # Разрешаем вход ТОЛЬКО если направление совпадает с первой сделкой
                    # И это действительно новый свип (is_fresh_signal == True)
                    if signal == last_trade_direction and is_fresh_signal:
                        can_enter = True

                # Исполняем ордер, если правила пройдены
                if can_enter:
                    order = self.portfolio_manager.generate_order(
                        current_time, signal, row, self.broker.get_total_equity()
                    )
                    if order: 
                        self.execution_engine.execute(order)
                        daily_trades_count += 1
                        last_trade_direction = signal

            # Запоминаем сигнал для следующего бара
            prev_bar_signal = signal

            # 4. Записываем эквити
            equity_curve.append(self.broker.get_total_equity())

        # Закрываем все в конце теста
        if self.broker.has_open_position():
            self.broker.close_all_positions(self.df.index[-1], self.df['close'].iloc[-1])

        self.completed_trades = self.broker.get_trade_history()

        df_equity = pd.DataFrame(index=self.df.index, data={'equity': equity_curve})
        df_equity['returns'] = df_equity['equity'].pct_change().fillna(0)
        return df_equity