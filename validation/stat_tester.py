import numpy as np
import pandas as pd
import scipy.stats as stats

class StatisticalValidator:
    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252 * 96):
        # 252 торговых дня * 96 баров (по 15 минут) = ~24192 периода в году
        self.rf = risk_free_rate
        self.ppy = periods_per_year

    def calculate_sharpe(self, returns: np.ndarray) -> float:
        """Классический Annualized Sharpe Ratio."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return (np.mean(returns) - self.rf) / np.std(returns) * np.sqrt(self.ppy)

    def monte_carlo_bootstrap(self, trade_returns: pd.Series, n_iterations: int = 10000) -> dict:
        """
        Monte Carlo Bootstrap Test.
        Проверяет, выживет ли стратегия, если последовательность сделок будет случайной.
        """
        if len(trade_returns) < 30:
            return {"error": "Not enough trades for statistical significance (minimum 30)."}

        actual_sharpe = self.calculate_sharpe(trade_returns.values)
        bootstrapped_sharpes = np.zeros(n_iterations)
        n = len(trade_returns)
        
        # Векторизованный бутстрап (выборка с возвратом)
        for i in range(n_iterations):
            # Симулируем альтернативные реальности
            sample = np.random.choice(trade_returns.values, size=n, replace=True)
            bootstrapped_sharpes[i] = self.calculate_sharpe(sample)
            
        # P-value: процент симуляций, где Шарп оказался нулевым или отрицательным
        p_value = np.sum(bootstrapped_sharpes <= 0) / n_iterations 
        
        return {
            "actual_sharpe": round(actual_sharpe, 3),
            "p_value": round(p_value, 5),
            "conf_interval_5%": round(np.percentile(bootstrapped_sharpes, 5), 3),
            "conf_interval_95%": round(np.percentile(bootstrapped_sharpes, 95), 3),
            "is_robust": p_value < 0.05  # Если < 5%, стратегия статистически надежна
        }

    def probabilistic_sharpe_ratio(self, trade_returns: pd.Series, benchmark_sharpe: float = 0.0) -> float:
        """
        Probabilistic Sharpe Ratio (PSR).
        Математически строгая оценка вероятности того, что Шарп больше нуля (или бенчмарка),
        с поправкой на толстые хвосты и асимметрию рынка.
        """
        if len(trade_returns) < 30:
            return 0.0

        sr = self.calculate_sharpe(trade_returns.values)
        skew = stats.skew(trade_returns.values)
        kurtosis = stats.kurtosis(trade_returns.values, fisher=False) # Pearson's kurtosis
        n = len(trade_returns)
        
        # Формула дисперсии Шарпа с учетом высших моментов (не-нормальности)
        sr_var = (1 - skew * sr + (kurtosis - 1) / 4 * sr**2) / (n - 1)
        
        # Вычисление Z-статистики и P-value (Кумулятивная функция распределения)
        z_stat = (sr - benchmark_sharpe) / np.sqrt(sr_var + 1e-8)
        psr = stats.norm.cdf(z_stat)
        
        return round(psr, 4)

    def evaluate_strategy(self, trade_history: list) -> dict:
        """Главный метод для вызова из Оркестратора/Бэктестера."""
        if not trade_history:
            return {"status": "REJECTED", "error": "No trades generated."}
            
        df = pd.DataFrame(trade_history)
        
        # ФИКС 1: Фильтруем только закрытые сделки (EXIT), потому что только у них есть зафиксированный PnL
        exits = df[df['type'] == 'EXIT'].copy()
        
        if len(exits) == 0:
            return {"status": "REJECTED", "error": "No closed trades."}
            
        # Если почему-то equity_at_exit нет, падаем на 100k
        exits['equity_at_exit'] = exits.get('equity_at_exit', 100000.0) 
        exits['return_pct'] = exits['pnl'] / exits['equity_at_exit']
        
        returns = exits['return_pct'].values
        
        # Считаем классический Sharpe Ratio (по сделкам)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        classic_sharpe = (mean_ret / std_ret) if std_ret > 0 else 0.0
        
        # Получаем метрики Монте-Карло и PSR
        mc_results = self.monte_carlo_bootstrap(exits['return_pct'])
        psr = self.probabilistic_sharpe_ratio(exits['return_pct'])
        
        # Защита от NaN (если формула сломалась из-за сильной асимметрии)
        if np.isnan(psr):
            psr = classic_sharpe

        win_rate = len(exits[exits['pnl'] > 0]) / len(exits) * 100

        # Система одобряется, если классический Шарп > 0 И Монте Карло пройдено
        is_approved = (classic_sharpe > 0.0) and mc_results.get('is_robust', False)

        return {
            "total_trades": len(exits),
            "win_rate": round(win_rate, 2),
            "classic_sharpe": round(classic_sharpe, 2),
            "psr_score": round(psr, 2),
            "monte_carlo": mc_results,
            "status": "APPROVED" if is_approved else "REJECTED"
        }