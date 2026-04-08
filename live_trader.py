import pandas as pd
import numpy as np
import time
import logging
from collections import deque
from catboost import CatBoostClassifier

# Импорты наших слоев (согласно Архитектурному Блюпринту)
from shared.features import htf, ml_features, sessions
from strategies.library.generic_strategy import InstitutionalSMCStrategy
from trading.portfolio.portfolio_manager import PortfolioManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class LiveTrader:
    def __init__(self, config: dict, max_bars: int = 2000):
        """
        max_bars: Размер кольцевого буфера. 
        1000 свечей 15m = примерно 10 торговых дней. Идеально для легковесного продакшена.
        """
        self.config = config
        self.max_bars = max_bars
        
        # Deque автоматически "выталкивает" старые данные при превышении maxlen.
        # Это исключает утечку памяти (Memory Leak) при аптайме в несколько месяцев.
        self.bar_buffer = deque(maxlen=self.max_bars)
        
        # Инициализация Мозгов (Слои 2 и 3)
        self.strategy = InstitutionalSMCStrategy(config)
        self.portfolio_manager = PortfolioManager(config)
        
        # Инициализация ML
        self.ml_model = None
        self.ml_features = config.get('ml_features', [
            'active_setup', 'volatility_z', 'changepoint_prob', 'trend_strength', 
            'cusum_signal', 'asia_intensity', 'london_intensity', 'ny_intensity', 
            'session_overlap_score', 'dist_to_pwh', 'dist_to_pwl', 'mtfa_score',"llm_sentiment_score", "is_macro_alignment"
        ])
        
        model_path = config.get('model_path', 'models/meta_model.cbm')
        try:
            self.ml_model = CatBoostClassifier().load_model(model_path)
            logging.info(f"Loaded CatBoost Meta-Labeler from {model_path}")
        except Exception as e:
            logging.warning(f"ML model not found: {e}. System will run with baseline Kelly confidence.")

    def warm_up(self, historical_bars: list[dict]):
        """
        Метод "разогрева". Перед тем как слушать вебсокет, 
        нужно загрузить последние 500 свечей по REST API, 
        иначе индикаторам не на чем будет считаться.
        """
        logging.info(f"Warming up state buffer with {len(historical_bars)} bars...")
        for bar in historical_bars:
            # Убеждаемся, что timestamp имеет нужный тип перед добавлением
            self.bar_buffer.append(bar)

    def _apply_feature_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет Layer 1 на лету к нашему срезу из 500 баров."""
        df = htf.add_daily_liquidity(df)
        df = htf.add_htf_fvg(df)
        df = htf.add_mtfa_trend(df)
        df = htf.add_advanced_liquidity_and_eq(df)
        df = ml_features.add_regime_and_changepoint_features(df)
        df = sessions.add_vector_sessions(df)
        return df

    def on_new_bar(self, current_bar: dict):
        """
        Event-Driven метод. Вызывается основным циклом (или вебсокетом) 
        в момент закрытия 15-минутной свечи.
        """
        # 1. Добавляем новый бар в память (самый старый автоматически удаляется)
        self.bar_buffer.append(current_bar)
        
        # Защита: нам нужно хотя бы 200 баров, чтобы длинные скользящие не выдали NaN
        if len(self.bar_buffer) < 200:
            return

        # 2. Конвертируем буфер в DataFrame (на 500 строках это занимает микросекунды)
        df = pd.DataFrame(self.bar_buffer)
        df.set_index('timestamp', inplace=True)
        
        # 3. Считаем наши тяжелые институциональные фичи
        try:
            enriched_df = self._apply_feature_pipeline(df)
        except Exception as e:
            logging.error(f"Feature calculation dropped a frame: {e}")
            return
            
        # 4. Берем ТОЛЬКО последнюю строку (именно она отражает рынок "прямо сейчас")
        current_state = enriched_df.iloc[-1].to_dict()
        current_state['timestamp'] = enriched_df.index[-1]
        
        # === ВЫЗОВ ЛОГИКИ ===
        # 5. Стратегия (Layer 2)
        base_signal = self.strategy.generate_signal(current_state)
        
        # 6. ML Модель (Layer 3)
        ml_confidence = self._get_ml_confidence(current_state) if base_signal != 0 else 0.0
        
        # 7. Келли Сайзинг и Риск (Layer 3)
        portfolio_command = self.portfolio_manager.process_signal(
            base_signal=base_signal,
            features=current_state,
            ml_confidence=ml_confidence
        )
        
        # 8. Исполнение (Layer 4)
        self._execute_real_order(portfolio_command, current_state)

    def _get_ml_confidence(self, features: dict) -> float:
        if not self.ml_model: 
            return 0.51 # Baseline

        # Формируем вектор в строгом порядке
        vector = []
        for f in self.ml_features:
            val = features.get(f, 0)
            if f == 'active_setup' and (val == 0 or val is None):
                val = "None"
            vector.append(val)
            
        return float(self.ml_model.predict_proba([vector])[0][1])

    def _execute_real_order(self, command: dict, state: dict):
        """Мост к API брокера (Binance, Bybit, MetaTrader)."""
        action = command.get('action')
        
        if action == 'LIQUIDATE_ALL':
            logging.error(f"🚨 KILL SWITCH ACTIVATED: {command.get('reason')}. Closing all positions!")
            # API call: broker.close_all_positions()
            return
            
        if action in ['HOLD', 'SKIP', 'BLOCK']:
            return
            
        if action == 'ENTER':
            direction = "LONG" if command['direction'] == 1 else "SHORT"
            risk = command['risk_fraction'] * 100
            conf = command['ml_confidence'] * 100
            
            logging.info(f"🟢 TRADE EXECUTED: {direction} | Risk: {risk:.2f}% | AI Confidence: {conf:.1f}%")
            # API call: broker.place_order(direction, size=..., stop_loss=state['atr_pct'])