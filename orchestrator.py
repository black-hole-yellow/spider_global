import pandas as pd
import numpy as np
import time
import logging
from catboost import CatBoostClassifier
import os

from research.engine.data_pipeline import DataQualityEngine
from trading.portfolio.portfolio_manager import PortfolioManager
from trading.execution.paper_broker import PaperBroker
from research.validation.stat_tester import StatisticalValidator

from shared.features import htf, ml_features, sessions, macro
from strategies.library.generic_strategy import InstitutionalSMCStrategy 

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class SystemOrchestrator:
    def __init__(self, config: dict):
        self.config = config
        logging.info("Initializing Quant System Components...")
        self.data_engine = DataQualityEngine(timeframe_mins=15)
        self.portfolio_manager = PortfolioManager(config)
        self.broker = PaperBroker(initial_capital=100000.0, config=config)
        self.validator = StatisticalValidator()
        self.strategy = InstitutionalSMCStrategy(config)
        
        self.model_path = config.get('model_path', 'models/meta_model.cbm')
        self.ml_features = config.get('ml_features', [
            'active_setup', 'volatility_z', 'changepoint_prob', 'trend_strength', 
            'cusum_signal', 'asia_intensity', 'london_intensity', 'ny_intensity', 
            'session_overlap_score', 'dist_to_pwh', 'dist_to_pwl', 'mtfa_score', 
            "llm_sentiment_score", "is_macro_alignment", "active_session_name"
        ])
        
        self.ml_model = None
        if os.path.exists(self.model_path):
            logging.info(f"Loading CatBoost Meta-Labeler from {self.model_path}...")
            self.ml_model = CatBoostClassifier().load_model(self.model_path)
        else:
            logging.warning(f"Model not found at {self.model_path}. Will use fallback baseline edge.")

    def get_ml_confidence(self, bar_features: dict) -> float:
        if self.ml_model is None:
            return 0.51 

        try:
            row = {}
            for f in self.ml_features:
                val = bar_features.get(f, 0)
                if f == 'active_setup' and (val == 0 or pd.isna(val)): val = "None"
                if f == 'active_session_name' and (val == 0 or pd.isna(val)): val = "Asian"
                row[f] = val
                
            df_input = pd.DataFrame([row]) 
            probabilities = self.ml_model.predict_proba(df_input)[0]
            return float(probabilities[1])
            
        except Exception as e:
            logging.error(f"ML Inference error: {e}")
            return 0.0 

    def run_pipeline(self, data_filepath: str):
        start_time = time.time()
        logging.info(f"Loading and cleaning data from {data_filepath}...")
        clean_df = self.data_engine.ingest_and_clean(data_filepath).to_pandas()
        
        df = clean_df.copy()
        df.set_index('timestamp', inplace=True)
        if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            
        try:
            if hasattr(htf, 'add_daily_liquidity'): df = htf.add_daily_liquidity(df)
            if hasattr(htf, 'add_htf_fvg'): df = htf.add_htf_fvg(df)
            if hasattr(htf, 'add_mtfa_trend'): df = htf.add_mtfa_trend(df)
            if hasattr(htf, 'add_advanced_liquidity_and_eq'): df = htf.add_advanced_liquidity_and_eq(df)
            if hasattr(ml_features, 'add_regime_and_changepoint_features'): df = ml_features.add_regime_and_changepoint_features(df)
            if hasattr(sessions, 'add_vector_sessions'): df = sessions.add_vector_sessions(df)
            if hasattr(macro, 'add_macro_events'): df = macro.add_macro_events(df)
            if hasattr(macro, 'add_llm_semantic_features'): df = macro.add_llm_semantic_features(df)
        except Exception as e:
            logging.error(f"Error during feature generation: {e}")
            
        df.reset_index(inplace=True)
        enriched_df = df.dropna()
        
        logging.info(f"Starting Event-Driven Backtest on {len(enriched_df)} bars...")
        records = enriched_df.to_dict('records')
        
        for bar in records:
            current_time = pd.to_datetime(bar['timestamp'])
            
            # --- ПРАВИЛО 3: ЗАКРЫВАЕМ ВСЁ В ПЯТНИЦУ ВЕЧЕРОМ ---
            if current_time.dayofweek == 4 and current_time.hour >= 21:
                self.broker.execute_command({'action': 'LIQUIDATE_ALL', 'reason': 'Friday_Close'})

            # --- ПРАВИЛО 1: ДИНАМИЧЕСКИЙ РИСК В ЗАВИСИМОСТИ ОТ ДНЯ ---
            # Четверг (3) -> 3.0. Все остальные -> 2.0
            dynamic_rr = 3.0 if current_time.dayofweek == 3 else 2.0
            self.portfolio_manager.win_loss_ratio = dynamic_rr
            
            self.broker.update_market_state(bar)
            self.portfolio_manager.update_drawdown(self.broker.current_daily_dd)
            
            base_signal = self.strategy.generate_signal(bar)
            bar['signal'] = base_signal
            
            ml_confidence = self.get_ml_confidence(bar) if base_signal != 0 else 0.0
            
            portfolio_command = self.portfolio_manager.process_signal(
                base_signal=base_signal, features=bar, ml_confidence=ml_confidence
            )
            self.broker.execute_command(portfolio_command)

        labeled_df = pd.DataFrame(records)
        labeled_df.to_csv("data/processed/strategy_labeled_data.csv", index=False)
        logging.info("Saved labeled dataset (with active setups) for ML training.")

        trade_history = self.broker.trade_history
        if not trade_history:
            logging.warning("No trades were executed.")
            return None
            
        report = self.validator.evaluate_strategy(trade_history)
        
        logging.info("=== SYSTEM REPORT ===")
        logging.info(f"Final Equity: ${self.broker.equity:,.2f}")
        logging.info(f"Total Trades: {report.get('total_trades', 0)}")
        logging.info(f"Win Rate: {report.get('win_rate', 0)}%")
        logging.info(f"Classic Sharpe Ratio: {report.get('classic_sharpe', 0)}")
        logging.info(f"Probabilistic Sharpe Ratio (PSR): {report.get('psr_score', 0)}")
        
        mc_res = report.get('monte_carlo', {})
        if 'error' in mc_res: logging.info(f"Monte Carlo Robustness: SKIPPED ({mc_res['error']})")
        else: logging.info(f"Monte Carlo Robustness: {'PASS' if mc_res.get('is_robust') else 'FAIL'}")
            
        logging.info(f"System Status: {report.get('status', 'UNKNOWN')}")
        logging.info(f"Time elapsed: {time.time() - start_time:.2f} seconds.")
        return report

if __name__ == "__main__":
    config = {
        'max_risk_per_trade': 0.02,
        'win_loss_ratio': 2.0, # Базовое значение, оно будет меняться Оркестратором в цикле
        'max_daily_drawdown': 0.05,
        'commission_per_100k': 2.50,
        'base_spread_pips': 0.5,
        'intensity_threshold': 0.3, 
        'min_trend_score': 5
    }
    orchestrator = SystemOrchestrator(config)
    orchestrator.run_pipeline("data/raw/gbpusd_data.csv")