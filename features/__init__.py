"""
Institutional Feature Engineering Graph.
Central registry and master wrapper exports for the ML pipeline.
"""

# =========================================================================
# 1. FEATURE MANIFEST (SCHEMA)
# Used by GlobalAlphaAgent and MetaLabeler to dynamically select/scale inputs
# =========================================================================
feature_manifest = {
    "continuous": [
        "atr", "volatility_zscore", "log_return", "norm_slope", 
        "premium_discount", "dist_to_liq_high", "dist_to_liq_low", 
        "fvg_bull_size", "fvg_bear_size", "inst_footprint_divergence", 
        "liquidity_absorption_ratio", "sm_conviction_decay", "asia_intensity", 
        "london_intensity", "ny_intensity", "session_overlap_score", 
        "session_liquidity_transfer", "trend_strength", "cusum_signal", 
        "changepoint_prob", "hurst_exponent", "fractal_dimension", 
        "volatility_skewness", "market_efficiency_index", "dist_to_pdh", 
        "dist_to_pdl", "mtfa_score", "trend_1d", "trend_4h", "trend_1h", 
        "linreg_score", "transfer_entropy_proxy", "vol_term_structure", 
        "ob_migration_velocity", "llm_sentiment_score", "macro_divergence_score"
    ],
    "categorical": [
        "active_session_name", "markov_regime"
    ],
    "probabilities": [
        "high_vol_prob", "regime_trend_prob", "regime_chop_prob", 
        "regime_break_prob", "llm_regime_prob", "macro_regime_1", "macro_regime_2"
    ],
    "binary_signals": [
        "algo_vol_crush_short", "ifvg_long", "ifvg_short", "bos_long", "bos_short",
        "sweep_bull", "sweep_bear", "is_anomaly", "pdh_sweep", "pdl_sweep", 
        "tap_4h_bull_fvg", "tap_4h_bear_fvg", "is_eqh", "is_eql", "is_macro_alignment"
    ],
    "macro_triggers": [
        "CPI_Momentum_Long", "CPI_Momentum_Short", "Macro_CPI_Div_Long", 
        "FOMC_Sell_News_Long", "CB_Divergence_Long", "BoE_Tone_Shift_Short", 
        "NFP_Resumption_Long", "NFP_Fade_Long"
    ]
}

# =========================================================================
# 2. MASTER WRAPPER EXPORTS
# These functions enforce the strict Layer 1 -> Layer 6 execution graph.
# =========================================================================

# Layer 1: Base Technical & M15 Structure
from .technical import add_technical_features

# Layer 2: SMC Structure & Institutional Footprints
from .structural import add_structural_features

# Layer 3: Higher Timeframe Context & Multi-Timeframe Alignment
from .htf import add_htf_features

# Layer 4: Cyclical Time & Session Liquidity
from .sessions import add_session_features

# Layer 5: Advanced ML Orchestration (Regimes, Chaos Theory, Anomaly)
from .ml_features import add_ml_features

# Layer 6: Macro Events, Semantic LLM Scores & Strategy Triggers
# Note: Ensure you create a wrapper `add_macro_features` in macro.py 
# that calls add_macro_events -> add_macro_strategy_triggers -> add_llm_semantic_features
from .macro import add_macro_features 

# Restricting what gets imported when someone uses `from features import *`
__all__ = [
    "feature_manifest",
    "add_technical_features",
    "add_structural_features",
    "add_htf_features",
    "add_session_features",
    "add_ml_features",
    "add_macro_features"
]