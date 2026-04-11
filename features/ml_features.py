import pandas as pd
import numpy as np
import warnings
from scipy.stats import skew
from sklearn.mixture import GaussianMixture

# Suppress sklearn/scipy warnings during rolling calculations
warnings.filterwarnings('ignore', category=UserWarning)

# =========================================================================
# 🧠 MASTER WRAPPER (LAYER 5)
# =========================================================================
def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master entry point for Machine Learning and Statistical features.
    Assumes Technical, Structural, HTF, and Session layers are already computed.
    """
    if df.empty or len(df) < 100: 
        return df

    df = _add_regime_and_changepoint(df)
    df = _add_hurst_and_fractal(df)
    df = _add_adaptive_volatility_skew(df)
    df = _add_market_efficiency_index(df)
    df = _add_bayesian_regime_probabilities(df)

    return df

# =========================================================================
# 🔬 INTERNAL MODULES
# =========================================================================

def _add_regime_and_changepoint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orthogonal features for ML and Hybrid Changepoint Detector (Variance Ratio + CUSUM).
    """
    # 1. Zero-lag Returns (Fallback if technical.py didn't compute it)
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)

    # 2. Orthogonal Features
    long_window = 96 # 1 day for 15m timeframe
    
    roll_std = df['log_return'].rolling(long_window).std()
    roll_mean = df['log_return'].rolling(long_window).mean()
    
    df['volatility_z'] = abs(df['log_return'] - roll_mean) / (roll_std + 1e-8)
    
    dir_move = abs(df['close'] - df['close'].shift(long_window))
    tot_move = abs(df['close'] - df['close'].shift(1)).rolling(long_window).sum()
    df['trend_strength'] = dir_move / (tot_move + 1e-8)

    # 3. Hybrid Changepoint Detection
    short_window = 8 # 2 hours
    short_std = df['log_return'].rolling(short_window).std()
    
    # 3.1 Variance Ratio
    variance_ratio = (short_std ** 2) / (roll_std ** 2 + 1e-8)
    vr_prob = 1 / (1 + np.exp(-2 * (variance_ratio - 2.5)))

    # 3.2 CUSUM (Cumulative Sum of Volatility Deviations)
    k_drift = 0.5 
    cusum_threshold = 3.0 

    deviation = df['volatility_z'] - k_drift
    cumulative_deviation = deviation.cumsum()
    
    local_min = cumulative_deviation.rolling(window=24, min_periods=1).min()
    cusum_score = cumulative_deviation - local_min

    df['cusum_signal'] = np.clip(cusum_score / cusum_threshold, 0, 1)
    df['changepoint_prob'] = np.maximum(vr_prob, df['cusum_signal'])

    # 4. Instant Kill Switch (Anomaly)
    df['is_anomaly'] = np.where((df['volatility_z'] > 4.0) | (df['changepoint_prob'] > 0.90), 1, 0)

    return df.fillna(0)

def _add_hurst_and_fractal(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """
    Local Hurst Exponent & Fractal Dimension Trajectories.
    H < 0.5 (Mean Reversion), H > 0.5 (Momentum). D = 2 - H.
    """
    def get_hurst(prices):
        if len(prices) < 20: return np.nan
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    df['hurst_exponent'] = df['close'].rolling(window=lookback).apply(get_hurst, raw=True)
    df['fractal_dimension'] = 2.0 - df['hurst_exponent']
    
    return df

def _add_adaptive_volatility_skew(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Adaptive Fractal Volatility Skewness.
    Measures asymmetry of volatility clusters (panic/expansion).
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    df['volatility_skewness'] = tr.rolling(window=lookback).apply(skew, raw=True)
    return df

def _add_market_efficiency_index(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Adaptive Market Efficiency Index.
    Ratio of N-bar variance to sum of 1-bar variances.
    """
    if 'log_return' not in df.columns:
        returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    else:
        returns = df['log_return']
    
    var_n = returns.rolling(window=lookback).sum().var() 
    var_1 = returns.rolling(window=lookback).var() * lookback
    
    vr = var_n / (var_1 + 1e-9)
    df['market_efficiency_index'] = np.abs(vr - 1.0)
    
    return df

def _add_bayesian_regime_probabilities(df: pd.DataFrame, lookback: int = 500) -> pd.DataFrame:
    """
    Bayesian Regime Probability Surface using Gaussian Mixture Models.
    Identifies probability of Chop, Trend, or Breakout regimes.
    """
    # Features for GMM
    ret = df['log_return'].values if 'log_return' in df.columns else np.log(df['close'] / df['close'].shift(1)).fillna(0).values
    tr = ((df['high'] - df['low']) / df['close'] * 100).values
    
    features = np.column_stack([ret, tr])
    
    prob_trend, prob_chop, prob_break = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
    
    # Step size for re-fitting GMM (1 day = 96 bars for 15m)
    step = 96 
    
    for i in range(lookback, len(df), step):
        window_data = features[i-lookback:i]
        end_idx = min(i+step, len(df))
        
        try:
            gmm = GaussianMixture(n_components=3, covariance_type='diag', random_state=42)
            gmm.fit(window_data)
            
            test_data = features[i:end_idx]
            probs = gmm.predict_proba(test_data)
            
            # Sort modes by volatility (TR is column 1)
            vol_centers = gmm.means_[:, 1]
            sorted_indices = np.argsort(vol_centers)
            
            chop_idx, trend_idx, break_idx = sorted_indices[0], sorted_indices[1], sorted_indices[2]
            
            prob_chop[i:end_idx] = probs[:, chop_idx]
            prob_trend[i:end_idx] = probs[:, trend_idx]
            prob_break[i:end_idx] = probs[:, break_idx]
        except Exception:
            # Fallback for non-convergence
            prob_chop[i:end_idx] = 0.33
            prob_trend[i:end_idx] = 0.33
            prob_break[i:end_idx] = 0.33

    df['regime_chop_prob'] = prob_chop
    df['regime_trend_prob'] = prob_trend
    df['regime_break_prob'] = prob_break
    
    # Mask initial lookback period
    cols = ['regime_chop_prob', 'regime_trend_prob', 'regime_break_prob']
    df.loc[df.index[:lookback], cols] = np.nan
    
    return df