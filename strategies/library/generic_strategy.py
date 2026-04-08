import pandas as pd
import numpy as np

class InstitutionalSMCStrategy:
    def __init__(self, config: dict):
        self.name = "Institutional_SMC_Alpha"
        self.config = config
        self.min_trend_score = config.get('min_trend_score', 5) 
        self.intensity_threshold = config.get('intensity_threshold', 0.3) 

    def generate_signal(self, bar: dict) -> int:
        if bar.get('is_anomaly', 0) == 1: return 0 

        signal = 0
        setup_name = "None"
        
        # ПРАВИЛО 2: Торгуем только во время сессий LO и NY
        is_london_active = bar.get('london_intensity', 0.0) > self.intensity_threshold
        is_ny_active = bar.get('ny_intensity', 0.0) > self.intensity_threshold
        is_session_active = is_london_active or is_ny_active
        
        if is_session_active:
            if bar.get('pdh_sweep', 0) == 1 or bar.get('london_judas_high', 0) == 1:
                signal = -1
                setup_name = "Liquidity_Sweep_Short"
            elif bar.get('pdl_sweep', 0) == 1 or bar.get('london_judas_low', 0) == 1:
                signal = 1
                setup_name = "Liquidity_Sweep_Long"
        
        if signal != 0:
            self._inject_dynamic_risk_params(bar, setup_name)
            return signal
        
        mtfa_score = bar.get('mtfa_score', 0)
        
        if bar.get('tap_4h_bull_fvg', 0) == 1 and mtfa_score > self.min_trend_score:
            signal = 1
            setup_name = "HTF_Bull_Continuation"
            
        elif bar.get('tap_4h_bear_fvg', 0) == 1 and mtfa_score < -self.min_trend_score:
            signal = -1
            setup_name = "HTF_Bear_Continuation"

        if signal != 0:
            self._inject_dynamic_risk_params(bar, setup_name)
            return signal

        return 0

    def _inject_dynamic_risk_params(self, bar: dict, setup_name: str):
        bar['active_setup'] = setup_name
        vol_z = bar.get('volatility_z', 0.0)
        dynamic_stop_pct = 0.0015 * (1.0 + (vol_z * 0.25))
        bar['atr_pct'] = min(max(dynamic_stop_pct, 0.001), 0.004)