import pandas as pd
import numpy as np
import logging
from catboost import CatBoostClassifier, Pool

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class MetaLabelingTrainer:
    def __init__(self):
        self.model = None

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Generating labels: TP, SL, or Friday Close limit...")
        labeled_df = df.copy()
        
        labeled_df['active_setup'] = labeled_df['active_setup'].astype(str)
        mask = (labeled_df['active_setup'] != 'None') & (labeled_df['active_setup'] != 'nan')
        
        signal_locs = np.where(mask)[0]
        close_arr = labeled_df['close'].values
        high_arr = labeled_df['high'].values
        low_arr = labeled_df['low'].values
        atr_arr = labeled_df['atr_pct'].values
        
        # Векторизация времени для поиска пятницы и дней недели
        timestamps = pd.to_datetime(labeled_df.index)
        dayofweek_arr = timestamps.dayofweek.values
        hour_arr = timestamps.hour.values
        
        if 'signal' in labeled_df.columns:
            direction_arr = labeled_df['signal'].values
        else:
            direction_arr = np.where(labeled_df.get('london_judas_low', pd.Series([0]*len(labeled_df))).values == 1, 1, -1)
            
        meta_labels = np.full(len(labeled_df), np.nan)
        
        # ПРАВИЛО 1: Четверг (index 3) = R/R 1:3. Остальные = 1:2.
        target_rr_arr = np.where(dayofweek_arr == 3, 3.0, 2.0)
        
        # Сканируем до 5 дней вперед (480 баров), так как мы ждем Тейк, Стоп или Пятницу
        horizon_bars = 480 
        
        for loc in signal_locs:
            if loc + 1 >= len(labeled_df): continue
                
            entry_price = close_arr[loc]
            direction = direction_arr[loc]
            
            if direction == 0 or pd.isna(direction): continue
                
            stop_distance = entry_price * atr_arr[loc]
            take_profit_dist = stop_distance * target_rr_arr[loc]
            
            stop_price = entry_price - (direction * stop_distance)
            take_price = entry_price + (direction * take_profit_dist)
            
            end_loc = min(loc + horizon_bars + 1, len(labeled_df))
            future_highs = high_arr[loc+1 : end_loc]
            future_lows = low_arr[loc+1 : end_loc]
            future_dow = dayofweek_arr[loc+1 : end_loc]
            future_hour = hour_arr[loc+1 : end_loc]
            
            if direction == 1:
                stops = np.where(future_lows <= stop_price)[0]
                takes = np.where(future_highs >= take_price)[0]
            else:
                stops = np.where(future_highs >= stop_price)[0]
                takes = np.where(future_lows <= take_price)[0]
                
            # ПРАВИЛО 3: Принудительное закрытие в пятницу (day=4) после 21:00
            fridays = np.where((future_dow == 4) & (future_hour >= 21))[0]
            
            stop_idx = stops[0] if len(stops) > 0 else 9999
            take_idx = takes[0] if len(takes) > 0 else 9999
            time_idx = fridays[0] if len(fridays) > 0 else 9999
            
            # Проверяем, что произошло раньше
            first_event = min(take_idx, stop_idx, time_idx)
            
            if first_event == 9999:
                meta_labels[loc] = 0 # Не дошли ни до чего (конец истории)
            elif first_event == take_idx:
                meta_labels[loc] = 1 # Успешный Тейк-Профит
            else:
                meta_labels[loc] = 0 # Убыток по Стопу или принудительное закрытие в Пятницу
                
        labeled_df['meta_label'] = meta_labels
        return labeled_df.dropna(subset=['meta_label'])

    def train(self, df: pd.DataFrame, features_list: list, save_path: str = 'models/meta_model.cbm'):
        train_ready_df = self.generate_labels(df)
        if len(train_ready_df) < 50:
            logging.error("Not enough setups to train!")
            return
            
        logging.info(f"Total training setups: {len(train_ready_df)}. Win Rate: {train_ready_df['meta_label'].mean():.2%}")

        train_ready_df['active_setup'] = train_ready_df['active_setup'].fillna("None").astype(str)
        if 'active_session_name' in train_ready_df.columns:
            train_ready_df['active_session_name'] = train_ready_df['active_session_name'].fillna("Asian").astype(str)
        else:
            train_ready_df['active_session_name'] = "Asian"
            
        X = train_ready_df[features_list]
        y = train_ready_df['meta_label'].astype(int)
        
        split_idx = int(len(train_ready_df) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        cat_features = [f for f in ['active_setup', 'active_session_name'] if f in features_list]

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        self.model = CatBoostClassifier(
            iterations=500, learning_rate=0.03, depth=5, 
            eval_metric='Logloss', random_seed=42, early_stopping_rounds=50, verbose=100
        )

        logging.info("Training CatBoost Meta-Labeler...")
        self.model.fit(train_pool, eval_set=val_pool)
        self.model.save_model(save_path)
        logging.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    ml_features = [
        'active_setup', 'volatility_z', 'changepoint_prob', 'trend_strength', 
        'cusum_signal', 'asia_intensity', 'london_intensity', 'ny_intensity', 
        'session_overlap_score', 'dist_to_pwh', 'dist_to_pwl', 'mtfa_score', 
        "llm_sentiment_score", "is_macro_alignment", "active_session_name"
    ]
    try:
        df = pd.read_csv("data/processed/strategy_labeled_data.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        trainer = MetaLabelingTrainer() 
        trainer.train(df, ml_features)
    except Exception as e:
        logging.error(f"Failed to run training: {e}")