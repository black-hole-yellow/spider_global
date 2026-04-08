import pandas as pd
import inspect
import traceback
from shared.features import technical, structural, htf, macro,sessions,ml_features


class ResearchPipeline:
    """
    Replaces the old LabEngine. 
    Strictly responsible for building the dataset for the Strategy.
    """

    _REGISTRY_CACHE = None

    @classmethod
    def get_registry(cls) -> dict:
        if cls._REGISTRY_CACHE is None:
            registry = {}
            # 2. ДОБАВИТЬ ml_features В ЭТОТ СПИСОК:
            for module in [technical, structural, htf, macro, sessions, ml_features]:
                for name, func in inspect.getmembers(module, inspect.isfunction):
                    if hasattr(func, '_provides_features'):
                        for feature_name in func._provides_features:
                            registry[feature_name] = func
            cls._REGISTRY_CACHE = registry
        return cls._REGISTRY_CACHE
    
    
    def __init__(self, data_file: str, start_date: str, end_date: str):
        self.data_file = data_file
        self.start_date = start_date
        self.end_date = end_date
        # Instantly grab the pre-built registry
        self.feature_registry = self.get_registry()

    _DATA_CACHE = {}
    def load_and_prepare(self, strategy) -> pd.DataFrame:
        cache_key = f"{self.data_file}_{self.start_date}_{self.end_date}"
        
        try:
            # 1. Load Data (Check RAM Cache First)
            if cache_key in self.__class__._DATA_CACHE:
                # Return a quick copy from RAM instead of hitting the hard drive
                df = self.__class__._DATA_CACHE[cache_key].copy()
            else:
                print(f"Loading data from disk: {self.start_date} to {self.end_date}...")
                df = pd.read_parquet(self.data_file)
                
                if 'Datetime' in df.columns:
                    df['Datetime'] = pd.to_datetime(df['Datetime'])
                    df.set_index('Datetime', inplace=True)
                df = df.loc[self.start_date:self.end_date].copy()
                
                if df.index.tz is None: 
                    df.index = df.index.tz_localize('UTC')
                    
                # Save to RAM for the next strategy
                self.__class__._DATA_CACHE[cache_key] = df.copy()

            df = technical.calculate_atr(df)
            df = structural.calculate_htf_sweep_stops(df)

            # 2. Extract required features from the Strategy
            required_features = strategy.get_required_features()
            unique_funcs = {self.feature_registry[f] for f in required_features if f in self.feature_registry}
            
            sorted_funcs = sorted(list(unique_funcs), key=lambda x: x.__name__)
            
            # 3. Apply the pipeline
            for func in sorted_funcs:
                df = func(df)
                
            return df
            
        except Exception as e:
            print(f"❌ Critical Error in Data Pipeline: {e}")
            traceback.print_exc()
            return pd.DataFrame()