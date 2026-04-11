import pandas as pd

def resample_1m_to_15m(input_csv_path: str, output_parquet_path: str):
    """
    Ingests 1m OHLCV data, aggregates it to 15m candles, and saves to Parquet.
    """
    print(f"📥 Loading 1m data from: {input_csv_path}")
    
    # 1. Load the CSV. 
    # Adjust 'sep' if your data is tab-separated (sep='\t') or comma-separated (sep=',')
    try:
        df = pd.read_csv(
            input_csv_path, 
            header=0, # Change to None if your CSV has no header row
            parse_dates=True, 
            index_col=0 # Assumes datetime is the first column
        )
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return

    # 2. Standardize columns
    # We force them to lower case. If your CSV lacks headers, you must rename them:
    # df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    if 'volume' not in df.columns:
        raise ValueError("❌ CRITICAL: The loaded data does not have a 'volume' column!")

    print(f"📊 Raw 1m data shape: {df.shape}")
    print("🔄 Resampling to 15-minute timeframe...")

    # 3. Define how OHLCV aggregates from 1m to 15m
    aggregation_dict = {
        'open': 'first',   # The open of the first 1m candle in the 15m block
        'high': 'max',     # The highest high of the 15m block
        'low': 'min',      # The lowest low of the 15m block
        'close': 'last',   # The close of the last 1m candle in the 15m block
        'volume': 'sum'    # Total volume traded within the 15m block
    }

    # 4. Perform the resample
    # '15min' is the frequency. 
    # label='left' and closed='left' means a candle from 10:00 to 10:14:59 is labeled 10:00
    df_15m = df.resample('15min', label='left', closed='left').agg(aggregation_dict)

    # 5. Clean up missing data (weekends, market holidays, gaps)
    # If a 15m block had zero 1m candles, pandas fills it with NaNs. We drop those.
    df_15m.dropna(how='all', inplace=True)
    
    # Sometimes volume is 0 during dead periods, replace with 0.0001 to prevent Division by Zero in math
    df_15m['volume'] = df_15m['volume'].replace(0, 0.0001)

    print(f"✅ Resampling complete. Final 15m shape: {df_15m.shape}")

    # 6. Save to Parquet (Lightning fast format required by our pipeline)
    df_15m.to_parquet(output_parquet_path)
    print(f"💾 Saved successfully to: {output_parquet_path}")

if __name__ == "__main__":
    # Example Usage: Update these paths to where you downloaded your 1m data
    INPUT_FILE = "data/raw/gbpusd_1m.csv"
    OUTPUT_FILE = "data/raw/gbpusd_15m.parquet"
    
    resample_1m_to_15m(INPUT_FILE, OUTPUT_FILE)