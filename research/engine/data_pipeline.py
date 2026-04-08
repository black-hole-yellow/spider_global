import polars as pl
import numpy as np

class DataQualityEngine:
    def __init__(self, timeframe_mins: int = 15, spike_threshold: float = 10.0):
        self.tf_mins = timeframe_mins
        self.spike_threshold = spike_threshold

    def ingest_and_clean(self, filepath: str) -> pl.DataFrame:
        if filepath.endswith('.csv'):
            # ФИКС 1: Указываем табуляцию, отключаем хидер и жестко задаем имена колонок
            lf = pl.scan_csv(
                filepath, 
                separator='\t', 
                has_header=False, 
                new_columns=['timestamp', 'open', 'high', 'low', 'close']
            )
        elif filepath.endswith('.parquet'):
            lf = pl.scan_parquet(filepath)
            # ФИКС 2: Убираем PerformanceWarning, используя правильный метод collect_schema().names()
            lf = lf.rename({col: col.lower() for col in lf.collect_schema().names()})
        else:
            raise ValueError("Unsupported file format. Use .csv or .parquet")

        # Строку с lf.rename для CSV мы удалили, так как задали new_columns выше
        
        # Строгий парсинг дат для Polars
        lf = lf.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False))
        lf = lf.sort("timestamp")

        lf = self._apply_quality_gates(lf)

        lf = lf.with_columns(
            pl.when(pl.col("is_bad_data_spike"))
            .then(pl.col("close").rolling_median(window_size=10, min_periods=1))
            .otherwise(pl.col("close"))
            .alias("close")
        )

        clean_df = lf.collect()
        self._report_quality(clean_df)
        return clean_df

    def _apply_quality_gates(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        lf = lf.unique(subset=["timestamp"], keep="last", maintain_order=True)
        lf = lf.with_columns(pl.col("timestamp").diff().alias("time_delta"))
        lf = lf.with_columns((pl.col("close") / pl.col("close").shift(1)).log().alias("log_ret"))
        lf = lf.with_columns(pl.col("log_ret").rolling_std(window_size=192).alias("roll_std"))
        lf = lf.with_columns(
            (pl.col("log_ret").abs() > (self.spike_threshold * pl.col("roll_std")))
            .fill_null(False)
            .alias("is_bad_data_spike")
        )
        return lf

    def _report_quality(self, df: pl.DataFrame):
        total_rows = df.height
        bad_spikes = df.filter(pl.col("is_bad_data_spike")).height
        gaps = df.filter(pl.col("time_delta") > pl.duration(hours=1)).height

        print(f"--- Data Quality Report ---")
        print(f"Total Bars: {total_rows}")
        print(f"Data Spikes Detected (>10x ATR proxy): {bad_spikes}")
        print(f"Significant Session Gaps (>1h): {gaps}")
        print(f"---------------------------")