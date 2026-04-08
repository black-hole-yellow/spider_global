import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Suppress pandas warnings during massive loops
warnings.filterwarnings('ignore')

from strategies.library.factory import StrategyFactory
from research.engine.pipeline import ResearchPipeline
from trading.execution.paper_broker import PaperBroker
from trading.portfolio.portfolio_manager import PortfolioManager
from trading.execution.execution_engine import ExecutionEngine
from trading.backtest.event_driven_backtester import EventDrivenBacktester

def run_batch_tests(
    testing_dir: str = "strategies/configs/testing", 
    data_path: str = "data/processed/gbpusd_15m.parquet",
    start_date: str = "2000-01-01",
    end_date: str = "2026-01-01"
):
    target_dir = Path(testing_dir)
    if not target_dir.exists():
        print(f"❌ Directory not found: {testing_dir}")
        return

    json_files = list(target_dir.glob("*.json"))
    if not json_files:
        print(f"⚠️ No JSON configs found in {testing_dir}. Move some from archive/ first!")
        return

    print(f"\n🚀 --- STARTING BATCH RUN: {len(json_files)} Hypotheses --- \n")
    
    results = []

    for config_path in json_files:
        with open(config_path, 'r') as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping {config_path.name} (Invalid JSON)")
                continue
        
        try:
            # 1. Initialize Strategy
            strategy = StrategyFactory.load_from_config(config)
            
            # 2. Build Data
            pipeline = ResearchPipeline(data_path, start_date, end_date)
            prepared_data = pipeline.load_and_prepare(strategy)
            
            if prepared_data.empty:
                print(f"⏭️ Skipping {strategy.name}: Pipeline returned empty data.")
                continue

            # 3. Setup Execution
            broker = PaperBroker(
                initial_cash=10000.0, 
                slippage_pips=config.get("execution", {}).get("slippage_pips", 1.0),
                commission_per_unit=config.get("execution", {}).get("commission_per_unit", 0.00002)
            )
            pm = PortfolioManager(config)
            engine = ExecutionEngine(broker)

            # 4. Run Backtest
            backtester = EventDrivenBacktester(
                data=prepared_data, strategy=strategy, portfolio_manager=pm, 
                execution_engine=engine, broker=broker
            )
            
            df_equity = backtester.run()

            # 5. Extract Metrics
            returns = df_equity['returns']
            total_ret = (df_equity['equity'].iloc[-1] / df_equity['equity'].iloc[0]) - 1.0
            
            # 24192 is the approx number of 15m bars in a trading year
            sharpe = np.sqrt(24192) * (returns.mean() / returns.std()) if returns.std() != 0 else 0
            
            rolling_max = df_equity['equity'].cummax()
            max_dd = (df_equity['equity'] / rolling_max - 1.0).min()

            # Save to leaderboard array
            results.append({
                "Strategy": strategy.name[:30], # Truncate long names for clean table
                "Total Return": total_ret,
                "Sharpe": sharpe,
                "Max DD": max_dd,
                "File": config_path.name
            })
            
        except Exception as e:
            print(f"❌ FAILED {config_path.name}: {str(e)}")

    # --- Generate Leaderboard ---
    if results:
        df_results = pd.DataFrame(results)
        
        # Sort by best Sharpe Ratio
        df_results = df_results.sort_values(by="Sharpe", ascending=False).reset_index(drop=True)
        
        # Format for display
        df_results['Total Return'] = df_results['Total Return'].apply(lambda x: f"{x:.2%}")
        df_results['Sharpe'] = df_results['Sharpe'].apply(lambda x: f"{x:.2f}")
        df_results['Max DD'] = df_results['Max DD'].apply(lambda x: f"{x:.2%}")
        
        print("\n" + "="*80)
        print("🏆 TESTING BATCH LEADERBOARD (Sorted by Sharpe)")
        print("="*80)
        print(df_results.to_string(index=False))
        print("="*80 + "\n")
        
        # Save to CSV for your records
        df_results.to_csv("batch_results_log.csv", index=False)
        print("💾 Saved full results to batch_results_log.csv")

if __name__ == "__main__":
    run_batch_tests()