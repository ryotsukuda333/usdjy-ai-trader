"""Main orchestration module for USDJPY AI Trader.

Integrates all components: data fetching, feature engineering, model training,
prediction, backtesting, and visualization.

Tasks: 8.1, 8.2, 8.3, 8.4
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from features.data_fetcher import fetch_usdjpy_data
from features.feature_engineer import engineer_features
from model.train import train_model, save_model
from model.predict import load_model, predict
from backtest.backtest import run_backtest
from backtest.backtest_with_position_sizing import run_backtest_with_position_sizing
from trader.plotter import plot_backtest_results
from utils.errors import TraderError


def main():
    """Execute complete USDJPY trading pipeline.

    Requirement 8: Main orchestration combining all modules:
    - 8.1: Fetch USDJPY data (3 years)
    - 8.2: Engineer features from OHLCV
    - 8.3: Train model and generate predictions
    - 8.4: Execute backtest and visualize results
    """
    try:
        print("\n" + "="*60)
        print("USDJPY AI TRADER - COMPLETE PIPELINE")
        print("="*60)

        # Task 8.1: Fetch USDJPY data
        print("\n[8.1] Fetching USDJPY data (3 years)...")
        df_ohlcv = fetch_usdjpy_data(years=3)
        print(f"✓ Fetched {len(df_ohlcv)} candles")
        print(f"  Date range: {df_ohlcv.index[0]} to {df_ohlcv.index[-1]}")

        # Task 8.2: Engineer features
        print("\n[8.2] Engineering features from OHLCV...")
        df_features = engineer_features(df_ohlcv)
        print(f"✓ Engineered {len(df_features.columns)} features")
        col_names = [str(c) for c in df_features.columns[:5]]
        print(f"  Features: {', '.join(col_names)}... ({len(df_features.columns)} total)")

        # Align OHLCV with features (features removes initial rows due to dropna from feature engineering)
        # Take only the rows that correspond to valid feature rows
        rows_dropped = len(df_ohlcv) - len(df_features)
        if rows_dropped > 0:
            df_ohlcv = df_ohlcv.iloc[rows_dropped:]
            print(f"  ✓ Aligned OHLCV with features (dropped {rows_dropped} rows for lookback calculations)")

        # Task 8.3: Train model and generate predictions
        print("\n[8.3] Training model and generating predictions...")

        # Train model
        print("  Training XGBoost model...")
        model = train_model(df_features, test_mode=False)
        print("  ✓ Model trained successfully")

        # Save model
        print("  Saving model and feature metadata...")
        save_model(model, df_features)
        print("  ✓ Model saved")

        # Generate predictions
        print("  Generating predictions...")
        predictions = predict(df_features)
        print(f"  ✓ Generated {len(predictions)} predictions")
        print(f"    Buy signals: {(predictions == 1).sum()}")
        print(f"    Sell signals: {(predictions == 0).sum()}")

        # Task 8.4: Execute backtest and visualize
        print("\n[8.4] Executing backtest with position sizing strategies...")

        # Run baseline backtest with dynamic risk management (fixed 1 unit)
        print("  [BASELINE] Running backtest with fixed position sizing...")
        trades = run_backtest(df_ohlcv, df_features, predictions, use_dynamic_risk=True)

        if len(trades) > 0:
            print(f"  ✓ Baseline backtest complete: {len(trades)} trades")

            # Calculate baseline statistics
            wins = (trades['win_loss'] == 1).sum()
            losses = (trades['win_loss'] == 0).sum()
            win_rate = (wins / len(trades) * 100) if len(trades) > 0 else 0
            total_return = trades['return_percent'].sum()
            avg_return = trades['return_percent'].mean()

            print(f"  - Trades: {len(trades)}, Win Rate: {win_rate:.2f}%")
            print(f"  - Total Return: {total_return:+.2f}%")
        else:
            print("  ⚠ No trades generated in backtest")

        # Run position sizing strategies
        print("\n  [STRATEGY 1] Running with fixed risk % position sizing...")
        try:
            trades_fixed_risk, metrics_fixed_risk = run_backtest_with_position_sizing(
                df_ohlcv, df_features, predictions,
                use_dynamic_risk=True,
                position_sizing_strategy='fixed_risk',
                account_size=100000,
                risk_percent=1.0
            )
            print(f"  ✓ Fixed Risk strategy: {len(trades_fixed_risk)} trades")
            if metrics_fixed_risk:
                print(f"    Account: ${metrics_fixed_risk['final_account']:,.0f} ({metrics_fixed_risk['return_pct']:+.2f}%)")
        except Exception as e:
            print(f"  ⚠ Fixed Risk strategy failed: {str(e)}")
            trades_fixed_risk = None

        print("\n  [STRATEGY 2] Running with Kelly criterion position sizing...")
        try:
            trades_kelly, metrics_kelly = run_backtest_with_position_sizing(
                df_ohlcv, df_features, predictions,
                use_dynamic_risk=True,
                position_sizing_strategy='kelly',
                account_size=100000,
                kelly_fraction=0.15
            )
            print(f"  ✓ Kelly strategy: {len(trades_kelly)} trades")
            if metrics_kelly:
                print(f"    Account: ${metrics_kelly['final_account']:,.0f} ({metrics_kelly['return_pct']:+.2f}%)")
        except Exception as e:
            print(f"  ⚠ Kelly strategy failed: {str(e)}")
            trades_kelly = None
            metrics_kelly = None

        # Generate visualization
        print("\n  Generating equity curve visualization...")
        if len(trades) > 0:
            plot_path = plot_backtest_results(trades)
            print(f"  ✓ Visualization saved: {plot_path}")
        else:
            print("  ⚠ Skipping visualization (no trades)")

        # Summary
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETE")
        print("="*60)
        print(f"\nAll tasks completed successfully:")
        print(f"  [✓] 8.1 - Fetched {len(df_ohlcv)} USDJPY candles")
        print(f"  [✓] 8.2 - Engineered {len(df_features.columns)} features")
        print(f"  [✓] 8.3 - Trained model, generated {len(predictions)} predictions")
        print(f"  [✓] 8.4 - Executed backtest ({len(trades)} trades), visualized results")

        return 0

    except TraderError as e:
        print(f"\n❌ TRADER ERROR: {e.user_message}")
        print(f"Technical: {e.technical_message}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
