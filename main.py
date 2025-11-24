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
from model.step12_hybrid_strategy import HybridTradingStrategy
from backtest.backtest import run_backtest
from backtest.backtest_seasonality_aware import run_backtest_seasonality_aware
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

        # Initialize risk results list
        risk_results = []

        # Step 10.4: Seasonality-aware risk adjustment
        print("\n  [STEP 10] Seasonality-aware risk adjustment...")
        print("  " + "-" * 70)
        try:
            trades_seasonality, metrics_seasonality = run_backtest_seasonality_aware(
                df_ohlcv, df_features, predictions,
                account_size=100000,
                use_dynamic_risk=True
            )
            print(f"  ✓ Seasonality-aware backtest complete: {len(trades_seasonality)} trades")
            print(f"    Account: ${metrics_seasonality['final_account']:,.0f} ({metrics_seasonality['return_pct']:+.2f}%)")

            # Add to comparison
            risk_results.append({
                'strategy': 'Seasonality-Aware (Step 10)',
                'trades': len(trades_seasonality),
                'return_pct': metrics_seasonality['return_pct'],
                'final_account': metrics_seasonality['final_account'],
                'metrics': metrics_seasonality
            })

            # Update visualization with seasonality-aware results
            if len(trades_seasonality) > 0:
                print(f"    Results saved to: backtest_results_seasonality_aware.csv")
        except Exception as e:
            print(f"    ⚠ Seasonality-aware backtest failed: {str(e)}")
            import traceback
            traceback.print_exc()

        # Step 12: Hybrid Strategy (Seasonality + XGBoost)
        print("\n  [STEP 12] Hybrid Trading Strategy (Seasonality + XGBoost)...")
        print("  " + "-" * 70)
        try:
            # Initialize hybrid strategy with current XGBoost model
            xgb_model_path = Path(project_root) / "model" / "xgb_model.json"
            hybrid_strategy = HybridTradingStrategy(
                xgb_model_path=str(xgb_model_path) if xgb_model_path.exists() else None,
                use_seasonality=True
            )

            # Generate hybrid predictions
            print("  Generating hybrid signals...")
            feature_cols = [col for col in df_features.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            hybrid_predictions = hybrid_strategy.generate_predictions(
                df_features,
                feature_cols=feature_cols,
                xgb_threshold=0.5
            )

            print(f"  ✓ Hybrid predictions generated:")
            print(f"    - Buy signals: {(hybrid_predictions['hybrid_signal'] == 1).sum()}")
            print(f"    - Sell signals: {(hybrid_predictions['hybrid_signal'] == 0).sum()}")
            print(f"    - Hold signals: {(hybrid_predictions['hybrid_signal'] == -1).sum()}")

            # Run hybrid backtest
            print("  Running hybrid backtest...")
            total_return, metrics_hybrid = hybrid_strategy.backtest_hybrid_strategy(
                df_ohlcv,
                hybrid_predictions,
                initial_capital=100000
            )

            print(f"  ✓ Hybrid backtest complete: {metrics_hybrid['num_trades']} trades")
            print(f"    Total Return: {total_return:+.2f}%")
            print(f"    Win Rate: {metrics_hybrid['win_rate']:.2f}%")
            print(f"    Final Account: ${metrics_hybrid['final_equity']:,.0f}")

            # Add to comparison
            risk_results.append({
                'strategy': 'Hybrid (Step 12)',
                'trades': metrics_hybrid['num_trades'],
                'return_pct': total_return,
                'final_account': metrics_hybrid['final_equity'],
                'metrics': metrics_hybrid
            })

        except Exception as e:
            print(f"    ⚠ Hybrid backtest failed: {str(e)}")
            import traceback
            traceback.print_exc()

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
