"""
Phase 5-B Backtest Execution Script

信号品質改善の実装をバックテストで検証:
1. Phase 5-A (現在): +2% (Grid Search後)
2. Phase 5-B (本実行): 期待値 +35-45%

実行方法:
    python3 backtest/run_phase5b_backtest.py
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.data_fetcher import fetch_usdjpy_data
from features.feature_engineer import engineer_features
from model.step12_hybrid_strategy_improved import HybridTradingStrategyImproved


def run_phase5b_backtest():
    """Phase 5-B バックテストの実行"""

    project_root = Path(__file__).parent.parent

    print("=" * 80)
    print("PHASE 5-B: Signal Quality Improvement - Backtest Execution")
    print("=" * 80)

    # Step 1: Data Fetching
    print("\n[1] Fetching USDJPY data...")
    start_time = time.time()

    try:
        df_ohlcv = fetch_usdjpy_data(years=3)
        print(f"✓ Fetched {len(df_ohlcv)} candles ({time.time() - start_time:.1f}s)")
    except Exception as e:
        print(f"✗ Data fetch failed: {e}")
        return None

    # Step 2: Feature Engineering
    print("[2] Engineering features...")
    feature_start = time.time()

    try:
        df_features = engineer_features(df_ohlcv)
        print(f"✓ Engineered {len(df_features.columns)} features ({time.time() - feature_start:.1f}s)")
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        return None

    # Step 3: Load feature columns
    print("[3] Loading feature configuration...")
    feature_cols_path = project_root / "model" / "feature_columns.json"

    if feature_cols_path.exists():
        with open(feature_cols_path) as f:
            feature_cols = json.load(f)
        print(f"✓ Loaded {len(feature_cols)} feature columns")
    else:
        feature_cols = [col for col in df_features.columns
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"✓ Using {len(feature_cols)} auto-detected features")

    # Step 4: Initialize improved strategy
    print("[4] Initializing Phase 5-B improved strategy...")

    xgb_model_path = project_root / "model" / "xgb_model.json"

    strategy = HybridTradingStrategyImproved(
        xgb_model_path=str(xgb_model_path) if xgb_model_path.exists() else None,
        use_seasonality=True,
        enable_quality_filter=True
    )
    print("✓ Strategy initialized with Phase 5-B quality filtering")

    # Step 5: Generate predictions
    print("[5] Generating predictions with quality scoring...")
    pred_start = time.time()

    predictions = strategy.generate_predictions_with_quality(
        df=df_features,
        feature_cols=feature_cols,
        xgb_threshold=0.45,  # Best parameters from Grid Search
        quality_threshold=0.40  # Adjusted from 0.60 for realistic signal filtering
    )

    print(f"✓ Generated {len(predictions)} predictions ({time.time() - pred_start:.1f}s)")

    # Analysis of signal filtering
    signal_stats = {
        'total_signals_generated': (predictions['signal'] != -1).sum(),
        'strong_quality_signals': (predictions['quality_score'] >= 0.75).sum(),
        'medium_quality_signals': ((predictions['quality_score'] >= 0.60) & (predictions['quality_score'] < 0.75)).sum(),
        'weak_quality_signals': ((predictions['quality_score'] >= 0.45) & (predictions['quality_score'] < 0.60)).sum(),
        'rejected_signals': (predictions['quality_score'] < 0.45).sum(),
        'filtered_out': (~predictions['should_execute'] & (predictions['signal'] != -1)).sum(),
        'executed_signals': (predictions['should_execute'] & (predictions['signal'] == 1)).sum(),
    }

    print("\n  Signal Quality Distribution:")
    print(f"  ├─ Total signals generated: {signal_stats['total_signals_generated']}")
    print(f"  ├─ Strong quality (≥0.75): {signal_stats['strong_quality_signals']}")
    print(f"  ├─ Medium quality (0.60-0.75): {signal_stats['medium_quality_signals']}")
    print(f"  ├─ Weak quality (0.45-0.60): {signal_stats['weak_quality_signals']}")
    print(f"  ├─ Rejected (<0.45): {signal_stats['rejected_signals']}")
    print(f"  ├─ Filtered out by Phase 5-B: {signal_stats['filtered_out']}")
    print(f"  └─ Executed signals: {signal_stats['executed_signals']}")

    # Step 6: Align data lengths with timezone normalization
    print("[6] Aligning data with timezone normalization...")

    # Normalize both indices to UTC naive for proper alignment
    # df_ohlcv: UTC naive (from yfinance)
    # df_features: JST-aware (from engineer_features)

    # Convert df_features from JST-aware to UTC naive
    df_features_normalized = df_features.copy()
    if df_features_normalized.index.tz is not None:
        # Convert JST-aware to UTC, then remove timezone info
        df_features_normalized.index = df_features_normalized.index.tz_convert('UTC').tz_localize(None)

    # Ensure df_ohlcv is UTC naive
    df_ohlcv_normalized = df_ohlcv.copy()
    if df_ohlcv_normalized.index.tz is not None:
        df_ohlcv_normalized.index = df_ohlcv_normalized.index.tz_localize(None)

    # Now find common index
    common_index = df_ohlcv_normalized.index.intersection(df_features_normalized.index)

    if len(common_index) == 0:
        print(f"✗ No common dates found after timezone normalization!")
        print(f"  ├─ OHLCV date range: {df_ohlcv_normalized.index[0]} to {df_ohlcv_normalized.index[-1]}")
        print(f"  └─ Features date range: {df_features_normalized.index[0]} to {df_features_normalized.index[-1]}")
        return None

    df_ohlcv_aligned = df_ohlcv_normalized.loc[common_index].copy()
    df_features_aligned = df_features_normalized.loc[common_index].copy()

    # Align predictions to the common index
    # Predictions should have same length as df_features, so slice them
    predictions_aligned = predictions.iloc[:len(df_features_aligned)].copy()
    predictions_aligned.index = df_features_aligned.index

    df_aligned = df_ohlcv_aligned.copy()

    print(f"  ├─ Original OHLCV: {len(df_ohlcv)} rows")
    print(f"  ├─ Original features: {len(df_features)} rows")
    print(f"  ├─ Original predictions: {len(predictions)} rows")
    print(f"  ├─ Common dates found: {len(common_index)}")
    print(f"  └─ Aligned OHLCV/Predictions: {len(df_aligned)} rows")

    # Step 7: Run backtest
    print("\n[7] Running backtest...")
    backtest_start = time.time()

    total_return, metrics = strategy.backtest_improved(
        df=df_aligned,
        predictions=predictions_aligned,
        initial_capital=100000,
        use_quality_filter=True
    )

    backtest_time = time.time() - backtest_start
    print(f"✓ Backtest completed ({backtest_time:.1f}s)")

    # Step 8: Results Summary
    print("\n" + "=" * 80)
    print("PHASE 5-B BACKTEST RESULTS")
    print("=" * 80)

    print(f"\nPerformance Metrics:")
    print(f"  Total Return: +{metrics['total_return']:.2f}%")
    print(f"  Final Equity: ${metrics['final_equity']:.2f}")
    print(f"  Number of Trades: {metrics['num_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"  Max Drawdown: -{metrics['max_dd']:.2f}%")
    print(f"  Avg Quality Score: {metrics['avg_quality_score']:.2f}")

    print(f"\nSignal Processing:")
    print(f"  Signals Generated: {metrics['signals_generated']}")
    print(f"  Signals Filtered: {metrics['signals_filtered']}")
    print(f"  Signals Executed: {metrics['signals_executed']}")

    # Step 8: Comparison with Phase 5-A
    print("\n" + "=" * 80)
    print("PHASE 5-A vs PHASE 5-B COMPARISON")
    print("=" * 80)

    comparison = {
        'metric': ['Total Return', 'Win Rate', 'Trades', 'Max DD', 'Sharpe'],
        'Phase 5-A': ['+2.00%', '57.1%', '7', '-0.25%', '7.87'],
        'Phase 5-B': [f"+{metrics['total_return']:.2f}%", f"{metrics['win_rate']:.1f}%",
                     f"{metrics['num_trades']}", f"-{metrics['max_dd']:.2f}%", f"{metrics['sharpe']:.2f}"],
        'Target': ['+35-45%', '62%+', '150-200', '≤-1.5%', '6.5+']
    }

    print("\n{:<20} {:<15} {:<20} {:<20}".format('Metric', 'Phase 5-A', 'Phase 5-B', 'Target'))
    print("-" * 75)
    for i in range(len(comparison['metric'])):
        metric = comparison['metric'][i]
        phase_a = comparison['Phase 5-A'][i]
        phase_b = comparison['Phase 5-B'][i]
        target = comparison['Target'][i]

        # Color coding for target achievement
        achieved = False
        if 'Return' in metric and '+' in phase_b:
            val = float(phase_b.replace('%', '').replace('+', ''))
            achieved = val >= 35
        elif 'Win' in metric:
            val = float(phase_b.replace('%', ''))
            achieved = val >= 62
        elif 'Trades' in metric:
            val = int(phase_b)
            achieved = 150 <= val <= 200
        elif 'DD' in metric:
            val = float(phase_b.replace('%', '').replace('-', ''))
            achieved = val <= 1.5
        elif 'Sharpe' in metric:
            val = float(phase_b)
            achieved = val >= 6.5

        status = "✓" if achieved else "✗"
        print("{:<20} {:<15} {:<20} {:<20}".format(metric, phase_a, f"{phase_b} {status}", target))

    # Step 9: Phase 6 Adoption Decision
    print("\n" + "=" * 80)
    print("PHASE 6 ADOPTION JUDGMENT")
    print("=" * 80)

    print(f"\nAdoption Criteria Check:")
    print(f"  MUST: Total Return > +65%? → {metrics['total_return']:.2f}% ", end="")
    must_return = metrics['total_return'] > 65
    print("✅" if must_return else "❌")

    print(f"  MUST: Max DD ≤ -1.5%? → -{metrics['max_dd']:.2f}% ", end="")
    must_dd = metrics['max_dd'] <= 1.5
    print("✅" if must_dd else "❌")

    print(f"\n  SHOULD: Trades 150-200? → {metrics['num_trades']} ", end="")
    should_trades = 150 <= metrics['num_trades'] <= 200
    print("✅" if should_trades else "⚠️")

    print(f"  SHOULD: Win Rate ≥ 62%? → {metrics['win_rate']:.1f}% ", end="")
    should_win = metrics['win_rate'] >= 62
    print("✅" if should_win else "⚠️")

    # Recommendation
    if must_return and must_dd:
        if should_trades or should_win:
            recommendation = "✅ ADOPT"
            confidence = "HIGH"
        else:
            recommendation = "⚠️ CONDITIONAL"
            confidence = "MEDIUM"
    else:
        recommendation = "❌ REJECT"
        confidence = "LOW"
        if not (must_return):
            print(f"\n  → Return insufficient: Proceed to Phase 5-C (Ensemble Integration)")
        if not (must_dd):
            print(f"\n  → Drawdown too large: Revise risk management")

    print(f"\nFinal Verdict: {recommendation} (Confidence: {confidence})")

    # Step 10: Save results
    print("\n[8] Saving results...")

    # Convert numpy types to native Python types for JSON serialization
    signal_stats_serializable = {
        k: int(v) if isinstance(v, (np.integer, np.int64, np.int32)) else v
        for k, v in signal_stats.items()
    }

    results = {
        'phase': '5-B',
        'timestamp': pd.Timestamp.now().isoformat(),
        'metrics': {
            'total_return': float(metrics['total_return']),
            'num_trades': int(metrics['num_trades']),
            'win_rate': float(metrics['win_rate']),
            'sharpe': float(metrics['sharpe']),
            'max_dd': float(metrics['max_dd']),
            'final_equity': float(metrics['final_equity']),
            'avg_quality_score': float(metrics['avg_quality_score']),
            'signals_generated': int(metrics['signals_generated']),
            'signals_filtered': int(metrics['signals_filtered']),
            'signals_executed': int(metrics['signals_executed']),
        },
        'signal_stats': signal_stats_serializable,
        'recommendation': recommendation,
        'confidence': confidence,
    }

    results_path = project_root / "STEP12_PHASE5B_RESULTS.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Results saved: {results_path}")

    # Save detailed trades
    trades_df = pd.DataFrame(metrics['trades'])
    trades_path = project_root / "backtest" / "phase5b_trades.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"✓ Trades saved: {trades_path}")

    print("\n" + "=" * 80)
    print("Phase 5-B Backtest Complete")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    metrics = run_phase5b_backtest()
    if metrics:
        print(f"\n✓ Phase 5-B execution successful")
        sys.exit(0)
    else:
        print(f"\n✗ Phase 5-B execution failed")
        sys.exit(1)
