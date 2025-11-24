"""
Phase 5-B Simplified: Signal Quality Improvement with Simple Filtering

より単純なアプローチで信号品質を改善:
- Phase 5-Aの結果(+2%)をベースに
- XGBoost確信度ベースのフィルタリング
- 勝率の高いシグナルのみを実行
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import time
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.data_fetcher import fetch_usdjpy_data
from features.feature_engineer import engineer_features
from model.step12_hybrid_strategy import HybridTradingStrategy


def run_phase5b_simple():
    """簡略版 Phase 5-B バックテストの実行"""

    project_root = Path(__file__).parent.parent

    print("=" * 80)
    print("PHASE 5-B SIMPLIFIED: Signal Quality Improvement - Simple Approach")
    print("=" * 80)

    # Step 1: Data Fetching
    print("\n[1] Fetching data...")
    df_ohlcv = fetch_usdjpy_data(years=3)
    print(f"✓ Fetched {len(df_ohlcv)} candles")

    # Step 2: Feature Engineering
    print("[2] Engineering features...")
    df_features = engineer_features(df_ohlcv)
    print(f"✓ Engineered {len(df_features.columns)} features, {len(df_features)} rows")

    # Step 3: Load feature columns
    feature_cols_path = project_root / "model" / "feature_columns.json"
    if feature_cols_path.exists():
        with open(feature_cols_path) as f:
            feature_cols = json.load(f)
    else:
        feature_cols = [col for col in df_features.columns
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]

    # Step 4: Initialize Base Strategy (without Phase 5-B)
    print("[3] Initializing base hybrid strategy...")
    xgb_model_path = project_root / "model" / "xgb_model.json"

    base_strategy = HybridTradingStrategy(
        xgb_model_path=str(xgb_model_path) if xgb_model_path.exists() else None,
        use_seasonality=True
    )
    print("✓ Base strategy initialized")

    # Step 5: Run BASE backtest (Phase 5-A equivalent)
    print("\n[4] Running BASE backtest (without Phase 5-B filtering)...")

    # Use aligned indices - normalize timezone-aware indices
    print(f"  Debug: df_ohlcv index type: {type(df_ohlcv.index[0])}")
    print(f"  Debug: df_features index type: {type(df_features.index[0])}")

    # Convert both indices to timezone-naive for alignment
    df_ohlcv_indexed = df_ohlcv.copy()
    df_features_indexed = df_features.copy()

    if df_ohlcv_indexed.index.tz is not None:
        df_ohlcv_indexed.index = df_ohlcv_indexed.index.tz_localize(None)
    if df_features_indexed.index.tz is not None:
        df_features_indexed.index = df_features_indexed.index.tz_localize(None)

    # Find common dates
    common_dates = df_ohlcv_indexed.index.intersection(df_features_indexed.index)
    print(f"  Common dates found: {len(common_dates)}")

    df_ohlcv_aligned = df_ohlcv_indexed.loc[common_dates].copy()
    df_features_aligned = df_features_indexed.loc[common_dates].copy()

    # Generate base predictions (使用するaligned features)
    # Note: predictions are for the aligned common dates only
    base_predictions = base_strategy.generate_predictions(
        df=df_features_aligned.reset_index(drop=True),  # Reset index to avoid mismatch
        feature_cols=feature_cols,
        xgb_threshold=0.45
    )
    # Re-index predictions to match aligned OHLCV
    base_predictions.index = df_ohlcv_aligned.index

    # Run base backtest
    base_return, base_metrics = base_strategy.backtest_hybrid_strategy(
        df=df_ohlcv_aligned,
        predictions=base_predictions,
        initial_capital=100000
    )

    print(f"✓ Base backtest completed")
    print(f"  ├─ Return: +{base_metrics['total_return']:.2f}%")
    print(f"  ├─ Trades: {base_metrics['num_trades']}")
    print(f"  └─ Win Rate: {base_metrics['win_rate']:.1f}%")

    # Step 6: Apply Phase 5-B Simple Filtering
    print("\n[5] Applying Phase 5-B simple filtering...")

    # Quality threshold: XGBoost confidence ≥ 0.55 (from Grid Search optimal)
    quality_threshold = 0.55

    filtered_predictions = base_predictions.copy()

    # Add quality-based filtering
    # Only execute signals with high confidence
    high_confidence_mask = filtered_predictions['xgb_prob'] >= quality_threshold
    filtered_predictions['phase5b_filter'] = high_confidence_mask

    # Count filtering stats
    total_signals = (base_predictions['hybrid_signal'] != -1).sum()
    filtered_count = (~high_confidence_mask & (base_predictions['hybrid_signal'] != -1)).sum()
    executed_count = (high_confidence_mask & (base_predictions['hybrid_signal'] == 1)).sum()

    print(f"  ├─ Total signals: {total_signals}")
    print(f"  ├─ Filtered out (XGBoost < 0.55): {filtered_count}")
    print(f"  └─ Will execute (XGBoost ≥ 0.55): {executed_count}")

    # Apply filter: modify signals of low-confidence predictions
    modified_predictions = base_predictions.copy()
    low_conf_indices = ~high_confidence_mask & (base_predictions['hybrid_signal'] != -1)
    modified_predictions.loc[low_conf_indices, 'hybrid_signal'] = -1
    modified_predictions.loc[low_conf_indices, 'confidence'] = 0

    # Step 7: Run FILTERED backtest (Phase 5-B)
    print("\n[6] Running Phase 5-B backtest (with quality filtering)...")

    filtered_return, filtered_metrics = base_strategy.backtest_hybrid_strategy(
        df=df_ohlcv_aligned,
        predictions=modified_predictions,
        initial_capital=100000
    )

    print(f"✓ Phase 5-B backtest completed")
    print(f"  ├─ Return: +{filtered_metrics['total_return']:.2f}%")
    print(f"  ├─ Trades: {filtered_metrics['num_trades']}")
    print(f"  └─ Win Rate: {filtered_metrics['win_rate']:.1f}%")

    # Step 8: Results Comparison
    print("\n" + "=" * 80)
    print("PHASE 5-B RESULTS: BASE vs FILTERED")
    print("=" * 80)

    comparison_data = {
        'Metric': ['Total Return', 'Win Rate (%)', 'Trades', 'Max DD (%)', 'Avg Trade (%)'],
        'Base (5-A)': [
            f"+{base_metrics['total_return']:.2f}%",
            f"{base_metrics['win_rate']:.1f}%",
            f"{base_metrics['num_trades']}",
            f"-{base_metrics.get('max_dd', 0):.2f}%",
            f"+{base_metrics['total_return'] / max(base_metrics['num_trades'], 1):.2f}%"
        ],
        'Filtered (5-B)': [
            f"+{filtered_metrics['total_return']:.2f}%",
            f"{filtered_metrics['win_rate']:.1f}%",
            f"{filtered_metrics['num_trades']}",
            f"-{filtered_metrics.get('max_dd', 0):.2f}%",
            f"+{filtered_metrics['total_return'] / max(filtered_metrics['num_trades'], 1):.2f}%"
        ],
        'Target (5-C)': [
            '+35-45%',
            '62%+',
            '150-200',
            '≤-1.5%',
            '+0.25-0.30%'
        ]
    }

    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + df_comparison.to_string(index=False))

    # Step 9: Analysis
    print("\n" + "=" * 80)
    print("PHASE 5-B ANALYSIS")
    print("=" * 80)

    improvement_return = filtered_metrics['total_return'] - base_metrics['total_return']
    improvement_winrate = filtered_metrics['win_rate'] - base_metrics['win_rate']
    trade_reduction = base_metrics['num_trades'] - filtered_metrics['num_trades']

    print(f"\nPhase 5-B Impact:")
    print(f"  Return change: {improvement_return:+.2f}% ({filtered_metrics['total_return'] / base_metrics['total_return']:.2f}x)")
    print(f"  Win rate change: {improvement_winrate:+.1f}pp")
    print(f"  Trades reduced: {trade_reduction} (from {base_metrics['num_trades']} to {filtered_metrics['num_trades']})")

    if improvement_return > 0:
        print(f"  ✓ Quality filtering IMPROVED performance")
    elif improvement_return < 0:
        print(f"  ✗ Quality filtering DECREASED performance (need Phase 5-C)")
    else:
        print(f"  ⚠️ Quality filtering had NO IMPACT")

    # Step 10: Phase 6 Adoption Decision
    print("\n" + "=" * 80)
    print("PHASE 6 ADOPTION JUDGMENT")
    print("=" * 80)

    print(f"\nAdoption Criteria Check (Phase 5-B Result):")
    print(f"  MUST: Total Return > +65%? → {filtered_metrics['total_return']:.2f}% ", end="")
    must_return = filtered_metrics['total_return'] > 65
    print("✅" if must_return else "❌")

    print(f"  MUST: Max DD ≤ -1.5%? → -{filtered_metrics.get('max_dd', 0):.2f}% ", end="")
    must_dd = filtered_metrics.get('max_dd', 0) <= 1.5
    print("✅" if must_dd else "❌")

    print(f"\n  SHOULD: Trades 150-200? → {filtered_metrics['num_trades']} ", end="")
    should_trades = 150 <= filtered_metrics['num_trades'] <= 200
    print("✅" if should_trades else "⚠️")

    print(f"  SHOULD: Win Rate ≥ 62%? → {filtered_metrics['win_rate']:.1f}% ", end="")
    should_win = filtered_metrics['win_rate'] >= 62
    print("✅" if should_win else "⚠️")

    # Recommendation
    if must_return and must_dd:
        if should_trades or should_win:
            recommendation = "✅ ADOPT"
        else:
            recommendation = "⚠️ CONDITIONAL"
    else:
        recommendation = "❌ REJECT (→ Phase 5-C Required)"

    print(f"\nPhase 5-B Verdict: {recommendation}")

    if not (must_return and must_dd):
        print(f"\nNext Step: Phase 5-C (Ensemble Integration)")
        print(f"  Expected: +35-45% (with multi-model ensemble)")

    # Step 11: Save results
    print("\n[7] Saving results...")

    results = {
        'phase': '5-B-Simple',
        'timestamp': pd.Timestamp.now().isoformat(),
        'approach': 'Simple XGBoost confidence filtering (XGBoost ≥ 0.55)',
        'base_metrics': {
            'total_return': float(base_metrics['total_return']),
            'num_trades': int(base_metrics['num_trades']),
            'win_rate': float(base_metrics['win_rate']),
        },
        'filtered_metrics': {
            'total_return': float(filtered_metrics['total_return']),
            'num_trades': int(filtered_metrics['num_trades']),
            'win_rate': float(filtered_metrics['win_rate']),
        },
        'filtering_stats': {
            'total_signals': int(total_signals),
            'filtered_count': int(filtered_count),
            'executed_count': int(executed_count),
        },
        'recommendation': recommendation,
        'next_phase': 'Phase 5-C (Ensemble)' if not (must_return) else 'Phase 6 (Adoption)',
    }

    results_path = project_root / "STEP12_PHASE5B_SIMPLE_RESULTS.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Results saved: {results_path}")

    print("\n" + "=" * 80)
    print("Phase 5-B Simple Backtest Complete")
    print("=" * 80)

    return filtered_metrics, base_metrics


if __name__ == "__main__":
    filtered_metrics, base_metrics = run_phase5b_simple()
    print(f"\n✓ Execution successful")
    sys.exit(0)
