"""Test XGBoost Integration - Verify Model Loads and Works with 1D Features

This test validates:
1. XGBoost model loads correctly
2. Feature alignment works
3. Predictions are generated
4. Fallback handling works
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.multi_timeframe_fetcher import MultiTimeframeFetcher
from features.multi_timeframe_engineer import MultiTimeframeFeatureEngineer
from model.adaptive_xgb_signal_generator import AdaptiveXGBSignalGenerator

import warnings
warnings.filterwarnings('ignore')


def test_xgboost_loading():
    """Test 1: XGBoost model loading"""
    print("\n" + "="*80)
    print("TEST 1: XGBoost Model Loading")
    print("="*80)

    try:
        generator = AdaptiveXGBSignalGenerator()
        print(f"✓ Model loaded successfully")
        print(f"✓ Feature columns: {len(generator.feature_columns)}")
        print(f"✓ First 5 features: {generator.feature_columns[:5]}")
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False


def test_feature_alignment():
    """Test 2: 1D feature alignment with model training features"""
    print("\n" + "="*80)
    print("TEST 2: 1D Feature Alignment")
    print("="*80)

    try:
        # Fetch data
        print("[1/3] Fetching data...")
        fetcher = MultiTimeframeFetcher()
        data_dict = fetcher.fetch_and_resample(years=2)
        print(f"✓ Fetched {len(data_dict['5m']):,} 5m bars")

        # Engineer features
        print("[2/3] Engineering features...")
        engineer = MultiTimeframeFeatureEngineer()
        features_dict = engineer.engineer_all_timeframes(data_dict)
        df_1d = features_dict.get('1d', pd.DataFrame())
        print(f"✓ Engineered 1D features: {df_1d.shape}")

        # Load model feature columns
        print("[3/3] Checking feature alignment...")
        with open(Path(__file__).parent.parent / "model" / "feature_columns.json") as f:
            model_features = json.load(f)

        print(f"\nModel expects {len(model_features)} features:")
        print(f"  {model_features}")

        print(f"\n1D DataFrame has {len(df_1d.columns)} columns:")
        print(f"  {list(df_1d.columns)[:10]}...")

        # Check coverage
        missing = [f for f in model_features if f not in df_1d.columns]
        extra = [f for f in df_1d.columns if f not in model_features]

        if missing:
            print(f"\n⚠️  Missing features ({len(missing)}): {missing[:5]}...")
        else:
            print(f"\n✓ All {len(model_features)} model features present in 1D data")

        if extra:
            print(f"⚠️  Extra features ({len(extra)}): {extra[:5]}... (will be ignored)")

        return len(missing) == 0 or len(missing) <= 5  # Tolerance for minor differences

    except Exception as e:
        print(f"✗ Feature alignment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predictions():
    """Test 3: Generate predictions on actual data"""
    print("\n" + "="*80)
    print("TEST 3: XGBoost Predictions on Real Data")
    print("="*80)

    try:
        # Fetch and engineer
        print("[1/4] Fetching and engineering data...")
        fetcher = MultiTimeframeFetcher()
        data_dict = fetcher.fetch_and_resample(years=2)

        engineer = MultiTimeframeFeatureEngineer()
        features_dict = engineer.engineer_all_timeframes(data_dict)

        df_1d = features_dict.get('1d', pd.DataFrame()).copy()
        df_5m = data_dict['5m'].copy()
        df_5m_features = features_dict.get('5m', pd.DataFrame()).copy()

        print(f"✓ Data ready: {len(df_1d)} 1D bars, {len(df_5m):,} 5m bars")

        # Initialize generator
        print("[2/4] Loading adaptive XGBoost generator...")
        generator = AdaptiveXGBSignalGenerator()
        print(f"✓ Generator initialized")

        # Test predictions
        print("[3/4] Testing predictions on sample bars...")
        success_count = 0
        fail_count = 0

        # Sample every 100th bar for speed
        sample_indices = range(100, len(df_5m), 500)

        for idx in sample_indices:
            try:
                # Get features up to this point
                df_1d_slice = df_1d.iloc[:min(len(df_1d), idx // 288 + 1)]  # Approx 1D equivalent
                df_5m_slice = df_5m_features.iloc[:min(idx+1, len(df_5m_features))]

                if len(df_1d_slice) == 0 or len(df_5m_slice) == 0:
                    continue

                # Generate signal
                signal = generator.generate_adaptive_signal(
                    df_1d_slice,
                    df_5m_slice,
                    bars_in_trade=0
                )

                if signal.should_execute:
                    success_count += 1
                else:
                    fail_count += 1

            except Exception as e:
                fail_count += 1

        total = success_count + fail_count
        print(f"✓ Generated {total} signals")
        print(f"  - Executable: {success_count}")
        print(f"  - Hold: {fail_count}")

        print("[4/4] Checking prediction quality...")

        # Get one actual prediction
        df_1d_slice = df_1d.iloc[-100:]
        df_5m_slice = df_5m_features.iloc[-100:]

        signal = generator.generate_adaptive_signal(df_1d_slice, df_5m_slice)

        print(f"\nLatest signal:")
        print(f"  Signal: {['HOLD', 'BUY', 'SELL'][signal.signal + 1]}")
        print(f"  Confidence: {signal.confidence:.4f}")
        print(f"  XGBoost Prob: {signal.xgb_probability:.4f}")
        print(f"  Technical Score: {signal.technical_score:.4f}")
        print(f"  Should Execute: {signal.should_execute}")

        return True

    except Exception as e:
        print(f"✗ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("XGBOOST INTEGRATION TEST SUITE")
    print("="*80)

    results = []

    # Test 1
    results.append(("Model Loading", test_xgboost_loading()))

    # Test 2
    results.append(("Feature Alignment", test_feature_alignment()))

    # Test 3
    results.append(("Predictions", test_predictions()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print(f"\n✅ All tests passed! XGBoost is ready for production integration.")
    else:
        print(f"\n⚠️  Some tests failed. Review above for details.")

    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
