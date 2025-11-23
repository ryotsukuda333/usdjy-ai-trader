"""Script to run hyperparameter tuning and save results.

Executes GridSearchCV hyperparameter tuning on the feature engineering pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from features.data_fetcher import fetch_usdjpy_data
from features.feature_engineer import engineer_features
from model.hyperparameter_tuning import tune_hyperparameters, save_tuning_results
from utils.errors import TraderError


def main():
    """Run hyperparameter tuning pipeline."""
    try:
        print("\n" + "="*60)
        print("USDJPY AI TRADER - HYPERPARAMETER TUNING")
        print("="*60)

        # Step 1: Fetch USDJPY data
        print("\n[Step 1] Fetching USDJPY data (3 years)...")
        df_ohlcv = fetch_usdjpy_data(years=3)
        print(f"✓ Fetched {len(df_ohlcv)} candles")

        # Step 2: Engineer features
        print("\n[Step 2] Engineering features from OHLCV...")
        df_features = engineer_features(df_ohlcv)
        print(f"✓ Engineered {len(df_features.columns)} features")

        # Step 3: Run hyperparameter tuning
        print("\n[Step 3] Running hyperparameter tuning...")
        tuning_results = tune_hyperparameters(df_features, cv_folds=3)

        # Step 4: Save results
        print("\n[Step 4] Saving tuning results...")
        save_tuning_results(tuning_results)

        # Final summary
        print("\n" + "="*60)
        print("✓ HYPERPARAMETER TUNING COMPLETE")
        print("="*60)
        print(f"\nResults saved to model/ directory:")
        print(f"  • best_hyperparameters.json")
        print(f"  • grid_search_results.csv")
        print(f"  • tuning_improvement.json")

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
