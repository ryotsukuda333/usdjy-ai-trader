"""Script to run hyperparameter tuning with focus on regularization parameters.

Executes GridSearchCV hyperparameter tuning on regularization parameters
(reg_alpha, reg_lambda, min_child_weight) while keeping other parameters fixed
to find optimal regularization strategy.

Tasks: Step 4.1, 4.2, 4.3
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from features.data_fetcher import fetch_usdjpy_data
from features.feature_engineer import engineer_features
from utils.errors import TraderError


def tune_regularization_parameters():
    """Run regularization parameter tuning pipeline.

    Focuses GridSearch on regularization parameters only:
    - reg_alpha: [0.0, 0.5, 1.0]
    - reg_lambda: [0.0, 0.5, 1.0]
    - min_child_weight: [1, 3, 5]

    Total combinations: 3 × 3 × 3 = 27 (much faster than full 3,888)
    """
    try:
        print("\n" + "="*60)
        print("REGULARIZATION PARAMETER TUNING")
        print("="*60)

        # Step 1: Fetch USDJPY data
        print("\n[Step 1] Fetching USDJPY data (3 years)...")
        df_ohlcv = fetch_usdjpy_data(years=3)
        print(f"✓ Fetched {len(df_ohlcv)} candles")

        # Step 2: Engineer features
        print("\n[Step 2] Engineering features from OHLCV...")
        df_features = engineer_features(df_ohlcv)
        print(f"✓ Engineered {len(df_features.columns)} features")

        # Step 3: Run regularization tuning
        print("\n[Step 3] Running regularization parameter tuning...")
        from model.hyperparameter_tuning import tune_regularization_only, save_tuning_results

        tuning_results = tune_regularization_only(df_features, cv_folds=3)

        # Step 4: Save results
        print("\n[Step 4] Saving tuning results...")
        save_tuning_results(tuning_results)

        # Final summary
        print("\n" + "="*60)
        print("✓ REGULARIZATION TUNING COMPLETE")
        print("="*60)
        print(f"\nResults saved to model/ directory:")
        print(f"  • best_hyperparameters.json")
        print(f"  • regularization_tuning_results.csv")
        print(f"  • regularization_improvement.json")

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
    exit_code = tune_regularization_parameters()
    sys.exit(exit_code)
