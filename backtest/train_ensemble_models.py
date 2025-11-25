"""Train Ensemble Models - Phase 5-C

Trains XGBoost, LightGBM, CatBoost, and Neural Network models on 2-year USDJPY data.
Saves trained models and weights for use in production trading.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from features.multi_timeframe_fetcher import MultiTimeframeFetcher
from features.multi_timeframe_engineer import MultiTimeframeFeatureEngineer
from utils.data_labeler import label_data
from model.ensemble_trainer import EnsembleTrainer


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("ENSEMBLE MODEL TRAINING - PHASE 5-C")
    print("=" * 80)

    try:
        # Step 1: Fetch data
        print("\n[Step 1/4] Fetching multi-timeframe data...")
        fetcher = MultiTimeframeFetcher()
        data_dict = fetcher.fetch_and_resample(years=2)
        print(f"✓ Data fetched: 1D={len(data_dict['1d'])}, 5m={len(data_dict['5m'])} bars")

        # Step 2: Engineer features
        print("\n[Step 2/4] Engineering features...")
        engineer = MultiTimeframeFeatureEngineer()
        features_dict = engineer.engineer_all_timeframes(data_dict)
        df_1d = features_dict['1d'].copy()
        print(f"✓ Features engineered: {len(df_1d)} rows × {len(df_1d.columns)} columns")

        # Step 3: Label data
        print("\n[Step 3/4] Labeling training data...")
        df_1d = label_data(df_1d)
        print(f"✓ Data labeled: {len(df_1d)} training samples")
        print(f"  Distribution: {df_1d['target'].value_counts().to_dict()}")

        # Step 4: Train ensemble
        print("\n[Step 4/4] Training ensemble models...")
        trainer = EnsembleTrainer()
        results = trainer.train_ensemble(df_1d)

        # Save models
        model_path = trainer.save_ensemble()
        print(f"✓ Models saved to {model_path}")

        # Save training results
        results_path = Path("backtest") / "ENSEMBLE_TRAINING_RESULTS.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✓ Results saved to {results_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total training time: {results['total_training_time']:.2f}s")
        print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"Test F1 Score: {results['test_metrics']['f1']:.4f}")
        print(f"Test AUC-ROC: {results['test_metrics']['auc']:.4f}")
        print(f"\nEnsemble Weights:")
        for model, weight in results['ensemble_weights'].items():
            print(f"  {model:20s}: {weight:.4f}")

        print("\n" + "=" * 80)
        print("✓ ENSEMBLE TRAINING COMPLETE")
        print("=" * 80)

        return results

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    sys.exit(0 if results else 1)
