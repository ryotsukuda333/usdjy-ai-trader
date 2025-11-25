"""
Phase 5-D Stage 1: Hyperparameter Optimization for XGBoost (Optimized Version)

Performs random grid search over XGBoost hyperparameters to improve Phase 5-B (+293.83%).
Simplified version with direct CSV loading for reliability.

Optimization Space:
├─ n_estimators: [50, 100, 150, 200]
├─ max_depth: [5, 6, 7, 8, 9]
├─ learning_rate: [0.01, 0.03, 0.05, 0.07, 0.1]
├─ subsample: [0.6, 0.7, 0.8, 0.9]
├─ colsample_bytree: [0.6, 0.7, 0.8, 0.9]
└─ gamma: [0, 0.1, 0.5, 1.0]

Total: 500 random samples (from 3,200 combinations)
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))

from features.multi_timeframe_engineer import MultiTimeframeFeatureEngineer
from utils.data_labeler import label_data


class FastHyperparameterOptimizer:
    """Optimized hyperparameter optimizer with timeout and progress tracking."""

    def __init__(self, random_state: int = 42, num_samples: int = 500, max_runtime: int = 3600):
        """
        Initialize optimizer.

        Args:
            random_state: Random seed for reproducibility
            num_samples: Number of random parameter combinations to test
            max_runtime: Maximum runtime in seconds (default 1 hour)
        """
        self.random_state = random_state
        self.num_samples = num_samples
        self.max_runtime = max_runtime
        self.results = []
        self.best_params = None
        self.best_score = 0
        self.start_time = None

        np.random.seed(random_state)

    def generate_parameter_space(self) -> List[Dict]:
        """Generate random parameter combinations from defined space."""
        param_space = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [5, 6, 7, 8, 9],
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.5, 1.0]
        }

        # Generate random combinations
        params_list = []
        for _ in range(self.num_samples):
            params = {}
            for param_name, values in param_space.items():
                params[param_name] = np.random.choice(values)
            params_list.append(params)

        return params_list

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        params: Dict
    ) -> Dict:
        """
        Train XGBoost with given parameters and evaluate on validation set.

        Args:
            X_train, X_val: Feature arrays
            y_train, y_val: Label arrays
            params: XGBoost hyperparameters

        Returns:
            Dictionary with metrics: accuracy, f1, auc
        """
        try:
            model = xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
                **params
            )

            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]

            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred, zero_division=0),
                'auc': roc_auc_score(y_val, y_proba),
            }

            return metrics
        except Exception as e:
            return {'accuracy': 0, 'f1': 0, 'auc': 0, 'error': str(e)}

    def optimize(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[Dict, List[Dict]]:
        """
        Run hyperparameter optimization with timeout.

        Args:
            X_train, X_val: Feature arrays
            y_train, y_val: Label arrays

        Returns:
            Tuple of (best_params, all_results)
        """
        print("=" * 80)
        print("PHASE 5-D STAGE 1: HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)
        print(f"\nGenerating {self.num_samples} random parameter combinations...")

        param_combinations = self.generate_parameter_space()

        print(f"Starting grid search ({len(param_combinations)} combinations)...")
        print(f"Max runtime: {self.max_runtime}s\n")

        self.start_time = time.time()

        for i, params in enumerate(param_combinations):
            # Check timeout
            elapsed = time.time() - self.start_time
            if elapsed > self.max_runtime:
                print(f"\n⚠️  TIMEOUT: Reached max runtime {self.max_runtime}s")
                print(f"Completed {i}/{self.num_samples} combinations")
                break

            if (i + 1) % 50 == 0:
                elapsed = time.time() - self.start_time
                avg_time = elapsed / (i + 1)
                remaining = (self.num_samples - i - 1) * avg_time
                print(f"  [{i+1:3d}/{self.num_samples}] {elapsed:6.1f}s | ETA: {remaining:6.1f}s | Best AUC: {self.best_score:.4f}")

            metrics = self.train_and_evaluate(X_train, X_val, y_train, y_val, params)

            result = {
                'index': i,
                'params': params,
                'metrics': metrics,
                'score': metrics.get('auc', 0)  # Primary metric: AUC
            }

            self.results.append(result)

            # Update best
            if result['score'] > self.best_score:
                self.best_score = result['score']
                self.best_params = params
                print(f"    ✓ New best: AUC = {self.best_score:.4f}")

        total_time = time.time() - self.start_time
        print(f"\n✓ Optimization completed in {total_time:.1f}s")
        print(f"  Total combinations tested: {len(self.results)}/{self.num_samples}")

        return self.best_params, self.results

    def print_results_summary(self):
        """Print summary of optimization results."""
        if not self.results:
            print("No results available")
            return

        # Sort by AUC score
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)

        print("\n" + "=" * 80)
        print("TOP 10 PARAMETER COMBINATIONS")
        print("=" * 80)
        print(f"\n{'Rank':<5} {'AUC':<8} {'Accuracy':<10} {'F1':<8}")
        print("-" * 80)

        for rank, result in enumerate(sorted_results[:10], 1):
            metrics = result['metrics']
            print(f"{rank:<5} {metrics['auc']:<8.4f} {metrics['accuracy']:<10.4f} {metrics['f1']:<8.4f}")

            params = result['params']
            print(f"  Parameters:")
            for param_name, value in params.items():
                print(f"    {param_name:20s}: {value}")
            print()

        # Best parameters summary
        print("\n" + "=" * 80)
        print("BEST PARAMETERS")
        print("=" * 80)
        print(f"\nBest AUC Score: {self.best_score:.4f}\n")
        for param_name, value in self.best_params.items():
            print(f"  {param_name:20s}: {value}")

    def save_results(self, output_dir: str = "backtest"):
        """Save optimization results to JSON."""
        output_path = Path(output_dir) / "phase5d_hyperparameter_results.json"
        output_path.parent.mkdir(exist_ok=True, parents=True)

        # Prepare results for JSON serialization
        results_to_save = {
            'num_samples': len(self.results),
            'best_score': float(self.best_score),
            'best_params': {k: float(v) if isinstance(v, (np.integer, np.floating)) else int(v)
                           for k, v in self.best_params.items()},
            'top_10_results': [
                {
                    'rank': i + 1,
                    'score': float(result['score']),
                    'params': {k: float(v) if isinstance(v, (np.integer, np.floating)) else int(v)
                             for k, v in result['params'].items()},
                    'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                              for k, v in result['metrics'].items()}
                }
                for i, result in enumerate(sorted(self.results, key=lambda x: x['score'], reverse=True)[:10])
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")
        return str(output_path)


def main():
    """Main optimization pipeline."""
    print("=" * 80)
    print("PHASE 5-D STAGE 1: HYPERPARAMETER OPTIMIZATION (V2)")
    print("=" * 80)

    try:
        # Step 1: Load existing features from cache
        print("\n[1/4] Loading pre-engineered features...")

        # Try to load from cached features
        feature_cache = Path("backtest/features_cache.pkl")
        if feature_cache.exists():
            import pickle
            with open(feature_cache, 'rb') as f:
                df_1d = pickle.load(f)
            print(f"✓ Features loaded from cache: {len(df_1d)} rows × {len(df_1d.columns)} columns")
        else:
            # Engineer features fresh
            print("  Fetching and engineering features...")
            from features.multi_timeframe_fetcher import MultiTimeframeFetcher
            fetcher = MultiTimeframeFetcher()
            data_dict = fetcher.fetch_and_resample(years=2)

            engineer = MultiTimeframeFeatureEngineer()
            features_dict = engineer.engineer_all_timeframes(data_dict)
            df_1d = features_dict['1d'].copy()

            # Cache for future use
            import pickle
            with open(feature_cache, 'wb') as f:
                pickle.dump(df_1d, f)
            print(f"✓ Features engineered: {len(df_1d)} rows × {len(df_1d.columns)} columns")

        # Step 2: Label data
        print("\n[2/4] Labeling data...")
        df_1d = label_data(df_1d)
        print(f"✓ {len(df_1d)} training samples labeled")

        # Step 3: Prepare data splits
        print("\n[3/4] Preparing data splits...")
        feature_cols = [c for c in df_1d.columns if c != 'target']
        X = df_1d[feature_cols].values
        y = df_1d['target'].values

        # Time-series aware split
        split_idx = int(len(X) * 0.8)
        X_trainval, X_test = X[:split_idx], X[split_idx:]
        y_trainval, y_test = y[:split_idx], y[split_idx:]

        split_idx2 = int(len(X_trainval) * 0.8)
        X_train, X_val = X_trainval[:split_idx2], X_trainval[split_idx2:]
        y_train, y_val = y_trainval[:split_idx2], y_trainval[split_idx2:]

        print(f"✓ Data splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        print(f"  Target distribution (train): {np.bincount(y_train)}")

        # Step 4: Optimize hyperparameters
        print("\n[4/4] Running hyperparameter optimization...")
        optimizer = FastHyperparameterOptimizer(num_samples=500, max_runtime=3600)
        best_params, all_results = optimizer.optimize(X_train, X_val, y_train, y_val)

        # Print results
        optimizer.print_results_summary()

        # Save results
        optimizer.save_results()

        print("\n" + "=" * 80)
        print("✓ OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"\nNext: Stage 2 - Feature Engineering Optimization")

        return 0

    except Exception as e:
        print(f"\n✗ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
