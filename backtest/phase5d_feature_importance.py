"""
Phase 5-D Stage 2: Feature Importance Analysis and Engineering Optimization

Analyzes feature importance using best XGBoost model from Stage 1.
Tests feature deletion impact and identifies low-value features for removal.
Also tests new feature engineering approaches.

Process:
1. Load best model parameters from Stage 1
2. Train model and extract feature importance
3. Test iterative feature deletion (backward elimination)
4. Analyze importance distribution and identify low-impact features
5. Generate engineering recommendations
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path.cwd()))

from features.multi_timeframe_engineer import MultiTimeframeFeatureEngineer
from utils.data_labeler import label_data


class FeatureImportanceAnalyzer:
    """Analyze feature importance and test feature engineering optimizations."""

    def __init__(self, best_params: Dict):
        """
        Initialize analyzer with best hyperparameters from Stage 1.

        Args:
            best_params: Best parameters from hyperparameter optimization
        """
        self.best_params = best_params
        self.feature_names = None
        self.importance_df = None
        self.deletion_results = []

    def train_and_extract_importance(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[pd.DataFrame, float]:
        """
        Train model and extract feature importance.

        Args:
            X_train, X_val: Feature arrays
            y_train, y_val: Label arrays
            feature_names: Names of features

        Returns:
            Tuple of (importance_df, val_auc)
        """
        # Train model with best parameters
        model = xgb.XGBClassifier(random_state=42, n_jobs=-1, verbosity=0, **self.best_params)
        model.fit(X_train, y_train)

        # Get validation AUC
        y_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_proba)

        # Extract importance
        importance = model.get_booster().get_score(importance_type='weight')

        # Convert to dataframe
        importance_df = pd.DataFrame(
            list(importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)

        # Add feature names (map f0, f1, ... to actual names)
        importance_df['Feature_Name'] = importance_df['Feature'].apply(
            lambda x: feature_names[int(x[1:])] if x.startswith('f') else x
        )

        return importance_df, val_auc

    def test_feature_deletion(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        num_features_to_test: int = 10
    ) -> List[Dict]:
        """
        Test impact of removing low-importance features.

        Args:
            X_train, X_val: Feature arrays
            y_train, y_val: Label arrays
            feature_names: Names of features
            num_features_to_test: Number of lowest-importance features to test

        Returns:
            List of deletion test results
        """
        results = []

        # Start with baseline (all features)
        model_base = xgb.XGBClassifier(random_state=42, n_jobs=-1, verbosity=0, **self.best_params)
        model_base.fit(X_train, y_train)
        y_proba_base = model_base.predict_proba(X_val)[:, 1]
        auc_base = roc_auc_score(y_val, y_proba_base)

        results.append({
            'deleted_features': [],
            'num_features': X_train.shape[1],
            'auc': auc_base,
            'auc_change': 0.0
        })

        # Get importance ranking
        importance = model_base.get_booster().get_score(importance_type='weight')
        importance_dict = {int(k[1:]): v for k, v in importance.items() if k.startswith('f')}

        # Sort by importance (ascending = lowest first)
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1])
        low_importance_indices = [idx for idx, _ in sorted_features[:num_features_to_test]]

        # Test deleting features one by one
        features_to_delete = []
        for delete_idx in low_importance_indices:
            features_to_delete.append(delete_idx)

            # Create mask for features to keep
            keep_mask = np.array([i not in features_to_delete for i in range(X_train.shape[1])])
            X_train_reduced = X_train[:, keep_mask]
            X_val_reduced = X_val[:, keep_mask]

            # Train model with reduced features
            model_reduced = xgb.XGBClassifier(random_state=42, n_jobs=-1, verbosity=0, **self.best_params)
            model_reduced.fit(X_train_reduced, y_train)

            y_proba_reduced = model_reduced.predict_proba(X_val_reduced)[:, 1]
            auc_reduced = roc_auc_score(y_val, y_proba_reduced)

            deleted_names = [feature_names[i] for i in features_to_delete]

            results.append({
                'deleted_features': deleted_names,
                'num_features': X_train_reduced.shape[1],
                'auc': auc_reduced,
                'auc_change': auc_reduced - auc_base
            })

        return results

    def print_importance_analysis(self):
        """Print feature importance analysis."""
        if self.importance_df is None or self.importance_df.empty:
            print("No importance data available")
            return

        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE RANKING (Top 20)")
        print("=" * 80)
        print(f"\n{'Rank':<5} {'Feature':<25} {'Importance':<12} {'%':<8}")
        print("-" * 80)

        total_importance = self.importance_df['Importance'].sum()

        for rank, row in self.importance_df.head(20).iterrows():
            pct = (row['Importance'] / total_importance * 100) if total_importance > 0 else 0
            print(f"{rank+1:<5} {row['Feature_Name']:<25} {row['Importance']:<12.4f} {pct:<8.2f}%")

        # Bottom 10
        print("\n" + "=" * 80)
        print("LOWEST IMPORTANCE FEATURES (Bottom 10)")
        print("=" * 80)
        print(f"\n{'Rank':<5} {'Feature':<25} {'Importance':<12} {'%':<8}")
        print("-" * 80)

        for rank, row in self.importance_df.tail(10).iterrows():
            pct = (row['Importance'] / total_importance * 100) if total_importance > 0 else 0
            print(f"{rank+1:<5} {row['Feature_Name']:<25} {row['Importance']:<12.4f} {pct:<8.2f}%")

    def print_deletion_analysis(self):
        """Print feature deletion test results."""
        if not self.deletion_results:
            print("No deletion test results")
            return

        print("\n" + "=" * 80)
        print("FEATURE DELETION TEST RESULTS")
        print("=" * 80)
        print(f"\n{'Features':<50} {'Num':<5} {'AUC':<8} {'Change':<8}")
        print("-" * 80)

        for result in self.deletion_results:
            deleted_str = ', '.join(result['deleted_features'][:2])
            if len(result['deleted_features']) > 2:
                deleted_str += f", +{len(result['deleted_features'])-2} more"

            auc_str = f"{result['auc']:.4f}"
            change_str = f"{result['auc_change']:+.4f}"

            print(f"{deleted_str:<50} {result['num_features']:<5} {auc_str:<8} {change_str:<8}")

    def save_results(self, output_dir: str = "backtest"):
        """Save analysis results to JSON."""
        output_path = Path(output_dir) / "phase5d_feature_importance_results.json"
        output_path.parent.mkdir(exist_ok=True, parents=True)

        results_to_save = {
            'feature_importance': [
                {
                    'rank': i + 1,
                    'feature_name': row['Feature_Name'],
                    'importance': float(row['Importance']),
                    'percentage': float(row['Importance'] / self.importance_df['Importance'].sum() * 100)
                }
                for i, row in self.importance_df.iterrows()
            ],
            'deletion_tests': self.deletion_results,
            'recommendations': self._generate_recommendations()
        }

        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")
        return str(output_path)

    def _generate_recommendations(self) -> List[str]:
        """Generate feature engineering recommendations."""
        if not self.importance_df.empty:
            # Bottom 5 features
            bottom_5 = self.importance_df.tail(5)['Feature_Name'].tolist()

            recommendations = [
                f"Consider removing low-importance features: {', '.join(bottom_5[:3])}",
                "Test combinations of feature deletion for further improvement",
                "Consider adding new derived features (volatility, momentum combinations)",
                "Test lag features for temporal dependencies",
                "Analyze correlation between low-importance features and target"
            ]
            return recommendations
        return []


def main():
    """Main feature importance analysis pipeline."""
    print("=" * 80)
    print("PHASE 5-D STAGE 2: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    try:
        # Load best parameters from Stage 1
        print("\n[1/4] Loading Stage 1 results...")
        results_file = Path("backtest/phase5d_hyperparameter_results.json")

        if not results_file.exists():
            print("❌ Stage 1 results not found")
            return 1

        with open(results_file) as f:
            stage1_results = json.load(f)

        best_params = stage1_results['best_params']
        # Convert n_estimators to int (required by XGBoost)
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        print(f"✓ Best parameters loaded: AUC = {stage1_results['best_score']:.4f}")

        # Load features from cache
        print("\n[2/4] Loading features...")
        feature_cache = Path("backtest/features_cache.pkl")

        if not feature_cache.exists():
            print("❌ Feature cache not found. Running Stage 1 first.")
            return 1

        import pickle
        with open(feature_cache, 'rb') as f:
            df_1d = pickle.load(f)

        print(f"✓ Features loaded: {len(df_1d)} rows × {len(df_1d.columns)} columns")

        # Label data
        print("\n[3/4] Preparing data...")
        df_1d = label_data(df_1d)
        feature_cols = [c for c in df_1d.columns if c != 'target']
        X = df_1d[feature_cols].values
        y = df_1d['target'].values

        # Time-series split
        split_idx = int(len(X) * 0.8)
        X_trainval, X_test = X[:split_idx], X[split_idx:]
        y_trainval, y_test = y[:split_idx], y[split_idx:]

        split_idx2 = int(len(X_trainval) * 0.8)
        X_train, X_val = X_trainval[:split_idx2], X_trainval[split_idx2:]
        y_train, y_val = y_trainval[:split_idx2], y_trainval[split_idx2:]

        print(f"✓ Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        # Analyze feature importance
        print("\n[4/4] Analyzing feature importance...")
        analyzer = FeatureImportanceAnalyzer(best_params)

        importance_df, val_auc = analyzer.train_and_extract_importance(
            X_train, X_val, y_train, y_val, feature_cols
        )
        analyzer.importance_df = importance_df
        print(f"✓ Feature importance extracted (Validation AUC: {val_auc:.4f})")

        # Test feature deletion
        print("\nTesting feature deletion impact...")
        deletion_results = analyzer.test_feature_deletion(
            X_train, X_val, y_train, y_val, feature_cols, num_features_to_test=10
        )
        analyzer.deletion_results = deletion_results
        print(f"✓ Deletion tests completed: {len(deletion_results)} scenarios tested")

        # Print results
        analyzer.print_importance_analysis()
        analyzer.print_deletion_analysis()

        # Save results
        analyzer.save_results()

        print("\n" + "=" * 80)
        print("✓ FEATURE IMPORTANCE ANALYSIS COMPLETE")
        print("=" * 80)
        print("\nNext: Stage 3 - Trading Conditions (TP/SL) Optimization")

        return 0

    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
