"""Hyperparameter tuning module for XGBoost model optimization.

Implements GridSearchCV to find optimal hyperparameters for XGBoost
classifier using time-series aware cross-validation.

Tasks: 1.3 (Hyperparameter tuning with GridSearch)
"""

import time
import json
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from utils.errors import ModelError


def tune_hyperparameters(df_features: pd.DataFrame, cv_folds: int = 3) -> Dict:
    """Tune XGBoost hyperparameters using GridSearchCV with time-series split.

    Performs grid search over key XGBoost hyperparameters:
    - n_estimators: [100, 200, 300, 400]
    - max_depth: [3, 4, 5, 6, 7]
    - learning_rate: [0.01, 0.05, 0.1]
    - subsample: [0.6, 0.8, 1.0]
    - colsample_bytree: [0.6, 0.8, 1.0]

    Uses TimeSeriesSplit for time-series aware cross-validation (no shuffling).

    Args:
        df_features: Feature DataFrame with target column for training
        cv_folds: Number of cross-validation folds (default: 3)

    Returns:
        Dict with keys:
            - 'best_params': Best hyperparameters found
            - 'best_score': Best cross-validation score
            - 'best_model': Trained model with best parameters
            - 'grid_results': DataFrame with all grid search results
            - 'elapsed_time': Time taken for grid search in seconds

    Raises:
        ModelError: If tuning fails or insufficient data
    """
    start_time = time.time()

    # Validate input
    if df_features is None or df_features.empty:
        raise ModelError(
            error_code="INVALID_INPUT",
            user_message="Feature DataFrame is empty or None",
            technical_message="df_features is empty"
        )

    if 'target' not in df_features.columns:
        raise ModelError(
            error_code="MISSING_TARGET",
            user_message="Target column 'target' not found in features",
            technical_message=f"Available columns: {list(df_features.columns)}"
        )

    try:
        # Separate features and target
        X = df_features.drop('target', axis=1)
        y = df_features['target']

        print(f"\n{'='*60}")
        print("HYPERPARAMETER TUNING - GRIDSEARCH")
        print(f"{'='*60}")
        print(f"Features: {len(X.columns)} | Samples: {len(X)} | Target distribution: {y.value_counts().to_dict()}")

        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8],
            'colsample_bytree': [0.6, 0.8],
        }

        # Calculate total combinations
        total_combinations = 1
        for key, values in param_grid.items():
            total_combinations *= len(values)

        print(f"\n🔍 Grid Search Configuration:")
        print(f"  Parameter combinations: {total_combinations}")
        print(f"  Cross-validation folds: {cv_folds} (TimeSeriesSplit - no shuffle)")
        print(f"  Total model trainings: {total_combinations * cv_folds}")
        print(f"  Estimated time: {total_combinations * cv_folds * 2 / 60:.1f} minutes (est.)")

        # Initialize base model
        base_model = XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            verbosity=0,
            eval_metric='logloss'
        )

        # Time-series aware cross-validation (no shuffle)
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        # GridSearchCV
        print(f"\n⏱  Starting grid search...")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring='f1',  # Use F1 score as primary metric
            n_jobs=-1,  # Use all available cores
            verbose=1
        )

        grid_search.fit(X, y)

        elapsed_time = time.time() - start_time

        # Extract results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_

        print(f"\n✅ Grid search completed in {elapsed_time:.1f} seconds")
        print(f"\n🏆 Best Parameters Found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

        print(f"\n📊 Best Cross-Validation Score (F1): {best_score:.4f}")

        # Evaluate on full training set
        y_pred = best_model.predict(X)
        y_pred_proba = best_model.predict_proba(X)[:, 1]
        train_accuracy = accuracy_score(y, y_pred)
        train_f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        print(f"\n📈 Training Set Performance (Best Model):")
        print(f"  Accuracy: {train_accuracy:.4f}")
        print(f"  F1 Score: {train_f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        # Compare with baseline (original parameters)
        baseline_params = {
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }

        improvement = {
            'f1_improvement': f1_score(y, best_model.predict(X)) - 0.6269,  # Original F1
            'accuracy_improvement': train_accuracy - 0.4898,  # Original accuracy
            'params_changed': {k: (baseline_params[k], best_params[k])
                              for k in best_params if best_params[k] != baseline_params[k]}
        }

        print(f"\n📈 Improvement vs Baseline:")
        print(f"  F1 improvement: {improvement['f1_improvement']:+.4f}")
        print(f"  Accuracy improvement: {improvement['accuracy_improvement']:+.4f}")
        print(f"  Changed parameters: {len(improvement['params_changed'])}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_model': best_model,
            'grid_results': pd.DataFrame(grid_search.cv_results_),
            'elapsed_time': elapsed_time,
            'improvement': improvement,
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
        }

    except Exception as e:
        raise ModelError(
            error_code="TUNING_FAILED",
            user_message="Hyperparameter tuning failed",
            technical_message=f"Error during tuning: {str(e)}"
        )


def save_tuning_results(tuning_results: Dict) -> None:
    """Save hyperparameter tuning results to JSON and CSV files.

    Args:
        tuning_results: Dictionary returned from tune_hyperparameters()

    Raises:
        ModelError: If saving fails
    """
    try:
        model_dir = Path(__file__).parent
        model_dir.mkdir(exist_ok=True)

        # Save best parameters as JSON
        best_params_path = model_dir / 'best_hyperparameters.json'
        with open(best_params_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            params_dict = {k: int(v) if isinstance(v, (np.integer, np.floating)) else float(v) if isinstance(v, np.floating) else v
                          for k, v in tuning_results['best_params'].items()}
            json.dump({
                'best_params': params_dict,
                'best_cv_score': float(tuning_results['best_score']),
                'train_accuracy': float(tuning_results['train_accuracy']),
                'train_f1': float(tuning_results['train_f1']),
            }, f, indent=2)
        print(f"✓ Best parameters saved to {best_params_path}")

        # Save full grid results as CSV
        grid_results_path = model_dir / 'grid_search_results.csv'
        grid_df = tuning_results['grid_results'].copy()
        # Select only relevant columns
        result_cols = [col for col in grid_df.columns if 'param_' in col or 'mean_test' in col or 'std_test' in col]
        grid_df[result_cols].to_csv(grid_results_path, index=False)
        print(f"✓ Grid search results saved to {grid_results_path}")

        # Save improvement summary
        improvement_path = model_dir / 'tuning_improvement.json'
        improvement = tuning_results['improvement']
        with open(improvement_path, 'w') as f:
            json.dump({
                'f1_improvement': float(improvement['f1_improvement']),
                'accuracy_improvement': float(improvement['accuracy_improvement']),
                'params_changed': str(improvement['params_changed']),
            }, f, indent=2)
        print(f"✓ Improvement summary saved to {improvement_path}")

    except Exception as e:
        raise ModelError(
            error_code="SAVE_FAILED",
            user_message="Failed to save tuning results",
            technical_message=f"Error saving results: {str(e)}"
        )
