"""Model Trainer module for XGBoost model training and serialization.

Implements time-series aware train/test split, XGBoost training with fixed hyperparameters,
evaluation metrics calculation, and model serialization to JSON format.

Tasks: 4.1, 4.2, 4.3
"""

import time
import json
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from utils.errors import ModelError


def train_model(df_features: pd.DataFrame, test_mode: bool = False) -> XGBClassifier:
    """Train XGBoost model using time-series aware train/test split.

    Implements time-series appropriate 8:2 split without shuffling to preserve
    temporal order. Uses fixed hyperparameters as specified in requirements.
    Requirement 3: Model training with train/test split and evaluation.

    Args:
        df_features: Feature DataFrame with target column
                    Must contain 'target' column for labels
                    Must have sufficient rows for 8:2 split
        test_mode: If True, uses fewer estimators (10) for faster testing (default: False)

    Returns:
        XGBClassifier: Trained model with evaluation metrics

    Raises:
        ModelError: If training fails, features missing, or target absent

    Process:
        1. Validate features DataFrame and required columns
        2. Separate features and target
        3. Perform 8:2 train/test split with shuffle=False (time-series)
        4. Initialize XGBClassifier with fixed hyperparameters
        5. Train on training set
        6. Evaluate on test set (accuracy, F1, confusion matrix)
        7. Calculate and visualize feature importance
        8. Log training metrics and timestamps
    """
    start_time = time.time()

    # Validate features DataFrame
    if df_features is None or df_features.empty:
        raise ModelError(
            error_code="INVALID_INPUT",
            user_message="Feature DataFrame is empty or None",
            technical_message="df_features is empty"
        )

    # Validate target column exists
    if 'target' not in df_features.columns:
        raise ModelError(
            error_code="MISSING_TARGET",
            user_message="Target column 'target' not found in features",
            technical_message=f"Available columns: {list(df_features.columns)}"
        )

    # Validate sufficient data for 8:2 split
    min_rows = 100  # Reasonable minimum for training
    if len(df_features) < min_rows:
        raise ModelError(
            error_code="INSUFFICIENT_DATA",
            user_message=f"Feature DataFrame has only {len(df_features)} rows (minimum: {min_rows})",
            technical_message=f"Not enough data for proper train/test split"
        )

    try:
        # Separate features and target
        X = df_features.drop('target', axis=1)
        y = df_features['target']

        # Validate feature columns
        required_features = [
            'ma5', 'ma20', 'ma50',
            'ma5_slope', 'ma20_slope', 'ma50_slope',
            'rsi14',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'pct_change',
            'lag1', 'lag2', 'lag3', 'lag4', 'lag5',
            'mon', 'tue', 'wed', 'thu', 'fri',
        ]

        missing_features = [col for col in required_features if col not in X.columns]
        if missing_features:
            raise ModelError(
                error_code="MISSING_FEATURES",
                user_message=f"Missing required feature columns: {missing_features}",
                technical_message=f"Expected: {required_features}, Got: {list(X.columns)}"
            )

        # Time-series aware 8:2 train/test split (NO SHUFFLING)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        print(f"✓ Data split (8:2, no shuffle): train={len(X_train)}, test={len(X_test)}")

        # Initialize XGBClassifier with optimized hyperparameters from GridSearch
        # Optimized hyperparameters (Step 2.2):
        # - n_estimators: 100 (reduced from 300 for efficiency)
        # - max_depth: 4 (reduced from 5 to prevent overfitting)
        # - learning_rate: 0.01 (reduced from 0.05 for better convergence)
        # - subsample: 0.6 (reduced from 0.8 to improve robustness)
        # - colsample_bytree: 0.8 (optimal from tuning)
        n_estimators = 10 if test_mode else 100  # Changed from 300 to 100
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=4,  # Changed from 5 to 4
            learning_rate=0.01,  # Changed from 0.05 to 0.01
            subsample=0.6,  # Changed from 0.8 to 0.6
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42,
            verbosity=0
        )

        # Train model
        print("⏱ Model training in progress...")
        model.fit(
            X_train, y_train,
            verbose=0
        )

        # Evaluate model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        elapsed_time = time.time() - start_time

        # Log metrics
        print(f"✓ Model training complete: {elapsed_time:.2f}s")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Confusion Matrix:\n{cm}")

        # Performance warning
        if elapsed_time > 300:
            print(f"⚠ Warning: Training took {elapsed_time:.2f}s (limit: 300s)")

        # Save feature importance visualization
        _plot_feature_importance(model, X_train.columns)

        return model

    except ModelError:
        raise
    except Exception as e:
        raise ModelError(
            error_code="TRAINING_FAILED",
            user_message="Model training failed",
            technical_message=f"Error during training: {str(e)}"
        )


def _plot_feature_importance(model: XGBClassifier, feature_names) -> None:
    """Save feature importance bar chart to model/feature_importance.png.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature column names
    """
    try:
        # Create model directory if it doesn't exist
        model_dir = Path(__file__).parent
        model_dir.mkdir(exist_ok=True)

        # Get feature importance
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:20]  # Top 20 features

        # Create bar chart
        plt.figure(figsize=(12, 6))
        plt.title('Feature Importance (Top 20)')
        plt.bar(range(20), importance[indices])
        plt.xticks(range(20), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Importance')
        plt.tight_layout()

        # Save to PNG
        plot_path = model_dir / 'feature_importance.png'
        plt.savefig(plot_path, dpi=100)
        plt.close()

        print(f"✓ Feature importance chart saved to {plot_path}")

    except Exception as e:
        print(f"⚠ Warning: Failed to save feature importance chart: {str(e)}")


def save_model(model: XGBClassifier, df_features: pd.DataFrame) -> None:
    """Save trained model and feature columns to JSON files.

    Serializes XGBoost model to JSON and saves the list of training feature
    columns for use by the Predictor module.
    Requirement 3.9, 3.10: Model serialization and column persistence.

    Args:
        model: Trained XGBoost model
        df_features: Original feature DataFrame (used to extract column names)

    Raises:
        ModelError: If serialization fails
    """
    try:
        model_dir = Path(__file__).parent
        model_dir.mkdir(exist_ok=True)

        # Save model to JSON
        model_path = model_dir / 'xgb_model.json'
        model.get_booster().save_model(str(model_path))
        print(f"✓ Model saved to {model_path}")

        # Save feature columns list (all features excluding target)
        # Convert tuple columns to simple strings (for multi-index DataFrames)
        feature_columns = []
        for col in df_features.columns:
            if col != 'target':
                if isinstance(col, tuple):
                    # For tuples, take first element (e.g., ('Close', 'USDJPY=X') -> 'Close')
                    feature_columns.append(col[0] if len(col) > 0 else str(col))
                else:
                    # For regular strings, keep as-is
                    feature_columns.append(str(col))

        columns_path = model_dir / 'feature_columns.json'
        with open(columns_path, 'w') as f:
            json.dump(feature_columns, f)

        print(f"✓ Feature columns saved: {len(feature_columns)} features to {columns_path}")

    except Exception as e:
        raise ModelError(
            error_code="SERIALIZATION_FAILED",
            user_message="Failed to save model",
            technical_message=f"Error saving model: {str(e)}"
        )
