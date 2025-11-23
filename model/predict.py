"""Predictor module for model loading and prediction generation.

Loads trained XGBoost model and feature column specifications, validates
input features, and generates predictions with probabilities.

Tasks: 5.1, 5.2
"""

import json
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb

from utils.errors import ModelError, FeatureEngineeringError


def load_model() -> xgb.Booster:
    """Load trained XGBoost model from JSON file.

    Loads the serialized model from model/xgb_model.json created during
    training. Requirement 4.1: Load model for predictions.

    Returns:
        xgb.Booster: Loaded model ready for prediction

    Raises:
        ModelError: If model file not found or loading fails
    """
    model_dir = Path(__file__).parent
    model_path = model_dir / 'xgb_model.json'

    if not model_path.exists():
        raise ModelError(
            error_code="MODEL_NOT_FOUND",
            user_message=f"Model file not found: {model_path}",
            technical_message=f"Expected model at {model_path} does not exist. "
                            f"Run training first."
        )

    try:
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        print(f"✓ Model loaded from {model_path}")
        return booster
    except Exception as e:
        raise ModelError(
            error_code="MODEL_LOAD_ERROR",
            user_message="Failed to load model from JSON",
            technical_message=f"Error loading model: {str(e)}"
        )


def load_feature_columns() -> list:
    """Load list of training feature columns from JSON.

    Loads the feature column list saved during training in model/feature_columns.json.
    This is used to validate and align prediction input features.
    Requirement 4.4: Load and validate feature columns.

    Returns:
        list: List of feature column names in training order

    Raises:
        ModelError: If feature columns file not found or loading fails
    """
    model_dir = Path(__file__).parent
    columns_path = model_dir / 'feature_columns.json'

    if not columns_path.exists():
        raise ModelError(
            error_code="COLUMNS_NOT_FOUND",
            user_message=f"Feature columns file not found: {columns_path}",
            technical_message=f"Expected feature_columns.json at {columns_path}. "
                            f"Run training first."
        )

    try:
        with open(columns_path, 'r') as f:
            feature_columns = json.load(f)

        if not isinstance(feature_columns, list):
            raise ValueError("feature_columns.json should contain a list of column names")

        print(f"✓ Loaded {len(feature_columns)} feature columns from {columns_path}")
        return feature_columns

    except Exception as e:
        raise ModelError(
            error_code="COLUMNS_LOAD_ERROR",
            user_message="Failed to load feature columns from JSON",
            technical_message=f"Error loading columns: {str(e)}"
        )


def predict(df_features: pd.DataFrame, model: xgb.Booster = None,
           feature_columns: list = None) -> np.ndarray:
    """Generate predictions from feature DataFrame.

    Validates input features match training features, aligns feature order,
    and generates binary predictions and probabilities.
    Requirement 4.1-4.5: Generate predictions with validation.

    Args:
        df_features: DataFrame with engineered features for prediction
        model: Loaded XGBoost booster model (optional, loads if not provided)
        feature_columns: Training feature column list (optional, loads if not provided)

    Returns:
        np.ndarray: Binary predictions (0 or 1) for each row

    Raises:
        ModelError: If model loading fails
        FeatureEngineeringError: If features missing or cannot be aligned
    """
    if df_features is None or df_features.empty:
        raise FeatureEngineeringError(
            error_code="INVALID_INPUT",
            user_message="Feature DataFrame is empty or None",
            technical_message="df_features is empty"
        )

    # Load model and feature columns if not provided
    if model is None:
        model = load_model()

    if feature_columns is None:
        feature_columns = load_feature_columns()

    try:
        # Validate required columns exist
        missing_columns = [col for col in feature_columns if col not in df_features.columns]
        if missing_columns:
            raise FeatureEngineeringError(
                error_code="MISSING_COLUMNS",
                user_message=f"Missing required feature columns: {missing_columns}",
                technical_message=f"Training columns: {feature_columns}, "
                                f"Prediction columns: {list(df_features.columns)}"
            )

        # Auto-align features to match training order
        X_aligned = df_features[feature_columns]

        print(f"✓ Features aligned for {len(X_aligned)} predictions")

        # Generate predictions using DMatrix
        dmatrix = xgb.DMatrix(X_aligned.values, feature_names=feature_columns)
        y_pred_proba = model.predict(dmatrix)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        print(f"✓ Predictions generated: {np.sum(y_pred)} buy signals out of {len(y_pred)}")

        return y_pred

    except FeatureEngineeringError:
        raise
    except Exception as e:
        raise ModelError(
            error_code="PREDICTION_FAILED",
            user_message="Prediction generation failed",
            technical_message=f"Error during prediction: {str(e)}"
        )
