"""Ensemble Signal Generator for Phase 5-C

Integrates ensemble model predictions with multi-timeframe technical analysis.
Combines ensemble bias signals (1D layer) with 5m precision entry signals.

Ensemble Architecture:
- 1D Layer: 4-model ensemble (XGBoost, LightGBM, CatBoost, Neural Network)
- 5m Layer: Technical indicators (MA, RSI, MACD)
- Integration: 50% ensemble + 50% technical, 0.55 confidence threshold
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


class EnsembleSignalGenerator:
    """Generate trading signals using ensemble models + technical indicators."""

    def __init__(
        self,
        ensemble_path: str = "model/ensemble_models",
        confidence_threshold: float = 0.55
    ):
        """Initialize ensemble signal generator.

        Args:
            ensemble_path: Path to saved ensemble models
            confidence_threshold: Confidence threshold for signal generation
        """
        self.ensemble_path = Path(ensemble_path)
        self.confidence_threshold = confidence_threshold
        self.models = {}
        self.weights = {}
        self.scalers = {}
        self._load_ensemble()

    def _load_ensemble(self):
        """Load pre-trained ensemble models."""
        print(f"Loading ensemble from {self.ensemble_path}...")

        with open(self.ensemble_path / "xgboost_model.pkl", "rb") as f:
            self.models['xgboost'] = pickle.load(f)

        with open(self.ensemble_path / "random_forest_model.pkl", "rb") as f:
            self.models['random_forest'] = pickle.load(f)

        with open(self.ensemble_path / "gradient_boosting_model.pkl", "rb") as f:
            self.models['gradient_boosting'] = pickle.load(f)

        with open(self.ensemble_path / "neural_network_model.pkl", "rb") as f:
            self.models['neural_network'] = pickle.load(f)

        with open(self.ensemble_path / "nn_scaler.pkl", "rb") as f:
            self.scalers['neural_network'] = pickle.load(f)

        with open(self.ensemble_path / "ensemble_weights.json", "r") as f:
            self.weights = json.load(f)

        print(f"✓ Ensemble loaded with weights: {self.weights}")

    def get_ensemble_probability(self, features_1d: np.ndarray) -> np.ndarray:
        """Get ensemble probability predictions.

        Args:
            features_1d: 1D features array (N, 40)

        Returns:
            Ensemble probability predictions (N,)
        """
        predictions = {}

        predictions['xgboost'] = self.models['xgboost'].predict_proba(features_1d)[:, 1]
        predictions['random_forest'] = self.models['random_forest'].predict_proba(features_1d)[:, 1]
        predictions['gradient_boosting'] = self.models['gradient_boosting'].predict_proba(features_1d)[:, 1]

        features_scaled = self.scalers['neural_network'].transform(features_1d)
        predictions['neural_network'] = self.models['neural_network'].predict_proba(features_scaled)[:, 1]

        # Weighted ensemble
        ensemble_proba = np.zeros_like(predictions['xgboost'])
        for model_name, weight in self.weights.items():
            ensemble_proba += predictions[model_name] * float(weight)

        return ensemble_proba

    def get_technical_score(self, df_5m: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate technical indicator scores for 5m timeframe.

        Args:
            df_5m: 5m OHLCV data with technical indicators

        Returns:
            Tuple of (entry_scores, support_scores)
        """
        n = len(df_5m)
        entry_scores = np.zeros(n)
        support_scores = np.zeros(n)

        # MA crossover signal
        if 'ma3' in df_5m.columns and 'ma8' in df_5m.columns:
            ma_crossover = ((df_5m['ma3'] > df_5m['ma8']).astype(int) -
                           (df_5m['ma3'] <= df_5m['ma8']).astype(int))
            entry_scores += ma_crossover * 0.40

        # RSI extremes signal
        if 'rsi14' in df_5m.columns:
            rsi = df_5m['rsi14'].values
            rsi_signal = np.zeros(n)
            rsi_signal[rsi < 30] = 1.0   # Oversold
            rsi_signal[rsi > 70] = -1.0  # Overbought
            entry_scores += rsi_signal * 0.30

        # MACD crossover signal
        if 'macd' in df_5m.columns and 'macd_signal' in df_5m.columns:
            macd_crossover = ((df_5m['macd'] > df_5m['macd_signal']).astype(int) -
                             (df_5m['macd'] <= df_5m['macd_signal']).astype(int))
            entry_scores += macd_crossover * 0.30

        # Normalize entry scores
        entry_scores = entry_scores / 3.0  # [−1, 1] range

        # Support score (buy signal when score > 0.5)
        support_scores = (entry_scores + 1.0) / 2.0  # Convert [−1, 1] to [0, 1]

        return entry_scores, support_scores

    def get_1d_bias(
        self,
        df_1d: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get 1D ensemble bias confirmation.

        Args:
            df_1d: 1D OHLCV data with engineered features (40 features including OHLCV)

        Returns:
            Tuple of (ensemble_proba, ensemble_bias)
        """
        # Extract feature columns (include all except target)
        feature_cols = [col for col in df_1d.columns if col != 'target']
        features = df_1d[feature_cols].values

        # Get ensemble predictions
        ensemble_proba = self.get_ensemble_probability(features)

        # Bias signal: 1.0 for buy, -1.0 for sell, 0.0 for neutral
        ensemble_bias = np.zeros_like(ensemble_proba)
        ensemble_bias[ensemble_proba >= self.confidence_threshold] = 1.0
        ensemble_bias[ensemble_proba < (1 - self.confidence_threshold)] = -1.0

        return ensemble_proba, ensemble_bias

    def generate_signals(
        self,
        df_1d: pd.DataFrame,
        df_5m: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generate integrated trading signals.

        Args:
            df_1d: 1D OHLCV data with engineered features (40 features)
            df_5m: 5m OHLCV data with technical indicators (35 features)

        Returns:
            Tuple of (signals, confidences, metadata)
        """
        # Get 1D ensemble probability (uses all columns including OHLCV)
        feature_cols = [col for col in df_1d.columns if col != 'target']
        features_1d = df_1d[feature_cols].values

        # Get ensemble probability for each 1D bar
        ensemble_proba_1d = self.get_ensemble_probability(features_1d)  # Shape: (num_1d_bars,)

        # Expand 1D ensemble probability to 5m frequency
        # Repeat each daily probability for each 5m bar in that day
        bars_per_day = len(df_5m) // len(df_1d) + 1
        ensemble_proba = np.repeat(ensemble_proba_1d, bars_per_day)[:len(df_5m)]

        # Get 5m technical scores
        entry_scores, support_scores = self.get_technical_score(df_5m)

        # Combine signals: 50% ensemble + 50% technical
        combined_confidence = 0.5 * ensemble_proba + 0.5 * support_scores

        # Generate signals
        signals = np.zeros(len(df_5m))
        signals[combined_confidence >= self.confidence_threshold] = 1.0
        signals[combined_confidence < (1 - self.confidence_threshold)] = -1.0

        metadata = {
            'ensemble_proba_mean': ensemble_proba.mean(),
            'ensemble_proba_std': ensemble_proba.std(),
            'technical_score_mean': support_scores.mean(),
            'signal_distribution': {
                'buy': (signals > 0).sum(),
                'sell': (signals < 0).sum(),
                'neutral': (signals == 0).sum()
            }
        }

        return signals, combined_confidence, metadata

    def get_signal_at_datetime(
        self,
        datetime_val: pd.Timestamp,
        df_1d: pd.DataFrame,
        df_5m: pd.DataFrame,
        signals: np.ndarray,
        confidences: np.ndarray
    ) -> Dict:
        """Get signal information at specific datetime.

        Args:
            datetime_val: Target datetime
            df_1d: 1D data
            df_5m: 5m data
            signals: Generated signals
            confidences: Signal confidences

        Returns:
            Dict with signal details at that datetime
        """
        # Find closest 5m bar
        if datetime_val not in df_5m.index:
            closest_idx = df_5m.index.get_indexer([datetime_val], method='nearest')[0]
        else:
            closest_idx = df_5m.index.get_loc(datetime_val)

        if closest_idx >= len(signals):
            return {'status': 'no_signal', 'reason': 'datetime out of range'}

        signal = signals[closest_idx]
        confidence = confidences[closest_idx]
        price = df_5m.iloc[closest_idx]['Close']

        return {
            'datetime': df_5m.index[closest_idx],
            'price': price,
            'signal': signal,
            'confidence': confidence,
            'threshold': self.confidence_threshold
        }


if __name__ == "__main__":
    # Example usage
    from features.multi_timeframe_fetcher import MultiTimeframeFetcher
    from features.multi_timeframe_engineer import MultiTimeframeFeatureEngineer

    # Fetch data
    fetcher = MultiTimeframeFetcher()
    data_dict = fetcher.fetch_and_resample(years=2)

    # Engineer features
    engineer = MultiTimeframeFeatureEngineer()
    features_dict = engineer.engineer_all_timeframes(data_dict)

    # Load ensemble and generate signals
    try:
        generator = EnsembleSignalGenerator()
        signals, confidences, metadata = generator.generate_signals(
            features_dict['1d'],
            features_dict['5m']
        )
        print(f"Generated {len(signals)} signals")
        print(f"Metadata: {metadata}")
    except FileNotFoundError:
        print("Ensemble models not yet trained. Run ensemble training first.")
