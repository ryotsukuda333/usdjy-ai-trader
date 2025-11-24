"""Multi-Timeframe Signal Generator with XGBoost Probability Weighting

Implements hierarchical confluence-based signal generation with XGBoost probability
weighting for improved signal discrimination and trade execution.

Architecture:
- Per-timeframe technical indicator scoring (technical signal)
- XGBoost probability weighting (ML signal strength)
- Confluence alignment calculation (multi-TF synchronization)
- Combined confidence = 0.50*xgb_prob + 0.30*seasonality + 0.10*alignment + 0.10*technical
- Execution threshold = 0.55 (vs 0.70 alignment-only)

Expected Impact:
- Executable signals: 30-50 (vs 0.2% with technical-only)
- Trade count: 3-8 trades
- Return: +2-3% (matching Phase 5-A baseline)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import xgboost as xgb
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')


@dataclass
class SignalResult:
    """Result of multi-timeframe signal generation with XGBoost integration."""
    signal: int  # 1 (BUY), -1 (SELL), 0 (HOLD)
    confidence: float  # 0.0-1.0
    xgb_probability: float  # 0.0-1.0 from XGBoost
    technical_score: float  # 0.0-1.0 from technical indicators
    alignment_score: float  # 0.0-1.0 from confluence
    seasonality_score: float  # 0.0-1.0 from time-based factors
    reason: str  # Explanation of signal
    should_execute: bool  # Whether to execute this signal


class MultiTimeframeSignalGeneratorXGB:
    """Generate multi-timeframe signals with XGBoost probability weighting."""

    # Timeframe importance weights
    TIMEFRAME_WEIGHTS = {
        '1d': 0.40,   # Primary strategic bias
        '4h': 0.25,   # Important confirmation
        '1h': 0.20,   # Secondary confirmation
        '15m': 0.10,  # Entry zone
        '5m': 0.05    # Entry precision
    }

    # Confidence weighting components
    CONFIDENCE_WEIGHTS = {
        'xgb_probability': 0.50,  # XGBoost probability (primary)
        'seasonality': 0.30,      # Day of week / time of day effects
        'alignment': 0.10,        # Multi-TF alignment score
        'technical': 0.10         # Technical indicator strength
    }

    # Execution thresholds
    EXECUTE_THRESHOLD = 0.55  # vs 0.70 for alignment-only

    def __init__(self, model_path: Optional[Path] = None,
                 feature_columns_path: Optional[Path] = None):
        """Initialize multi-timeframe signal generator with XGBoost.

        Args:
            model_path: Path to xgb_model.json (default: model/xgb_model.json)
            feature_columns_path: Path to feature_columns.json (default: model/feature_columns.json)
        """
        self.project_root = Path(__file__).parent.parent

        # Model paths
        if model_path is None:
            model_path = self.project_root / "model" / "xgb_model.json"
        if feature_columns_path is None:
            feature_columns_path = self.project_root / "model" / "feature_columns.json"

        # Load XGBoost model and feature columns
        self.model = self._load_model(model_path)
        self.feature_columns = self._load_feature_columns(feature_columns_path)

        print(f"✓ XGBoost model loaded with {len(self.feature_columns)} features")

    def _load_model(self, model_path: Path) -> xgb.Booster:
        """Load XGBoost model from JSON."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            return booster
        except Exception as e:
            raise RuntimeError(f"Failed to load XGBoost model: {e}")

    def _load_feature_columns(self, columns_path: Path) -> list:
        """Load feature column names from JSON."""
        if not columns_path.exists():
            raise FileNotFoundError(f"Feature columns file not found: {columns_path}")

        try:
            with open(columns_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load feature columns: {e}")

    def get_xgb_probability(self, features: pd.DataFrame) -> np.ndarray:
        """Generate XGBoost probabilities for a feature set.

        Args:
            features: DataFrame with engineered features

        Returns:
            np.ndarray: Probability array (values 0.0-1.0)
        """
        try:
            # Validate and align features
            missing_cols = [col for col in self.feature_columns if col not in features.columns]
            if missing_cols:
                print(f"⚠️ Warning: Missing columns for XGBoost: {missing_cols}")
                # Fill missing columns with default values
                for col in missing_cols:
                    if col in features.columns:
                        features[col] = 0.0

            # Align features to training order
            X_aligned = features[self.feature_columns].fillna(0.0)

            # Generate predictions
            dmatrix = xgb.DMatrix(X_aligned.values, feature_names=self.feature_columns)
            y_proba = self.model.predict(dmatrix)

            return y_proba

        except Exception as e:
            print(f"✗ XGBoost prediction error: {e}")
            return np.full(len(features), 0.5)  # Default to neutral

    def generate_signal_per_timeframe(self,
                                    features_dict: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """Generate per-timeframe technical indicator signals.

        Args:
            features_dict: Dict with timeframe keys ('1d', '4h', etc)

        Returns:
            Dict: Signal per timeframe (1=BUY, -1=SELL, 0=HOLD)
        """
        signals = {}

        for tf, df_features in features_dict.items():
            if df_features.empty or len(df_features) == 0:
                signals[tf] = 0
                continue

            # Get last row for current signal
            last_row = df_features.iloc[-1]

            # Technical scoring: MA crossover, RSI, MACD
            score = 0.0

            # Moving average analysis
            ma_cols = [col for col in df_features.columns if col.startswith('ma') and not col.endswith('_slope') and '_' not in col]
            if len(ma_cols) >= 2:
                price = last_row.get('Close', 0)
                # Extract MA periods (e.g., 'ma5' -> 5)
                try:
                    ma_periods = sorted([int(col[2:]) for col in ma_cols if col[2:].isdigit()])
                    if len(ma_periods) >= 2:
                        short_ma = last_row.get(f'ma{ma_periods[0]}', price)
                        long_ma = last_row.get(f'ma{ma_periods[-1]}', price)
                        if short_ma > long_ma:
                            score += 0.3
                        elif short_ma < long_ma:
                            score -= 0.3
                except (ValueError, IndexError):
                    pass  # Skip if MA columns can't be parsed

            # RSI analysis
            if 'rsi14' in df_features.columns:
                rsi = last_row.get('rsi14', 50)
                if rsi < 30:
                    score += 0.3  # Oversold = bullish
                elif rsi > 70:
                    score -= 0.3  # Overbought = bearish
                elif 40 < rsi < 60:
                    score += 0.1  # Neutral lean bullish

            # MACD analysis
            if 'macd' in df_features.columns and 'macd_signal' in df_features.columns:
                macd = last_row.get('macd', 0)
                macd_signal = last_row.get('macd_signal', 0)
                if macd > macd_signal:
                    score += 0.2
                else:
                    score -= 0.2

            # Convert to signal
            if score > 0.2:
                signals[tf] = 1  # BUY
            elif score < -0.2:
                signals[tf] = -1  # SELL
            else:
                signals[tf] = 0  # HOLD

        return signals

    def calculate_timeframe_alignment(self, signals: Dict[str, int]) -> float:
        """Calculate alignment score across timeframes.

        Args:
            signals: Per-timeframe signals

        Returns:
            float: Alignment score (0.0-1.0)
        """
        if not signals:
            return 0.0

        # Calculate weighted signal
        weighted_sum = 0.0
        total_weight = 0.0

        for tf, signal in signals.items():
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 0.0)
            weighted_sum += signal * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        # Normalized weighted signal (range: -1 to 1)
        normalized_signal = weighted_sum / total_weight

        # Convert to alignment score (0 to 1)
        # High absolute value = high alignment (all agree)
        # Low absolute value = low alignment (disagreement)
        alignment = abs(normalized_signal)

        return alignment

    def calculate_seasonality_score(self, timestamp: datetime) -> float:
        """Calculate seasonality/time-based factors.

        Args:
            timestamp: Current bar timestamp

        Returns:
            float: Seasonality score (0.0-1.0)
        """
        try:
            # Day of week effects (0-1)
            dow = timestamp.weekday()  # 0=Monday, 6=Sunday
            day_score = {
                0: 0.55,  # Monday - often strong
                1: 0.50,  # Tuesday
                2: 0.52,  # Wednesday
                3: 0.50,  # Thursday
                4: 0.48,  # Friday - profit-taking
                5: 0.45,  # Saturday (low volume)
                6: 0.47   # Sunday (low volume)
            }.get(dow, 0.50)

            # Hour of day effects (trading hours)
            hour = timestamp.hour
            if hour < 8 or hour > 22:
                hour_score = 0.40  # Low volume hours
            elif 8 <= hour < 11:
                hour_score = 0.58  # Asian morning strong
            elif 11 <= hour < 14:
                hour_score = 0.52  # Mid-day
            elif 14 <= hour < 17:
                hour_score = 0.55  # Afternoon strong
            else:
                hour_score = 0.50

            # Combined seasonality
            seasonality = 0.6 * day_score + 0.4 * hour_score

            return seasonality

        except Exception as e:
            print(f"⚠️ Seasonality calculation error: {e}")
            return 0.50  # Neutral default

    def generate_confluence_signal(self,
                                 features_dict: Dict[str, pd.DataFrame],
                                 current_idx: int,
                                 timestamp: Optional[datetime] = None) -> SignalResult:
        """Generate confluence-based signal with XGBoost probability weighting.

        Args:
            features_dict: Dict with features per timeframe
            current_idx: Current bar index (for DataFrame row selection)
            timestamp: Current timestamp (for seasonality calculation)

        Returns:
            SignalResult: Complete signal with confidence and execution decision
        """
        # Get per-timeframe signals
        signals = self.generate_signal_per_timeframe(features_dict)

        # Calculate technical strength (0-1)
        signal_strength = sum(1 for s in signals.values() if s == 1) / len(signals) if signals else 0.5
        technical_score = signal_strength

        # Get XGBoost probability (requires aligned features)
        try:
            # Use 5m features as primary (most recent)
            if '5m' in features_dict and len(features_dict['5m']) > 0:
                xgb_features = features_dict['5m'].iloc[[-1]] if len(features_dict['5m']) > 0 else None
                if xgb_features is not None and not xgb_features.empty:
                    xgb_proba = self.get_xgb_probability(xgb_features)
                    xgb_prob = float(xgb_proba[0]) if len(xgb_proba) > 0 else 0.5
                else:
                    xgb_prob = 0.5
            else:
                xgb_prob = 0.5
        except Exception as e:
            print(f"⚠️ XGBoost probability error: {e}")
            xgb_prob = 0.5

        # Calculate alignment
        alignment = self.calculate_timeframe_alignment(signals)

        # Calculate seasonality
        if timestamp is None:
            seasonality = 0.50
        else:
            seasonality = self.calculate_seasonality_score(timestamp)

        # Determine primary signal (1D or 4H)
        primary_signal = signals.get('1d', 0) or signals.get('4h', 0)

        if primary_signal == 0:
            # No primary signal = HOLD
            confidence = 0.0
            combined_confidence = 0.0
            should_execute = False
            reason = "No primary signal (1D/4H BUY needed)"

        else:
            # Calculate combined confidence
            combined_confidence = (
                self.CONFIDENCE_WEIGHTS['xgb_probability'] * xgb_prob +
                self.CONFIDENCE_WEIGHTS['seasonality'] * seasonality +
                self.CONFIDENCE_WEIGHTS['alignment'] * alignment +
                self.CONFIDENCE_WEIGHTS['technical'] * technical_score
            )

            confidence = combined_confidence
            should_execute = combined_confidence >= self.EXECUTE_THRESHOLD

            if primary_signal == 1:
                if should_execute:
                    reason = f"BUY: XGB={xgb_prob:.2f}, Align={alignment:.2f}, Confidence={combined_confidence:.2f}"
                else:
                    reason = f"BUY (Low Conf): XGB={xgb_prob:.2f}, Align={alignment:.2f}, Confidence={combined_confidence:.2f}"
            else:
                if should_execute:
                    reason = f"SELL: XGB={xgb_prob:.2f}, Align={alignment:.2f}, Confidence={combined_confidence:.2f}"
                else:
                    reason = f"SELL (Low Conf): XGB={xgb_prob:.2f}, Align={alignment:.2f}, Confidence={combined_confidence:.2f}"

        return SignalResult(
            signal=primary_signal,
            confidence=confidence,
            xgb_probability=xgb_prob,
            technical_score=technical_score,
            alignment_score=alignment,
            seasonality_score=seasonality,
            reason=reason,
            should_execute=should_execute
        )

    def generate_multi_timeframe_signals(self,
                                        data_dict: Dict[str, pd.DataFrame],
                                        features_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate signals for all timeframes using XGBoost probability weighting.

        Args:
            data_dict: Dict with OHLCV per timeframe
            features_dict: Dict with features per timeframe

        Returns:
            Dict: DataFrames with signals per timeframe plus confluence signals
        """
        result = {}

        # For each timeframe, generate signals
        for tf in ['1d', '4h', '1h', '15m', '5m']:
            if tf not in features_dict:
                continue

            features = features_dict[tf]
            if features.empty:
                continue

            signals_list = []

            # Generate signal for each row
            for idx in range(len(features)):
                # Get features up to current index
                features_slice = {
                    tf_key: df.iloc[:idx+1] if tf_key == tf else df
                    for tf_key, df in features_dict.items()
                }

                # Get timestamp
                if tf in data_dict and len(data_dict[tf]) > idx:
                    timestamp = data_dict[tf].index[idx] if hasattr(data_dict[tf].index, '__getitem__') else None
                else:
                    timestamp = None

                # Generate signal
                signal_result = self.generate_confluence_signal(
                    features_slice,
                    idx,
                    timestamp
                )

                signals_list.append({
                    'index': idx,
                    'signal': signal_result.signal,
                    'confidence': signal_result.confidence,
                    'xgb_probability': signal_result.xgb_probability,
                    'alignment': signal_result.alignment_score,
                    'seasonality': signal_result.seasonality_score,
                    'should_execute': signal_result.should_execute,
                    'reason': signal_result.reason
                })

            # Convert to DataFrame
            df_signals = pd.DataFrame(signals_list)
            result[tf] = df_signals

        return result


def generate_multi_timeframe_signals_xgb(
    data_dict: Dict[str, pd.DataFrame],
    features_dict: Dict[str, pd.DataFrame],
    model_path: Optional[Path] = None,
    feature_columns_path: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """Convenience function to generate multi-timeframe signals with XGBoost.

    Args:
        data_dict: Dict with OHLCV per timeframe
        features_dict: Dict with features per timeframe
        model_path: Optional path to XGBoost model
        feature_columns_path: Optional path to feature columns

    Returns:
        Dict: Signal DataFrames per timeframe
    """
    generator = MultiTimeframeSignalGeneratorXGB(model_path, feature_columns_path)
    return generator.generate_multi_timeframe_signals(data_dict, features_dict)


if __name__ == "__main__":
    # Example usage
    print("Multi-Timeframe Signal Generator with XGBoost Integration")
    print("Module ready for use in backtest pipeline")
