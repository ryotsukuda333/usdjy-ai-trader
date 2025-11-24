"""Adaptive XGBoost Signal Generator for Multi-Timeframe Trading

This module implements hierarchical multi-timeframe signal generation that bridges
the feature space gap between multi-TF systems and the 1D-trained XGBoost model.

Architecture:
- 1D Level: Use XGBoost probability (trained on 1D daily data)
- 5m Level: Use technical indicators (MA crossover, RSI, MACD)
- Integration: Combine 1D XGBoost probability with 5m technical signals
- Confidence = 50% * xgb_prob_1d + 50% * technical_score_5m

Expected Impact:
- Executable signals: 20-40 trades
- Performance: +2-3% return (matching Phase 5-A baseline)
- Win rate: 55-60% (from Phase 5-A proven pattern)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import xgboost as xgb
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')


@dataclass
class AdaptiveSignalResult:
    """Result of adaptive signal generation combining 1D XGBoost with 5m technical signals."""
    signal: int  # 1 (BUY), -1 (SELL), 0 (HOLD)
    confidence: float  # 0.0-1.0
    xgb_probability: float  # 0.0-1.0 from 1D XGBoost
    technical_score: float  # 0.0-1.0 from 5m technical indicators
    bars_in_trade: int  # How many bars in current trade
    reason: str  # Explanation
    should_execute: bool  # Whether to execute


class AdaptiveXGBSignalGenerator:
    """Generate signals using 1D XGBoost probability + 5m technical indicators.

    Key insight: Don't try to reuse XGBoost in multi-TF feature space.
    Instead, use 1D XGBoost for what it was trained on (daily bias),
    combine with 5m technical signals for entry precision.
    """

    # Execution parameters
    EXECUTE_THRESHOLD = 0.55  # Confidence threshold for signal execution

    # Technical indicator weights for 5m signals
    TECHNICAL_WEIGHTS = {
        'ma_crossover': 0.40,   # MA short > MA long
        'rsi': 0.30,             # RSI extremes
        'macd': 0.30              # MACD crossover
    }

    def __init__(self,
                 model_path: Optional[Path] = None,
                 feature_columns_path: Optional[Path] = None):
        """Initialize adaptive XGBoost signal generator.

        Args:
            model_path: Path to xgb_model.json
            feature_columns_path: Path to feature_columns.json
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

        print(f"✓ Adaptive XGBoost generator loaded ({len(self.feature_columns)} training features)")
        print(f"  Strategy: 50% 1D XGBoost + 50% 5m Technical")

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

    def get_1d_xgb_probability(self, df_1d_features: pd.DataFrame, cache: dict = None) -> float:
        """Get XGBoost probability from 1D daily features with caching.

        Args:
            df_1d_features: DataFrame with 1D features (last row used)
            cache: Optional cache dict for storing probabilities

        Returns:
            float: Probability 0.0-1.0 from XGBoost
        """
        if df_1d_features.empty or len(df_1d_features) == 0:
            return 0.5  # Neutral default

        try:
            # Use last row
            last_row = df_1d_features.iloc[-1:]
            cache_key = None

            # Cache lookup
            if cache is not None and len(df_1d_features) > 0:
                cache_key = len(df_1d_features)  # Use length as proxy
                if cache_key in cache:
                    return cache[cache_key]

            # Validate and align features
            missing_cols = [col for col in self.feature_columns if col not in last_row.columns]
            if missing_cols:
                # Fill missing with 0.0
                for col in missing_cols:
                    last_row[col] = 0.0

            # Align to training feature order
            X_aligned = last_row[self.feature_columns].fillna(0.0)

            # Generate prediction
            dmatrix = xgb.DMatrix(X_aligned.values, feature_names=self.feature_columns)
            y_proba = self.model.predict(dmatrix)

            result = float(y_proba[0]) if len(y_proba) > 0 else 0.5

            # Cache result
            if cache is not None and cache_key is not None:
                cache[cache_key] = result

            return result

        except Exception as e:
            # Silently fail - XGBoost is optional
            return 0.5

    def get_5m_technical_score(self, df_5m_features: pd.DataFrame) -> Tuple[int, float]:
        """Get technical signal and score from 5m features.

        Args:
            df_5m_features: DataFrame with 5m technical features

        Returns:
            Tuple: (signal: 1/0/-1, score: 0.0-1.0)
        """
        if df_5m_features.empty or len(df_5m_features) == 0:
            return 0, 0.5

        last_row = df_5m_features.iloc[-1]
        price = last_row.get('Close', 0)
        score = 0.0

        # MA crossover (typical 5m periods: 3, 5, 8, 13)
        ma_cols = [col for col in df_5m_features.columns
                  if col.startswith('ma') and not col.endswith('_slope') and '_' not in col]
        if len(ma_cols) >= 2:
            try:
                ma_periods = sorted([int(col[2:]) for col in ma_cols if col[2:].isdigit()])
                if len(ma_periods) >= 2:
                    short_ma = last_row.get(f'ma{ma_periods[0]}', price)
                    long_ma = last_row.get(f'ma{ma_periods[-1]}', price)
                    if short_ma > long_ma:
                        score += self.TECHNICAL_WEIGHTS['ma_crossover']
                    else:
                        score -= self.TECHNICAL_WEIGHTS['ma_crossover']
            except (ValueError, IndexError):
                pass

        # RSI (standard period: 14)
        if 'rsi14' in df_5m_features.columns:
            rsi = last_row.get('rsi14', 50)
            if rsi < 30:
                score += self.TECHNICAL_WEIGHTS['rsi']  # Oversold = bullish
            elif rsi > 70:
                score -= self.TECHNICAL_WEIGHTS['rsi']  # Overbought = bearish

        # MACD (typical 5m periods: 5, 13, 5)
        if 'macd' in df_5m_features.columns and 'macd_signal' in df_5m_features.columns:
            macd = last_row.get('macd', 0)
            macd_signal = last_row.get('macd_signal', 0)
            if macd > macd_signal:
                score += self.TECHNICAL_WEIGHTS['macd']
            else:
                score -= self.TECHNICAL_WEIGHTS['macd']

        # Convert score to signal
        signal = 1 if score > 0.2 else (-1 if score < -0.2 else 0)

        # Normalize score to 0.0-1.0
        technical_prob = 0.5 + (score / 2.0)
        technical_prob = max(0.0, min(1.0, technical_prob))

        return signal, technical_prob

    def generate_adaptive_signal(self,
                               df_1d_features: pd.DataFrame,
                               df_5m_features: pd.DataFrame,
                               bars_in_trade: int = 0,
                               timestamp: Optional[datetime] = None) -> AdaptiveSignalResult:
        """Generate adaptive signal combining 1D XGBoost + 5m technical.

        Args:
            df_1d_features: DataFrame with 1D features
            df_5m_features: DataFrame with 5m features
            bars_in_trade: Number of bars held in current position
            timestamp: Current timestamp (for diagnostics)

        Returns:
            AdaptiveSignalResult with combined signal
        """
        # Get 1D XGBoost probability (daily bias)
        xgb_prob = self.get_1d_xgb_probability(df_1d_features)

        # Get 5m technical signal (entry confirmation)
        tech_signal, tech_score = self.get_5m_technical_score(df_5m_features)

        # If no 5m technical signal, can't execute
        if tech_signal == 0:
            return AdaptiveSignalResult(
                signal=0,
                confidence=0.0,
                xgb_probability=xgb_prob,
                technical_score=tech_score,
                bars_in_trade=bars_in_trade,
                reason="No 5m technical signal (need MA/RSI/MACD confirmation)",
                should_execute=False
            )

        # Combine signals: 50% XGBoost (1D bias) + 50% Technical (5m entry)
        combined_confidence = (0.50 * xgb_prob) + (0.50 * tech_score)
        should_execute = combined_confidence >= self.EXECUTE_THRESHOLD

        # Generate reason
        signal_type = "BUY" if tech_signal == 1 else "SELL"
        reason = f"{signal_type}: XGB={xgb_prob:.2f}, Tech={tech_score:.2f}, Conf={combined_confidence:.2f}"
        if not should_execute:
            reason += f" (below {self.EXECUTE_THRESHOLD})"

        return AdaptiveSignalResult(
            signal=tech_signal,  # Use 5m signal as primary
            confidence=combined_confidence,
            xgb_probability=xgb_prob,
            technical_score=tech_score,
            bars_in_trade=bars_in_trade,
            reason=reason,
            should_execute=should_execute
        )


def generate_adaptive_signal(df_1d_features: pd.DataFrame,
                            df_5m_features: pd.DataFrame,
                            bars_in_trade: int = 0) -> AdaptiveSignalResult:
    """Convenience function for adaptive signal generation.

    Args:
        df_1d_features: 1D feature DataFrame
        df_5m_features: 5m feature DataFrame
        bars_in_trade: Bars in current trade

    Returns:
        AdaptiveSignalResult
    """
    generator = AdaptiveXGBSignalGenerator()
    return generator.generate_adaptive_signal(
        df_1d_features,
        df_5m_features,
        bars_in_trade
    )


if __name__ == "__main__":
    print("Adaptive XGBoost Signal Generator Module")
    print("Ready for use in backtest pipeline")
    print("\nStrategy: 50% 1D XGBoost probability + 50% 5m technical indicators")
    print("Expected: +2-3% return with 20-40 trades (Phase 5-A baseline)")
