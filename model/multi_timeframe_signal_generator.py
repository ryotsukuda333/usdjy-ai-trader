"""Multi-Timeframe Signal Generator

Implements confluence-based signal generation using hierarchical analysis:
1. Strategic Layer: 1D, 4H, 1H signals establish trend direction
2. Tactical Layer: 15m, 5m signals provide entry point precision

Signal Confidence Calculation:
- Strong (0.80+): Multiple timeframes aligned (e.g., 1D + 4H + 1H)
- Medium (0.60-0.80): Two timeframes aligned (e.g., 1D + 4H)
- Weak (0.40-0.60): Single timeframe or divergence
- Reject (<0.40): No alignment or conflicting signals

Entry Rules:
1. Strategic alignment: 1D trend + 4H confirmation required
2. Tactical entry: 15m/5m signal + high confidence
3. Exit: Trend reversal at higher timeframe or technical levels
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import xgboost as xgb
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class MultiTimeframeSignalGenerator:
    """Generate confluence-based signals from multiple timeframes."""

    def __init__(self, xgb_model_path: Optional[str] = None):
        """Initialize signal generator.

        Args:
            xgb_model_path: Path to trained XGBoost model (optional)
        """
        self.xgb_model_path = xgb_model_path
        self.xgb_model = None
        self.feature_cols = None

        if xgb_model_path and Path(xgb_model_path).exists():
            self._load_xgb_model(xgb_model_path)

    def _load_xgb_model(self, model_path: str):
        """Load XGBoost model and feature columns.

        Args:
            model_path: Path to xgb_model.json
        """
        try:
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(model_path)

            # Load feature columns
            feature_path = Path(model_path).parent / "feature_columns.json"
            if feature_path.exists():
                with open(feature_path) as f:
                    self.feature_cols = json.load(f)

            print(f"✓ Loaded XGBoost model with {len(self.feature_cols)} features")
        except Exception as e:
            print(f"⚠️ Failed to load XGBoost model: {e}")

    def generate_signal_per_timeframe(
        self,
        df_features: pd.DataFrame,
        timeframe: str,
        threshold_high: float = 0.55,
        threshold_low: float = 0.45
    ) -> pd.DataFrame:
        """Generate basic buy/sell signals for a single timeframe.

        Uses simple technical indicator-based rules:
        - BUY (1): Multiple bullish indicators aligned
        - SELL (0): Multiple bearish indicators aligned
        - HOLD (-1): Mixed or no clear signal

        Args:
            df_features: Engineered features for timeframe
            timeframe: Timeframe key
            threshold_high: BUY signal threshold
            threshold_low: SELL signal threshold

        Returns:
            pd.DataFrame: Predictions with columns [signal, confidence, reason]
        """
        results = []

        for idx in range(len(df_features)):
            row = df_features.iloc[idx]

            # Calculate bullish/bearish score from technical indicators
            bullish_score = 0.5  # Start at neutral

            # Moving average alignment (check if price > MAs)
            ma_cols = [col for col in df_features.columns if col.startswith('ma') and col[2:3].isdigit()]
            if ma_cols:
                ma_values = [row.get(col, 0) for col in ma_cols]
                if ma_values:
                    # Count how many MAs price is above
                    price = row['Close']
                    above_count = sum(1 for ma in ma_values if price > ma)
                    bullish_score += (above_count / len(ma_values)) * 0.2

            # RSI signal
            rsi_col = f"rsi{14 if timeframe in ['1d', '4h'] else 14}"
            if rsi_col in df_features.columns:
                rsi = row.get(rsi_col, 50)
                if rsi > 70:  # Overbought (potential sell)
                    bullish_score -= 0.1
                elif rsi < 30:  # Oversold (potential buy)
                    bullish_score += 0.1
                elif rsi > 50:  # Neutral-bullish
                    bullish_score += 0.05
                else:  # Neutral-bearish
                    bullish_score -= 0.05

            # MACD signal
            if 'macd' in df_features.columns and 'macd_signal' in df_features.columns:
                macd = row.get('macd', 0)
                macd_sig = row.get('macd_signal', 0)
                macd_hist = row.get('macd_histogram', 0)

                if macd > macd_sig:  # Bullish crossover
                    bullish_score += 0.15
                elif macd < macd_sig:  # Bearish crossover
                    bullish_score -= 0.15

            # Clip to valid range [0, 1]
            confidence = np.clip(bullish_score, 0.0, 1.0)

            # Determine signal
            if confidence >= threshold_high:
                signal = 1  # BUY
                reason = "Bullish alignment"
            elif confidence <= threshold_low:
                signal = 0  # SELL
                reason = "Bearish alignment"
            else:
                signal = -1  # HOLD
                reason = "Mixed signals"

            results.append({
                'signal': signal,
                'confidence': confidence,
                'reason': reason
            })

        return pd.DataFrame(results)

    def calculate_timeframe_alignment(
        self,
        signals_dict: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Calculate multi-timeframe alignment score.

        Measures how well signals align across timeframes. Higher alignment
        means signals are more consistent, indicating higher confidence.

        Args:
            signals_dict: Dict with keys ['1d', '4h', '1h', '15m', '5m']
                         Each value is Series of signal values (1, 0, -1)
            weights: Optional custom weights per timeframe

        Returns:
            np.ndarray: Alignment scores [0.0-1.0] per bar
        """
        if weights is None:
            # Default weights: higher timeframes more important
            weights = {
                '1d': 0.40,
                '4h': 0.25,
                '1h': 0.20,
                '15m': 0.10,
                '5m': 0.05
            }

        alignment_scores = []
        max_len = max(len(signals_dict[tf]) for tf in signals_dict if tf in signals_dict)

        for i in range(max_len):
            # Get signals at this index across timeframes (may be shorter in intraday)
            signals_at_i = {}
            for tf in ['1d', '4h', '1h', '15m', '5m']:
                if tf in signals_dict and i < len(signals_dict[tf]):
                    signals_at_i[tf] = signals_dict[tf].iloc[i]

            if not signals_at_i:
                alignment_scores.append(0.5)  # No data, neutral
                continue

            # Calculate alignment based on signal consistency
            signal_values = list(signals_at_i.values())

            # Count matching signals
            if len(signal_values) > 1:
                # Check if all signals are same direction (all BUY or all SELL)
                all_same = len(set(signal_values)) == 1

                if all_same and signal_values[0] != -1:
                    # Perfect alignment on BUY or SELL
                    base_alignment = 0.85
                else:
                    # Mixed signals - penalize based on variance
                    variance = np.var(signal_values)
                    base_alignment = 1.0 / (1.0 + variance)
            else:
                # Single signal, neutral alignment
                base_alignment = 0.5

            # Apply weights (higher timeframes influence more)
            weighted_alignment = sum(
                signals_at_i.get(tf, 0.5) * weights.get(tf, 0.1) / 3.0 + base_alignment * 0.5
                for tf in weights.keys()
            ) / len(weights)

            alignment = np.clip(weighted_alignment, 0.0, 1.0)
            alignment_scores.append(alignment)

        return np.array(alignment_scores)

    def generate_confluence_signal(
        self,
        signals_dict: Dict[str, pd.DataFrame],
        alignment_threshold: float = 0.70
    ) -> pd.DataFrame:
        """Generate final confluence-based signals.

        Combines individual timeframe signals using alignment score.

        Args:
            signals_dict: Dict with engineered signals per timeframe
            alignment_threshold: Minimum alignment for execution

        Returns:
            pd.DataFrame: Confluence signals with columns:
                [signal, confidence, alignment, should_execute, details]
        """
        # Extract signal series from dict
        signal_series = {
            tf: df['signal'].reset_index(drop=True)
            for tf, df in signals_dict.items()
        }
        conf_series = {
            tf: df['confidence'].reset_index(drop=True)
            for tf, df in signals_dict.items()
        }

        # Calculate alignment
        alignment = self.calculate_timeframe_alignment(signal_series)

        # Get longest series length
        max_len = max(len(signals_dict[tf]) for tf in signals_dict)

        results = []

        for i in range(max_len):
            # Get signals at index i
            signals_at_i = {}
            confidence_at_i = {}

            for tf in ['1d', '4h', '1h', '15m', '5m']:
                if tf in signals_dict:
                    if i < len(signals_dict[tf]):
                        signals_at_i[tf] = signals_dict[tf].iloc[i]['signal']
                        confidence_at_i[tf] = signals_dict[tf].iloc[i]['confidence']

            # Determine final signal (weighted by timeframe importance)
            if '1d' in signals_at_i:
                # 1D signal is primary
                signal_1d = signals_at_i['1d']
                conf_1d = confidence_at_i['1d']

                # Check for 4H confirmation
                signal_4h = signals_at_i.get('4h', -1)
                conf_4h = confidence_at_i.get('4h', 0.5)

                # Determine execution eligibility
                alignment_score = alignment[i] if i < len(alignment) else 0.5

                # Entry logic
                if signal_1d == 1 and signal_4h == 1:
                    # Strong alignment - look for 15m/5m entry
                    final_signal = 1
                    final_confidence = (conf_1d + conf_4h) / 2
                    should_execute = alignment_score >= alignment_threshold
                    details = "1D+4H alignment"

                elif signal_1d == 1 and alignment_score > 0.60:
                    # Moderate alignment - wait for confirmation
                    final_signal = 1
                    final_confidence = conf_1d
                    should_execute = alignment_score >= alignment_threshold
                    details = "1D signal, awaiting confirmation"

                elif signal_1d == 0:
                    # Sell signal
                    final_signal = 0
                    final_confidence = conf_1d
                    should_execute = True
                    details = "1D sell signal"

                else:
                    # Hold
                    final_signal = -1
                    final_confidence = 0.5
                    should_execute = False
                    details = "No clear signal"

            else:
                # Fallback if no 1D data
                final_signal = -1
                final_confidence = 0.5
                should_execute = False
                details = "Missing 1D data"

            results.append({
                'signal': final_signal,
                'confidence': final_confidence,
                'alignment': alignment[i] if i < len(alignment) else 0.5,
                'should_execute': should_execute,
                'details': details
            })

        return pd.DataFrame(results)


def generate_multi_timeframe_signals(
    features_dict: Dict[str, pd.DataFrame],
    xgb_model_path: Optional[str] = None,
    alignment_threshold: float = 0.70
) -> Dict[str, pd.DataFrame]:
    """Generate multi-timeframe confluence signals.

    Args:
        features_dict: Dict with engineered features per timeframe
        xgb_model_path: Optional path to XGBoost model
        alignment_threshold: Minimum alignment for execution

    Returns:
        Dict[str, pd.DataFrame]: Signals per timeframe + final confluence
    """
    generator = MultiTimeframeSignalGenerator(xgb_model_path)

    print("=" * 80)
    print("MULTI-TIMEFRAME SIGNAL GENERATION")
    print("=" * 80)

    # Generate signals per timeframe
    signals_dict = {}
    for tf in ['1d', '4h', '1h', '15m', '5m']:
        if tf in features_dict:
            print(f"\n[{tf.upper()}] Generating signals...")
            signals = generator.generate_signal_per_timeframe(features_dict[tf], tf)
            signals_dict[tf] = signals

            # Print signal distribution
            signal_counts = signals['signal'].value_counts().sort_index()
            print(f"  Signal distribution:")
            for sig_val in [-1, 0, 1]:
                count = signal_counts.get(sig_val, 0)
                pct = 100 * count / len(signals)
                sig_name = {-1: 'HOLD', 0: 'SELL', 1: 'BUY'}.get(sig_val, '?')
                print(f"    {sig_name}: {count:5d} ({pct:5.1f}%)")

    # Generate confluence signals
    print("\n[CONFLUENCE] Generating multi-timeframe alignment signals...")
    confluence = generator.generate_confluence_signal(signals_dict, alignment_threshold)

    signals_dict['confluence'] = confluence

    # Print summary
    print(f"\n[CONFLUENCE] Final signals:")
    conf_counts = confluence['signal'].value_counts().sort_index()
    for sig_val in [-1, 0, 1]:
        count = conf_counts.get(sig_val, 0)
        pct = 100 * count / len(confluence)
        sig_name = {-1: 'HOLD', 0: 'SELL', 1: 'BUY'}.get(sig_val, '?')
        print(f"  {sig_name}: {count:5d} ({pct:5.1f}%)")

    execute_count = confluence['should_execute'].sum()
    print(f"  Executable signals: {execute_count} ({100*execute_count/len(confluence):.1f}%)")

    print("\n" + "=" * 80)
    print("SIGNAL GENERATION COMPLETE")
    print("=" * 80)

    return signals_dict


if __name__ == "__main__":
    # Example usage
    from features.multi_timeframe_fetcher import fetch_multi_timeframe_usdjpy
    from features.multi_timeframe_engineer import engineer_features_multi_timeframe

    # Fetch data
    data = fetch_multi_timeframe_usdjpy(years=1)

    # Engineer features
    features = engineer_features_multi_timeframe(data)

    # Generate signals
    signals = generate_multi_timeframe_signals(features)

    # Display sample
    print("\n[CONFLUENCE] Sample predictions:")
    print(signals['confluence'].head(10))
