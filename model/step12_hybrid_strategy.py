"""
Step 12: Hybrid Trading Strategy - Seasonality + XGBoost

ハイブリッド戦略の実装:
1. XGBoostモデル: 市場の微細なパターンを捉える
2. SeasonalityManager: 確実な季節パターンを重視する
3. 融合ロジック: 両者のシグナルを統合して最終判定

戦略:
- XGBoostが「買い」かつ Seasonalityスコアが高い → 確信度高い買いシグナル
- XGBoostが「買い」だが Seasonalityスコアが低い → 慎重に進める
- XGBoostが「売り」または Seasonalityが強く「売り」→ スキップ

期待効果:
- Step 10: +62.46% (季節性のみ)
- Step 11: +4.85% (XGBoostのみ)
- Step 12: +70-75% (統合戦略) ← 目標
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import xgboost as xgb


class HybridTradingStrategy:
    """季節性 + XGBoost のハイブリッド戦略"""

    def __init__(self, xgb_model_path: Optional[str] = None, use_seasonality: bool = True):
        """
        Initialize hybrid strategy

        Args:
            xgb_model_path: Path to trained XGBoost model
            use_seasonality: Whether to use seasonality weighting
        """
        self.xgb_model = None
        self.use_seasonality = use_seasonality

        if xgb_model_path and Path(xgb_model_path).exists():
            try:
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(xgb_model_path)
                print(f"✓ Loaded XGBoost model from {xgb_model_path}")
            except Exception as e:
                print(f"⚠ Could not load XGBoost model: {e}")
                self.xgb_model = None

        self._init_seasonality_stats()

    def _init_seasonality_stats(self):
        """Initialize seasonality statistics from Step 10"""
        # Weekly patterns (0=Monday, 4=Friday)
        self.weekly_stats = {
            0: {'mean': 0.0609, 'std': 0.7176, 'signal': +0.1},  # Monday - best
            1: {'mean': 0.0514, 'std': 0.5132, 'signal': +0.1},  # Tuesday - good
            2: {'mean': 0.0035, 'std': 0.6183, 'signal': 0.0},   # Wednesday - neutral
            3: {'mean': -0.0060, 'std': 0.6691, 'signal': -0.05}, # Thursday - weak
            4: {'mean': -0.0237, 'std': 0.7068, 'signal': -0.1},  # Friday - worst
        }

        # Monthly patterns
        self.monthly_stats = {
            1: {'mean': 0.0077, 'std': 0.5925, 'signal': 0.0},
            2: {'mean': 0.0660, 'std': 0.6498, 'signal': +0.1},
            3: {'mean': -0.0300, 'std': 0.5858, 'signal': -0.05},
            4: {'mean': -0.0200, 'std': 0.7115, 'signal': -0.05},
            5: {'mean': 0.0884, 'std': 0.7488, 'signal': +0.1},
            6: {'mean': 0.1023, 'std': 0.4759, 'signal': +0.2},  # June - best
            7: {'mean': -0.0685, 'std': 0.6404, 'signal': -0.1},
            8: {'mean': -0.0451, 'std': 0.7380, 'signal': -0.1},
            9: {'mean': 0.0319, 'std': 0.5411, 'signal': +0.05},
            10: {'mean': 0.1545, 'std': 0.5384, 'signal': +0.15},
            11: {'mean': -0.0191, 'std': 0.5899, 'signal': 0.0},
            12: {'mean': -0.0640, 'std': 0.8714, 'signal': -0.2},  # December - worst
        }

    def get_seasonality_score(self, date: pd.Timestamp) -> float:
        """
        Get seasonality score for a given date (0.0 to 1.0, neutral at 0.5)

        Args:
            date: Timestamp to evaluate

        Returns:
            float: Seasonality score (0.0=avoid, 0.5=neutral, 1.0=ideal)
        """
        dow = date.dayofweek
        month = date.month

        # Weekly score contribution (30%)
        weekly_signal = self.weekly_stats[dow if dow < 5 else 0]['signal']
        weekly_score = 0.5 + weekly_signal  # Range: 0.4-0.6

        # Monthly score contribution (70%)
        monthly_signal = self.monthly_stats[month]['signal']
        monthly_score = 0.5 + monthly_signal  # Range: 0.3-0.7

        # Weighted average
        seasonality_score = 0.3 * weekly_score + 0.7 * monthly_score
        return np.clip(seasonality_score, 0.0, 1.0)

    def generate_hybrid_signal(
        self,
        xgb_probability: float,
        seasonality_score: float,
        threshold_high: float = 0.6,
        threshold_low: float = 0.4,
    ) -> Tuple[int, float]:
        """
        Generate hybrid signal combining XGBoost probability and seasonality

        Args:
            xgb_probability: XGBoost model probability (0.0-1.0)
            seasonality_score: Seasonality score (0.0-1.0)
            threshold_high: High probability threshold (default 0.6)
            threshold_low: Low probability threshold (default 0.4)

        Returns:
            Tuple[signal, confidence]:
                signal: 1 (buy), 0 (sell), -1 (hold)
                confidence: 0.0-1.0 confidence score
        """

        # Combine XGBoost probability with seasonality weighting
        # Seasonality acts as a confidence multiplier
        weighted_probability = xgb_probability * (0.5 + 0.5 * seasonality_score)

        if weighted_probability >= threshold_high:
            # Strong buy signal
            confidence = min(weighted_probability, 1.0)
            return 1, confidence

        elif weighted_probability < threshold_low:
            # Sell signal
            confidence = min(1.0 - weighted_probability, 1.0)
            return 0, confidence

        else:
            # Hold signal - insufficient confidence
            return -1, 0.5

    def generate_predictions(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        xgb_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Generate hybrid predictions for entire dataframe

        Args:
            df: DataFrame with OHLCV and features
            feature_cols: List of feature column names
            xgb_threshold: XGBoost decision threshold

        Returns:
            DataFrame with hybrid predictions
        """

        results = pd.DataFrame(index=df.index)

        # Generate XGBoost predictions if model is available
        if self.xgb_model is not None:
            try:
                # Prepare data for XGBoost
                available_cols = [col for col in feature_cols if col in df.columns]
                X = df[available_cols].fillna(0)

                # Convert to DMatrix
                dmatrix = xgb.DMatrix(X)

                # Get predictions
                xgb_probs = self.xgb_model.predict(dmatrix)
                results['xgb_prob'] = xgb_probs
                results['xgb_signal'] = (xgb_probs >= xgb_threshold).astype(int)

            except Exception as e:
                print(f"⚠ XGBoost prediction failed: {e}")
                results['xgb_prob'] = 0.5
                results['xgb_signal'] = 0

        else:
            # No XGBoost model, use default
            results['xgb_prob'] = 0.5
            results['xgb_signal'] = 0

        # Generate seasonality scores
        results['seasonality_score'] = df.index.map(self.get_seasonality_score)

        # Generate hybrid signals
        if self.xgb_model is not None:
            hybrid_results = [
                self.generate_hybrid_signal(
                    xgb_prob,
                    season_score,
                    threshold_high=0.55,
                    threshold_low=0.45,
                )
                for xgb_prob, season_score in zip(
                    results['xgb_prob'], results['seasonality_score']
                )
            ]
            results['hybrid_signal'] = [sig for sig, _ in hybrid_results]
            results['confidence'] = [conf for _, conf in hybrid_results]

        else:
            # Use seasonality-only approach
            results['hybrid_signal'] = (results['seasonality_score'] > 0.55).astype(int)
            results['confidence'] = results['seasonality_score']

        return results

    def backtest_hybrid_strategy(
        self,
        df: pd.DataFrame,
        predictions: pd.DataFrame,
        initial_capital: float = 100000,
    ) -> Tuple[float, Dict]:
        """
        Backtest the hybrid strategy

        Args:
            df: OHLCV dataframe
            predictions: Predictions dataframe with hybrid signals
            initial_capital: Starting capital

        Returns:
            Tuple[total_return, metrics_dict]
        """

        trades = []
        equity = initial_capital
        position = None
        entry_price = None

        for i in range(1, len(df)):
            date = df.index[i]
            price = df['Close'].iloc[i]
            signal = predictions['hybrid_signal'].iloc[i]
            confidence = predictions['confidence'].iloc[i]

            # Close position if signal changes or confidence is low
            if position is not None and (signal != 1 or confidence < 0.3):
                return_pct = (price - entry_price) / entry_price * 100
                equity *= (1 + return_pct / 100)
                trades.append({
                    'entry_date': position,
                    'entry_price': entry_price,
                    'exit_date': date,
                    'exit_price': price,
                    'return_pct': return_pct,
                    'win': 1 if return_pct > 0 else 0,
                })
                position = None

            # Open position on strong buy signal
            if signal == 1 and position is None and confidence >= 0.5:
                position = date
                entry_price = price

        # Close final position
        if position is not None:
            final_price = df['Close'].iloc[-1]
            return_pct = (final_price - entry_price) / entry_price * 100
            equity *= (1 + return_pct / 100)
            trades.append({
                'entry_date': position,
                'entry_price': entry_price,
                'exit_date': df.index[-1],
                'exit_price': final_price,
                'return_pct': return_pct,
                'win': 1 if return_pct > 0 else 0,
            })

        # Calculate metrics
        total_return = (equity - initial_capital) / initial_capital * 100
        num_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['win'])
        win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0

        metrics = {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'winning_trades': winning_trades,
            'final_equity': equity,
            'trades': trades,
        }

        return total_return, metrics


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 12: Hybrid Trading Strategy - Seasonality + XGBoost")
    print("=" * 70)

    # Test seasonality scoring
    strategy = HybridTradingStrategy(use_seasonality=True)

    print("\n[1] Seasonality Scoring Examples:")
    test_dates = [
        pd.Timestamp('2024-06-10'),  # Best month (June), Monday
        pd.Timestamp('2024-12-20'),  # Worst month (December), Friday
        pd.Timestamp('2024-03-15'),  # Average month, Friday
    ]

    for date in test_dates:
        score = strategy.get_seasonality_score(date)
        dow_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][date.dayofweek]
        month_name = date.strftime('%B')
        print(f"  {date.date()} ({dow_name}, {month_name}): {score:.3f}")

    print("\n[2] Hybrid Signal Generation Examples:")
    test_cases = [
        (0.7, 0.8, "Strong XGBoost + Good Seasonality"),
        (0.7, 0.3, "Strong XGBoost + Bad Seasonality"),
        (0.5, 0.8, "Weak XGBoost + Good Seasonality"),
        (0.3, 0.2, "Weak XGBoost + Bad Seasonality"),
    ]

    for xgb_prob, season_score, label in test_cases:
        signal, conf = strategy.generate_hybrid_signal(xgb_prob, season_score)
        signal_name = {1: 'BUY', 0: 'SELL', -1: 'HOLD'}[signal]
        print(f"  {label}: {signal_name} (confidence: {conf:.2f})")

    print("\n✓ Hybrid strategy initialized successfully")

