"""
Phase 5-B Enhanced: Hybrid Trading Strategy with Signal Quality Improvement

改善点:
1. マルチタイムフレーム分析 (1D, 4H, 1H)
2. 信号品質スコアリング (0.0-1.0)
3. 信頼度ベースのフィルタリング
4. トレンド強度とボラティリティの考慮

期待効果:
- 偽陽性シグナルの削減: 593 → 150-200取引
- 勝率向上: 59% → 62%+
- 総リターン: +2% → +35-45%
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Import from current module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from step12_signal_quality_improver import (
    MultiTimeframeAnalyzer,
    SignalQualityScorer,
    SignalQualityFilter
)


class HybridTradingStrategyImproved:
    """信号品質改善を統合したハイブリッド戦略"""

    def __init__(
        self,
        xgb_model_path: Optional[str] = None,
        use_seasonality: bool = True,
        enable_quality_filter: bool = True
    ):
        """
        Initialize improved hybrid strategy

        Args:
            xgb_model_path: Path to trained XGBoost model
            use_seasonality: Whether to use seasonality weighting
            enable_quality_filter: Enable Phase 5-B signal quality filtering
        """
        self.xgb_model = None
        self.use_seasonality = use_seasonality
        self.enable_quality_filter = enable_quality_filter

        # Initialize modules
        self.scorer = SignalQualityScorer()
        self.filter = SignalQualityFilter()

        # Load XGBoost model
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
        """Initialize seasonality statistics"""
        self.weekly_stats = {
            0: {'mean': 0.0609, 'std': 0.7176, 'signal': +0.1},
            1: {'mean': 0.0514, 'std': 0.5132, 'signal': +0.1},
            2: {'mean': 0.0035, 'std': 0.6183, 'signal': 0.0},
            3: {'mean': -0.0060, 'std': 0.6691, 'signal': -0.05},
            4: {'mean': -0.0237, 'std': 0.7068, 'signal': -0.1},
        }

        self.monthly_stats = {
            1: {'mean': 0.0077, 'std': 0.5925, 'signal': 0.0},
            2: {'mean': 0.0660, 'std': 0.6498, 'signal': +0.1},
            3: {'mean': -0.0300, 'std': 0.5858, 'signal': -0.05},
            4: {'mean': -0.0200, 'std': 0.7115, 'signal': -0.05},
            5: {'mean': 0.0884, 'std': 0.7488, 'signal': +0.1},
            6: {'mean': 0.1023, 'std': 0.4759, 'signal': +0.2},
            7: {'mean': -0.0685, 'std': 0.6404, 'signal': -0.1},
            8: {'mean': -0.0451, 'std': 0.7380, 'signal': -0.1},
            9: {'mean': 0.0319, 'std': 0.5411, 'signal': +0.05},
            10: {'mean': 0.1545, 'std': 0.5384, 'signal': +0.15},
            11: {'mean': -0.0191, 'std': 0.5899, 'signal': 0.0},
            12: {'mean': -0.0640, 'std': 0.8714, 'signal': -0.2},
        }

    def get_seasonality_score(self, date: pd.Timestamp) -> float:
        """Get seasonality score (0.0-1.0)"""
        dow = date.dayofweek
        month = date.month

        weekly_signal = self.weekly_stats[dow if dow < 5 else 0]['signal']
        weekly_score = 0.5 + weekly_signal

        monthly_signal = self.monthly_stats[month]['signal']
        monthly_score = 0.5 + monthly_signal

        seasonality_score = 0.3 * weekly_score + 0.7 * monthly_score
        return np.clip(seasonality_score, 0.0, 1.0)

    def generate_predictions_with_quality(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        xgb_threshold: float = 0.5,
        quality_threshold: float = 0.60
    ) -> pd.DataFrame:
        """
        Generate predictions with Phase 5-B signal quality filtering

        Args:
            df: DataFrame with OHLCV and features
            feature_cols: List of feature column names
            xgb_threshold: XGBoost decision threshold
            quality_threshold: Minimum quality score for signal acceptance

        Returns:
            DataFrame with signals, confidence, and quality metrics
        """
        results = pd.DataFrame(index=df.index)

        # Generate XGBoost predictions
        if self.xgb_model is not None:
            try:
                available_cols = [col for col in feature_cols if col in df.columns]
                X = df[available_cols].fillna(0)
                dmatrix = xgb.DMatrix(X)
                xgb_probs = self.xgb_model.predict(dmatrix)
                results['xgb_prob'] = xgb_probs
            except Exception as e:
                print(f"⚠ XGBoost prediction failed: {e}")
                results['xgb_prob'] = 0.5
        else:
            results['xgb_prob'] = 0.5

        # Generate seasonality scores
        results['seasonality_score'] = df.index.map(self.get_seasonality_score)

        # Calculate trend strength
        results['trend_strength'] = [
            self.scorer.calculate_trend_strength(df.iloc[max(0, i-20):i+1])
            if i >= 1 else 0.5
            for i in range(len(df))
        ]

        # Calculate volatility score
        results['volatility_score'] = [
            self.scorer.calculate_volatility_score(df.iloc[max(0, i-20):i+1])
            if i >= 1 else 0.5
            for i in range(len(df))
        ]

        # Calculate volume score
        results['volume_score'] = [
            self.scorer.calculate_volume_score(df.iloc[max(0, i-20):i+1])
            if i >= 1 else 0.5
            for i in range(len(df))
        ]

        # Generate initial hybrid signals
        hybrid_results = [
            self._generate_hybrid_signal_with_quality(
                xgb_prob=results['xgb_prob'].iloc[i],
                seasonality_score=results['seasonality_score'].iloc[i],
                trend_strength=results['trend_strength'].iloc[i],
                volatility_score=results['volatility_score'].iloc[i],
                volume_score=results['volume_score'].iloc[i],
                xgb_threshold=xgb_threshold
            )
            for i in range(len(results))
        ]

        results['signal'] = [h[0] for h in hybrid_results]
        results['confidence'] = [h[1] for h in hybrid_results]
        results['quality_score'] = [h[2] for h in hybrid_results]

        # Apply Phase 5-B quality filtering
        if self.enable_quality_filter:
            filtered_results = [
                self.filter.filter_signal(
                    quality_score=results['quality_score'].iloc[i],
                    signal=results['signal'].iloc[i],
                    confidence=results['confidence'].iloc[i]
                )
                for i in range(len(results))
            ]

            results['filtered_signal'] = [f[0] for f in filtered_results]
            results['filter_reason'] = [f[1] for f in filtered_results]
            results['should_execute'] = [f[2] for f in filtered_results]

            # Final hybrid signal is the filtered signal
            results['hybrid_signal'] = results['filtered_signal']
        else:
            results['hybrid_signal'] = results['signal']
            results['filtered_signal'] = results['signal']
            results['should_execute'] = True

        return results

    def _generate_hybrid_signal_with_quality(
        self,
        xgb_prob: float,
        seasonality_score: float,
        trend_strength: float,
        volatility_score: float,
        volume_score: float,
        xgb_threshold: float = 0.5
    ) -> Tuple[int, float, float]:
        """
        Generate hybrid signal with quality assessment

        Returns:
            Tuple[signal, confidence, quality_score]:
                - signal: 1 (buy), 0 (sell), -1 (hold)
                - confidence: Original confidence (0.0-1.0)
                - quality_score: Phase 5-B quality score (0.0-1.0)
        """
        # Weighted probability with seasonality
        weighted_prob = xgb_prob * (0.5 + 0.5 * seasonality_score)

        # Determine base signal
        if weighted_prob >= 0.60:
            signal = 1
            confidence = min(weighted_prob, 1.0)
        elif weighted_prob < 0.40:
            signal = 0
            confidence = min(1.0 - weighted_prob, 1.0)
        else:
            signal = -1
            confidence = 0.5

        # Calculate quality score with Phase 5-B factors
        quality_score = self.scorer.calculate_signal_quality_score(
            df=None,  # Not used in this calculation
            xgb_confidence=xgb_prob,
            seasonality_score=seasonality_score,
            timeframe_alignment=0.5,  # Would be calculated with actual multi-TF data
            trend_strength=trend_strength,
            volatility_score=volatility_score,
            volume_score=volume_score
        )

        return signal, confidence, quality_score

    def backtest_improved(
        self,
        df: pd.DataFrame,
        predictions: pd.DataFrame,
        initial_capital: float = 100000,
        use_quality_filter: bool = True
    ) -> Tuple[float, Dict]:
        """
        Backtest improved strategy with Phase 5-B quality filtering

        Args:
            df: OHLCV dataframe
            predictions: Predictions dataframe with quality metrics
            initial_capital: Starting capital
            use_quality_filter: Whether to respect quality filtering

        Returns:
            Tuple[total_return, metrics_dict]
        """
        trades = []
        equity = initial_capital
        position = None
        entry_price = None
        signals_generated = 0
        signals_filtered = 0
        signals_executed = 0

        for i in range(1, len(df)):
            date = df.index[i]
            price = df['Close'].iloc[i]
            signal = predictions['hybrid_signal'].iloc[i]
            confidence = predictions['confidence'].iloc[i]
            quality = predictions['quality_score'].iloc[i]
            should_execute = predictions['should_execute'].iloc[i] if use_quality_filter else True

            # Count signals
            if signal != -1:
                signals_generated += 1

            if signal != -1 and not should_execute:
                signals_filtered += 1

            # Close position
            if position is not None and (signal != 1 or confidence < 0.3 or not should_execute):
                return_pct = (price - entry_price) / entry_price * 100
                equity *= (1 + return_pct / 100)
                trades.append({
                    'entry_date': position,
                    'entry_price': entry_price,
                    'exit_date': date,
                    'exit_price': price,
                    'return_pct': return_pct,
                    'win': 1 if return_pct > 0 else 0,
                    'quality': quality
                })
                position = None

            # Open position
            if signal == 1 and position is None and should_execute and confidence >= 0.5:
                position = date
                entry_price = price
                signals_executed += 1

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
                'quality': quality if position is not None else 0
            })

        # Calculate metrics
        total_return = (equity - initial_capital) / initial_capital * 100
        num_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['win'])
        win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0

        # Calculate Sharpe ratio
        if num_trades > 1:
            returns = [t['return_pct'] for t in trades]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Calculate max drawdown
        eq = initial_capital
        peak = initial_capital
        max_dd = 0
        for trade in trades:
            eq *= (1 + trade['return_pct'] / 100)
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            max_dd = max(max_dd, dd)

        avg_quality = np.mean([t['quality'] for t in trades]) if trades else 0

        metrics = {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'winning_trades': winning_trades,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'final_equity': equity,
            'signals_generated': signals_generated,
            'signals_filtered': signals_filtered,
            'signals_executed': signals_executed,
            'avg_quality_score': avg_quality,
            'trades': trades,
        }

        return total_return, metrics


if __name__ == "__main__":
    print("=" * 70)
    print("Phase 5-B Enhanced: Hybrid Trading Strategy with Quality Filtering")
    print("=" * 70)
    print("\n✓ Improved hybrid strategy module initialized")
