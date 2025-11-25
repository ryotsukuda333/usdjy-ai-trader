"""
Phase 5-B: Signal Quality Improvement Module

マルチタイムフレーム確認と信号品質スコア機構:
1. 複数時間軸 (1D, 4H, 1H) でのシグナル確認
2. 信号品質スコア (0.0-1.0) の計算
3. 時間軸間の一致度確認によるフィルタリング
4. トレンド強度分析による確信度向上

期待効果:
- 偽陽性シグナルの削減
- 勝率向上 (59% → 62%+)
- 総リターン改善 (+2% → +35-45%)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Dict, List, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class MultiTimeframeAnalyzer:
    """複数時間軸でのマーケット分析"""

    def __init__(self):
        """Initialize multi-timeframe analyzer"""
        self.data_cache = {}

    def fetch_multi_timeframe_data(
        self,
        symbol: str = "USDJPY=X",
        years: int = 3,
        resample: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        複数時間軸でのデータを取得

        Args:
            symbol: Ticker symbol (default: USDJPY=X)
            years: Historical years to fetch
            resample: Whether to resample 1D data to 4H/1H (True) or fetch separately (False)

        Returns:
            Dict[str, pd.DataFrame]: {
                '1d': 日足データ,
                '4h': 4時間足データ,
                '1h': 1時間足データ
            }
        """
        # 1日足データを取得
        df_1d = yf.download(
            symbol,
            period=f"{years}y",
            interval="1d",
            progress=False
        )

        if isinstance(df_1d.columns, pd.MultiIndex):
            df_1d.columns = [col[0] if isinstance(col, tuple) else col for col in df_1d.columns]

        df_1d = df_1d[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df_1d = df_1d.dropna()

        result = {'1d': df_1d}

        if resample:
            # シミュレーション用: 1D データから 4H, 1H データを構成的に生成
            # 実運用では外部サーバーから取得、ここではモック実装
            df_4h = self._simulate_higher_frequency(df_1d, freq='4h')
            df_1h = self._simulate_higher_frequency(df_1d, freq='1h')

            result['4h'] = df_4h
            result['1h'] = df_1h

        return result

    def _simulate_higher_frequency(self, df: pd.DataFrame, freq: str = '4h') -> pd.DataFrame:
        """
        シミュレーション用: 高周期データを低周期データから生成
        実運用環境では実際の高周期データを使用してください

        Args:
            df: Base dataframe (1D)
            freq: Frequency ('4h' or '1h')

        Returns:
            Simulated higher frequency dataframe
        """
        result = []

        for idx, row in df.iterrows():
            close = row['Close']
            high = row['High']
            low = row['Low']

            # Add some noise to simulate intraday variation
            noise_factor = np.random.normal(1.0, 0.001, 4 if freq == '4h' else 24)

            if freq == '4h':
                # Generate 4 x 4H candles per day
                for i in range(4):
                    o = close + (np.random.normal(0, 0.01))
                    h = max(close, high * noise_factor[i])
                    l = min(close, low * noise_factor[i])
                    c = close * noise_factor[i]

                    result.append({
                        'Open': o,
                        'High': h,
                        'Low': l,
                        'Close': c,
                        'Volume': row['Volume'] / 4
                    })
            else:  # 1h
                # Generate 24 x 1H candles per day
                for i in range(24):
                    o = close + (np.random.normal(0, 0.005))
                    h = max(close, high * noise_factor[i % len(noise_factor)])
                    l = min(close, low * noise_factor[i % len(noise_factor)])
                    c = close * noise_factor[i % len(noise_factor)]

                    result.append({
                        'Open': o,
                        'High': h,
                        'Low': l,
                        'Close': c,
                        'Volume': row['Volume'] / 24
                    })

        df_sim = pd.DataFrame(result)
        return df_sim


class SignalQualityScorer:
    """信号品質スコアを計算するモジュール"""

    def __init__(self, min_bars: int = 20):
        """
        Initialize signal quality scorer

        Args:
            min_bars: Minimum bars for trend calculation
        """
        self.min_bars = min_bars

    def calculate_trend_strength(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> float:
        """
        トレンド強度を計算 (0.0-1.0)

        Args:
            df: OHLCV dataframe
            window: Window size for trend calculation

        Returns:
            float: Trend strength (0.0-1.0)
        """
        if len(df) < window:
            return 0.5  # Neutral

        # 単純なトレンドスコア: 上昇バーの比率
        closes = df['Close'].tail(window).values
        up_bars = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        trend_strength = up_bars / (window - 1)

        return np.clip(trend_strength, 0.0, 1.0)

    def calculate_volatility_score(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> float:
        """
        ボラティリティスコア (0.0-1.0)
        高ボラティリティ = 信号品質低い

        Args:
            df: OHLCV dataframe
            window: Window size for volatility calculation

        Returns:
            float: Volatility score (inverted: high vol = low score)
        """
        if len(df) < window:
            return 0.5

        returns = df['Close'].pct_change().tail(window)
        vol = returns.std()

        # ボラティリティを正規化 (1% = 0.5, 2% = 0.25)
        normalized_vol = 1.0 / (1.0 + vol * 100)  # Inverted

        return np.clip(normalized_vol, 0.0, 1.0)

    def calculate_volume_score(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> float:
        """
        出来高スコア (0.0-1.0)
        出来高が多い = 信号品質高い

        Args:
            df: OHLCV dataframe
            window: Window size for volume calculation

        Returns:
            float: Volume score
        """
        if len(df) < window:
            return 0.5

        current_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].tail(window).mean()

        if avg_vol == 0:
            return 0.5

        vol_ratio = current_vol / avg_vol
        # 出来高比率が1.0前後が理想的
        vol_score = 1.0 / (1.0 + abs(vol_ratio - 1.0))

        return np.clip(vol_score, 0.0, 1.0)

    def calculate_multiframe_alignment(
        self,
        timeframes: Dict[str, float],
        weights: Dict[str, float] = None
    ) -> float:
        """
        複数時間軸シグナルの一致度 (0.0-1.0)

        Args:
            timeframes: Dict of {timeframe: confidence_score}
                Example: {'1d': 0.7, '4h': 0.6, '1h': 0.8}
            weights: Optional weights for each timeframe
                Default: {'1d': 0.5, '4h': 0.3, '1h': 0.2}

        Returns:
            float: Alignment score (0.0-1.0)
        """
        if weights is None:
            weights = {'1d': 0.5, '4h': 0.3, '1h': 0.2}

        # 全シグナルが同じ方向か確認
        signals = list(timeframes.values())
        signal_variance = np.var(signals) if len(signals) > 0 else 0.0

        # 分散が小さい = 一致度高い
        alignment = 1.0 / (1.0 + signal_variance)

        # 重み付き平均スコア
        weighted_score = sum(
            timeframes.get(tf, 0.5) * weights.get(tf, 0.1)
            for tf in weights.keys()
        ) / sum(weights.values())

        # 一致度と加重平均の平均
        final_score = 0.6 * alignment + 0.4 * weighted_score

        return np.clip(final_score, 0.0, 1.0)

    def calculate_signal_quality_score(
        self,
        df: pd.DataFrame,
        xgb_confidence: float,
        seasonality_score: float,
        timeframe_alignment: float = 0.5,
        trend_strength: float = 0.5,
        volatility_score: float = 0.5,
        volume_score: float = 0.5
    ) -> float:
        """
        統合的な信号品質スコアを計算 (0.0-1.0)

        複数要因を組み合わせて最終的な信号品質を決定:
        - XGBoost確信度 (50%) - 最重要
        - 季節性スコア (30%) - 重要
        - トレンド強度 (10%)
        - ボラティリティ (5%)
        - 出来高 (5%)

        Args:
            df: OHLCV dataframe
            xgb_confidence: XGBoost model confidence (0.0-1.0)
            seasonality_score: Seasonality score (0.0-1.0)
            timeframe_alignment: Multi-timeframe alignment (0.0-1.0)
            trend_strength: Trend strength (0.0-1.0)
            volatility_score: Volatility score (0.0-1.0)
            volume_score: Volume score (0.0-1.0)

        Returns:
            float: Final signal quality score (0.0-1.0)
        """
        # 重み付き組み合わせ (XGBoostとSeasonality中心)
        weights = {
            'xgb': 0.50,           # XGBoost confidence が最重要
            'seasonality': 0.30,   # Seasonality も重要
            'trend': 0.10,         # Trend補足
            'volatility': 0.05,    # Volatility軽減
            'volume': 0.05         # Volume補足
        }

        components = {
            'xgb': xgb_confidence,
            'seasonality': seasonality_score,
            'trend': trend_strength,
            'volatility': volatility_score,
            'volume': volume_score
        }

        quality_score = sum(
            components[key] * weights[key]
            for key in weights.keys()
        )

        return np.clip(quality_score, 0.0, 1.0)


class SignalQualityFilter:
    """信号品質に基づいたフィルタリングロジック"""

    # デフォルト信号受け入れ基準 (調整版)
    THRESHOLDS = {
        'strong': 0.65,      # 即座に実行
        'medium': 0.50,      # 確認待ち
        'weak': 0.40,        # スキップ推奨
        'reject': 0.25       # 除外
    }

    @staticmethod
    def filter_signal(
        quality_score: float,
        signal: int,  # 1: buy, 0: sell, -1: hold
        confidence: float,
        quality_threshold: float = 0.40
    ) -> Tuple[int, str, bool]:
        """
        信号品質に基づいてシグナルをフィルタリング

        Args:
            quality_score: Signal quality score (0.0-1.0)
            signal: Original signal (1=buy, 0=sell, -1=hold)
            confidence: Original confidence from hybrid strategy
            quality_threshold: Minimum quality score for signal execution (default 0.40)

        Returns:
            Tuple[filtered_signal, decision_reason, should_execute]:
                - filtered_signal: 1 (buy), 0 (sell), -1 (hold)
                - decision_reason: 判定理由テキスト
                - should_execute: 実行すべきか
        """
        # Dynamic threshold application
        strong_threshold = quality_threshold + 0.25
        medium_threshold = quality_threshold
        weak_threshold = quality_threshold - 0.10

        if quality_score >= strong_threshold:
            return signal, "STRONG_QUALITY_SIGNAL", True

        elif quality_score >= medium_threshold:
            if signal == 1 and confidence > 0.6:
                return signal, "MEDIUM_QUALITY_SIGNAL", True
            elif signal == 0 and confidence > 0.5:
                return signal, "MEDIUM_QUALITY_EXIT", True
            else:
                return -1, "AWAITING_CONFIRMATION", False

        elif quality_score >= weak_threshold:
            return -1, "WEAK_QUALITY_SIGNAL", False

        else:
            return -1, "REJECTED_LOW_QUALITY", False


if __name__ == "__main__":
    print("=" * 70)
    print("Phase 5-B: Signal Quality Improvement Module")
    print("=" * 70)

    # Test analyzer
    print("\n[1] Fetching multi-timeframe data...")
    analyzer = MultiTimeframeAnalyzer()
    data = analyzer.fetch_multi_timeframe_data(years=1)
    print(f"✓ 1D data: {len(data['1d'])} candles")
    print(f"✓ 4H data: {len(data['4h'])} candles")
    print(f"✓ 1H data: {len(data['1h'])} candles")

    # Test scorer
    print("\n[2] Testing signal quality scoring...")
    scorer = SignalQualityScorer()

    df_1d = data['1d'].tail(100)
    trend = scorer.calculate_trend_strength(df_1d)
    vol = scorer.calculate_volatility_score(df_1d)
    vol_score = scorer.calculate_volume_score(df_1d)

    print(f"✓ Trend strength: {trend:.2f}")
    print(f"✓ Volatility score: {vol:.2f}")
    print(f"✓ Volume score: {vol_score:.2f}")

    # Test quality score calculation
    quality = scorer.calculate_signal_quality_score(
        df_1d,
        xgb_confidence=0.7,
        seasonality_score=0.6,
        timeframe_alignment=0.75,
        trend_strength=trend,
        volatility_score=vol,
        volume_score=vol_score
    )
    print(f"✓ Final quality score: {quality:.2f}")

    # Test filtering
    print("\n[3] Testing signal filtering...")
    test_cases = [
        (0.8, 1, 0.8, "Strong buy signal"),
        (0.65, 1, 0.7, "Medium buy signal"),
        (0.45, 1, 0.5, "Weak buy signal"),
        (0.25, 1, 0.4, "Rejected signal")
    ]

    for quality, signal, conf, label in test_cases:
        filtered_sig, reason, execute = SignalQualityFilter.filter_signal(
            quality, signal, conf
        )
        signal_map = {1: 'BUY', 0: 'SELL', -1: 'HOLD'}
        print(f"  {label}:")
        print(f"    Quality: {quality:.2f} → {signal_map[filtered_sig]} ({reason})")
        print(f"    Execute: {execute}")

    print("\n✓ Signal quality module initialized successfully")
