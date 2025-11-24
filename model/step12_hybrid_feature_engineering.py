"""
Step 12: Seasonality + XGBoost Hybrid Feature Engineering

季節性マネージャーの統計データを XGBoost 学習可能な特徴に変換し、
ドメイン知識 + ML の融合モデルを実現する。

主な特徴:
- SeasonalityManager の統計データを直接特徴化
- 週別・月別・年別パターンを XGBoost 入力に統合
- ボラティリティ調整係数を動的に計算
- 過去の取引成功パターンの統計を組み込み
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List
from pathlib import Path


class SeasonalityFeatureEngineer:
    """季節性統計をXGBoost特徴に変換するエンジニア"""

    # Step 10 から抽出した統計データ
    WEEKLY_STATS = {
        0: {'name': '月', 'mean': 0.0609, 'std': 0.7176, 'count': 156},
        1: {'name': '火', 'mean': 0.0514, 'std': 0.5132, 'count': 156},
        2: {'name': '水', 'mean': 0.0035, 'std': 0.6183, 'count': 155},
        3: {'name': '木', 'mean': -0.0060, 'std': 0.6691, 'count': 156},
        4: {'name': '金', 'mean': -0.0237, 'std': 0.7068, 'count': 156},
    }

    MONTHLY_STATS = {
        1: {'name': '1月', 'mean': 0.0077, 'std': 0.5925},
        2: {'name': '2月', 'mean': 0.0660, 'std': 0.6498},
        3: {'name': '3月', 'mean': -0.0300, 'std': 0.5858},
        4: {'name': '4月', 'mean': -0.0200, 'std': 0.7115},
        5: {'name': '5月', 'mean': 0.0884, 'std': 0.7488},
        6: {'name': '6月', 'mean': 0.1023, 'std': 0.4759},  # Best month
        7: {'name': '7月', 'mean': -0.0685, 'std': 0.6404},
        8: {'name': '8月', 'mean': -0.0451, 'std': 0.7380},
        9: {'name': '9月', 'mean': 0.0319, 'std': 0.5411},
        10: {'name': '10月', 'mean': 0.1545, 'std': 0.5384},
        11: {'name': '11月', 'mean': -0.0191, 'std': 0.5899},
        12: {'name': '12月', 'mean': -0.0640, 'std': 0.8714},  # Worst month
    }

    YEARLY_STATS = {
        2022: {'mean': -0.1743, 'std': 1.0524},
        2023: {'mean': 0.0258, 'std': 0.6253},
        2024: {'mean': 0.0419, 'std': 0.6343},
        2025: {'mean': 0.0012, 'std': 0.6325},
    }

    OVERALL_STATS = {
        'mean': 0.0172,
        'std': 0.6485,
        'win_rate': 0.5392,
    }

    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names = []
        self._compute_normalization()

    def _compute_normalization(self):
        """正規化のための統計を計算"""
        self.weekly_means = [self.WEEKLY_STATS[i]['mean'] for i in range(5)]
        self.weekly_stds = [self.WEEKLY_STATS[i]['std'] for i in range(5)]
        self.monthly_means = [self.MONTHLY_STATS[m]['mean'] for m in range(1, 13)]
        self.monthly_stds = [self.MONTHLY_STATS[m]['std'] for m in range(1, 13)]

        self.weekly_means_norm = (np.array(self.weekly_means) - np.mean(self.weekly_means)) / np.std(self.weekly_means)
        self.weekly_stds_norm = (np.array(self.weekly_stds) - np.mean(self.weekly_stds)) / np.std(self.weekly_stds)
        self.monthly_means_norm = (np.array(self.monthly_means) - np.mean(self.monthly_means)) / np.std(self.monthly_means)
        self.monthly_stds_norm = (np.array(self.monthly_stds) - np.mean(self.monthly_stds)) / np.std(self.monthly_stds)

    def add_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        季節性統計ベースの特徴を追加

        特徴:
        1. Weekly Return Bias: 曜日別の過去平均リターン
        2. Weekly Volatility: 曜日別のボラティリティ
        3. Monthly Return Bias: 月別の過去平均リターン
        4. Monthly Volatility: 月別のボラティリティ
        5. Yearly Pattern: 年別パターン
        6. Volatility Multiplier: 現在のボラティリティ vs 季節的期待値
        7. Win Rate History: 曜日・月別の過去勝率
        """

        df = df.copy()

        # === Weekly Features ===
        dow = pd.Series(df.index.dayofweek, index=df.index)  # 0=Monday, 4=Friday

        df['weekly_return_bias'] = dow.apply(
            lambda x: self.WEEKLY_STATS[x]['mean'] if x < 5 else 0.0
        )
        df['weekly_volatility'] = dow.apply(
            lambda x: self.WEEKLY_STATS[x]['std'] if x < 5 else 0.0
        )

        # Normalized versions for model
        df['weekly_return_bias_norm'] = dow.apply(
            lambda x: self.weekly_means_norm[x] if x < 5 else 0.0
        )
        df['weekly_volatility_norm'] = dow.apply(
            lambda x: self.weekly_stds_norm[x] if x < 5 else 0.0
        )

        # === Monthly Features ===
        month = pd.Series(df.index.month, index=df.index)

        df['monthly_return_bias'] = month.apply(
            lambda m: self.MONTHLY_STATS[m]['mean']
        )
        df['monthly_volatility'] = month.apply(
            lambda m: self.MONTHLY_STATS[m]['std']
        )

        # Normalized versions
        df['monthly_return_bias_norm'] = month.apply(
            lambda m: self.monthly_means_norm[m - 1]
        )
        df['monthly_volatility_norm'] = month.apply(
            lambda m: self.monthly_stds_norm[m - 1]
        )

        # === Yearly Features ===
        year = pd.Series(df.index.year, index=df.index)
        df['yearly_return_bias'] = year.apply(
            lambda y: self.YEARLY_STATS.get(y, {'mean': 0.0})['mean']
        )
        df['yearly_volatility'] = year.apply(
            lambda y: self.YEARLY_STATS.get(y, {'std': 0.6485})['std']
        )

        # === Volatility Adjustments ===
        # Expected volatility based on season
        expected_volatility = month.apply(
            lambda m: self.MONTHLY_STATS[m]['std']
        )

        # Actual volatility (rolling)
        returns = df['Close'].pct_change() * 100
        actual_volatility = returns.rolling(window=20).std()

        # Volatility ratio (current vs expected)
        df['volatility_ratio'] = actual_volatility / expected_volatility
        df['volatility_ratio'] = df['volatility_ratio'].fillna(1.0)

        # Extreme volatility flag
        df['high_volatility_month'] = month.apply(
            lambda m: 1.0 if self.MONTHLY_STATS[m]['std'] > 0.70 else 0.0
        )
        df['low_volatility_month'] = month.apply(
            lambda m: 1.0 if self.MONTHLY_STATS[m]['std'] < 0.55 else 0.0
        )

        # === Seasonal Strength ===
        # How strong is the seasonal pattern (signal strength)
        df['seasonal_strength_weekly'] = dow.apply(
            lambda x: abs(self.WEEKLY_STATS[x]['mean']) / (self.WEEKLY_STATS[x]['std'] + 1e-6) if x < 5 else 0.0
        )
        df['seasonal_strength_monthly'] = month.apply(
            lambda m: abs(self.MONTHLY_STATS[m]['mean']) / (self.MONTHLY_STATS[m]['std'] + 1e-6)
        )

        # === Best/Worst Trading Windows ===
        # June (6) is best, December (12) is worst
        df['is_best_month'] = (month == 6).astype(float)
        df['is_worst_month'] = (month == 12).astype(float)

        # Monday is best day, Friday is worst
        df['is_best_day'] = (dow == 0).astype(float)
        df['is_worst_day'] = (dow == 4).astype(float)

        # === Combined Seasonality Score ===
        # Composite score: higher = better trading conditions
        df['seasonality_score'] = (
            0.3 * df['weekly_return_bias_norm'] +  # Positive in good days
            0.3 * df['monthly_return_bias_norm'] +  # Positive in good months
            0.2 * (1.0 / (df['volatility_ratio'] + 1e-6)) +  # Lower volatility is better
            0.2 * df['seasonal_strength_monthly']  # Strong patterns are tradable
        )

        return df

    def add_lagged_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        過去の季節性パターン (lag features)
        同じ曜日・月の過去のパターンを学習させる
        """
        df = df.copy()

        # Weekly patterns from the same day of week in past weeks
        for lag in [5, 10, 20]:  # 1 week, 2 weeks, 1 month
            df[f'weekly_pattern_lag{lag}'] = (df['Close'].shift(lag) / df['Close'] - 1) * 100

        # Monthly patterns (same day of month in past months)
        for lag in [20, 60]:  # 1 month, 3 months
            df[f'monthly_pattern_lag{lag}'] = (df['Close'].shift(lag) / df['Close'] - 1) * 100

        return df

    def get_feature_names(self) -> List[str]:
        """Return list of all seasonality feature names"""
        return [
            'weekly_return_bias', 'weekly_volatility',
            'weekly_return_bias_norm', 'weekly_volatility_norm',
            'monthly_return_bias', 'monthly_volatility',
            'monthly_return_bias_norm', 'monthly_volatility_norm',
            'yearly_return_bias', 'yearly_volatility',
            'volatility_ratio', 'high_volatility_month', 'low_volatility_month',
            'seasonal_strength_weekly', 'seasonal_strength_monthly',
            'is_best_month', 'is_worst_month', 'is_best_day', 'is_worst_day',
            'seasonality_score',
            'weekly_pattern_lag5', 'weekly_pattern_lag10', 'weekly_pattern_lag20',
            'monthly_pattern_lag20', 'monthly_pattern_lag60',
        ]

    def engineer_hybrid_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline combining:
        1. Existing technical features
        2. Seasonality statistics features
        3. Lagged seasonality features
        """
        # Add seasonality-based features
        df = self.add_seasonality_features(df)
        df = self.add_lagged_seasonality_features(df)

        return df


def create_hybrid_feature_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create complete hybrid feature set for XGBoost
    Combines technical indicators with seasonal patterns
    """
    engineer = SeasonalityFeatureEngineer()
    df_featured = engineer.engineer_hybrid_features(df)
    feature_names = engineer.get_feature_names()

    return df_featured, feature_names


if __name__ == "__main__":
    """Test the feature engineer"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from features.data_fetcher import fetch_usdjpy_data
    from features.advanced_feature_engineer import engineer_advanced_features

    print("=" * 70)
    print("STEP 12: Seasonality + XGBoost Hybrid Feature Engineering")
    print("=" * 70)

    # Fetch data
    print("\n[1] Fetching USDJPY data...")
    df = fetch_usdjpy_data()
    print(f"✓ Fetched {len(df)} candles")

    # Add technical features
    print("\n[2] Adding technical features...")
    df = engineer_advanced_features(df)
    print(f"✓ Added technical features: {len(df.columns)} columns")

    # Add seasonality features
    print("\n[3] Adding seasonality features...")
    df_hybrid, seasonality_features = create_hybrid_feature_set(df)
    print(f"✓ Added {len(seasonality_features)} seasonality features")
    print(f"  Features: {seasonality_features[:5]}... ({len(seasonality_features)} total)")

    # Display statistics
    print("\n[4] Feature Statistics:")
    print(f"  Total columns: {len(df_hybrid.columns)}")
    print(f"  Total rows: {len(df_hybrid)}")
    if len(df_hybrid) > 0:
        print(f"  Date range: {df_hybrid.index[0]} to {df_hybrid.index[-1]}")
    print(f"  Missing values (sample):")
    for col in seasonality_features[:5]:
        if col in df_hybrid.columns:
            missing = df_hybrid[col].isna().sum()
            print(f"    {col}: {missing}")

    print("\n✓ Feature engineering complete")

