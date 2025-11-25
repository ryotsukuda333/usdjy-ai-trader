"""Data Labeler Module

Generates training labels for supervised learning based on price movements.
Labels are generated using a forward-looking window approach.
"""

import numpy as np
import pandas as pd


def label_data(df: pd.DataFrame, lookforward_periods: int = 5, threshold_pct: float = 0.5) -> pd.DataFrame:
    """Label data based on forward-looking price movement.

    Args:
        df: DataFrame with 'Close' price column
        lookforward_periods: Number of periods to look forward for labeling
        threshold_pct: Price change threshold (%) for classification

    Returns:
        DataFrame with 'target' column added (0: no movement, 1: up movement)
    """
    df = df.copy()

    # Shift prices forward to get future prices
    future_close = df['Close'].shift(-lookforward_periods)

    # Calculate price change percentage
    price_change_pct = ((future_close - df['Close']) / df['Close'] * 100)

    # Label: 1 if future price goes up by threshold, 0 otherwise
    df['target'] = (price_change_pct >= threshold_pct).astype(int)

    # Remove last N rows (NaN targets due to forward shift)
    df = df.dropna(subset=['target']).copy()

    return df


if __name__ == "__main__":
    # Example usage
    from features.multi_timeframe_fetcher import MultiTimeframeFetcher
    from features.multi_timeframe_engineer import MultiTimeframeFeatureEngineer

    # Fetch and engineer data
    fetcher = MultiTimeframeFetcher()
    data = fetcher.fetch_and_resample(years=1)

    engineer = MultiTimeframeFeatureEngineer()
    features = engineer.engineer_all_timeframes(data)
    df_1d = features['1d']

    # Label data
    df_labeled = label_data(df_1d)

    print(f"Total samples: {len(df_labeled)}")
    print(f"Target distribution:\n{df_labeled['target'].value_counts()}")
    print(f"\nFirst few rows:")
    print(df_labeled.head())
