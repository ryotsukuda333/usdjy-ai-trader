"""
Advanced Feature Engineering for USDJPY AI Trader
Adds sophisticated features to improve model prediction accuracy

New features include:
- Seasonal indicators (month, week patterns)
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility patterns (ATR, Bollinger Band signals)
- Pattern recognition (candle patterns, support/resistance)
- Market regime indicators
"""

import pandas as pd
import numpy as np
from typing import Tuple
import warnings

warnings.filterwarnings('ignore')


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add seasonal and temporal features based on date patterns."""

    df = df.copy()

    # Seasonal encoding (Sin/Cos for circular nature of months/days)
    month = df.index.month
    day_of_year = df.index.dayofyear

    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    df['day_sin'] = np.sin(2 * np.pi * day_of_year / 365)
    df['day_cos'] = np.cos(2 * np.pi * day_of_year / 365)

    # Quarter and season
    df['quarter'] = (df.index.month - 1) // 3 + 1
    seasons = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]  # 0:winter, 1:spring, 2:summer, 3:autumn
    df['season'] = [seasons[m - 1] for m in month]

    # Seasonal volatility pattern (high in Dec/Jan, low in Jun/Jul)
    seasonal_vol_pattern = {1: 0.95, 2: 0.90, 3: 0.85, 4: 0.80, 5: 0.75, 6: 0.70,
                           7: 0.75, 8: 0.80, 9: 0.85, 10: 0.90, 11: 0.95, 12: 1.05}
    df['seasonal_vol_factor'] = [seasonal_vol_pattern[m] for m in month]

    return df


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum indicators: RSI, Stochastic, Rate of Change."""

    df = df.copy()

    close = df['Close']

    # RSI (Relative Strength Index)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Stochastic Oscillator
    low_14 = close.rolling(window=14).min()
    high_14 = close.rolling(window=14).max()
    df['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    # Rate of Change
    df['roc_5'] = (close - close.shift(5)) / close.shift(5) * 100
    df['roc_10'] = (close - close.shift(10)) / close.shift(10) * 100

    # CCI (Commodity Channel Index)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci_20'] = (tp - sma_tp) / (0.015 * mad)

    return df


def add_volatility_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility-based signals and ATR."""

    df = df.copy()

    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()

    # Normalized ATR (as % of close)
    df['atr_pct'] = df['atr_14'] / df['Close'] * 100

    # Bollinger Band signals
    close = df['Close']
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    bb_width = 2 * std_20
    bb_position = (close - (sma_20 - std_20)) / bb_width

    df['bb_position'] = bb_position
    df['bb_squeeze'] = (bb_width / sma_20) < 0.1  # Bollinger Band squeeze

    # Volatility regime (high/low volatility periods)
    volatility_30 = close.pct_change().rolling(window=30).std() * np.sqrt(252) * 100
    volatility_median = volatility_30.rolling(window=60).median()
    df['volatility_regime'] = volatility_30 / volatility_median

    return df


def _calc_hma(close, period):
    """Calculate Hull Moving Average."""
    wma1 = close.rolling(window=period//2).mean()
    wma2 = close.rolling(window=period).mean()
    hma = (2*wma1 - wma2).rolling(window=int(np.sqrt(period))).mean()
    return hma


def add_trend_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend-following signals."""

    df = df.copy()

    close = df['Close']

    # ADX (Average Directional Index) - simplified version
    high_diff = df['High'].diff()
    low_diff = -df['Low'].diff()

    up_move = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    down_move = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

    atr_14 = (pd.concat([df['High'] - df['Low'],
                         np.abs(df['High'] - df['Close'].shift()),
                         np.abs(df['Low'] - df['Close'].shift())], axis=1).max(axis=1)
                .rolling(window=14).mean())

    di_plus = 100 * (up_move.rolling(window=14).mean() / atr_14)
    di_minus = 100 * (down_move.rolling(window=14).mean() / atr_14)

    df['di_plus'] = di_plus
    df['di_minus'] = di_minus

    # Trend strength (DX)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-6)
    df['adx'] = dx.rolling(window=14).mean()

    # HMA (Hull Moving Average) for trend confirmation
    hma_9 = _calc_hma(close, 9)
    hma_20 = _calc_hma(close, 20)
    df['hma_trend'] = hma_9 > hma_20

    return df


def add_support_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """Add support and resistance level signals."""

    df = df.copy()

    close = df['Close']
    high = df['High']
    low = df['Low']

    # Find local support/resistance (swing highs/lows)
    window = 5

    df['swing_high'] = high.rolling(window=2*window+1, center=True).max() == high
    df['swing_low'] = low.rolling(window=2*window+1, center=True).min() == low

    # Distance to recent swing high/low
    recent_high = high.rolling(window=20).max()
    recent_low = low.rolling(window=20).min()

    df['distance_to_high'] = (recent_high - close) / close * 100
    df['distance_to_low'] = (close - recent_low) / close * 100

    # Price position between recent high/low
    df['price_position'] = (close - recent_low) / (recent_high - recent_low)

    return df


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based indicators."""

    df = df.copy()

    volume = df['Volume']
    close = df['Close']

    # Volume Moving Average
    df['volume_ma_20'] = volume.rolling(window=20).mean()
    df['volume_ratio'] = volume / df['volume_ma_20']

    # On Balance Volume (OBV)
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df['obv'] = obv
    df['obv_ma_20'] = obv.rolling(window=20).mean()

    # Volume Rate of Change
    df['vroc_10'] = (volume - volume.shift(10)) / volume.shift(10) * 100

    # Accumulation/Distribution Line
    ad = ((close - close.shift(1)) / (close.shift(1))) * volume
    df['ad_line'] = ad.cumsum()

    return df


def add_order_flow_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add order flow and price action signals."""

    df = df.copy()

    open_p = df['Open']
    close = df['Close']
    high = df['High']
    low = df['Low']

    # Body size (as % of range)
    body_size = np.abs(close - open_p) / (high - low + 1e-6)
    df['candle_body_size'] = body_size

    # Upper/Lower wick size
    df['upper_wick'] = (high - np.maximum(open_p, close)) / (high - low + 1e-6)
    df['lower_wick'] = (np.minimum(open_p, close) - low) / (high - low + 1e-6)

    # Close position within range
    df['close_position'] = (close - low) / (high - low)

    # Consecutive up/down closes
    up_closes = (close > close.shift(1)).astype(int)
    df['consecutive_up'] = up_closes.rolling(window=3).sum()
    df['consecutive_down'] = (~(close > close.shift(1))).astype(int).rolling(window=3).sum()

    return df


def add_mean_reversion_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add mean reversion indicators."""

    df = df.copy()

    close = df['Close']

    # Distance from 20/50/200 MA
    ma_20 = close.rolling(window=20).mean()
    ma_50 = close.rolling(window=50).mean()
    ma_200 = close.rolling(window=200).mean()

    df['distance_from_ma20'] = (close - ma_20) / ma_20 * 100
    df['distance_from_ma50'] = (close - ma_50) / ma_50 * 100
    df['distance_from_ma200'] = (close - ma_200) / ma_200 * 100

    # Z-score (distance from mean in standard deviations)
    df['zscore_20'] = (close - ma_20) / close.rolling(window=20).std()
    df['zscore_50'] = (close - ma_50) / close.rolling(window=50).std()

    return df




def engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to apply all advanced feature engineering.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with additional advanced features
    """

    print("🔧 Applying advanced feature engineering...")

    # Apply feature engineering in order
    df = add_seasonal_features(df)
    print("  ✓ Seasonal features added")

    df = add_momentum_indicators(df)
    print("  ✓ Momentum indicators added")

    df = add_volatility_signals(df)
    print("  ✓ Volatility signals added")

    df = add_trend_signals(df)
    print("  ✓ Trend signals added")

    df = add_support_resistance(df)
    print("  ✓ Support/resistance features added")

    df = add_volume_indicators(df)
    print("  ✓ Volume indicators added")

    df = add_order_flow_signals(df)
    print("  ✓ Order flow signals added")

    df = add_mean_reversion_signals(df)
    print("  ✓ Mean reversion signals added")

    # Drop NaN rows
    df = df.dropna()

    print(f"✅ Advanced features complete: {len(df.columns)} total columns")
    print(f"   New advanced features: {len(df.columns) - 47}")  # Original had ~47 features
    print(f"   Data rows: {len(df)}")

    return df


if __name__ == "__main__":
    # Example usage
    from data_fetcher import fetch_usdjpy_data

    df = fetch_usdjpy_data(years=3)
    df_advanced = engineer_advanced_features(df)
    print(f"\nResulting DataFrame shape: {df_advanced.shape}")
    print(f"Columns: {list(df_advanced.columns)}")
