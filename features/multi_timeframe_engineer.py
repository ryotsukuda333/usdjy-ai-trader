"""Multi-Timeframe Feature Engineering

Implements technical indicator calculation per timeframe with adjusted periods.
Each timeframe uses optimized indicator periods reflecting its resolution:
- 1D: Longer periods (MA 5/20/50, RSI 14, MACD 12/26)
- 4H: Standard periods (MA 5/20/50, RSI 14, MACD 12/26)
- 1H: Medium periods (MA 5/13/50, RSI 14, MACD 12/26)
- 15m: Shorter periods (MA 5/13/20, RSI 14, MACD 5/13)
- 5m: Shortest periods (MA 3/8/13, RSI 14, MACD 5/13)

Features generated per timeframe:
- Moving averages (adjusted per timeframe)
- Moving average slopes
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands (including width)
- Volatility metrics (5/10/20 period)
- Price features (pct_change, hl_ratio, price_range)
- Autocorrelation features
- Lag features (lag 1-5)
- Day-of-week dummy variables (1D layer)
"""

import pandas as pd
import numpy as np
import ta
import pytz
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


class MultiTimeframeFeatureEngineer:
    """Engineer technical features for multiple timeframes."""

    # Indicator periods per timeframe
    TIMEFRAME_PARAMS = {
        '1d': {
            'ma_periods': [5, 20, 50],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
        },
        '4h': {
            'ma_periods': [5, 20, 50],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
        },
        '1h': {
            'ma_periods': [5, 13, 50],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
        },
        '15m': {
            'ma_periods': [5, 13, 20],
            'rsi_period': 14,
            'macd_fast': 5,
            'macd_slow': 13,
            'macd_signal': 5,
        },
        '5m': {
            'ma_periods': [3, 8, 13],
            'rsi_period': 14,
            'macd_fast': 5,
            'macd_slow': 13,
            'macd_signal': 5,
        },
    }

    def engineer_features(
        self,
        df: pd.DataFrame,
        timeframe: str,
        lag_features: bool = True,
        lag_periods: List[int] = [1, 2, 3, 4, 5],
        extended_features: bool = True
    ) -> pd.DataFrame:
        """Engineer technical features for a single timeframe.

        Args:
            df: OHLCV DataFrame with columns [Open, High, Low, Close, Volume]
            timeframe: Timeframe key ('1d', '4h', '1h', '15m', '5m')
            lag_features: Whether to generate lag features
            lag_periods: Lag periods to generate [1, 2, 3, 4, 5]
            extended_features: Whether to generate volatility, autocorr, price features (for XGBoost compatibility)

        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        if timeframe not in self.TIMEFRAME_PARAMS:
            raise ValueError(f"Unknown timeframe: {timeframe}. Valid: {list(self.TIMEFRAME_PARAMS.keys())}")

        df_eng = df.copy()
        params = self.TIMEFRAME_PARAMS[timeframe]

        # Moving averages and slopes
        for period in params['ma_periods']:
            ma_col = f'ma{period}'
            slope_col = f'ma{period}_slope'

            df_eng[ma_col] = df_eng['Close'].rolling(window=period).mean()
            df_eng[slope_col] = df_eng[ma_col].pct_change() * 100

        # RSI
        rsi_col = f"rsi{params['rsi_period']}"
        df_eng[rsi_col] = ta.momentum.rsi(df_eng['Close'], window=params['rsi_period'])

        # MACD
        macd = ta.trend.MACD(
            df_eng['Close'],
            window_fast=params['macd_fast'],
            window_slow=params['macd_slow'],
            window_sign=params['macd_signal']
        )
        df_eng['macd'] = macd.macd()
        df_eng['macd_signal'] = macd.macd_signal()
        df_eng['macd_histogram'] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df_eng['Close'], window=20, window_dev=2)
        df_eng['bb_upper'] = bb.bollinger_hband()
        df_eng['bb_middle'] = bb.bollinger_mavg()
        df_eng['bb_lower'] = bb.bollinger_lband()
        df_eng['bb_width'] = df_eng['bb_upper'] - df_eng['bb_lower']

        # Price changes and ratios
        df_eng['pct_change'] = df_eng['Close'].pct_change() * 100
        df_eng['hl_ratio'] = (df_eng['High'] - df_eng['Low']) / df_eng['Close']
        df_eng['hl_ratio_5'] = (df_eng['High'].rolling(5).max() - df_eng['Low'].rolling(5).min()) / df_eng['Close']
        df_eng['price_range'] = (df_eng['High'] - df_eng['Low'])
        df_eng['price_range_10'] = (df_eng['High'].rolling(10).max() - df_eng['Low'].rolling(10).min())

        # Extended features for XGBoost (volatility, autocorrelation, etc.)
        if extended_features:
            # Volatility metrics
            df_eng['volatility_5'] = df_eng['Close'].pct_change().rolling(5).std() * 100
            df_eng['volatility_10'] = df_eng['Close'].pct_change().rolling(10).std() * 100
            df_eng['volatility_20'] = df_eng['Close'].pct_change().rolling(20).std() * 100

            # Autocorrelation and correlation with MAs (optimized with vectorized operations)
            # Autocorrelation using numpy for better performance
            def compute_autocorr(x):
                try:
                    return np.corrcoef(x[:-1], x[1:])[0, 1]
                except:
                    return 0.0

            df_eng['autocorr_5'] = df_eng['Close'].rolling(5).apply(compute_autocorr, raw=True)

            # Correlation with moving averages (simplified: lag-0 correlation)
            if 'ma5' in df_eng.columns:
                df_eng['close_ma5_corr'] = df_eng['Close'].rolling(5).cov(df_eng['ma5']) / (
                    df_eng['Close'].rolling(5).std() * df_eng['ma5'].rolling(5).std()
                )
            else:
                df_eng['close_ma5_corr'] = 0.0

            if 'ma20' in df_eng.columns:
                df_eng['close_ma20_corr'] = df_eng['Close'].rolling(5).cov(df_eng['ma20']) / (
                    df_eng['Close'].rolling(5).std() * df_eng['ma20'].rolling(5).std()
                )
            else:
                df_eng['close_ma20_corr'] = 0.0

            # Day-of-week dummy variables (only for 1D timeframe, for XGBoost)
            if timeframe == '1d':
                df_eng['mon'] = (df_eng.index.dayofweek == 0).astype(int)
                df_eng['tue'] = (df_eng.index.dayofweek == 1).astype(int)
                df_eng['wed'] = (df_eng.index.dayofweek == 2).astype(int)
                df_eng['thu'] = (df_eng.index.dayofweek == 3).astype(int)
                df_eng['fri'] = (df_eng.index.dayofweek == 4).astype(int)

        # Lag features (extended to 5 periods)
        if lag_features:
            for period in lag_periods:
                df_eng[f'lag{period}'] = df_eng['Close'].shift(period)
                # Note: Removing volume lag for compatibility; can be re-added if needed
                # df_eng[f'volume_lag{period}'] = df_eng['Volume'].shift(period)

        # Day of week (if daily or higher, for non-XGBoost features)
        if timeframe in ['1d', '4h', '1h'] and not extended_features:
            df_eng['day_of_week'] = df_eng.index.dayofweek
            df_eng['hour_of_day'] = df_eng.index.hour

        # Drop initial NaN values (from MAs and indicators)
        max_lookback = max(params['ma_periods'] + [params['rsi_period'], params['macd_slow'], 20])
        df_eng = df_eng.iloc[max_lookback:].copy()

        return df_eng

    def engineer_all_timeframes(
        self,
        timeframe_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Engineer features for all timeframes.

        Args:
            timeframe_dict: Dict with keys ['1d', '4h', '1h', '15m', '5m']

        Returns:
            Dict[str, pd.DataFrame]: Dict with engineered features per timeframe
        """
        print("=" * 80)
        print("MULTI-TIMEFRAME FEATURE ENGINEERING")
        print("=" * 80)

        result = {}

        for timeframe in ['1d', '4h', '1h', '15m', '5m']:
            if timeframe not in timeframe_dict:
                print(f"[{timeframe.upper()}] Skipping (not in input dict)")
                continue

            print(f"\n[{timeframe.upper()}] Engineering features...")
            df_input = timeframe_dict[timeframe]

            try:
                df_eng = self.engineer_features(df_input, timeframe)
                result[timeframe] = df_eng
                print(f"✓ Engineered {len(df_eng)} rows × {len(df_eng.columns)} features")
            except Exception as e:
                print(f"✗ Failed to engineer {timeframe}: {e}")
                raise

        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING COMPLETE")
        print("=" * 80)

        # Print feature summary
        print("\nFeature Summary:")
        for tf, df in result.items():
            print(f"\n{tf.upper()}:")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Column names: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")

        return result

    def get_feature_names(self, timeframe: str) -> List[str]:
        """Get list of engineered feature names for a timeframe.

        Args:
            timeframe: Timeframe key

        Returns:
            List[str]: Feature column names (excluding OHLCV)
        """
        # Sample engineered features to determine column names
        sample_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000, 1000, 1000]
        })
        sample_df.index = pd.date_range('2020-01-01', periods=3, freq='D')

        # Engineer features on sample
        engineered = self.engineer_features(sample_df, timeframe)

        # Return non-OHLCV columns
        ohlcv_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        feature_cols = [col for col in engineered.columns if col not in ohlcv_cols]

        return feature_cols


def engineer_features_multi_timeframe(
    timeframe_dict: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """Convenience function to engineer features for all timeframes.

    Args:
        timeframe_dict: Dict with keys ['1d', '4h', '1h', '15m', '5m']

    Returns:
        Dict[str, pd.DataFrame]: Engineered features per timeframe
    """
    engineer = MultiTimeframeFeatureEngineer()
    return engineer.engineer_all_timeframes(timeframe_dict)


if __name__ == "__main__":
    # Example usage
    from features.multi_timeframe_fetcher import fetch_multi_timeframe_usdjpy

    # Fetch data
    data = fetch_multi_timeframe_usdjpy(years=1)

    # Engineer features
    engineer = MultiTimeframeFeatureEngineer()
    features = engineer.engineer_all_timeframes(data)

    # Display summary
    for tf, df in features.items():
        print(f"\n{tf.upper()} Features Sample:")
        print(df.head())
        print(f"\nColumns ({len(df.columns)}):")
        print(list(df.columns))
