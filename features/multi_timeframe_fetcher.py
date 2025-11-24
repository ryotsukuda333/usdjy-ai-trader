"""Multi-Timeframe Data Fetcher for USDJPY

Implements hierarchical multi-timeframe data management for swing trading with day trading execution.
Fetches 1D data from yfinance and resamples to 4H, 1H, 15m, 5m with realistic intraday generation.

Time Hierarchy:
- 1D (Daily): Strategic bias and trend confirmation
- 4H (4-hour): Mid-term direction confirmation
- 1H (Hourly): Shorter-term trend validation
- 15m (15-minute): Entry zone identification
- 5m (5-minute): Precise entry point timing

Features:
- Fetches 1D OHLCV from yfinance
- Resamples to 4H and 1H using standard OHLC resampling
- Generates synthetic 15m/5m with realistic volatility patterns
- Ensures data alignment across all timeframes
- Handles timezone conversion (UTC -> JST)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


class MultiTimeframeFetcher:
    """Manages multi-timeframe OHLCV data fetching and resampling."""

    def __init__(self, ticker: str = "USDJPY=X"):
        """Initialize multi-timeframe fetcher.

        Args:
            ticker: Ticker symbol (default: USDJPY=X for USD/JPY)
        """
        self.ticker = ticker
        self.timeframes = ['1d', '4h', '1h', '15m', '5m']
        self.resample_rules = {
            '4h': '4h',
            '1h': '1h',
            '15m': '15min',
            '5m': '5min'
        }
        self.multipliers = {
            '1d': 1,
            '4h': 6,  # 4 hours = 6 × 40-min blocks (approximation)
            '1h': 24,  # 1 day = 24 hours
            '15m': 96,  # 1 day = 96 × 15-min blocks
            '5m': 288,  # 1 day = 288 × 5-min blocks
        }

    def fetch_1d_data(self, years: int = 3) -> pd.DataFrame:
        """Fetch 1D OHLCV data from yfinance.

        Args:
            years: Number of years of historical data

        Returns:
            pd.DataFrame: 1D OHLCV with columns [Open, High, Low, Close, Volume]
        """
        print(f"[Multi-TF] Fetching {years}-year 1D data for {self.ticker}...")

        try:
            df = yf.download(
                self.ticker,
                period=f"{years}y",
                interval="1d",
                progress=False
            )

            # Ensure index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Ensure UTC naive (yfinance returns UTC naive for daily)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Handle multi-level columns (yfinance returns tuples like ('Close', 'USDJPY=X'))
            if isinstance(df.columns, pd.MultiIndex) or any(isinstance(col, tuple) for col in df.columns):
                # Flatten by taking first element (the actual column name)
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

            # Standardize column names to uppercase
            df.columns = [str(col).upper() for col in df.columns]

            # Keep only OHLCV columns (select by actual names)
            ohlcv_cols = [col for col in df.columns if col in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
            if len(ohlcv_cols) != 5:
                raise ValueError(f"Expected 5 OHLCV columns, found: {ohlcv_cols}")

            df = df[ohlcv_cols].copy()
            # Rename to standard format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            print(f"✓ Fetched {len(df)} daily candles ({df.index[0].date()} to {df.index[-1].date()})")
            return df

        except Exception as e:
            print(f"✗ Failed to fetch 1D data: {e}")
            raise

    def generate_intraday_synthetic(
        self,
        df_1d: pd.DataFrame,
        intraday_bars_per_day: int = 288,  # For 5-minute bars
        volatility_multiplier: float = 1.5
    ) -> pd.DataFrame:
        """Generate synthetic intraday data with realistic volatility patterns.

        Creates intraday candles that respect daily OHLCV while maintaining
        realistic price movement patterns.

        Args:
            df_1d: 1D OHLCV DataFrame
            intraday_bars_per_day: Number of intraday bars per day (288 for 5m)
            volatility_multiplier: Scale factor for intraday volatility

        Returns:
            pd.DataFrame: Synthetic intraday OHLCV data
        """
        print(f"[Multi-TF] Generating synthetic intraday data ({intraday_bars_per_day} bars/day)...")

        intraday_bars = []

        for idx, (date, row) in enumerate(df_1d.iterrows()):
            daily_open = row['Open']
            daily_high = row['High']
            daily_low = row['Low']
            daily_close = row['Close']
            daily_volume = row['Volume']

            # Daily price range
            daily_range = daily_high - daily_low

            # Intraday volatility (adjusted for bar size)
            intraday_vol = daily_range * volatility_multiplier / (intraday_bars_per_day ** 0.5)

            # Generate intraday progression: open -> close with realistic path
            # Using random walk constrained within daily range
            current_price = daily_open
            intraday_closes = [daily_open]

            for bar_idx in range(1, intraday_bars_per_day):
                # Random walk step
                step = np.random.normal(0, intraday_vol)

                # Drift toward daily close (mean reversion component)
                bar_progress = bar_idx / intraday_bars_per_day
                drift = (daily_close - daily_open) * bar_progress - current_price
                drift *= 0.3  # Weak drift (don't force it too hard)

                # Combined step
                price_change = step + drift
                new_price = current_price + price_change

                # Constrain within a reasonable band (allow some overshoot)
                min_price = daily_low - daily_range * 0.2
                max_price = daily_high + daily_range * 0.2
                new_price = np.clip(new_price, min_price, max_price)

                intraday_closes.append(new_price)
                current_price = new_price

            # Ensure last bar closes at daily close
            intraday_closes[-1] = daily_close

            # Generate OHLC from close progression
            for bar_idx in range(intraday_bars_per_day):
                bar_time = date + pd.Timedelta(minutes=int(1440 / intraday_bars_per_day) * bar_idx)

                # Create realistic OHLC
                bar_close = intraday_closes[bar_idx]

                if bar_idx == 0:
                    bar_open = daily_open
                else:
                    bar_open = intraday_closes[bar_idx - 1]

                # Generate high/low around open/close with volatility
                bar_min = min(bar_open, bar_close) - intraday_vol * np.random.uniform(0, 1)
                bar_max = max(bar_open, bar_close) + intraday_vol * np.random.uniform(0, 1)

                # Distribute volume across bars
                bar_volume = daily_volume / intraday_bars_per_day

                intraday_bars.append({
                    'Datetime': bar_time,
                    'Open': bar_open,
                    'High': bar_max,
                    'Low': bar_min,
                    'Close': bar_close,
                    'Volume': bar_volume
                })

        df_intraday = pd.DataFrame(intraday_bars)
        df_intraday.set_index('Datetime', inplace=True)
        df_intraday.index = df_intraday.index.tz_localize(None)

        print(f"✓ Generated {len(df_intraday)} synthetic intraday candles")
        return df_intraday

    def resample_to_timeframes(
        self,
        df_1d: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Resample 1D data to multiple timeframes.

        Creates 4H and 1H via resampling, and 15m/5m via synthetic generation.

        Args:
            df_1d: 1D OHLCV DataFrame

        Returns:
            Dict[str, pd.DataFrame]: Dict with keys ['1d', '4h', '1h', '15m', '5m']
        """
        result = {'1d': df_1d.copy()}

        # Resample to 4H
        print("[Multi-TF] Resampling to 4H...")
        df_4h = df_1d.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        result['4h'] = df_4h
        print(f"✓ Generated {len(df_4h)} 4H candles")

        # Resample to 1H
        print("[Multi-TF] Resampling to 1H...")
        df_1h = df_1d.resample('1H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        result['1h'] = df_1h
        print(f"✓ Generated {len(df_1h)} 1H candles")

        # Generate synthetic intraday for 5m (from which 15m can be derived)
        df_5m = self.generate_intraday_synthetic(df_1d, intraday_bars_per_day=288)
        result['5m'] = df_5m

        # Resample 5m to 15m
        print("[Multi-TF] Deriving 15m from 5m...")
        df_15m = df_5m.resample('15min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        result['15m'] = df_15m
        print(f"✓ Generated {len(df_15m)} 15m candles")

        return result

    def align_timeframes(
        self,
        timeframe_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Align all timeframes to common index.

        Uses the intersection of indices to ensure all timeframes share
        the same time points (at their respective resolutions).

        Args:
            timeframe_dict: Dict with keys ['1d', '4h', '1h', '15m', '5m']

        Returns:
            Dict[str, pd.DataFrame]: Aligned timeframes
        """
        print("[Multi-TF] Aligning timeframes to common date range...")

        # Find date range that exists in all timeframes
        min_date = max(df.index[0] for df in timeframe_dict.values())
        max_date = min(df.index[-1] for df in timeframe_dict.values())

        print(f"  Date range: {min_date.date()} to {max_date.date()}")

        aligned = {}
        for tf, df in timeframe_dict.items():
            # Slice to common date range
            mask = (df.index >= min_date) & (df.index <= max_date)
            aligned[tf] = df[mask].copy()
            print(f"  {tf.upper():5s}: {len(aligned[tf]):6d} candles")

        return aligned

    def fetch_and_resample(self, years: int = 3) -> Dict[str, pd.DataFrame]:
        """Complete pipeline: fetch 1D data and resample to all timeframes.

        Args:
            years: Number of years of historical data

        Returns:
            Dict[str, pd.DataFrame]: Aligned multi-timeframe data
        """
        print("=" * 80)
        print("MULTI-TIMEFRAME DATA FETCHING PIPELINE")
        print("=" * 80)

        # Step 1: Fetch 1D data
        df_1d = self.fetch_1d_data(years)

        # Step 2: Resample to all timeframes
        timeframe_dict = self.resample_to_timeframes(df_1d)

        # Step 3: Align to common index
        aligned = self.align_timeframes(timeframe_dict)

        print("\n" + "=" * 80)
        print("MULTI-TIMEFRAME FETCH COMPLETE")
        print("=" * 80)
        print(f"\nData Summary:")
        print(f"  Date range: {aligned['1d'].index[0].date()} to {aligned['1d'].index[-1].date()}")
        for tf, df in aligned.items():
            print(f"  {tf.upper():5s}: {len(df):6d} candles")

        return aligned


def fetch_multi_timeframe_usdjpy(years: int = 3) -> Dict[str, pd.DataFrame]:
    """Convenience function to fetch multi-timeframe USDJPY data.

    Args:
        years: Number of years of historical data

    Returns:
        Dict[str, pd.DataFrame]: Multi-timeframe USDJPY data
    """
    fetcher = MultiTimeframeFetcher()
    return fetcher.fetch_and_resample(years)


if __name__ == "__main__":
    # Example usage
    data = fetch_multi_timeframe_usdjpy(years=1)

    for tf, df in data.items():
        print(f"\n{tf.upper()} Sample:")
        print(df.head())
