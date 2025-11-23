"""Data Fetcher module for USDJPY OHLCV data retrieval.

Implements yfinance-based data fetching with retry logic and exponential backoff.
Validates and persists OHLCV data to CSV format.

Tasks: 2.1, 2.2
"""

import time
from pathlib import Path
from typing import Optional
import pandas as pd
import yfinance as yf

from utils.errors import DataError


def fetch_usdjpy_data(years: int = 3) -> pd.DataFrame:
    """Fetch USDJPY daily OHLCV data from yfinance with retry logic.

    Implements exponential backoff retry strategy (5s -> 10s -> 30s) for network
    resilience. Requirement 1.1: Fetch USDJPY=X daily data for configurable period.

    Args:
        years: Number of years of historical data to fetch (default: 3)

    Returns:
        pd.DataFrame: OHLCV data with DatetimeIndex, columns=['Open', 'High', 'Low', 'Close', 'Volume']

    Raises:
        DataError: If data fetch fails after maximum retries (3 attempts)

    Performance:
        Must complete within 60 seconds (Requirement: パフォーマンス)
    """
    max_retries = 3
    backoff_delays = [5, 10, 30]  # Exponential backoff: 5s -> 10s -> 30s
    ticker = "USDJPY=X"
    period = f"{years}y"

    last_error = None

    for attempt in range(max_retries):
        try:
            # Fetch data from yfinance
            df = yf.download(
                ticker,
                period=period,
                interval="1d",
                progress=False
            )

            if df is None or df.empty:
                raise DataError(
                    error_code="DATA_EMPTY",
                    user_message=f"Failed to fetch {ticker} data",
                    technical_message=f"yfinance returned empty DataFrame for {ticker}"
                )

            # Flatten multi-index columns if present (yfinance may return multi-index)
            if isinstance(df.columns, pd.MultiIndex):
                # If multi-index, take only the first level (column names)
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

            # Verify required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise DataError(
                    error_code="MISSING_COLUMNS",
                    user_message=f"Fetched data missing required columns: {missing_columns}",
                    technical_message=f"Expected columns: {required_columns}, got: {list(df.columns)}"
                )

            # Verify no null values in OHLCV
            null_check = df[required_columns].isnull()
            if null_check.any().any():
                raise DataError(
                    error_code="NULL_VALUES",
                    user_message="Fetched data contains null values in OHLCV columns",
                    technical_message=f"Null values found in:\n{df[required_columns].isnull().sum()}"
                )

            # Log performance
            print(f"✓ Successfully fetched {ticker} data: {len(df)} rows")
            return df

        except DataError:
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = backoff_delays[attempt]
                print(f"⚠ Fetch attempt {attempt + 1} failed: {str(e)}. "
                      f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"✗ All {max_retries} fetch attempts failed")

    # All retries exhausted
    raise DataError(
        error_code="DATA_FETCH_FAILED",
        user_message="Failed to fetch USDJPY data after maximum retries",
        technical_message=f"Last error: {str(last_error)}. "
                         f"Retries: {max_retries}, Backoff delays: {backoff_delays}"
    )


def validate_and_save_ohlcv_data(df: pd.DataFrame) -> str:
    """Validate OHLCV DataFrame and save to CSV.

    Validates that DataFrame contains minimum 750 rows (approximately 3 years
    of daily trading data). Requirement 1.6: Save data to data/ohlcv_usdjpy.csv

    Args:
        df: OHLCV DataFrame with DatetimeIndex

    Returns:
        str: Path to saved CSV file

    Raises:
        DataError: If DataFrame has fewer than 750 rows or validation fails
    """
    # Validate minimum row count (approximately 3 years of daily data)
    min_rows = 750
    if len(df) < min_rows:
        raise DataError(
            error_code="INSUFFICIENT_DATA",
            user_message=f"Insufficient historical data: {len(df)} rows (minimum: {min_rows})",
            technical_message=f"DataFrame has only {len(df)} rows, need at least {min_rows} "
                            f"(approximately 3 years of daily trading data)"
        )

    # Validate required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataError(
            error_code="MISSING_COLUMNS",
            user_message=f"Missing required columns: {missing_columns}",
            technical_message=f"Expected: {required_columns}, Got: {list(df.columns)}"
        )

    # Validate no null values
    if df[required_columns].isnull().any().any():
        null_summary = df[required_columns].isnull().sum()
        raise DataError(
            error_code="NULL_VALUES",
            user_message="Data contains null values in OHLCV columns",
            technical_message=f"Null value counts:\n{null_summary}"
        )

    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    # Save to CSV
    csv_path = data_dir / 'ohlcv_usdjpy.csv'
    df.to_csv(csv_path)

    print(f"✓ OHLCV data saved to {csv_path}: {len(df)} rows")
    return str(csv_path)


def load_ohlcv_data() -> pd.DataFrame:
    """Load OHLCV data from CSV file.

    Used by Feature Engineer to load previously fetched data without
    repeated yfinance calls.

    Returns:
        pd.DataFrame: OHLCV data with DatetimeIndex

    Raises:
        DataError: If CSV file doesn't exist or data is invalid
    """
    csv_path = Path(__file__).parent.parent / 'data' / 'ohlcv_usdjpy.csv'

    if not csv_path.exists():
        raise DataError(
            error_code="FILE_NOT_FOUND",
            user_message=f"OHLCV data file not found: {csv_path}",
            technical_message=f"Expected file at {csv_path} does not exist. "
                            f"Run fetch_usdjpy_data() and validate_and_save_ohlcv_data() first."
        )

    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded OHLCV data from {csv_path}: {len(df)} rows")
        return df
    except Exception as e:
        raise DataError(
            error_code="CSV_READ_ERROR",
            user_message=f"Failed to load OHLCV data from CSV",
            technical_message=f"Error reading {csv_path}: {str(e)}"
        )
