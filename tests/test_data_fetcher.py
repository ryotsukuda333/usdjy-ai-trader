"""Tests for data fetcher module (Task 2.1, 2.2).

Test-Driven Development phase for yfinance retry logic and OHLCV validation.
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.errors import DataError


class TestFetchUsdjpyData:
    """Tests for fetch_usdjpy_data function (Task 2.1)."""

    def test_fetch_usdjpy_data_returns_dataframe(self):
        """
        Given: yfinance data is available for USDJPY=X
        When: fetch_usdjpy_data() is called
        Then: Should return a pandas DataFrame with OHLCV columns
        """
        from features.data_fetcher import fetch_usdjpy_data

        df = fetch_usdjpy_data(years=3)

        assert isinstance(df, pd.DataFrame), "Should return a pandas DataFrame"
        assert len(df) > 0, "DataFrame should not be empty"

    def test_fetch_usdjpy_data_has_ohlcv_columns(self):
        """
        Given: DataFrame is fetched from yfinance
        When: fetch_usdjpy_data() completes
        Then: DataFrame should contain OHLCV columns
        """
        from features.data_fetcher import fetch_usdjpy_data

        df = fetch_usdjpy_data(years=3)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

    def test_fetch_usdjpy_data_default_years(self):
        """
        Given: No years parameter specified
        When: fetch_usdjpy_data() is called with default
        Then: Should fetch 3 years of data by default
        """
        from features.data_fetcher import fetch_usdjpy_data

        df = fetch_usdjpy_data()

        # Verify approximately 3 years of daily data (roughly 750 trading days)
        assert len(df) >= 700, "Should fetch approximately 3 years of data (min 700 rows)"

    def test_fetch_usdjpy_data_custom_years(self):
        """
        Given: Custom years parameter
        When: fetch_usdjpy_data(years=1) is called
        Then: Should fetch 1 year of data instead
        """
        from features.data_fetcher import fetch_usdjpy_data

        df = fetch_usdjpy_data(years=1)

        # Verify approximately 1 year of daily data (roughly 250 trading days)
        assert len(df) >= 200, "Should fetch approximately 1 year of data (min 200 rows)"

    def test_fetch_usdjpy_data_chronological_order(self):
        """
        Given: Data is fetched from yfinance
        When: fetch_usdjpy_data() completes
        Then: Data should be chronologically ordered (ascending dates)
        """
        from features.data_fetcher import fetch_usdjpy_data

        df = fetch_usdjpy_data(years=3)

        # Verify index is sorted in ascending order
        assert df.index.is_monotonic_increasing, "Data should be chronologically ordered"

    def test_fetch_usdjpy_data_performance(self):
        """
        Given: fetch_usdjpy_data() is called
        When: Function executes
        Then: Should complete within 60 seconds (performance requirement)
        """
        from features.data_fetcher import fetch_usdjpy_data

        start_time = time.time()
        df = fetch_usdjpy_data(years=3)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 60, f"fetch_usdjpy_data took {elapsed_time:.2f}s (limit: 60s)"

    @patch('yfinance.download')
    def test_fetch_usdjpy_data_retry_on_network_error(self, mock_download):
        """
        Given: yfinance.download raises a network error
        When: fetch_usdjpy_data() is called
        Then: Should retry with exponential backoff
        """
        from features.data_fetcher import fetch_usdjpy_data

        # Simulate network error on first call, success on second
        mock_download.side_effect = [
            Exception("Network error"),
            pd.DataFrame({
                'Open': [100.0],
                'High': [101.0],
                'Low': [99.0],
                'Close': [100.5],
                'Volume': [1000000]
            })
        ]

        # Should retry and succeed
        start_time = time.time()
        df = fetch_usdjpy_data(years=3)
        elapsed_time = time.time() - start_time

        # Should have called download twice (first failed, second succeeded)
        assert mock_download.call_count == 2, "Should retry on network error"
        # First retry delay should be ~5 seconds
        assert elapsed_time >= 5, "Should have exponential backoff delay"

    @patch('yfinance.download')
    def test_fetch_usdjpy_data_exponential_backoff(self, mock_download):
        """
        Given: yfinance.download fails multiple times
        When: fetch_usdjpy_data() retries
        Then: Should use exponential backoff (5s -> 10s -> 30s)
        """
        from features.data_fetcher import fetch_usdjpy_data

        # Simulate persistent network errors, then success
        mock_download.side_effect = [
            Exception("Network error"),
            Exception("Network error"),
            pd.DataFrame({
                'Open': [100.0],
                'High': [101.0],
                'Low': [99.0],
                'Close': [100.5],
                'Volume': [1000000]
            })
        ]

        start_time = time.time()
        df = fetch_usdjpy_data(years=3)
        elapsed_time = time.time() - start_time

        # Should retry 3 times with backoff: 5s + 10s = ~15s minimum
        assert mock_download.call_count == 3, "Should retry up to 3 times"
        assert elapsed_time >= 15, "Should have backoff delays (5s + 10s)"

    @patch('yfinance.download')
    def test_fetch_usdjpy_data_raises_after_max_retries(self, mock_download):
        """
        Given: yfinance.download fails 3 times
        When: fetch_usdjpy_data() retries all attempts
        Then: Should raise DataError after exhausting retries
        """
        from features.data_fetcher import fetch_usdjpy_data

        # Simulate persistent failures
        mock_download.side_effect = Exception("Persistent network error")

        with pytest.raises(DataError) as exc_info:
            fetch_usdjpy_data(years=3)

        assert "DATA_FETCH_FAILED" in str(exc_info.value.error_code) or \
               "network" in str(exc_info.value.technical_message).lower(), \
               "Should raise DataError with appropriate error code"

    def test_fetch_usdjpy_data_index_is_datetime(self):
        """
        Given: Data is fetched from yfinance
        When: fetch_usdjpy_data() completes
        Then: DataFrame index should be DatetimeIndex
        """
        from features.data_fetcher import fetch_usdjpy_data

        df = fetch_usdjpy_data(years=3)

        assert isinstance(df.index, pd.DatetimeIndex), "Index should be DatetimeIndex"

    def test_fetch_usdjpy_data_data_quality(self):
        """
        Given: Data is fetched from yfinance
        When: fetch_usdjpy_data() completes
        Then: OHLCV data should have reasonable quality (most values non-null)
        """
        from features.data_fetcher import fetch_usdjpy_data

        df = fetch_usdjpy_data(years=3)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Each OHLCV column should have at least 90% non-null values (market data may have occasional gaps)
        for col in required_columns:
            non_null_count = int(df[col].notna().sum())
            total_count = len(df)
            non_null_ratio = non_null_count / total_count
            assert non_null_ratio > 0.90, \
                f"Column '{col}' has only {int(non_null_ratio*100)}% non-null values"


class TestValidateAndSaveOhlcvData:
    """Tests for validate_and_save_ohlcv_data function (Task 2.2)."""

    def test_validate_and_save_ohlcv_data_creates_csv(self):
        """
        Given: OHLCV DataFrame exists with sufficient rows
        When: validate_and_save_ohlcv_data() is called
        Then: Should create data/ohlcv_usdjpy.csv file
        """
        from features.data_fetcher import validate_and_save_ohlcv_data

        # Create sample DataFrame with 751 rows (minimum 750 required)
        df = pd.DataFrame({
            'Open': [100.0 + i*0.01 for i in range(751)],
            'High': [101.0 + i*0.01 for i in range(751)],
            'Low': [99.0 + i*0.01 for i in range(751)],
            'Close': [100.5 + i*0.01 for i in range(751)],
            'Volume': [1000000 + i*100 for i in range(751)]
        }, index=pd.date_range('2020-01-01', periods=751))

        csv_path = validate_and_save_ohlcv_data(df)

        assert Path(csv_path).exists(), "CSV file should be created"
        assert csv_path.endswith('ohlcv_usdjpy.csv'), "Should save to correct file"

    def test_validate_and_save_ohlcv_data_validates_row_count(self):
        """
        Given: OHLCV DataFrame with insufficient rows
        When: validate_and_save_ohlcv_data() is called
        Then: Should raise DataError if rows < 750
        """
        from features.data_fetcher import validate_and_save_ohlcv_data

        # Create DataFrame with only 100 rows (insufficient)
        df = pd.DataFrame({
            'Open': [100.0] * 100,
            'High': [101.0] * 100,
            'Low': [99.0] * 100,
            'Close': [100.5] * 100,
            'Volume': [1000000] * 100
        }, index=pd.date_range('2023-01-01', periods=100))

        with pytest.raises(DataError) as exc_info:
            validate_and_save_ohlcv_data(df)

        assert "rows" in str(exc_info.value.technical_message).lower() or \
               "insufficient" in str(exc_info.value.technical_message).lower(), \
               "Should mention row count issue"

    def test_validate_and_save_ohlcv_data_sufficient_rows(self):
        """
        Given: OHLCV DataFrame with 750+ rows
        When: validate_and_save_ohlcv_data() is called
        Then: Should successfully save the file
        """
        from features.data_fetcher import validate_and_save_ohlcv_data

        # Create DataFrame with 750 rows (minimum requirement)
        df = pd.DataFrame({
            'Open': [100.0] * 750,
            'High': [101.0] * 750,
            'Low': [99.0] * 750,
            'Close': [100.5] * 750,
            'Volume': [1000000] * 750
        }, index=pd.date_range('2020-01-01', periods=750))

        csv_path = validate_and_save_ohlcv_data(df)

        assert Path(csv_path).exists(), "CSV file should be created for sufficient data"

    def test_validate_and_save_ohlcv_data_csv_readability(self):
        """
        Given: OHLCV data is saved to CSV
        When: CSV file is read back
        Then: Should match original DataFrame structure
        """
        from features.data_fetcher import validate_and_save_ohlcv_data

        # Create sample DataFrame
        df = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0] * 250 + [100.0],  # 751 rows
            'High': [101.0, 102.0, 103.0] * 250 + [101.0],
            'Low': [99.0, 100.0, 101.0] * 250 + [99.0],
            'Close': [100.5, 101.5, 102.5] * 250 + [100.5],
            'Volume': [1000000, 1100000, 1200000] * 250 + [1000000]
        }, index=pd.date_range('2020-01-01', periods=751))
        df.index.name = 'Date'

        csv_path = validate_and_save_ohlcv_data(df)

        # Read back the CSV
        df_read = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        assert len(df_read) == 751, "CSV should preserve all rows"
        assert set(df_read.columns) == {'Open', 'High', 'Low', 'Close', 'Volume'}, \
            "CSV should have all OHLCV columns"

    def test_validate_and_save_ohlcv_data_overwrites_existing(self):
        """
        Given: CSV file already exists
        When: validate_and_save_ohlcv_data() is called with new data
        Then: Should overwrite the existing file
        """
        from features.data_fetcher import validate_and_save_ohlcv_data

        # Create and save first DataFrame
        df1 = pd.DataFrame({
            'Open': [100.0] * 750,
            'High': [101.0] * 750,
            'Low': [99.0] * 750,
            'Close': [100.5] * 750,
            'Volume': [1000000] * 750
        }, index=pd.date_range('2020-01-01', periods=750))

        csv_path1 = validate_and_save_ohlcv_data(df1)

        # Create and save second DataFrame
        df2 = pd.DataFrame({
            'Open': [110.0] * 750,  # Different values
            'High': [111.0] * 750,
            'Low': [109.0] * 750,
            'Close': [110.5] * 750,
            'Volume': [2000000] * 750
        }, index=pd.date_range('2020-01-01', periods=750))

        csv_path2 = validate_and_save_ohlcv_data(df2)

        # Should be the same file
        assert csv_path1 == csv_path2, "Should use the same CSV file path"

        # Read back and verify it has new data
        df_read = pd.read_csv(csv_path2, index_col=0, parse_dates=True)
        assert df_read['Open'].iloc[0] == 110.0, "Should have updated data"
