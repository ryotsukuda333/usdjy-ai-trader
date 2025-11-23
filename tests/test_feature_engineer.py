"""Tests for feature engineer module (Task 3.1, 3.2, 3.3).

Test-Driven Development phase for feature generation and validation.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest
import pytz

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.errors import FeatureEngineeringError


class TestEngineerFeatures:
    """Tests for engineer_features function (Task 3.1-3.3)."""

    @staticmethod
    def create_sample_ohlcv_data(rows=800):
        """Create sample OHLCV data for testing.

        Args:
            rows: Number of rows to create

        Returns:
            pd.DataFrame: Sample OHLCV data with UTC timezone
        """
        dates = pd.date_range('2021-01-01', periods=rows, freq='D', tz='UTC')
        base_price = 100.0

        df = pd.DataFrame({
            'Open': [base_price + i*0.01 for i in range(rows)],
            'High': [base_price + 1.0 + i*0.01 for i in range(rows)],
            'Low': [base_price - 0.5 + i*0.01 for i in range(rows)],
            'Close': [base_price + 0.5 + i*0.01 for i in range(rows)],
            'Volume': [1000000 + i*1000 for i in range(rows)]
        }, index=dates)

        return df

    def test_engineer_features_returns_dataframe(self):
        """
        Given: OHLCV DataFrame with UTC timezone
        When: engineer_features() is called
        Then: Should return a pandas DataFrame
        """
        from features.feature_engineer import engineer_features

        df_ohlcv = self.create_sample_ohlcv_data()
        df_features = engineer_features(df_ohlcv)

        assert isinstance(df_features, pd.DataFrame), "Should return a pandas DataFrame"
        assert len(df_features) > 0, "DataFrame should not be empty"

    def test_engineer_features_converts_timezone_to_jst(self):
        """
        Given: OHLCV DataFrame with UTC timezone
        When: engineer_features() is called
        Then: Index should be converted to JST (UTC+9)
        """
        from features.feature_engineer import engineer_features

        df_ohlcv = self.create_sample_ohlcv_data()
        df_features = engineer_features(df_ohlcv)

        # Check that index is in JST timezone
        assert df_features.index.tz is not None, "Index should have timezone info"
        assert str(df_features.index.tz) == 'Asia/Tokyo', \
            f"Expected Asia/Tokyo timezone, got {df_features.index.tz}"

    def test_engineer_features_calculates_moving_averages(self):
        """
        Given: OHLCV DataFrame
        When: engineer_features() is called
        Then: Should calculate ma5, ma20, ma50 columns
        """
        from features.feature_engineer import engineer_features

        df_ohlcv = self.create_sample_ohlcv_data()
        df_features = engineer_features(df_ohlcv)

        # Check that MA columns exist
        assert 'ma5' in df_features.columns, "Should have ma5 column"
        assert 'ma20' in df_features.columns, "Should have ma20 column"
        assert 'ma50' in df_features.columns, "Should have ma50 column"

    def test_engineer_features_calculates_ma_slopes(self):
        """
        Given: OHLCV DataFrame
        When: engineer_features() is called
        Then: Should calculate ma5_slope, ma20_slope, ma50_slope (day-over-day %)
        """
        from features.feature_engineer import engineer_features

        df_ohlcv = self.create_sample_ohlcv_data()
        df_features = engineer_features(df_ohlcv)

        # Check that MA slope columns exist
        assert 'ma5_slope' in df_features.columns, "Should have ma5_slope column"
        assert 'ma20_slope' in df_features.columns, "Should have ma20_slope column"
        assert 'ma50_slope' in df_features.columns, "Should have ma50_slope column"

    def test_engineer_features_calculates_rsi(self):
        """
        Given: OHLCV DataFrame
        When: engineer_features() is called
        Then: Should calculate RSI14 indicator
        """
        from features.feature_engineer import engineer_features

        df_ohlcv = self.create_sample_ohlcv_data()
        df_features = engineer_features(df_ohlcv)

        assert 'rsi14' in df_features.columns, "Should have rsi14 column"
        # RSI values should be between 0 and 100
        rsi_values = df_features['rsi14'].dropna()
        if len(rsi_values) > 0:
            assert (rsi_values >= 0).all() and (rsi_values <= 100).all(), \
                "RSI values should be between 0 and 100"

    def test_engineer_features_calculates_macd(self):
        """
        Given: OHLCV DataFrame
        When: engineer_features() is called
        Then: Should calculate MACD columns (macd, macd_signal, macd_histogram)
        """
        from features.feature_engineer import engineer_features

        df_ohlcv = self.create_sample_ohlcv_data()
        df_features = engineer_features(df_ohlcv)

        assert 'macd' in df_features.columns, "Should have macd column"
        assert 'macd_signal' in df_features.columns, "Should have macd_signal column"
        assert 'macd_histogram' in df_features.columns, "Should have macd_histogram column"

    def test_engineer_features_calculates_bollinger_bands(self):
        """
        Given: OHLCV DataFrame
        When: engineer_features() is called
        Then: Should calculate Bollinger Bands columns
        """
        from features.feature_engineer import engineer_features

        df_ohlcv = self.create_sample_ohlcv_data()
        df_features = engineer_features(df_ohlcv)

        # Check Bollinger Bands columns
        assert 'bb_upper' in df_features.columns, "Should have bb_upper column"
        assert 'bb_middle' in df_features.columns, "Should have bb_middle column"
        assert 'bb_lower' in df_features.columns, "Should have bb_lower column"
        assert 'bb_width' in df_features.columns, "Should have bb_width column"

    def test_engineer_features_calculates_pct_change(self):
        """
        Given: OHLCV DataFrame
        When: engineer_features() is called
        Then: Should calculate daily percentage change (pct_change)
        """
        from features.feature_engineer import engineer_features

        df_ohlcv = self.create_sample_ohlcv_data()
        df_features = engineer_features(df_ohlcv)

        assert 'pct_change' in df_features.columns, "Should have pct_change column"

    def test_engineer_features_calculates_lag_features(self):
        """
        Given: OHLCV DataFrame
        When: engineer_features() is called
        Then: Should calculate lag1 through lag5 features
        """
        from features.feature_engineer import engineer_features

        df_ohlcv = self.create_sample_ohlcv_data()
        df_features = engineer_features(df_ohlcv)

        # Check lag columns
        for lag in range(1, 6):
            col_name = f'lag{lag}'
            assert col_name in df_features.columns, f"Should have {col_name} column"

    def test_engineer_features_calculates_day_of_week(self):
        """
        Given: OHLCV DataFrame
        When: engineer_features() is called
        Then: Should calculate day-of-week one-hot encoding
        """
        from features.feature_engineer import engineer_features

        df_ohlcv = self.create_sample_ohlcv_data()
        df_features = engineer_features(df_ohlcv)

        # Check day-of-week columns
        weekdays = ['mon', 'tue', 'wed', 'thu', 'fri']
        for day in weekdays:
            assert day in df_features.columns, f"Should have {day} column"

    def test_engineer_features_calculates_target_variable(self):
        """
        Given: OHLCV DataFrame
        When: engineer_features() is called
        Then: Should calculate target variable (1 if next_close > current_close else 0)
        """
        from features.feature_engineer import engineer_features

        df_ohlcv = self.create_sample_ohlcv_data()
        df_features = engineer_features(df_ohlcv)

        assert 'target' in df_features.columns, "Should have target column"
        # Target should be 0 or 1 (ignoring NaN)
        target_values = df_features['target'].dropna()
        assert set(target_values).issubset({0, 1, 0.0, 1.0}), \
            "Target should be 0 or 1"

    def test_engineer_features_drops_nan_values(self):
        """
        Given: OHLCV DataFrame with initial NaN values
        When: engineer_features() is called
        Then: Should drop initial rows with NaN (from MA50 lookback, etc.)
        """
        from features.feature_engineer import engineer_features

        df_ohlcv = self.create_sample_ohlcv_data(rows=800)
        original_rows = len(df_ohlcv)

        df_features = engineer_features(df_ohlcv)

        # After feature engineering, rows should be reduced due to dropna()
        # MA50 requires 50 rows, MACD requires 34 rows, so we expect significant reduction
        assert len(df_features) < original_rows, \
            "Should drop initial NaN rows from technical indicator lookback"
        assert len(df_features) > original_rows * 0.85, \
            "Should retain most rows (at least 85%)"

    def test_engineer_features_validates_required_columns(self):
        """
        Given: OHLCV DataFrame
        When: engineer_features() is called
        Then: Should verify all required feature columns exist
        """
        from features.feature_engineer import engineer_features

        df_ohlcv = self.create_sample_ohlcv_data()
        df_features = engineer_features(df_ohlcv)

        # Required feature columns
        required_features = [
            'ma5', 'ma20', 'ma50',
            'ma5_slope', 'ma20_slope', 'ma50_slope',
            'rsi14',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'pct_change',
            'lag1', 'lag2', 'lag3', 'lag4', 'lag5',
            'mon', 'tue', 'wed', 'thu', 'fri',
            'target'
        ]

        for col in required_features:
            assert col in df_features.columns, \
                f"Missing required feature column: {col}"

    def test_engineer_features_rejects_invalid_ohlcv_data(self):
        """
        Given: DataFrame missing required OHLCV columns
        When: engineer_features() is called
        Then: Should raise FeatureEngineeringError
        """
        from features.feature_engineer import engineer_features

        # DataFrame missing 'Close' column
        df_invalid = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [101.0, 102.0],
            'Low': [99.0, 100.0],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2023-01-01', periods=2, tz='UTC'))

        with pytest.raises(FeatureEngineeringError):
            engineer_features(df_invalid)

    def test_engineer_features_jst_index_consistency(self):
        """
        Given: OHLCV DataFrame is processed
        When: engineer_features() completes
        Then: Index should be in JST timezone consistently
        """
        from features.feature_engineer import engineer_features

        df_ohlcv = self.create_sample_ohlcv_data(rows=100)
        df_features = engineer_features(df_ohlcv)

        # Verify index is in JST timezone
        assert df_features.index.tz is not None, "Index should have timezone"
        assert str(df_features.index.tz) == 'Asia/Tokyo', \
            f"Index should be in Asia/Tokyo timezone, got {df_features.index.tz}"

    def test_engineer_features_handles_ohlcv_without_explicit_timezone(self):
        """
        Given: OHLCV DataFrame without explicit timezone (naive datetime)
        When: engineer_features() is called
        Then: Should treat as UTC and convert to JST
        """
        from features.feature_engineer import engineer_features

        # Create naive (no timezone) OHLCV data
        dates = pd.date_range('2021-01-01', periods=100, freq='D')  # No tz
        df_ohlcv = pd.DataFrame({
            'Open': [100.0 + i*0.01 for i in range(100)],
            'High': [101.0 + i*0.01 for i in range(100)],
            'Low': [99.0 + i*0.01 for i in range(100)],
            'Close': [100.5 + i*0.01 for i in range(100)],
            'Volume': [1000000 + i*1000 for i in range(100)]
        }, index=dates)

        df_features = engineer_features(df_ohlcv)

        # Should be converted to JST
        assert str(df_features.index.tz) == 'Asia/Tokyo', \
            "Naive datetime should be treated as UTC and converted to JST"
