"""Feature Engineer module for technical indicator calculation and feature generation.

Implements timezone-aware feature engineering using ta library for technical indicators.
Converts UTC OHLCV data to JST and generates comprehensive feature set for ML training.

Tasks: 3.1, 3.2, 3.3
"""

from typing import Optional
import pandas as pd
import ta
import pytz

from utils.errors import FeatureEngineeringError


def engineer_features(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from OHLCV data with timezone conversion and technical indicators.

    Converts yfinance (UTC) data to JST, calculates technical indicators via ta library,
    generates lag and cyclical features, and returns enriched feature DataFrame.
    Requirement 2: Complete feature engineering with all technical indicators.

    Args:
        df_ohlcv: OHLCV DataFrame with DatetimeIndex (UTC or naive)
                  Columns: ['Open', 'High', 'Low', 'Close', 'Volume']

    Returns:
        pd.DataFrame: Feature-rich DataFrame with JST timezone index
                     Columns include all technical indicators, lag features, day-of-week encoding, and target

    Raises:
        FeatureEngineeringError: If required columns missing or data validation fails

    Process:
        1. Validate OHLCV columns exist
        2. Convert to JST timezone
        3. Calculate moving averages (5, 20, 50)
        4. Calculate MA slopes (day-over-day %)
        5. Calculate RSI14
        6. Calculate MACD (macd, signal, histogram)
        7. Calculate Bollinger Bands
        8. Calculate daily percentage change
        9. Generate lag features (lag1-lag5)
        10. Generate day-of-week one-hot encoding
        11. Calculate target variable (next day direction)
        12. Drop initial NaN values (MA50, MACD lookback)
        13. Return JST-indexed feature DataFrame
    """
    # Validate required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df_ohlcv.columns]
    if missing_columns:
        raise FeatureEngineeringError(
            error_code="MISSING_COLUMNS",
            user_message=f"OHLCV data missing required columns: {missing_columns}",
            technical_message=f"Expected: {required_columns}, Got: {list(df_ohlcv.columns)}"
        )

    # Make a copy to avoid modifying original
    df = df_ohlcv.copy()

    # Step 2: Handle timezone conversion
    # If index has no timezone, treat as UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    # Convert UTC to JST (Asia/Tokyo)
    try:
        df.index = df.index.tz_convert('Asia/Tokyo')
    except Exception as e:
        raise FeatureEngineeringError(
            error_code="TIMEZONE_ERROR",
            user_message="Failed to convert timezone to JST",
            technical_message=f"Timezone conversion error: {str(e)}"
        )

    # Step 3-4: Calculate moving averages and slopes
    try:
        df['ma5'] = df['Close'].rolling(window=5).mean()
        df['ma20'] = df['Close'].rolling(window=20).mean()
        df['ma50'] = df['Close'].rolling(window=50).mean()

        # MA slopes (day-over-day percentage change)
        df['ma5_slope'] = df['ma5'].pct_change() * 100
        df['ma20_slope'] = df['ma20'].pct_change() * 100
        df['ma50_slope'] = df['ma50'].pct_change() * 100
    except Exception as e:
        raise FeatureEngineeringError(
            error_code="MA_CALCULATION_ERROR",
            user_message="Failed to calculate moving averages",
            technical_message=f"Error calculating MAs: {str(e)}"
        )

    # Step 5: Calculate RSI14
    try:
        # Convert Close to 1D numpy array if needed for ta library compatibility
        close_data = df['Close'].values.flatten() if hasattr(df['Close'].values, 'flatten') else df['Close'].values
        df['rsi14'] = ta.momentum.rsi(pd.Series(close_data, index=df.index), window=14)
    except Exception as e:
        raise FeatureEngineeringError(
            error_code="RSI_CALCULATION_ERROR",
            user_message="Failed to calculate RSI14",
            technical_message=f"Error calculating RSI: {str(e)}"
        )

    # Step 6: Calculate MACD (ta.trend.macd, macd_signal, macd_diff return separate Series)
    try:
        # Ensure Close is 1D Series for ta library
        close_series = pd.Series(df['Close'].values.flatten() if hasattr(df['Close'].values, 'flatten') else df['Close'].values, index=df.index)
        df['macd'] = ta.trend.macd(close_series, window_fast=12, window_slow=26, fillna=False)
        df['macd_signal'] = ta.trend.macd_signal(close_series, window_fast=12, window_slow=26, window_sign=9, fillna=False)
        df['macd_histogram'] = ta.trend.macd_diff(close_series, window_fast=12, window_slow=26, window_sign=9, fillna=False)
    except Exception as e:
        raise FeatureEngineeringError(
            error_code="MACD_CALCULATION_ERROR",
            user_message="Failed to calculate MACD",
            technical_message=f"Error calculating MACD: {str(e)}"
        )

    # Step 7: Calculate Bollinger Bands
    try:
        # Ensure Close is 1D Series for ta library
        close_series = pd.Series(df['Close'].values.flatten() if hasattr(df['Close'].values, 'flatten') else df['Close'].values, index=df.index)
        bb_indicator = ta.volatility.BollingerBands(close=close_series, window=20, window_dev=2, fillna=False)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        # Band width
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
    except Exception as e:
        raise FeatureEngineeringError(
            error_code="BB_CALCULATION_ERROR",
            user_message="Failed to calculate Bollinger Bands",
            technical_message=f"Error calculating Bollinger Bands: {str(e)}"
        )

    # Step 8: Calculate daily percentage change
    try:
        df['pct_change'] = df['Close'].pct_change() * 100
    except Exception as e:
        raise FeatureEngineeringError(
            error_code="PCT_CHANGE_ERROR",
            user_message="Failed to calculate percentage change",
            technical_message=f"Error calculating pct_change: {str(e)}"
        )

    # Step 8.5: Calculate VIX-like volatility indicators
    try:
        # Rolling volatility (standard deviation of returns)
        returns = df['Close'].pct_change()
        df['volatility_5'] = returns.rolling(window=5).std() * 100
        df['volatility_10'] = returns.rolling(window=10).std() * 100
        df['volatility_20'] = returns.rolling(window=20).std() * 100

        # High-Low ratio (intraday volatility proxy)
        df['hl_ratio'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['hl_ratio_5'] = df['hl_ratio'].rolling(window=5).mean()

        # Average True Range (ATR) alternative: (High - Low) range
        df['price_range'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['price_range_10'] = df['price_range'].rolling(window=10).mean()
    except Exception as e:
        raise FeatureEngineeringError(
            error_code="VOLATILITY_CALCULATION_ERROR",
            user_message="Failed to calculate volatility indicators",
            technical_message=f"Error calculating volatility: {str(e)}"
        )

    # Step 8.6: Calculate correlation features (autocorrelation)
    try:
        # Autocorrelation of returns (market momentum persistence)
        df['autocorr_5'] = returns.rolling(window=6).apply(lambda x: x.autocorr(), raw=False)

        # Close-to-MA correlation (trend strength)
        df['close_ma5_corr'] = df['Close'].rolling(window=10).corr(df['ma5'])
        df['close_ma20_corr'] = df['Close'].rolling(window=20).corr(df['ma20'])
    except Exception as e:
        raise FeatureEngineeringError(
            error_code="CORRELATION_CALCULATION_ERROR",
            user_message="Failed to calculate correlation features",
            technical_message=f"Error calculating correlations: {str(e)}"
        )

    # Step 9: Generate lag features (past 5 days)
    try:
        for lag in range(1, 6):
            df[f'lag{lag}'] = df['Close'].shift(lag)
    except Exception as e:
        raise FeatureEngineeringError(
            error_code="LAG_CALCULATION_ERROR",
            user_message="Failed to calculate lag features",
            technical_message=f"Error calculating lag features: {str(e)}"
        )

    # Step 10: Day-of-week encoding (one-hot)
    try:
        # Get day of week (0=Monday, 4=Friday)
        day_of_week = df.index.dayofweek
        day_names = ['mon', 'tue', 'wed', 'thu', 'fri']

        for i, day_name in enumerate(day_names):
            df[day_name] = (day_of_week == i).astype(int)
    except Exception as e:
        raise FeatureEngineeringError(
            error_code="DOW_ENCODING_ERROR",
            user_message="Failed to encode day-of-week",
            technical_message=f"Error encoding day-of-week: {str(e)}"
        )

    # Step 11: Calculate target variable (next day direction)
    try:
        # Target: 1 if next close > current close, else 0
        # Shift Close down by 1 to get next day's closing price
        next_close = df['Close'].shift(-1)
        df['target'] = (next_close > df['Close']).astype(int)
    except Exception as e:
        raise FeatureEngineeringError(
            error_code="TARGET_CALCULATION_ERROR",
            user_message="Failed to calculate target variable",
            technical_message=f"Error calculating target: {str(e)}"
        )

    # Step 12: Drop initial NaN values
    # MA50 requires 50 rows, MACD requires 34 rows of lookback
    # Also drop last row due to target variable shift
    try:
        df = df.dropna()
    except Exception as e:
        raise FeatureEngineeringError(
            error_code="DROPNA_ERROR",
            user_message="Failed to drop NaN values",
            technical_message=f"Error dropping NaN: {str(e)}"
        )

    # Validate final feature set
    required_features = [
        'ma5', 'ma20', 'ma50',
        'ma5_slope', 'ma20_slope', 'ma50_slope',
        'rsi14',
        'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'pct_change',
        # New volatility features
        'volatility_5', 'volatility_10', 'volatility_20',
        'hl_ratio', 'hl_ratio_5',
        'price_range', 'price_range_10',
        # New correlation features
        'autocorr_5', 'close_ma5_corr', 'close_ma20_corr',
        # Lag and day-of-week features
        'lag1', 'lag2', 'lag3', 'lag4', 'lag5',
        'mon', 'tue', 'wed', 'thu', 'fri',
        'target'
    ]

    missing_features = [col for col in required_features if col not in df.columns]
    if missing_features:
        raise FeatureEngineeringError(
            error_code="MISSING_FEATURES",
            user_message=f"Missing required feature columns: {missing_features}",
            technical_message=f"Expected all features: {required_features}"
        )

    print(f"✓ Feature engineering complete: {len(df)} rows with {len(df.columns)} features")
    return df


def load_features_from_csv(csv_path: str) -> pd.DataFrame:
    """Load pre-engineered features from CSV file.

    Args:
        csv_path: Path to CSV file with engineered features

    Returns:
        pd.DataFrame: Loaded features with JST timezone index

    Raises:
        FeatureEngineeringError: If file not found or loading fails
    """
    from pathlib import Path

    if not Path(csv_path).exists():
        raise FeatureEngineeringError(
            error_code="FILE_NOT_FOUND",
            user_message=f"Features file not found: {csv_path}",
            technical_message=f"Expected file at {csv_path}"
        )

    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        # Ensure timezone is set to JST
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Tokyo')
        else:
            df.index = df.index.tz_convert('Asia/Tokyo')
        return df
    except Exception as e:
        raise FeatureEngineeringError(
            error_code="CSV_READ_ERROR",
            user_message="Failed to load features from CSV",
            technical_message=f"Error reading CSV: {str(e)}"
        )
