"""Dynamic Risk Management module for volatility-aware position sizing.

Implements dynamic take-profit and stop-loss calculation based on market
volatility indicators. Adjusts risk parameters in real-time as volatility
changes, improving risk-adjusted returns.

Tasks: 5.1, 5.2
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from utils.errors import TraderError


class DynamicRiskManager:
    """Manages dynamic take-profit and stop-loss based on volatility.

    Transforms static risk parameters (+0.60% TP, -0.30% SL) into
    volatility-adjusted levels that adapt to market conditions.

    Key Algorithm:
        - Base TP: 0.60%, Base SL: 0.30%
        - Volatility Factor = current_volatility / median_volatility
        - Dynamic TP = base_TP * volatility_factor
        - Dynamic SL = base_SL * volatility_factor

    Behavior:
        - High volatility → Wider TP/SL (increase position hold time)
        - Low volatility → Tighter TP/SL (faster profit-taking)
    """

    def __init__(self,
                 base_tp_percent: float = 0.60,
                 base_sl_percent: float = 0.30,
                 vol_window: int = 20,
                 vol_min_percentile: int = 25,
                 vol_max_percentile: int = 75):
        """Initialize Dynamic Risk Manager.

        Args:
            base_tp_percent: Base take-profit percentage (default: 0.60%)
            base_sl_percent: Base stop-loss percentage (default: 0.30%)
            vol_window: Window for calculating volatility statistics (default: 20)
            vol_min_percentile: Minimum volatility percentile threshold (default: 25)
            vol_max_percentile: Maximum volatility percentile threshold (default: 75)
        """
        self.base_tp_percent = base_tp_percent
        self.base_sl_percent = base_sl_percent
        self.vol_window = vol_window
        self.vol_min_percentile = vol_min_percentile
        self.vol_max_percentile = vol_max_percentile

        # Statistics will be calculated from data
        self.vol_stats = {
            'median': None,
            'q25': None,
            'q75': None,
            'mean': None,
            'std': None
        }
        self.is_initialized = False

    def initialize_from_features(self, df_features: pd.DataFrame) -> None:
        """Calculate volatility statistics from historical features.

        Must be called before using dynamic risk manager to establish
        baseline volatility metrics.

        Args:
            df_features: Feature DataFrame containing volatility columns

        Raises:
            TraderError: If volatility features missing
        """
        try:
            # Check for volatility features
            vol_columns = ['volatility_5', 'volatility_10', 'volatility_20']
            missing = [col for col in vol_columns if col not in df_features.columns]

            if missing:
                raise TraderError(
                    error_code="MISSING_VOLATILITY_FEATURES",
                    user_message=f"Missing volatility features: {missing}",
                    technical_message=f"Expected {vol_columns}, got {list(df_features.columns)}"
                )

            # Use 20-period volatility as primary indicator
            volatility = df_features['volatility_20'].dropna()

            if len(volatility) < self.vol_window:
                raise TraderError(
                    error_code="INSUFFICIENT_DATA",
                    user_message=f"Insufficient data for volatility calculation",
                    technical_message=f"Need {self.vol_window} rows, got {len(volatility)}"
                )

            # Calculate statistics
            self.vol_stats = {
                'median': volatility.median(),
                'q25': volatility.quantile(0.25),
                'q75': volatility.quantile(0.75),
                'mean': volatility.mean(),
                'std': volatility.std(),
                'min': volatility.min(),
                'max': volatility.max()
            }

            self.is_initialized = True

            print(f"✓ Risk Manager initialized from {len(volatility)} volatility samples")
            print(f"  Volatility Stats:")
            print(f"    Median: {self.vol_stats['median']:.4f}%")
            print(f"    Range: {self.vol_stats['q25']:.4f}% - {self.vol_stats['q75']:.4f}%")
            print(f"    Min/Max: {self.vol_stats['min']:.4f}% - {self.vol_stats['max']:.4f}%")

        except TraderError:
            raise
        except Exception as e:
            raise TraderError(
                error_code="INITIALIZATION_FAILED",
                user_message="Risk manager initialization failed",
                technical_message=f"Error: {str(e)}"
            )

    def get_dynamic_tp_sl(self,
                         entry_price: float,
                         current_volatility: float,
                         index: int = None,
                         df_features: pd.DataFrame = None) -> Tuple[float, float]:
        """Calculate dynamic take-profit and stop-loss levels.

        Volatility-adjusted TP/SL calculation that widens or tightens
        profit targets based on current market volatility.

        Args:
            entry_price: Entry price of position
            current_volatility: Current volatility percentage
            index: (Optional) Row index in dataframe for enhanced analysis
            df_features: (Optional) Full features dataframe

        Returns:
            Tuple[float, float]: (take_profit_level, stop_loss_level)

        Raises:
            TraderError: If manager not initialized or inputs invalid
        """
        if not self.is_initialized:
            raise TraderError(
                error_code="NOT_INITIALIZED",
                user_message="Risk manager not initialized",
                technical_message="Call initialize_from_features() first"
            )

        if entry_price <= 0:
            raise TraderError(
                error_code="INVALID_PRICE",
                user_message="Entry price must be positive",
                technical_message=f"Got {entry_price}"
            )

        if current_volatility < 0:
            raise TraderError(
                error_code="INVALID_VOLATILITY",
                user_message="Volatility cannot be negative",
                technical_message=f"Got {current_volatility}"
            )

        try:
            # Calculate volatility factor with clipping
            median_vol = self.vol_stats['median']
            vol_factor = current_volatility / median_vol if median_vol > 0 else 1.0

            # Clip volatility factor to reasonable range [0.5, 2.0]
            # Prevents extreme adjustments from outlier volatility
            vol_factor = np.clip(vol_factor, 0.5, 2.0)

            # Calculate dynamic TP and SL
            dynamic_tp_pct = self.base_tp_percent * vol_factor
            dynamic_sl_pct = self.base_sl_percent * vol_factor

            # Calculate absolute price levels
            tp_level = entry_price * (1 + dynamic_tp_pct / 100)
            sl_level = entry_price * (1 - dynamic_sl_pct / 100)

            return tp_level, sl_level

        except Exception as e:
            raise TraderError(
                error_code="CALCULATION_FAILED",
                user_message="Failed to calculate dynamic TP/SL",
                technical_message=f"Error: {str(e)}"
            )

    def calculate_position_size(self,
                               account_balance: float,
                               entry_price: float,
                               sl_price: float,
                               risk_percent: float = 1.0) -> float:
        """Calculate position size based on risk management.

        Determines how many units to trade based on:
        - Account balance
        - Entry and stop-loss prices
        - Target risk percentage per trade

        Formula:
            Position Size = (Account * Risk%) / (Entry - SL)

        Args:
            account_balance: Current account balance
            entry_price: Entry price level
            sl_price: Stop-loss price level
            risk_percent: Risk percentage per trade (default: 1.0%)

        Returns:
            float: Position size in units

        Raises:
            TraderError: If calculation invalid
        """
        if account_balance <= 0:
            raise TraderError(
                error_code="INVALID_BALANCE",
                user_message="Account balance must be positive",
                technical_message=f"Got {account_balance}"
            )

        if entry_price <= 0 or sl_price <= 0:
            raise TraderError(
                error_code="INVALID_PRICE",
                user_message="Entry and SL prices must be positive",
                technical_message=f"Entry: {entry_price}, SL: {sl_price}"
            )

        if risk_percent <= 0 or risk_percent > 10:
            raise TraderError(
                error_code="INVALID_RISK",
                user_message="Risk percent must be between 0 and 10%",
                technical_message=f"Got {risk_percent}"
            )

        try:
            # Risk amount in account currency
            risk_amount = account_balance * (risk_percent / 100)

            # Price difference per unit
            price_diff = abs(entry_price - sl_price)

            if price_diff == 0:
                raise TraderError(
                    error_code="INVALID_SL",
                    user_message="Stop-loss equals entry price",
                    technical_message=f"Entry: {entry_price}, SL: {sl_price}"
                )

            # Position size = risk_amount / price_diff
            position_size = risk_amount / price_diff

            return position_size

        except TraderError:
            raise
        except Exception as e:
            raise TraderError(
                error_code="SIZE_CALCULATION_FAILED",
                user_message="Position size calculation failed",
                technical_message=f"Error: {str(e)}"
            )

    def get_risk_metrics(self, current_volatility: float) -> Dict:
        """Get current risk metrics and adjustment summary.

        Args:
            current_volatility: Current market volatility

        Returns:
            Dict with keys:
                - vol_factor: Volatility adjustment multiplier
                - tp_multiplier: TP adjustment from base (%)
                - sl_multiplier: SL adjustment from base (%)
                - volatility_regime: 'low', 'normal', or 'high'
        """
        if not self.is_initialized:
            return {'error': 'Not initialized'}

        median_vol = self.vol_stats['median']
        vol_factor = current_volatility / median_vol if median_vol > 0 else 1.0
        vol_factor = np.clip(vol_factor, 0.5, 2.0)

        # Classify volatility regime
        if current_volatility < self.vol_stats['q25']:
            regime = 'low'
        elif current_volatility > self.vol_stats['q75']:
            regime = 'high'
        else:
            regime = 'normal'

        return {
            'vol_factor': vol_factor,
            'tp_multiplier': vol_factor * self.base_tp_percent,
            'sl_multiplier': vol_factor * self.base_sl_percent,
            'volatility_regime': regime,
            'current_volatility': current_volatility,
            'median_volatility': median_vol
        }


def initialize_risk_manager(df_features: pd.DataFrame) -> DynamicRiskManager:
    """Convenience function to initialize and return risk manager.

    Args:
        df_features: Feature DataFrame with volatility columns

    Returns:
        DynamicRiskManager: Initialized risk manager instance

    Raises:
        TraderError: If initialization fails
    """
    manager = DynamicRiskManager()
    manager.initialize_from_features(df_features)
    return manager
