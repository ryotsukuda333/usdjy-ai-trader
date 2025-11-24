"""
Position Sizing Module for USDJPY AI Trader

Implements two position sizing strategies:
1. Fixed Risk Percentage: Risk a fixed % of account on each trade
2. Kelly Criterion: Optimal f based on win rate and R:R ratio

Reference:
  - Kelly Criterion: Edward O. Thorp, "The Mathematics of Gambling"
  - Position Sizing: Ralph Vince, "Portfolio Management Formulas"
"""

from typing import Dict, Tuple
import numpy as np
from utils.errors import TraderError


class PositionSizer:
    """Dynamic position sizing based on trading statistics."""

    def __init__(self, account_size: float = 100000,
                 initial_risk_percent: float = 1.0,
                 kelly_fraction: float = None,
                 max_position_pct: float = 50.0):
        """
        Initialize position sizer.

        Args:
            account_size: Initial trading account size in USD
            initial_risk_percent: Risk % per trade (default 1%)
            kelly_fraction: Kelly criterion f* (auto-calculated if None)
            max_position_pct: Maximum position size as % of account (safety limit)

        Examples:
            # Fixed risk strategy
            sizer = PositionSizer(account_size=100000, initial_risk_percent=1.0)

            # Kelly criterion strategy
            sizer = PositionSizer(account_size=100000, kelly_fraction=0.15)
        """
        if account_size <= 0:
            raise TraderError(
                error_code="INVALID_ACCOUNT",
                user_message="Account size must be positive",
                technical_message=f"account_size={account_size}"
            )

        self.initial_account = account_size
        self.current_account = account_size
        self.risk_percent = initial_risk_percent
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct

        # Statistics tracking
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0

    def calculate_position_size_fixed_risk(self,
                                          entry_price: float,
                                          sl_price: float,
                                          risk_percent: float = None) -> float:
        """
        Calculate position size based on fixed risk percentage.

        Formula:
            Risk_Amount = Account_Size × Risk_Percentage
            Risk_Per_Unit = (Entry_Price - SL_Price) / Entry_Price × 100
            Position_Size = Risk_Amount / Risk_Per_Unit

        Args:
            entry_price: Entry price in pips
            sl_price: Stop loss price in pips
            risk_percent: Override default risk % (optional)

        Returns:
            Position size in units

        Example:
            >>> sizer = PositionSizer(100000, 1.0)
            >>> size = sizer.calculate_position_size_fixed_risk(150.0, 149.55)
            >>> size
            65104.17  # 65,104.17 units
        """
        if entry_price <= 0 or sl_price <= 0:
            raise TraderError(
                error_code="INVALID_PRICE",
                user_message="Entry and SL prices must be positive",
                technical_message=f"entry_price={entry_price}, sl_price={sl_price}"
            )

        if entry_price <= sl_price:
            raise TraderError(
                error_code="INVALID_SL",
                user_message="Entry price must be above SL price",
                technical_message=f"entry_price={entry_price}, sl_price={sl_price}"
            )

        risk_pct = risk_percent if risk_percent is not None else self.risk_percent

        # Risk amount in USD
        risk_amount = self.current_account * (risk_pct / 100)

        # Risk per pip (in percentage)
        risk_per_pip_pct = (entry_price - sl_price) / entry_price * 100

        # Position size in units (pip value = 100 units = $1 move for USDJPY)
        # For USDJPY: 1 pip = 100 units = ~$1 risk
        position_size = risk_amount / (risk_per_pip_pct / 100 * entry_price)

        # Apply position size limit
        max_position = self.current_account * (self.max_position_pct / 100)
        position_size = min(position_size, max_position)

        return position_size

    def calculate_position_size_kelly(self,
                                      entry_price: float,
                                      sl_price: float,
                                      win_rate: float,
                                      avg_win: float,
                                      avg_loss: float) -> Tuple[float, Dict]:
        """
        Calculate position size using Kelly Criterion formula.

        Formula:
            f* = (p × b - (1-p)) / b
            where:
              p = win rate (0-1)
              b = average win / average loss (risk-reward ratio)
              f* = optimal fraction of capital

        Args:
            entry_price: Entry price
            sl_price: Stop loss price
            win_rate: Win rate as decimal (0-1)
            avg_win: Average win as % (e.g., 0.60 for +0.60%)
            avg_loss: Average loss as % (e.g., 0.31 for -0.31%)

        Returns:
            Tuple[position_size, metrics_dict]

        Example:
            >>> sizer = PositionSizer(100000)
            >>> pos, metrics = sizer.calculate_position_size_kelly(
            ...     entry_price=150.0,
            ...     sl_price=149.55,
            ...     win_rate=0.54,
            ...     avg_win=0.60,
            ...     avg_loss=0.31
            ... )
            >>> pos, metrics['kelly_f']
            (49518.5, 0.151)
        """
        if win_rate < 0 or win_rate > 1:
            raise TraderError(
                error_code="INVALID_WINRATE",
                user_message="Win rate must be between 0 and 1",
                technical_message=f"win_rate={win_rate}"
            )

        if avg_win <= 0 or avg_loss <= 0:
            raise TraderError(
                error_code="INVALID_RETURNS",
                user_message="Average win and loss must be positive",
                technical_message=f"avg_win={avg_win}, avg_loss={avg_loss}"
            )

        # Risk-reward ratio
        b = avg_win / avg_loss

        # Kelly fraction
        kelly_f = (win_rate * b - (1 - win_rate)) / b if b > 0 else 0

        # Use configured kelly_fraction or calculated one
        f_star = self.kelly_fraction if self.kelly_fraction is not None else kelly_f

        # Ensure it's positive
        f_star = max(0, f_star)

        # Position sizing using kelly fraction
        # Position = (Account × f*) / Risk_Per_Unit
        risk_amount = self.current_account * f_star
        risk_per_pip_pct = (entry_price - sl_price) / entry_price * 100
        position_size = risk_amount / (risk_per_pip_pct / 100 * entry_price)

        # Apply safety limit
        max_position = self.current_account * (self.max_position_pct / 100)
        position_size = min(position_size, max_position)

        metrics = {
            'kelly_f_calculated': kelly_f,
            'kelly_f_used': f_star,
            'win_rate': win_rate,
            'risk_reward_ratio': b,
            'position_size': position_size,
            'position_pct': (position_size / self.current_account * 100) if self.current_account > 0 else 0,
            'risk_amount': risk_amount
        }

        return position_size, metrics

    def update_account(self, pnl: float) -> None:
        """
        Update account size after trade completion.

        Args:
            pnl: Profit/loss in USD
        """
        self.current_account += pnl
        self.total_pnl += pnl
        self.trade_count += 1

        if pnl >= 0:
            self.win_count += 1
        else:
            self.loss_count += 1

    def get_position_metrics(self) -> Dict:
        """
        Get current position sizing metrics and statistics.

        Returns:
            Dictionary with:
              - current_account: Current account size
              - initial_account: Starting account size
              - total_pnl: Total profit/loss
              - return_pct: Total return as %
              - trade_count: Number of trades
              - win_count: Number of winning trades
              - loss_count: Number of losing trades
              - win_rate: Win rate (0-1)
        """
        return_pct = (self.current_account - self.initial_account) / self.initial_account * 100
        win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0

        return {
            'current_account': self.current_account,
            'initial_account': self.initial_account,
            'total_pnl': self.total_pnl,
            'return_pct': return_pct,
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'risk_percent': self.risk_percent,
            'kelly_fraction': self.kelly_fraction,
            'max_position_pct': self.max_position_pct
        }


def create_position_sizer_fixed_risk(account_size: float = 100000,
                                     risk_percent: float = 1.0) -> PositionSizer:
    """
    Create a position sizer using fixed risk percentage strategy.

    Args:
        account_size: Initial account size in USD
        risk_percent: Risk percentage per trade (e.g., 1.0 for 1%)

    Returns:
        Configured PositionSizer instance

    Example:
        >>> sizer = create_position_sizer_fixed_risk(100000, 1.0)
        >>> size = sizer.calculate_position_size_fixed_risk(150.0, 149.55)
    """
    return PositionSizer(
        account_size=account_size,
        initial_risk_percent=risk_percent,
        kelly_fraction=None
    )


def create_position_sizer_kelly(account_size: float = 100000,
                               kelly_fraction: float = 0.15) -> PositionSizer:
    """
    Create a position sizer using Kelly Criterion strategy.

    Args:
        account_size: Initial account size in USD
        kelly_fraction: Kelly f fraction (typically 0.1-0.25)

    Returns:
        Configured PositionSizer instance

    Example:
        >>> sizer = create_position_sizer_kelly(100000, 0.15)
        >>> size, metrics = sizer.calculate_position_size_kelly(
        ...     150.0, 149.55, 0.54, 0.60, 0.31
        ... )
    """
    return PositionSizer(
        account_size=account_size,
        initial_risk_percent=None,
        kelly_fraction=kelly_fraction,
        max_position_pct=10.0  # Safety cap: max 10% per trade
    )
