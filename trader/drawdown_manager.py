"""
Drawdown Management System for USDJPY AI Trader

Implements adaptive risk reduction based on maximum drawdown:
- Monitors peak account value
- Calculates current drawdown
- Automatically scales risk when drawdown exceeds threshold
"""

from typing import Dict, Tuple


class DrawdownManager:
    """Manages trading account drawdown and risk adaptation."""

    def __init__(self, 
                 initial_account: float = 100000,
                 max_drawdown_threshold: float = 5.0,
                 max_absolute_drawdown: float = 10.0):
        """
        Initialize drawdown manager.

        Args:
            initial_account: Starting account size in USD
            max_drawdown_threshold: Maximum % drawdown before risk reduction (default 5%)
            max_absolute_drawdown: Maximum % absolute drawdown allowed (default 10%)
        """
        self.initial_account = initial_account
        self.peak_account = initial_account
        self.current_account = initial_account
        self.max_drawdown_threshold = max_drawdown_threshold
        self.max_absolute_drawdown = max_absolute_drawdown

    def update_account(self, new_account_value: float) -> None:
        """
        Update account value and track peak.

        Args:
            new_account_value: Current account value
        """
        self.current_account = new_account_value
        if new_account_value > self.peak_account:
            self.peak_account = new_account_value

    def get_current_drawdown(self) -> float:
        """
        Calculate current drawdown as %.

        Returns:
            Drawdown percentage (0-100)
        """
        if self.peak_account == 0:
            return 0.0
        return ((self.peak_account - self.current_account) / self.peak_account) * 100

    def get_absolute_drawdown(self) -> float:
        """
        Calculate absolute drawdown from initial account.

        Returns:
            Absolute drawdown percentage
        """
        if self.initial_account == 0:
            return 0.0
        return ((self.initial_account - self.current_account) / self.initial_account) * 100

    def should_reduce_risk(self) -> bool:
        """
        Check if risk should be reduced based on drawdown.

        Returns:
            True if current drawdown exceeds threshold
        """
        current_dd = self.get_current_drawdown()
        absolute_dd = self.get_absolute_drawdown()
        
        # Reduce risk if either threshold is exceeded
        return (current_dd > self.max_drawdown_threshold or 
                absolute_dd > self.max_absolute_drawdown)

    def get_risk_multiplier(self) -> float:
        """
        Calculate risk multiplier based on drawdown level.

        Returns:
            Risk multiplier (0.0 to 1.0)
            1.0 = full risk, 0.5 = half risk, 0.0 = no trading
        """
        current_dd = self.get_current_drawdown()
        absolute_dd = self.get_absolute_drawdown()
        
        # Most restrictive condition determines risk reduction
        max_dd = max(current_dd, absolute_dd)
        
        if max_dd <= self.max_drawdown_threshold:
            return 1.0  # Full risk
        elif max_dd >= self.max_absolute_drawdown:
            return 0.0  # Stop trading
        else:
            # Linear reduction between thresholds
            reduction_range = self.max_absolute_drawdown - self.max_drawdown_threshold
            excess_dd = max_dd - self.max_drawdown_threshold
            multiplier = 1.0 - (excess_dd / reduction_range)
            return max(0.0, multiplier)

    def get_metrics(self) -> Dict:
        """
        Get complete drawdown metrics.

        Returns:
            Dictionary with drawdown statistics
        """
        current_dd = self.get_current_drawdown()
        absolute_dd = self.get_absolute_drawdown()
        risk_mult = self.get_risk_multiplier()

        return {
            'peak_account': self.peak_account,
            'current_account': self.current_account,
            'initial_account': self.initial_account,
            'current_drawdown_pct': current_dd,
            'absolute_drawdown_pct': absolute_dd,
            'max_drawdown_threshold': self.max_drawdown_threshold,
            'max_absolute_drawdown': self.max_absolute_drawdown,
            'risk_multiplier': risk_mult,
            'should_reduce_risk': self.should_reduce_risk()
        }


def create_drawdown_manager(account_size: float = 100000,
                           max_dd_threshold: float = 5.0,
                           max_absolute_dd: float = 10.0) -> DrawdownManager:
    """
    Factory function to create drawdown manager.

    Args:
        account_size: Initial trading account in USD
        max_dd_threshold: Max % drawdown before risk reduction
        max_absolute_dd: Max % absolute drawdown allowed

    Returns:
        Configured DrawdownManager instance
    """
    return DrawdownManager(
        initial_account=account_size,
        max_drawdown_threshold=max_dd_threshold,
        max_absolute_drawdown=max_absolute_dd
    )
