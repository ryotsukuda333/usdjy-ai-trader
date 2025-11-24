"""
Seasonality Manager for USDJPY AI Trader

Analyzes and manages seasonal patterns in FX trading:
- Daily patterns (time of day effects)
- Weekly patterns (day-of-week effects)
- Monthly patterns (beginning/middle/end of month)
- Quarterly and yearly patterns

Reference data from 3 years of USDJPY historical analysis.
"""

from typing import Dict, Tuple, Optional, List
from datetime import datetime
import numpy as np


class SeasonalityManager:
    """Manage seasonal patterns and apply dynamic adjustments based on calendar features."""

    # Based on 3-year USDJPY analysis (2022-2025)

    # Weekly statistics (曜日別統計)
    WEEKLY_STATS = {
        0: {'name': '月', 'count': 156, 'mean': 0.0609, 'std': 0.7176},  # Monday
        1: {'name': '火', 'count': 156, 'mean': 0.0514, 'std': 0.5132},  # Tuesday
        2: {'name': '水', 'count': 155, 'mean': 0.0035, 'std': 0.6183},  # Wednesday
        3: {'name': '木', 'count': 156, 'mean': -0.0060, 'std': 0.6691}, # Thursday
        4: {'name': '金', 'count': 156, 'mean': -0.0237, 'std': 0.7068}, # Friday
        5: {'name': '土', 'count': 0, 'mean': 0.0, 'std': 0.0},          # Saturday (no trading)
        6: {'name': '日', 'count': 0, 'mean': 0.0, 'std': 0.0},          # Sunday (no trading)
    }

    # Monthly statistics (月別統計)
    MONTHLY_STATS = {
        1: {'name': '1月', 'mean': 0.0077, 'std': 0.5925},
        2: {'name': '2月', 'mean': 0.0660, 'std': 0.6498},
        3: {'name': '3月', 'mean': -0.0300, 'std': 0.5858},
        4: {'name': '4月', 'mean': -0.0200, 'std': 0.7115},
        5: {'name': '5月', 'mean': 0.0884, 'std': 0.7488},
        6: {'name': '6月', 'mean': 0.1023, 'std': 0.4759},
        7: {'name': '7月', 'mean': -0.0685, 'std': 0.6404},
        8: {'name': '8月', 'mean': -0.0451, 'std': 0.7380},
        9: {'name': '9月', 'mean': 0.0319, 'std': 0.5411},
        10: {'name': '10月', 'mean': 0.1545, 'std': 0.5384},
        11: {'name': '11月', 'mean': -0.0191, 'std': 0.5899},
        12: {'name': '12月', 'mean': -0.0640, 'std': 0.8714},
    }

    # Yearly statistics (年別統計)
    YEARLY_STATS = {
        2022: {'mean': -0.1743, 'std': 1.0524},
        2023: {'mean': 0.0258, 'std': 0.6253},
        2024: {'mean': 0.0419, 'std': 0.6343},
        2025: {'mean': 0.0012, 'std': 0.6325},
    }

    # Global statistics
    OVERALL_STATS = {
        'mean': 0.0172,
        'std': 0.6485,
        'win_rate': 0.5392,  # 53.92%
    }

    def __init__(self):
        """Initialize SeasonalityManager."""
        self.enabled = True
        self.volatility_threshold = 0.65  # High volatility months threshold

        # Identify high/low volatility periods
        self.high_vol_months = set()  # December, July, August (8月以上)
        self.low_vol_months = set()   # June (best for stability)
        self.high_vol_days = set()    # Friday (金曜日)

        for month, stats in self.MONTHLY_STATS.items():
            if stats['std'] > self.volatility_threshold:
                self.high_vol_months.add(month)
            if stats['std'] < 0.55:
                self.low_vol_months.add(month)

        # High volatility on Friday (金: 0.7068%)
        self.high_vol_days.add(4)

    def get_weekly_adjustment(self, check_date: datetime) -> float:
        """
        Get volatility adjustment factor based on day of week.

        Returns: 1.0 (baseline) to adjust for weekly patterns
        - Monday (月): +0.0609% mean → 1.01x volatility
        - Tuesday (火): +0.0514% mean → 1.00x volatility
        - Wednesday (水): +0.0035% mean → 1.00x volatility
        - Thursday (木): -0.0060% mean → 1.00x volatility
        - Friday (金): -0.0237% mean → 1.08x volatility (high vol)

        Args:
            check_date: datetime to check

        Returns:
            volatility multiplier (1.0 = baseline)
        """
        dayofweek = check_date.weekday()  # 0=Monday, 6=Sunday

        if dayofweek == 4:  # Friday - high volatility
            return 1.08
        elif dayofweek == 0:  # Monday - second highest
            return 1.01
        else:
            return 1.00

    def get_monthly_adjustment(self, check_date: datetime) -> float:
        """
        Get volatility adjustment factor based on month.

        Returns: 1.0 (baseline) to adjust for monthly patterns
        - December (12月): std=0.8714% → 1.35x volatility
        - July (7月): std=0.6404% → 1.02x volatility
        - August (8月): std=0.7380% → 1.14x volatility
        - June (6月): std=0.4759% → 0.75x volatility (low vol)

        Args:
            check_date: datetime to check

        Returns:
            volatility multiplier (1.0 = baseline)
        """
        month = check_date.month
        overall_std = self.OVERALL_STATS['std']

        if month in self.MONTHLY_STATS:
            month_std = self.MONTHLY_STATS[month]['std']
            # Multiplier = month_volatility / baseline_volatility
            multiplier = month_std / overall_std
            # Clamp between 0.7 and 1.3
            return max(0.70, min(1.30, multiplier))

        return 1.00

    def get_seasonal_volatility_adjustment(self, check_date: datetime) -> float:
        """
        Get combined seasonal volatility adjustment.

        Combines weekly and monthly patterns for total seasonal adjustment.

        Args:
            check_date: datetime to check

        Returns:
            combined volatility multiplier
        """
        weekly = self.get_weekly_adjustment(check_date)
        monthly = self.get_monthly_adjustment(check_date)

        # Combined multiplier (multiplicative)
        combined = weekly * monthly

        # Clamp between 0.65 and 1.50
        return max(0.65, min(1.50, combined))

    def get_seasonal_bias(self, check_date: datetime) -> float:
        """
        Get expected direction bias based on seasonal patterns.

        Positive = upward bias, negative = downward bias

        Args:
            check_date: datetime to check

        Returns:
            seasonal bias multiplier (-0.2 to +0.2)
        """
        month = check_date.month

        if month in self.MONTHLY_STATS:
            # Normalize monthly mean
            mean_return = self.MONTHLY_STATS[month]['mean']
            # Bias is mean_return normalized to ±0.2 range
            # Typical range is -0.0685 to +0.1545
            return max(-0.20, min(0.20, mean_return / 0.2))

        return 0.0

    def is_high_volatility_period(self, check_date: datetime) -> bool:
        """Check if date falls in high volatility period."""
        return (check_date.month in self.high_vol_months or
                check_date.weekday() in self.high_vol_days)

    def is_low_volatility_period(self, check_date: datetime) -> bool:
        """Check if date falls in low volatility period."""
        return check_date.month in self.low_vol_months

    def get_recommended_risk_adjustment(self, check_date: datetime) -> float:
        """
        Get recommended position sizing adjustment for the date.

        High volatility periods → reduce position (0.7-0.8)
        Low volatility periods → increase position (1.1-1.2)
        Normal periods → maintain baseline (1.0)

        Args:
            check_date: datetime to check

        Returns:
            position sizing multiplier
        """
        if self.is_high_volatility_period(check_date):
            return 0.75  # 25% position reduction
        elif self.is_low_volatility_period(check_date):
            return 1.15  # 15% position increase
        else:
            return 1.00

    def get_seasonal_quality_score(self, check_date: datetime) -> Tuple[float, Dict]:
        """
        Get overall quality score for trading on this date (0-100).

        Considers:
        - Volatility level (prefer moderate)
        - Weekly pattern (Monday/Tuesday better than Friday)
        - Monthly pattern (June better than December)
        - Seasonal bias

        Args:
            check_date: datetime to check

        Returns:
            (quality_score, details_dict)
        """
        dayofweek = check_date.weekday()
        month = check_date.month

        # Base score
        score = 50.0
        details = {
            'day_of_week': dayofweek,
            'day_name': self.WEEKLY_STATS[dayofweek]['name'],
            'month': month,
            'month_name': self.MONTHLY_STATS.get(month, {}).get('name', 'Unknown'),
            'adjustments': []
        }

        # Weekly adjustments
        if dayofweek == 0:  # Monday
            score += 8
            details['adjustments'].append('月曜日: +8 (positive mean)')
        elif dayofweek == 4:  # Friday
            score -= 10
            details['adjustments'].append('金曜日: -10 (high volatility)')

        # Monthly adjustments
        if month in self.MONTHLY_STATS:
            monthly_mean = self.MONTHLY_STATS[month]['mean']
            monthly_std = self.MONTHLY_STATS[month]['std']

            if monthly_mean > 0.08:
                score += 12
                details['adjustments'].append(f'{self.MONTHLY_STATS[month]["name"]}: +12 (strong positive)')
            elif monthly_mean < -0.05:
                score -= 8
                details['adjustments'].append(f'{self.MONTHLY_STATS[month]["name"]}: -8 (negative bias)')

            if monthly_std > 0.70:
                score -= 10
                details['adjustments'].append(f'{self.MONTHLY_STATS[month]["name"]}: -10 (high volatility)')
            elif monthly_std < 0.55:
                score += 8
                details['adjustments'].append(f'{self.MONTHLY_STATS[month]["name"]}: +8 (low volatility)')

        # Clamp score 0-100
        score = max(0, min(100, score))
        details['final_score'] = round(score, 1)

        return score, details

    def get_seasonality_summary(self, check_date: datetime) -> Dict:
        """
        Get comprehensive seasonality summary for a date.

        Args:
            check_date: datetime to check

        Returns:
            Dictionary with all seasonality information
        """
        quality_score, quality_details = self.get_seasonal_quality_score(check_date)

        return {
            'date': check_date,
            'volatility_adjustment': round(self.get_seasonal_volatility_adjustment(check_date), 3),
            'position_adjustment': round(self.get_recommended_risk_adjustment(check_date), 3),
            'seasonal_bias': round(self.get_seasonal_bias(check_date), 3),
            'quality_score': quality_score,
            'quality_details': quality_details,
            'is_high_vol': self.is_high_volatility_period(check_date),
            'is_low_vol': self.is_low_volatility_period(check_date),
        }
