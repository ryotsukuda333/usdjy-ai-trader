"""
Session Manager for USDJPY AI Trader

Manages trading session information and applies session-based risk adjustments:
- Tokyo Session: 00:00-09:00 UTC (09:00-18:00 JST) - Low volatility, 1% risk
- London Session: 08:00-17:00 UTC (17:00-02:00 JST next day) - Medium volatility, 3% risk
- New York Session: 13:00-22:00 UTC (22:00-07:00 JST next day) - High volatility, 5% risk

Reference data from Step 8 session analysis.
"""

from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta


class SessionManager:
    """Manage trading sessions and apply session-based risk adjustments."""

    # Session definitions (based on Step 8 analysis)
    SESSIONS = {
        'TOKYO': {
            'name': 'Tokyo',
            'utc_start': 0,  # 00:00 UTC
            'utc_end': 9,     # 09:00 UTC
            'risk_percent': 1.0,
            'volatility': 0.6383,  # Low volatility
            'mean_return': 0.0168
        },
        'LONDON': {
            'name': 'London',
            'utc_start': 8,   # 08:00 UTC
            'utc_end': 17,    # 17:00 UTC
            'risk_percent': 3.0,
            'volatility': 0.6481,  # Medium volatility
            'mean_return': 0.0192
        },
        'NEW_YORK': {
            'name': 'New York',
            'utc_start': 13,  # 13:00 UTC
            'utc_end': 22,    # 22:00 UTC
            'risk_percent': 5.0,
            'volatility': 0.6658,  # High volatility
            'mean_return': 0.0168
        }
    }

    def __init__(self):
        """Initialize SessionManager."""
        self.enabled = True

    def get_current_session(self, check_date: datetime) -> Dict:
        """
        Get current trading session information based on datetime (UTC).

        Args:
            check_date: datetime to check (interpreted as UTC)

        Returns:
            Dictionary with session information
        """
        hour_utc = check_date.hour

        # Determine primary session
        current_session = 'TOKYO'  # Default
        active_sessions = []

        # Check Tokyo Session (00:00-09:00 UTC)
        if hour_utc >= 0 and hour_utc < 9:
            active_sessions.append('TOKYO')

        # Check London Session (08:00-17:00 UTC)
        if hour_utc >= 8 and hour_utc < 17:
            active_sessions.append('LONDON')

        # Check New York Session (13:00-22:00 UTC)
        if hour_utc >= 13 and hour_utc < 22:
            active_sessions.append('NEW_YORK')

        # If no session active (22:00-00:00 UTC), mark as afterhours
        if not active_sessions:
            active_sessions = ['TOKYO']  # Next Tokyo session
            current_session = 'TOKYO'
        else:
            # Use highest risk session if overlapping
            risk_order = ['TOKYO', 'LONDON', 'NEW_YORK']
            for session in risk_order:
                if session in active_sessions:
                    current_session = session

        session_info = self.SESSIONS[current_session].copy()
        session_info['active_sessions'] = active_sessions
        session_info['is_high_volatility'] = current_session == 'NEW_YORK'
        session_info['is_low_volatility'] = current_session == 'TOKYO'
        session_info['session_name'] = current_session

        return session_info

    def get_session_volatility(self, check_date: datetime) -> float:
        """
        Get session-based volatility for the datetime.

        Args:
            check_date: datetime to check

        Returns:
            session volatility estimate
        """
        session_info = self.get_current_session(check_date)
        return session_info['volatility']

    def get_session_risk_multiplier(self, check_date: datetime) -> float:
        """
        Get session-based risk multiplier.

        Returns: 1.0 (baseline) to scale position sizing
        - Tokyo: 0.2x (1/5 risk)
        - London: 0.6x (3/5 risk)
        - New York: 1.0x (5/5 baseline risk)

        Args:
            check_date: datetime to check

        Returns:
            risk multiplier
        """
        session_info = self.get_current_session(check_date)
        risk_percent = session_info['risk_percent']

        # Normalize to New York = 1.0
        return risk_percent / 5.0

    def get_session_quality_score(self, check_date: datetime) -> Tuple[float, Dict]:
        """
        Get quality score for trading in current session (0-100).

        Args:
            check_date: datetime to check

        Returns:
            (quality_score, details_dict)
        """
        session_info = self.get_current_session(check_date)
        current_session = session_info['session_name']

        # Base scores
        session_scores = {
            'TOKYO': 60,      # Stable but low volatility
            'LONDON': 75,     # Good balance
            'NEW_YORK': 70    # High volatility (slightly lower)
        }

        score = session_scores.get(current_session, 50)

        details = {
            'session': current_session,
            'session_name': session_info['name'],
            'risk_percent': session_info['risk_percent'],
            'volatility': session_info['volatility'],
            'quality_score': score,
            'recommendations': []
        }

        # Add recommendations
        if current_session == 'TOKYO':
            details['recommendations'].append('Good for conservative trading')
            details['recommendations'].append('Low volatility environment')
        elif current_session == 'LONDON':
            details['recommendations'].append('Balanced volatility')
            details['recommendations'].append('Good for range-bound strategies')
        elif current_session == 'NEW_YORK':
            details['recommendations'].append('High volatility')
            details['recommendations'].append('Best for volatility-based strategies')

        return score, details

    def is_major_session_overlap(self, check_date: datetime) -> bool:
        """
        Check if current time is during major session overlap.

        Major overlaps:
        - 08:00-09:00 UTC: Tokyo-London overlap
        - 13:00-17:00 UTC: London-New York overlap

        Args:
            check_date: datetime to check

        Returns:
            True if in overlap period
        """
        hour_utc = check_date.hour

        # Tokyo-London overlap (08:00-09:00 UTC)
        if hour_utc == 8:
            return True

        # London-New York overlap (13:00-17:00 UTC)
        if 13 <= hour_utc < 17:
            return True

        return False

    def get_next_session_change(self, check_date: datetime) -> Tuple[datetime, str]:
        """
        Get time of next session change and the new session.

        Args:
            check_date: datetime to check

        Returns:
            (datetime_of_next_session_change, new_session_name)
        """
        hour_utc = check_date.hour
        minute_utc = check_date.minute

        if hour_utc < 8:
            # Next session: London at 08:00 UTC
            next_change = check_date.replace(hour=8, minute=0, second=0, microsecond=0)
            next_session = 'LONDON'
        elif hour_utc < 9:
            # Next session: New York at 13:00 UTC
            next_change = check_date.replace(hour=13, minute=0, second=0, microsecond=0)
            next_session = 'NEW_YORK'
        elif hour_utc < 13:
            # Next session: New York at 13:00 UTC
            next_change = check_date.replace(hour=13, minute=0, second=0, microsecond=0)
            next_session = 'NEW_YORK'
        elif hour_utc < 17:
            # Next session: Tokyo at 00:00 UTC (next day)
            next_change = check_date.replace(hour=0, minute=0, second=0, microsecond=0)
            next_change += timedelta(days=1)
            next_session = 'TOKYO'
        else:
            # Next session: Tokyo at 00:00 UTC (next day)
            next_change = check_date.replace(hour=0, minute=0, second=0, microsecond=0)
            next_change += timedelta(days=1)
            next_session = 'TOKYO'

        return next_change, next_session
