"""
Economic Calendar for USDJPY Trading

Implements economic indicator event detection and management.
Includes:
- Major US economic indicators (NFP, CPI, PPI, Fed Funds Rate)
- Japanese economic indicators (Nikkei, Unemployment, Trade Balance)
- Event timing detection
- Volatility adjustment based on event proximity

Tasks: 9.1, 9.2, 9.3
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd


class EconomicEvent:
    """Represents a single economic indicator event."""

    def __init__(
        self,
        name: str,
        country: str,
        importance: str,
        expected_impact: float,
        event_date: datetime,
        hours_before_event: float = 1.0,
        hours_after_event: float = 2.0
    ):
        """
        Initialize economic event.

        Args:
            name: Event name (e.g., "Non-Farm Payrolls")
            country: Country code (US, JP)
            importance: Importance level (HIGH, MEDIUM, LOW)
            expected_impact: Expected impact on volatility (0.0-2.0)
            event_date: Event datetime (UTC)
            hours_before_event: Hours before event to avoid trading
            hours_after_event: Hours after event to avoid trading
        """
        self.name = name
        self.country = country
        self.importance = importance
        self.expected_impact = expected_impact
        self.event_date = event_date
        self.hours_before_event = hours_before_event
        self.hours_after_event = hours_after_event

    def get_impact_window(self) -> Tuple[datetime, datetime]:
        """Get the time window affected by this event."""
        start = self.event_date - timedelta(hours=self.hours_before_event)
        end = self.event_date + timedelta(hours=self.hours_after_event)
        return start, end

    def is_impact_time(self, check_datetime: datetime) -> bool:
        """Check if a datetime is within the event impact window."""
        start, end = self.get_impact_window()
        return start <= check_datetime <= end

    def get_volatility_multiplier(self, check_datetime: datetime) -> float:
        """
        Get volatility adjustment multiplier based on proximity to event.

        Args:
            check_datetime: Time to check against event

        Returns:
            Volatility multiplier (1.0 = no impact, 1.5+ = high impact)
        """
        if not self.is_impact_time(check_datetime):
            return 1.0

        # Calculate proximity to event
        time_to_event = (self.event_date - check_datetime).total_seconds() / 3600.0
        hours_window = self.hours_before_event + self.hours_after_event

        # Closer to event = higher multiplier
        if time_to_event >= 0:  # Before event
            proximity_factor = 1.0 - (time_to_event / self.hours_before_event) if time_to_event < self.hours_before_event else 0.0
        else:  # After event
            proximity_factor = 1.0 + (time_to_event / self.hours_after_event) if time_to_event > -self.hours_after_event else 0.0

        # Maximum multiplier at event time
        max_multiplier = 1.0 + self.expected_impact
        return 1.0 + (proximity_factor * self.expected_impact)


class EconomicCalendar:
    """Manages economic events and their impact on trading."""

    # Major US Economic Indicators (Monthly/Weekly)
    US_INDICATORS = {
        'NFP': {
            'name': 'Non-Farm Payrolls',
            'frequency': 'monthly',
            'release_day': 'first_friday',
            'release_time_utc': 13.5,  # 13:30 UTC = 22:30 JST
            'importance': 'HIGH',
            'expected_impact': 0.80
        },
        'CPI': {
            'name': 'Consumer Price Index',
            'frequency': 'monthly',
            'release_day': 'second_week',
            'release_time_utc': 13.0,  # 13:00 UTC = 22:00 JST
            'importance': 'HIGH',
            'expected_impact': 0.70
        },
        'PPI': {
            'name': 'Producer Price Index',
            'frequency': 'monthly',
            'release_day': 'second_week',
            'release_time_utc': 13.0,
            'importance': 'MEDIUM',
            'expected_impact': 0.50
        },
        'FED_FUNDS_DECISION': {
            'name': 'Fed Funds Rate Decision',
            'frequency': 'quarterly',
            'release_day': 'quarterly_decision',
            'release_time_utc': 18.0,  # 18:00 UTC = 03:00 JST next day
            'importance': 'HIGH',
            'expected_impact': 1.00
        },
        'ISM_MANUFACTURING': {
            'name': 'ISM Manufacturing PMI',
            'frequency': 'monthly',
            'release_day': 'first_business_day',
            'release_time_utc': 14.0,
            'importance': 'MEDIUM',
            'expected_impact': 0.60
        }
    }

    # Major Japanese Economic Indicators
    JP_INDICATORS = {
        'NIKKEI_OPEN': {
            'name': 'Nikkei 225 Opening',
            'frequency': 'daily',
            'release_day': 'daily',
            'release_time_jst': 9.0,  # 09:00 JST = 00:00 UTC
            'importance': 'MEDIUM',
            'expected_impact': 0.40
        },
        'UNEMPLOYMENT': {
            'name': 'Unemployment Rate',
            'frequency': 'monthly',
            'release_day': 'last_friday',
            'release_time_jst': 14.3,  # 14:30 JST = 05:30 UTC
            'importance': 'MEDIUM',
            'expected_impact': 0.50
        },
        'TRADE_BALANCE': {
            'name': 'Trade Balance',
            'frequency': 'monthly',
            'release_day': 'monthly_data_day',
            'release_time_jst': 8.5,  # 08:30 JST = 23:30 UTC previous day
            'importance': 'LOW',
            'expected_impact': 0.30
        },
        'BOJ_DECISION': {
            'name': 'BOJ Interest Rate Decision',
            'frequency': 'quarterly',
            'release_day': 'quarterly_decision',
            'release_time_jst': 16.0,  # 16:00 JST = 07:00 UTC
            'importance': 'HIGH',
            'expected_impact': 1.00
        }
    }

    def __init__(self):
        """Initialize economic calendar."""
        self.events: List[EconomicEvent] = []

    def add_event(self, event: EconomicEvent) -> None:
        """Add an economic event to the calendar."""
        self.events.append(event)

    def get_events_in_range(
        self,
        start_date: datetime,
        end_date: datetime,
        country: Optional[str] = None,
        importance: Optional[str] = None
    ) -> List[EconomicEvent]:
        """
        Get events within a date range.

        Args:
            start_date: Start of range
            end_date: End of range
            country: Filter by country (US, JP)
            importance: Filter by importance (HIGH, MEDIUM, LOW)

        Returns:
            List of matching events
        """
        filtered_events = []

        for event in self.events:
            # Check date range
            if not (start_date <= event.event_date <= end_date):
                continue

            # Check country filter
            if country and event.country != country:
                continue

            # Check importance filter
            if importance and event.importance != importance:
                continue

            filtered_events.append(event)

        return filtered_events

    def get_next_event(
        self,
        current_datetime: datetime,
        hours_ahead: float = 24.0,
        country: Optional[str] = None
    ) -> Optional[EconomicEvent]:
        """
        Get the next upcoming event.

        Args:
            current_datetime: Current time
            hours_ahead: Look ahead window
            country: Filter by country

        Returns:
            Next event or None
        """
        look_ahead = current_datetime + timedelta(hours=hours_ahead)
        upcoming = self.get_events_in_range(current_datetime, look_ahead, country=country)

        if not upcoming:
            return None

        # Return the closest event
        return min(upcoming, key=lambda e: (e.event_date - current_datetime).total_seconds())

    def is_trading_restricted(
        self,
        check_datetime: datetime,
        restriction_level: str = 'HIGH'
    ) -> bool:
        """
        Check if trading is restricted at a given time.

        Args:
            check_datetime: Time to check
            restriction_level: Restriction level (HIGH, MEDIUM, LOW)

        Returns:
            True if trading is restricted
        """
        for event in self.events:
            # Only check HIGH importance events unless lower level specified
            if restriction_level == 'HIGH' and event.importance != 'HIGH':
                continue
            if restriction_level == 'MEDIUM' and event.importance not in ['HIGH', 'MEDIUM']:
                continue

            if event.is_impact_time(check_datetime):
                return True

        return False

    def get_volatility_adjustment(
        self,
        check_datetime: datetime
    ) -> float:
        """
        Get combined volatility adjustment for all active events.

        Args:
            check_datetime: Time to check

        Returns:
            Volatility multiplier (1.0 = baseline, 1.5+ = elevated)
        """
        max_multiplier = 1.0

        for event in self.events:
            multiplier = event.get_volatility_multiplier(check_datetime)
            max_multiplier = max(max_multiplier, multiplier)

        return max_multiplier

    def get_event_statistics(self) -> Dict:
        """Get statistics about loaded events."""
        if not self.events:
            return {
                'total_events': 0,
                'us_events': 0,
                'jp_events': 0,
                'high_importance': 0,
                'date_range': None
            }

        us_count = sum(1 for e in self.events if e.country == 'US')
        jp_count = sum(1 for e in self.events if e.country == 'JP')
        high_count = sum(1 for e in self.events if e.importance == 'HIGH')

        dates = [e.event_date for e in self.events]

        return {
            'total_events': len(self.events),
            'us_events': us_count,
            'jp_events': jp_count,
            'high_importance': high_count,
            'date_range': f"{min(dates)} to {max(dates)}"
        }


def create_economic_calendar() -> EconomicCalendar:
    """
    Factory function to create EconomicCalendar.

    Returns:
        Configured EconomicCalendar instance
    """
    return EconomicCalendar()
