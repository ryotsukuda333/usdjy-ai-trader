"""
Event-Based Volatility Management

Manages dynamic volatility adjustments based on economic events.
Implements:
- Real-time volatility adjustment
- Risk scaling around events
- Position sizing adjustment
- Trade restriction enforcement

Tasks: 9.2, 9.3
"""

from typing import Optional, Dict, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from trader.economic_calendar import EconomicCalendar, EconomicEvent


class EventVolatilityManager:
    """Manages volatility adjustments based on economic events."""

    def __init__(
        self,
        economic_calendar: EconomicCalendar,
        base_volatility: float = 0.60
    ):
        """
        Initialize event volatility manager.

        Args:
            economic_calendar: EconomicCalendar instance
            base_volatility: Baseline volatility (%)
        """
        self.calendar = economic_calendar
        self.base_volatility = base_volatility
        self.event_history: Dict[str, list] = {}

    def get_event_adjusted_volatility(
        self,
        current_datetime: datetime,
        current_volatility: float
    ) -> float:
        """
        Get volatility adjusted for active economic events.

        Args:
            current_datetime: Current time
            current_volatility: Current measured volatility (%)

        Returns:
            Adjusted volatility considering event impacts
        """
        # Get base volatility adjustment from events
        vol_multiplier = self.calendar.get_volatility_adjustment(current_datetime)

        # Apply adjustment to current volatility
        adjusted_vol = current_volatility * vol_multiplier

        return adjusted_vol

    def get_event_risk_adjustment(
        self,
        current_datetime: datetime,
        restriction_level: str = 'HIGH'
    ) -> float:
        """
        Get risk adjustment factor for position sizing.

        Reduces position size as events approach.

        Args:
            current_datetime: Current time
            restriction_level: Restriction level (HIGH, MEDIUM, LOW)

        Returns:
            Risk adjustment multiplier (0.0-1.0):
            - 1.0: Normal risk
            - 0.5: 50% risk during medium event
            - 0.0: No trading near HIGH events
        """
        # Find active HIGH importance events
        high_events = []
        for event in self.calendar.events:
            if event.importance != 'HIGH':
                continue
            if not event.is_impact_time(current_datetime):
                continue
            high_events.append(event)

        # If HIGH event is active and restriction is HIGH, block trading
        if high_events and restriction_level == 'HIGH':
            return 0.0

        # Check MEDIUM importance events
        medium_events = []
        for event in self.calendar.events:
            if event.importance != 'MEDIUM':
                continue
            if not event.is_impact_time(current_datetime):
                continue
            medium_events.append(event)

        # Reduce risk during MEDIUM events
        if medium_events:
            # Find closest event and calculate proximity factor
            closest_event = min(medium_events, key=lambda e: abs((e.event_date - current_datetime).total_seconds()))
            time_to_event = (closest_event.event_date - current_datetime).total_seconds() / 3600.0

            # 1 hour before/after = 50% risk, further away = higher risk
            if abs(time_to_event) < 1.0:
                return 0.5
            elif abs(time_to_event) < 2.0:
                return 0.7
            else:
                return 1.0

        return 1.0

    def should_trade(
        self,
        current_datetime: datetime,
        restriction_level: str = 'HIGH'
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if trading is allowed at current time.

        Args:
            current_datetime: Current time
            restriction_level: Restriction level (HIGH, MEDIUM, LOW)

        Returns:
            Tuple of (should_trade: bool, reason: Optional[str])
        """
        # Check for active HIGH importance events
        for event in self.calendar.events:
            if event.importance == 'HIGH' and event.is_impact_time(current_datetime):
                reason = f"HIGH event '{event.name}' active until {event.get_impact_window()[1]}"
                return False, reason

        # Check for MEDIUM importance events if restriction is strict
        if restriction_level == 'MEDIUM':
            for event in self.calendar.events:
                if event.importance in ['HIGH', 'MEDIUM'] and event.is_impact_time(current_datetime):
                    reason = f"MEDIUM event '{event.name}' active until {event.get_impact_window()[1]}"
                    return False, reason

        return True, None

    def get_next_event_info(
        self,
        current_datetime: datetime,
        hours_ahead: float = 24.0
    ) -> Optional[Dict]:
        """
        Get information about the next upcoming event.

        Args:
            current_datetime: Current time
            hours_ahead: Look ahead window

        Returns:
            Event info dict or None if no upcoming events
        """
        next_event = self.calendar.get_next_event(current_datetime, hours_ahead)

        if not next_event:
            return None

        time_until = (next_event.event_date - current_datetime).total_seconds() / 3600.0

        return {
            'name': next_event.name,
            'country': next_event.country,
            'importance': next_event.importance,
            'time_until_hours': time_until,
            'event_datetime': next_event.event_date,
            'expected_impact': next_event.expected_impact,
            'volatility_adjustment': self.calendar.get_volatility_adjustment(next_event.event_date)
        }

    def get_trading_window_quality(
        self,
        current_datetime: datetime,
        hours_ahead: float = 4.0
    ) -> Dict:
        """
        Assess trading window quality based on upcoming events.

        Args:
            current_datetime: Current time
            hours_ahead: Hours to evaluate ahead

        Returns:
            Quality assessment dict
        """
        # Check for events in the next N hours
        end_time = current_datetime + pd.Timedelta(hours=hours_ahead)
        upcoming_events = self.calendar.get_events_in_range(current_datetime, end_time)

        if not upcoming_events:
            return {
                'quality': 'EXCELLENT',
                'events_ahead': 0,
                'risk_adjustment': 1.0,
                'recommendation': 'Clear trading window'
            }

        # Categorize upcoming events
        high_events = [e for e in upcoming_events if e.importance == 'HIGH']
        medium_events = [e for e in upcoming_events if e.importance == 'MEDIUM']

        if high_events:
            return {
                'quality': 'POOR',
                'events_ahead': len(upcoming_events),
                'high_events': len(high_events),
                'medium_events': len(medium_events),
                'risk_adjustment': 0.0,
                'recommendation': 'HIGH impact event upcoming - avoid trading'
            }

        if medium_events:
            return {
                'quality': 'MODERATE',
                'events_ahead': len(upcoming_events),
                'high_events': 0,
                'medium_events': len(medium_events),
                'risk_adjustment': 0.7,
                'recommendation': 'MEDIUM event upcoming - reduce position size'
            }

        return {
            'quality': 'GOOD',
            'events_ahead': len(upcoming_events),
            'high_events': 0,
            'medium_events': 0,
            'risk_adjustment': 0.9,
            'recommendation': 'Safe to trade, minor events ahead'
        }

    def record_event_impact(
        self,
        event_name: str,
        actual_volatility_change: float,
        actual_move_pips: float
    ) -> None:
        """
        Record actual impact of an event for analysis.

        Args:
            event_name: Name of the event
            actual_volatility_change: Actual volatility change (%)
            actual_move_pips: Actual price movement (pips)
        """
        if event_name not in self.event_history:
            self.event_history[event_name] = []

        self.event_history[event_name].append({
            'timestamp': datetime.utcnow(),
            'volatility_change': actual_volatility_change,
            'price_move_pips': actual_move_pips
        })

    def get_event_impact_statistics(self, event_name: Optional[str] = None) -> Dict:
        """
        Get statistics about recorded event impacts.

        Args:
            event_name: Specific event to analyze, or None for all

        Returns:
            Statistics dict
        """
        if event_name:
            if event_name not in self.event_history:
                return {'events_recorded': 0}

            impacts = self.event_history[event_name]
            volatility_changes = [i['volatility_change'] for i in impacts]
            price_moves = [i['price_move_pips'] for i in impacts]

            return {
                'event_name': event_name,
                'occurrences': len(impacts),
                'avg_volatility_change': np.mean(volatility_changes),
                'std_volatility_change': np.std(volatility_changes),
                'avg_price_move': np.mean(price_moves),
                'max_price_move': max(price_moves),
                'min_price_move': min(price_moves)
            }

        else:
            # All events
            total_impacts = sum(len(impacts) for impacts in self.event_history.values())
            return {
                'total_events_recorded': total_impacts,
                'unique_events': len(self.event_history),
                'events': {
                    name: self.get_event_impact_statistics(name)
                    for name in self.event_history.keys()
                }
            }


def create_event_volatility_manager(
    economic_calendar: EconomicCalendar,
    base_volatility: float = 0.60
) -> EventVolatilityManager:
    """
    Factory function to create EventVolatilityManager.

    Args:
        economic_calendar: EconomicCalendar instance
        base_volatility: Baseline volatility (%)

    Returns:
        Configured EventVolatilityManager instance
    """
    return EventVolatilityManager(economic_calendar, base_volatility)
