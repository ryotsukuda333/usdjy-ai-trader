"""
Load Economic Events for USDJPY Trading

Loads major economic events from 2023-2025 for US and Japan.
Populates EconomicCalendar with historical events.

Tasks: 9.1
"""

from datetime import datetime, timedelta
from trader.economic_calendar import EconomicCalendar, EconomicEvent


def load_us_nfp_events(calendar: EconomicCalendar) -> None:
    """Load US Non-Farm Payrolls events (First Friday of each month, 13:30 UTC = 22:30 JST)."""
    nfp_dates = [
        # 2023
        datetime(2023, 1, 6, 13, 30),   # Jan
        datetime(2023, 2, 3, 13, 30),   # Feb
        datetime(2023, 3, 10, 12, 30),  # Mar (Daylight Saving)
        datetime(2023, 4, 7, 12, 30),   # Apr
        datetime(2023, 5, 5, 12, 30),   # May
        datetime(2023, 6, 2, 12, 30),   # Jun
        datetime(2023, 7, 7, 12, 30),   # Jul
        datetime(2023, 8, 4, 12, 30),   # Aug
        datetime(2023, 9, 1, 12, 30),   # Sep
        datetime(2023, 10, 6, 12, 30),  # Oct
        datetime(2023, 11, 3, 13, 30),  # Nov
        datetime(2023, 12, 8, 13, 30),  # Dec
        # 2024
        datetime(2024, 1, 5, 13, 30),   # Jan
        datetime(2024, 2, 2, 13, 30),   # Feb
        datetime(2024, 3, 8, 12, 30),   # Mar
        datetime(2024, 4, 5, 12, 30),   # Apr
        datetime(2024, 5, 3, 12, 30),   # May
        datetime(2024, 6, 7, 12, 30),   # Jun
        datetime(2024, 7, 5, 12, 30),   # Jul
        datetime(2024, 8, 2, 12, 30),   # Aug
        datetime(2024, 9, 6, 12, 30),   # Sep
        datetime(2024, 10, 4, 12, 30),  # Oct
        datetime(2024, 11, 1, 13, 30),  # Nov
        datetime(2024, 12, 6, 13, 30),  # Dec
        # 2025
        datetime(2025, 1, 10, 13, 30),  # Jan
        datetime(2025, 2, 7, 13, 30),   # Feb
        datetime(2025, 3, 7, 13, 30),   # Mar
    ]

    for date in nfp_dates:
        event = EconomicEvent(
            name='Non-Farm Payrolls (US)',
            country='US',
            importance='HIGH',
            expected_impact=0.80,
            event_date=date,
            hours_before_event=1.0,
            hours_after_event=2.0
        )
        calendar.add_event(event)


def load_us_cpi_events(calendar: EconomicCalendar) -> None:
    """Load US CPI events (Usually 2nd week, 13:00 UTC = 22:00 JST)."""
    cpi_dates = [
        # 2023
        datetime(2023, 1, 12, 13, 0),   # Jan
        datetime(2023, 2, 14, 13, 0),   # Feb
        datetime(2023, 3, 10, 12, 0),   # Mar
        datetime(2023, 4, 12, 12, 0),   # Apr
        datetime(2023, 5, 10, 12, 0),   # May
        datetime(2023, 6, 13, 12, 0),   # Jun
        datetime(2023, 7, 12, 12, 0),   # Jul
        datetime(2023, 8, 9, 12, 0),    # Aug
        datetime(2023, 9, 13, 12, 0),   # Sep
        datetime(2023, 10, 11, 12, 0),  # Oct
        datetime(2023, 11, 14, 13, 0),  # Nov
        datetime(2023, 12, 12, 13, 0),  # Dec
        # 2024
        datetime(2024, 1, 10, 13, 0),   # Jan
        datetime(2024, 2, 13, 13, 0),   # Feb
        datetime(2024, 3, 12, 12, 0),   # Mar
        datetime(2024, 4, 10, 12, 0),   # Apr
        datetime(2024, 5, 14, 12, 0),   # May
        datetime(2024, 6, 12, 12, 0),   # Jun
        datetime(2024, 7, 10, 12, 0),   # Jul
        datetime(2024, 8, 13, 12, 0),   # Aug
        datetime(2024, 9, 11, 12, 0),   # Sep
        datetime(2024, 10, 9, 12, 0),   # Oct
        datetime(2024, 11, 12, 13, 0),  # Nov
        datetime(2024, 12, 10, 13, 0),  # Dec
        # 2025
        datetime(2025, 1, 14, 13, 0),   # Jan
        datetime(2025, 2, 11, 13, 0),   # Feb
        datetime(2025, 3, 12, 13, 0),   # Mar
    ]

    for date in cpi_dates:
        event = EconomicEvent(
            name='CPI (US)',
            country='US',
            importance='HIGH',
            expected_impact=0.70,
            event_date=date,
            hours_before_event=0.5,
            hours_after_event=1.5
        )
        calendar.add_event(event)


def load_fed_decision_events(calendar: EconomicCalendar) -> None:
    """Load Federal Reserve FOMC Decision events (8 times per year, 18:00 UTC = 03:00 JST next day)."""
    fed_dates = [
        # 2023
        datetime(2023, 2, 1, 18, 0),    # Feb
        datetime(2023, 3, 22, 18, 0),   # Mar
        datetime(2023, 5, 3, 18, 0),    # May
        datetime(2023, 6, 14, 18, 0),   # Jun
        datetime(2023, 7, 26, 18, 0),   # Jul
        datetime(2023, 9, 20, 18, 0),   # Sep
        datetime(2023, 11, 1, 18, 0),   # Nov
        datetime(2023, 12, 13, 18, 0),  # Dec
        # 2024
        datetime(2024, 1, 31, 18, 0),   # Jan
        datetime(2024, 3, 20, 18, 0),   # Mar
        datetime(2024, 5, 1, 18, 0),    # May
        datetime(2024, 6, 12, 18, 0),   # Jun
        datetime(2024, 7, 31, 18, 0),   # Jul
        datetime(2024, 9, 18, 18, 0),   # Sep
        datetime(2024, 11, 6, 18, 0),   # Nov
        datetime(2024, 12, 18, 18, 0),  # Dec
        # 2025
        datetime(2025, 1, 29, 18, 0),   # Jan
    ]

    for date in fed_dates:
        event = EconomicEvent(
            name='Fed FOMC Decision (US)',
            country='US',
            importance='HIGH',
            expected_impact=1.00,
            event_date=date,
            hours_before_event=1.0,
            hours_after_event=2.0
        )
        calendar.add_event(event)


def load_jp_boj_events(calendar: EconomicCalendar) -> None:
    """Load BOJ (Bank of Japan) Policy Decision events (16:00 JST = 07:00 UTC)."""
    boj_dates = [
        # 2023
        datetime(2023, 3, 10, 7, 0),    # Mar
        datetime(2023, 4, 28, 7, 0),    # Apr
        datetime(2023, 6, 16, 7, 0),    # Jun
        datetime(2023, 7, 28, 7, 0),    # Jul
        datetime(2023, 9, 22, 7, 0),    # Sep
        datetime(2023, 10, 27, 7, 0),   # Oct
        datetime(2023, 12, 22, 7, 0),   # Dec
        # 2024
        datetime(2024, 1, 26, 7, 0),    # Jan
        datetime(2024, 3, 19, 7, 0),    # Mar
        datetime(2024, 4, 26, 7, 0),    # Apr
        datetime(2024, 6, 14, 7, 0),    # Jun
        datetime(2024, 7, 31, 7, 0),    # Jul
        datetime(2024, 9, 20, 7, 0),    # Sep
        datetime(2024, 10, 31, 7, 0),   # Oct
        datetime(2024, 12, 20, 7, 0),   # Dec
        # 2025
        datetime(2025, 1, 24, 7, 0),    # Jan
    ]

    for date in boj_dates:
        event = EconomicEvent(
            name='BOJ Policy Decision (JP)',
            country='JP',
            importance='HIGH',
            expected_impact=0.90,
            event_date=date,
            hours_before_event=1.0,
            hours_after_event=2.0
        )
        calendar.add_event(event)


def load_jp_unemployment_events(calendar: EconomicCalendar) -> None:
    """Load Japan Unemployment Rate events (Usually last Friday, 08:30 JST = 23:30 UTC previous day)."""
    jp_unemployment_dates = [
        # 2023
        datetime(2023, 1, 27, 23, 30),  # Jan (released 08:30 JST)
        datetime(2023, 2, 24, 23, 30),  # Feb
        datetime(2023, 3, 31, 23, 30),  # Mar
        datetime(2023, 4, 28, 23, 30),  # Apr
        datetime(2023, 5, 26, 23, 30),  # May
        datetime(2023, 6, 30, 23, 30),  # Jun
        datetime(2023, 7, 28, 23, 30),  # Jul
        datetime(2023, 8, 25, 23, 30),  # Aug
        datetime(2023, 9, 29, 23, 30),  # Sep
        datetime(2023, 10, 27, 23, 30), # Oct
        datetime(2023, 11, 24, 23, 30), # Nov
        datetime(2023, 12, 22, 23, 30), # Dec
        # 2024
        datetime(2024, 1, 26, 23, 30),  # Jan
        datetime(2024, 2, 23, 23, 30),  # Feb
        datetime(2024, 3, 29, 23, 30),  # Mar
        datetime(2024, 4, 26, 23, 30),  # Apr
        datetime(2024, 5, 31, 23, 30),  # May
        datetime(2024, 6, 28, 23, 30),  # Jun
        datetime(2024, 7, 26, 23, 30),  # Jul
        datetime(2024, 8, 30, 23, 30),  # Aug
        datetime(2024, 9, 27, 23, 30),  # Sep
        datetime(2024, 10, 25, 23, 30), # Oct
        datetime(2024, 11, 29, 23, 30), # Nov
        datetime(2024, 12, 27, 23, 30), # Dec
        # 2025
        datetime(2025, 1, 31, 23, 30),  # Jan
    ]

    for date in jp_unemployment_dates:
        event = EconomicEvent(
            name='Unemployment Rate (JP)',
            country='JP',
            importance='MEDIUM',
            expected_impact=0.50,
            event_date=date,
            hours_before_event=0.5,
            hours_after_event=1.0
        )
        calendar.add_event(event)


def load_economic_calendar_for_period(
    calendar: EconomicCalendar,
    start_date: datetime = None,
    end_date: datetime = None
) -> int:
    """
    Load all major economic events for the specified period.

    Args:
        calendar: EconomicCalendar instance
        start_date: Start date (default: 2023-01-01)
        end_date: End date (default: 2025-12-31)

    Returns:
        Number of events loaded
    """
    if start_date is None:
        start_date = datetime(2023, 1, 1)
    if end_date is None:
        end_date = datetime(2025, 12, 31)

    initial_count = len(calendar.events)

    # Load all event types
    load_us_nfp_events(calendar)
    load_us_cpi_events(calendar)
    load_fed_decision_events(calendar)
    load_jp_boj_events(calendar)
    load_jp_unemployment_events(calendar)

    # Filter by date range
    all_events = calendar.events.copy()
    calendar.events = [
        e for e in all_events
        if start_date <= e.event_date <= end_date
    ]

    return len(calendar.events)
