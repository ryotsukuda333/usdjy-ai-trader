"""
Event-Aware Backtest with Economic Indicator Management

Implements backtest with dynamic risk adjustment based on economic events.
Integrates economic calendar to:
- Restrict trading during HIGH importance events
- Reduce position size during MEDIUM events
- Track event impacts on volatility and prices

Task: 9.4
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

from utils.errors import BacktestError
from trader.dynamic_risk_manager import initialize_risk_manager
from trader.position_sizer import create_position_sizer_fixed_risk
from trader.drawdown_manager import create_drawdown_manager
from trader.session_analyzer import create_session_analyzer
from trader.economic_calendar import create_economic_calendar
from trader.load_economic_events import load_economic_calendar_for_period
from trader.event_volatility_manager import create_event_volatility_manager


def run_backtest_event_aware(
    df_ohlcv: pd.DataFrame,
    df_features: pd.DataFrame,
    predictions: np.ndarray,
    account_size: float = 100000,
    use_dynamic_risk: bool = True,
    restriction_level: str = 'HIGH'
) -> Tuple[pd.DataFrame, dict]:
    """
    Run backtest with economic event-aware risk adjustment.

    Args:
        df_ohlcv: OHLCV DataFrame with DatetimeIndex
        df_features: Features DataFrame
        predictions: Prediction array
        account_size: Initial account size
        use_dynamic_risk: Use volatility-adjusted TP/SL
        restriction_level: Event restriction level (HIGH, MEDIUM, LOW)

    Returns:
        Tuple[trades_df, metrics_dict]
    """
    # Initialize risk manager
    print("  ⚙️  Initializing dynamic risk manager...")
    risk_manager = initialize_risk_manager(df_features)

    # Initialize position sizer (base 5% fixed risk)
    print("  📊 Initializing event-aware position sizer")
    sizer = create_position_sizer_fixed_risk(account_size, risk_percent=5.0)

    # Initialize drawdown manager
    dd_manager = create_drawdown_manager(
        account_size=account_size,
        max_dd_threshold=5.0,
        max_absolute_dd=10.0
    )

    # Initialize session analyzer
    session_analyzer = create_session_analyzer()

    # Initialize economic calendar
    print("  📅 Loading economic calendar...")
    economic_calendar = create_economic_calendar()
    num_events = load_economic_calendar_for_period(economic_calendar)
    print(f"     Loaded {num_events} economic events")

    # Initialize event volatility manager
    event_vol_manager = create_event_volatility_manager(
        economic_calendar,
        base_volatility=0.595  # From analysis
    )

    trades = []
    position_open = False
    entry_date = None
    entry_price = None
    position_size = 0
    dynamic_tp_level = None
    dynamic_sl_level = None
    events_avoided = 0
    events_impacted = 0

    try:
        for i in range(len(df_ohlcv)):
            current_date = df_ohlcv.index[i]
            current_close = df_ohlcv['Close'].iloc[i]
            pred = predictions[i]
            rsi = df_features['rsi14'].iloc[i]
            ma20_slope = df_features['ma20_slope'].iloc[i]

            if not position_open:
                # Evaluate BUY signal
                if pred == 1 and rsi < 50 and ma20_slope > 0:
                    # Check if trading is restricted by events
                    should_trade, event_reason = event_vol_manager.should_trade(
                        current_date,
                        restriction_level=restriction_level
                    )

                    if not should_trade:
                        events_avoided += 1
                        continue

                    position_open = True
                    entry_date = current_date
                    entry_price = current_close

                    # Calculate dynamic TP/SL with event-aware volatility
                    if risk_manager is not None:
                        current_vol = df_features['volatility_20'].iloc[i]
                        # Adjust volatility for events
                        event_adjusted_vol = event_vol_manager.get_event_adjusted_volatility(
                            current_date, current_vol
                        )
                        dynamic_tp_level, dynamic_sl_level = risk_manager.get_dynamic_tp_sl(
                            entry_price, event_adjusted_vol
                        )
                        tp_pct = ((dynamic_tp_level - entry_price) / entry_price) * 100
                        sl_pct = ((entry_price - dynamic_sl_level) / entry_price) * 100
                    else:
                        dynamic_tp_level = entry_price * 1.006
                        dynamic_sl_level = entry_price * 0.997
                        tp_pct = 0.6
                        sl_pct = 0.3

                    # Calculate base position size (5% fixed risk)
                    position_size = sizer.calculate_position_size_fixed_risk(
                        entry_price, dynamic_sl_level, 5.0
                    )

                    # Apply event-based risk adjustment
                    event_risk_adj = event_vol_manager.get_event_risk_adjustment(
                        current_date,
                        restriction_level=restriction_level
                    )
                    if event_risk_adj < 1.0:
                        events_impacted += 1

                    position_size = position_size * event_risk_adj

                    # Apply drawdown-based risk multiplier
                    risk_multiplier = dd_manager.get_risk_multiplier()
                    position_size = position_size * risk_multiplier

                    position_pct = (position_size / sizer.current_account) * 100
                    current_hour = current_date.hour

                    session_name = "NY" if current_hour < 9 or current_hour > 23 else (
                        "Tokyo" if 9 <= current_hour <= 15 else "London"
                    )

                    print(f"📈 BUY signal at {entry_date} [{session_name}]: price={entry_price:.2f}")
                    print(f"   TP: {dynamic_tp_level:.5f} (+{tp_pct:.3f}%), SL: {dynamic_sl_level:.5f} (-{sl_pct:.3f}%)")
                    print(f"   Position: {position_size:,.0f} units ({position_pct:.2f}% of account) [Event Adj: {event_risk_adj:.2f}x]")

            else:
                # Position is open - check exit conditions
                # Priority: SL > TP > SELL signal

                # 1. Check Stop Loss
                if current_close <= dynamic_sl_level:
                    exit_price = dynamic_sl_level
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    pnl_usd = (return_pct / 100) * position_size

                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': current_date,
                        'exit_price': exit_price,
                        'return_percent': return_pct,
                        'position_size': position_size,
                        'pnl_usd': pnl_usd,
                        'win_loss': 1 if return_pct >= 0 else 0,
                        'exit_reason': 'stop_loss',
                        'tp_level': dynamic_tp_level,
                        'sl_level': dynamic_sl_level
                    })

                    sizer.update_account(pnl_usd)
                    dd_manager.update_account(sizer.current_account)
                    print(f"🛑 STOP LOSS at {current_date}: {exit_price:.2f}, return={return_pct:.3f}%, account=${sizer.current_account:,.0f}")
                    position_open = False

                # 2. Check Take Profit
                elif current_close >= dynamic_tp_level:
                    exit_price = dynamic_tp_level
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    pnl_usd = (return_pct / 100) * position_size

                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': current_date,
                        'exit_price': exit_price,
                        'return_percent': return_pct,
                        'position_size': position_size,
                        'pnl_usd': pnl_usd,
                        'win_loss': 1,
                        'exit_reason': 'take_profit',
                        'tp_level': dynamic_tp_level,
                        'sl_level': dynamic_sl_level
                    })

                    sizer.update_account(pnl_usd)
                    dd_manager.update_account(sizer.current_account)
                    print(f"💰 TAKE PROFIT at {current_date}: {exit_price:.2f}, return={return_pct:.3f}%, account=${sizer.current_account:,.0f}")
                    position_open = False

                # 3. Check SELL signal
                elif pred == 0 and rsi > 50 and ma20_slope < 0:
                    exit_price = current_close
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    pnl_usd = (return_pct / 100) * position_size

                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': current_date,
                        'exit_price': exit_price,
                        'return_percent': return_pct,
                        'position_size': position_size,
                        'pnl_usd': pnl_usd,
                        'win_loss': 1 if return_pct >= 0 else 0,
                        'exit_reason': 'signal',
                        'tp_level': dynamic_tp_level,
                        'sl_level': dynamic_sl_level
                    })

                    sizer.update_account(pnl_usd)
                    dd_manager.update_account(sizer.current_account)
                    print(f"📉 SELL signal at {current_date}: {exit_price:.2f}, return={return_pct:.3f}%, account=${sizer.current_account:,.0f}")
                    position_open = False

        # Close any remaining open position
        if position_open and len(df_ohlcv) > 0:
            final_price = df_ohlcv['Close'].iloc[-1]
            return_pct = ((final_price - entry_price) / entry_price) * 100
            pnl_usd = (return_pct / 100) * position_size

            trades.append({
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': df_ohlcv.index[-1],
                'exit_price': final_price,
                'return_percent': return_pct,
                'position_size': position_size,
                'pnl_usd': pnl_usd,
                'win_loss': 1 if return_pct >= 0 else 0,
                'exit_reason': 'end_of_data',
                'tp_level': dynamic_tp_level,
                'sl_level': dynamic_sl_level
            })

            sizer.update_account(pnl_usd)
            dd_manager.update_account(sizer.current_account)

        # Create trades DataFrame
        trades_df = pd.DataFrame(trades)

        # Compile metrics
        metrics = {
            'strategy': 'event_aware_fixed_risk',
            'account': sizer.get_position_metrics(),
            'trades': len(trades_df),
            'wins': (trades_df['win_loss'] == 1).sum() if len(trades_df) > 0 else 0,
            'losses': (trades_df['win_loss'] == 0).sum() if len(trades_df) > 0 else 0,
            'total_pnl_usd': sizer.total_pnl,
            'final_account': sizer.current_account,
            'return_pct': ((sizer.current_account - account_size) / account_size * 100),
            'events_avoided': events_avoided,
            'events_impacted': events_impacted
        }

        # Save results
        if len(trades_df) > 0:
            results_path = Path(__file__).parent / 'backtest_results_event_aware.csv'
            trades_df.to_csv(results_path, index=False)
            print(f"\n✓ Results saved to {results_path}")

        print(f"✓ Backtest complete: {len(trades_df)} trades")
        print(f"  Account: ${account_size:,.0f} → ${sizer.current_account:,.0f} ({metrics['return_pct']:.2f}%)")
        print(f"  Events Avoided: {events_avoided}, Events Impacted: {events_impacted}")

        return trades_df, metrics

    except BacktestError:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise BacktestError(
            error_code="BACKTEST_FAILED",
            user_message="Event-aware backtest execution failed",
            technical_message=f"Error: {str(e)}"
        )
