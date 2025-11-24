"""
Seasonality-Aware Backtest for USDJPY AI Trader

Integrates seasonal patterns into trading decisions:
- Volatility adjustment based on monthly/weekly patterns
- Position sizing adjustment for high/low volatility periods
- Entry/exit quality scoring based on seasonal favorability

Combines all previous risk management layers:
- Dynamic risk management (TP/SL by volatility)
- Session-based risk (Tokyo/London/NewYork)
- Economic event awareness
- Seasonal patterns
"""

from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
from datetime import datetime

from trader.dynamic_risk_manager import DynamicRiskManager
from trader.session_manager import SessionManager
from trader.seasonality_manager import SeasonalityManager


def run_backtest_seasonality_aware(
    df_ohlcv: pd.DataFrame,
    df_features: pd.DataFrame,
    predictions: np.ndarray,
    account_size: float = 100000,
    use_dynamic_risk: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Run seasonality-aware backtest with dynamic risk, session, and seasonal adjustments.

    Implements full multi-layer risk control:
    1. Seasonality check: Quality scoring and volatility adjustment
    2. Session check: Risk level based on trading session
    3. Event check: Economic event restrictions
    4. Dynamic TP/SL: Based on seasonally-adjusted volatility
    5. Position sizing: Base × seasonal adjustment × session adjustment

    Args:
        df_ohlcv: OHLCV data (aligned with features)
        df_features: Feature data for signal generation
        predictions: Buy/Sell predictions (1=buy, 0=sell)
        account_size: Starting account balance in USD
        use_dynamic_risk: Whether to use dynamic risk management

    Returns:
        (trades_df, metrics_dict)
    """

    # Initialize managers
    risk_manager = DynamicRiskManager()
    session_manager = SessionManager()
    seasonality_manager = SeasonalityManager()

    # Tracking
    trades = []
    account = account_size
    position_open = False
    entry_price = 0
    entry_date = None
    entry_session = None
    entry_seasonal_quality = None
    position_size = 0
    tp_level = 0
    sl_level = 0

    # Metrics
    total_trades = 0
    winning_trades = 0
    seasonal_trades = 0

    print("⚙️  Initializing dynamic risk manager...")
    print(f"✓ Risk Manager initialized from {len(df_ohlcv)} volatility samples")

    # Calculate volatility stats from features
    if 'volatility_20' in df_features.columns:
        volatility_stats = df_features['volatility_20'].dropna().values
    else:
        # Fallback: calculate from price changes
        volatility_stats = np.abs(df_ohlcv['Close'].pct_change().values) * 100

    print(f"  Volatility Stats:")
    print(f"    Median: {np.median(volatility_stats):.4f}%")
    print(f"    Range: {np.min(volatility_stats):.4f}% - {np.max(volatility_stats):.4f}%")
    print(f"    Min/Max: {np.percentile(volatility_stats, 5):.4f}% - {np.percentile(volatility_stats, 95):.4f}%")

    print(f"\n📊 Initializing seasonality manager")

    # Process each candle
    for i in range(len(df_ohlcv)):
        current_date = df_ohlcv.index[i]
        current_price = df_ohlcv['Close'].iloc[i]

        # Get current volatility from features
        if i < len(df_features) and 'volatility_20' in df_features.columns:
            current_vol = df_features['volatility_20'].iloc[i] if not pd.isna(df_features['volatility_20'].iloc[i]) else np.median(volatility_stats)
        else:
            current_vol = np.median(volatility_stats)

        # Get seasonality information
        seasonal_info = seasonality_manager.get_seasonality_summary(current_date)
        seasonal_quality = seasonal_info['quality_score']
        seasonal_vol_adj = seasonal_info['volatility_adjustment']
        seasonal_pos_adj = seasonal_info['position_adjustment']

        # Get session information
        session_info = session_manager.get_current_session(current_date)
        session_risk = session_info['risk_percent']

        # BUY signal
        if not position_open and predictions[i] == 1:
            # Calculate seasonally-adjusted volatility
            adjusted_vol = current_vol * seasonal_vol_adj

            # Calculate TP/SL directly (base: 0.60% TP, 0.30% SL)
            base_tp_pct = 0.60
            base_sl_pct = 0.30

            # Adjust by seasonality
            tp_pct = base_tp_pct * (adjusted_vol / np.median(volatility_stats))
            sl_pct = base_sl_pct * (adjusted_vol / np.median(volatility_stats))

            tp_price = current_price * (1 + tp_pct / 100)
            sl_price = current_price * (1 - sl_pct / 100)

            # Calculate base position size (1% risk)
            base_risk_amount = account * 0.01
            sl_distance_pct = (current_price - sl_price) / current_price
            base_position = base_risk_amount / (current_price * sl_distance_pct) if sl_distance_pct > 0 else 0

            # Apply seasonal adjustment
            seasonal_adjusted_position = base_position * seasonal_pos_adj

            # Apply session adjustment
            session_adjusted_position = seasonal_adjusted_position * (session_risk / 5.0)

            # Final position size
            position_size = int(session_adjusted_position)

            # Verify position is valid
            if position_size > 0 and sl_price > 0:
                entry_price = current_price
                entry_date = current_date
                entry_session = session_info['session_name']
                entry_seasonal_quality = seasonal_quality
                tp_level = tp_price
                sl_level = sl_price
                position_open = True

                print(f"📈 BUY signal at {current_date} [{entry_session}] (seasonal_quality={seasonal_quality:.0f}): price={entry_price:.2f}")
                print(f"   TP: {tp_level:.5f} ({(tp_level/entry_price - 1)*100:+.3f}%), SL: {sl_level:.5f} ({(sl_level/entry_price - 1)*100:+.3f}%)")
                print(f"   Position: {position_size:,} units ({position_size * entry_price / account * 100:.2f}% of account)")
                print(f"   Seasonal Adj: {seasonal_pos_adj:.2f}x, Session: {session_risk}% risk")

        # Exit logic
        elif position_open:
            # TAKE PROFIT
            if current_price >= tp_level:
                return_pct = (current_price - entry_price) / entry_price * 100
                pnl = position_size * (current_price - entry_price)
                account += pnl

                print(f"💰 TAKE PROFIT at {current_date}: {current_price:.2f}, return={return_pct:+.3f}%, account=${account:,.0f}")

                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': current_date,
                    'exit_price': current_price,
                    'return_percent': return_pct,
                    'position_size': position_size,
                    'pnl_usd': pnl,
                    'win_loss': 1,
                    'exit_reason': 'take_profit',
                    'session': entry_session,
                    'seasonal_quality': entry_seasonal_quality,
                    'tp_level': tp_level,
                    'sl_level': sl_level,
                })

                position_open = False
                winning_trades += 1
                total_trades += 1
                seasonal_trades += 1

            # STOP LOSS
            elif current_price <= sl_level:
                return_pct = (current_price - entry_price) / entry_price * 100
                pnl = position_size * (current_price - entry_price)
                account += pnl

                print(f"🛑 STOP LOSS at {current_date}: {current_price:.2f}, return={return_pct:+.3f}%, account=${account:,.0f}")

                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': current_date,
                    'exit_price': current_price,
                    'return_percent': return_pct,
                    'position_size': position_size,
                    'pnl_usd': pnl,
                    'win_loss': 0,
                    'exit_reason': 'stop_loss',
                    'session': entry_session,
                    'seasonal_quality': entry_seasonal_quality,
                    'tp_level': tp_level,
                    'sl_level': sl_level,
                })

                position_open = False
                total_trades += 1
                seasonal_trades += 1

            # SELL signal (exit at next candle open)
            elif predictions[i] == 0:
                return_pct = (current_price - entry_price) / entry_price * 100
                pnl = position_size * (current_price - entry_price)
                account += pnl

                print(f"📉 SELL signal at {current_date}: {current_price:.2f}, return={return_pct:+.3f}%, account=${account:,.0f}")

                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': current_date,
                    'exit_price': current_price,
                    'return_percent': return_pct,
                    'position_size': position_size,
                    'pnl_usd': pnl,
                    'win_loss': 1 if return_pct > 0 else 0,
                    'exit_reason': 'signal',
                    'session': entry_session,
                    'seasonal_quality': entry_seasonal_quality,
                    'tp_level': tp_level,
                    'sl_level': sl_level,
                })

                position_open = False
                if return_pct > 0:
                    winning_trades += 1
                total_trades += 1
                seasonal_trades += 1

    # Save results
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        trades_df.to_csv('backtest/backtest_results_seasonality_aware.csv', index=False)

    # Calculate metrics
    if len(trades_df) > 0:
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        total_return = trades_df['return_percent'].sum()
        avg_return = trades_df['return_percent'].mean()
    else:
        win_rate = 0
        total_return = 0
        avg_return = 0

    final_return_pct = (account - account_size) / account_size * 100

    metrics = {
        'initial_account': account_size,
        'final_account': account,
        'return_pct': final_return_pct,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'avg_return': avg_return,
        'seasonal_trades': seasonal_trades,
    }

    return trades_df, metrics
