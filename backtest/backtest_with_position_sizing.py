"""
Extended Backtest Engine with Position Sizing

Implements backtest with dynamic position sizing strategies:
1. Fixed Risk Percentage: Risk fixed % of account per trade
2. Kelly Criterion: Optimal f based on trading statistics

Tasks: 6.2, 6.3, 6.4 - Position sizing optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from utils.errors import BacktestError
from trader.dynamic_risk_manager import DynamicRiskManager, initialize_risk_manager
from trader.position_sizer import PositionSizer, create_position_sizer_fixed_risk, create_position_sizer_kelly
from trader.drawdown_manager import create_drawdown_manager


def run_backtest_with_position_sizing(
    df_ohlcv: pd.DataFrame,
    df_features: pd.DataFrame,
    predictions: np.ndarray,
    use_dynamic_risk: bool = True,
    position_sizing_strategy: str = 'fixed_risk',
    account_size: float = 100000,
    risk_percent: float = 1.0,
    kelly_fraction: float = 0.15,
    max_position_pct: float = 50.0
) -> Tuple[pd.DataFrame, dict]:
    """
    Run backtest simulation with dynamic position sizing.

    Combines:
    - Dynamic risk management (volatility-aware TP/SL from Step 5)
    - Dynamic position sizing (optimal position size based on strategy)

    Args:
        df_ohlcv: OHLCV DataFrame with DatetimeIndex
        df_features: Features DataFrame with same length as df_ohlcv
        predictions: Array of binary predictions (0 or 1)
        use_dynamic_risk: Use volatility-adjusted TP/SL
        position_sizing_strategy: 'fixed_risk' or 'kelly'
        account_size: Initial trading account size in USD
        risk_percent: Risk % per trade (for fixed_risk strategy)
        kelly_fraction: Kelly f fraction (for kelly strategy)
        max_position_pct: Maximum position size as % of account

    Returns:
        Tuple[trades_df, metrics_dict]
    """
    # Validate inputs
    if len(df_ohlcv) != len(df_features) or len(df_ohlcv) != len(predictions):
        raise BacktestError(
            error_code="DATA_MISMATCH",
            user_message="Input data lengths do not match",
            technical_message=f"ohlcv={len(df_ohlcv)}, features={len(df_features)}, predictions={len(predictions)}"
        )

    required_features = ['rsi14', 'ma20_slope', 'volatility_20']
    missing_features = [f for f in required_features if f not in df_features.columns]
    if missing_features:
        raise BacktestError(
            error_code="MISSING_FEATURES",
            user_message=f"Missing required features: {missing_features}",
            technical_message=f"Available: {list(df_features.columns)}"
        )

    # Initialize risk manager
    risk_manager = None
    if use_dynamic_risk:
        if 'volatility_20' not in df_features.columns:
            raise BacktestError(
                error_code="MISSING_VOLATILITY",
                user_message="Dynamic risk management requires volatility_20 feature",
                technical_message=f"Available features: {list(df_features.columns)}"
            )
        print("  ⚙️  Initializing dynamic risk manager...")
        risk_manager = initialize_risk_manager(df_features)

    # Initialize position sizer
    print(f"  📊 Initializing position sizer: {position_sizing_strategy}")
    if position_sizing_strategy == 'fixed_risk':
        sizer = create_position_sizer_fixed_risk(account_size, risk_percent)
    elif position_sizing_strategy == 'kelly':
        sizer = create_position_sizer_kelly(account_size, kelly_fraction)
    else:
        raise BacktestError(
            error_code="INVALID_STRATEGY",
            user_message=f"Unknown position sizing strategy: {position_sizing_strategy}",
            technical_message="Use 'fixed_risk' or 'kelly'"
        )

    trades = []
    position_open = False
    entry_date = None
    entry_price = None
    position_size = 0
    dynamic_tp_level = None
    dynamic_sl_level = None

    # Initialize drawdown manager
    dd_manager = create_drawdown_manager(
        account_size=account_size,
        max_dd_threshold=5.0,  # 5% drawdown threshold
        max_absolute_dd=10.0   # 10% absolute maximum drawdown
    )

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
                    position_open = True
                    entry_date = current_date
                    entry_price = current_close

                    # Calculate dynamic or static TP/SL
                    if risk_manager is not None:
                        current_vol = df_features['volatility_20'].iloc[i]
                        dynamic_tp_level, dynamic_sl_level = risk_manager.get_dynamic_tp_sl(
                            entry_price, current_vol
                        )
                        tp_pct = ((dynamic_tp_level - entry_price) / entry_price) * 100
                        sl_pct = ((entry_price - dynamic_sl_level) / entry_price) * 100
                    else:
                        dynamic_tp_level = entry_price * 1.006
                        dynamic_sl_level = entry_price * 0.997
                        tp_pct = 0.6
                        sl_pct = 0.3

                    # Calculate position size
                    if position_sizing_strategy == 'fixed_risk':
                        position_size = sizer.calculate_position_size_fixed_risk(
                            entry_price, dynamic_sl_level, risk_percent
                        )
                    else:  # kelly
                        # Use historical statistics from completed trades
                        win_rate = (sizer.win_count / sizer.trade_count) if sizer.trade_count > 0 else 0.54
                        avg_win = 0.60  # From analysis
                        avg_loss = 0.315  # From analysis
                        position_size, _ = sizer.calculate_position_size_kelly(
                            entry_price, dynamic_sl_level, win_rate, avg_win, avg_loss
                        )
                        # Additional safety cap for Kelly strategy
                        # Kelly can be aggressive at start, so cap at 10% of account
                        max_kelly_position = sizer.current_account * 0.10
                        position_size = min(position_size, max_kelly_position)

                    # Apply drawdown-based risk multiplier
                    risk_multiplier = dd_manager.get_risk_multiplier()
                    position_size = position_size * risk_multiplier

                    position_pct = (position_size / sizer.current_account) * 100
                    print(f"📈 BUY signal at {entry_date}: price={entry_price:.2f}")
                    print(f"   TP: {dynamic_tp_level:.5f} (+{tp_pct:.3f}%), SL: {dynamic_sl_level:.5f} (-{sl_pct:.3f}%)")
                    print(f"   Position: {position_size:,.0f} units ({position_pct:.2f}% of account)")

            else:
                # Position is open - check exit conditions
                # Priority: SL > TP > SELL signal

                # 1. Check Stop Loss
                if current_close <= dynamic_sl_level:
                    exit_price = dynamic_sl_level
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    # PNL = (return_pct / 100) * (position_size / account_size) * account
                    position_pct_of_account = position_size / (sizer.current_account + 0.0001)  # avoid division by zero
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

                    try:
                        sizer.update_account(pnl_usd)
                        dd_manager.update_account(sizer.current_account)
                    except Exception as e:
                        print(f"⚠️ Account update error: {e}")
                        raise
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
            'strategy': position_sizing_strategy,
            'account': sizer.get_position_metrics(),
            'trades': len(trades_df),
            'wins': (trades_df['win_loss'] == 1).sum() if len(trades_df) > 0 else 0,
            'losses': (trades_df['win_loss'] == 0).sum() if len(trades_df) > 0 else 0,
            'total_pnl_usd': sizer.total_pnl,
            'final_account': sizer.current_account,
            'return_pct': ((sizer.current_account - account_size) / account_size * 100)
        }

        # Save results
        if len(trades_df) > 0:
            _save_backtest_results_with_sizing(trades_df, position_sizing_strategy)

        print(f"✓ Backtest complete: {len(trades_df)} trades")
        print(f"  Account: ${account_size:,.0f} → ${sizer.current_account:,.0f} ({metrics['return_pct']:.2f}%)")

        return trades_df, metrics

    except BacktestError:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise BacktestError(
            error_code="BACKTEST_FAILED",
            user_message="Backtest execution failed",
            technical_message=f"Error: {str(e)}"
        )


def _save_backtest_results_with_sizing(trades_df: pd.DataFrame, strategy: str) -> None:
    """Save backtest results with position sizing data."""
    try:
        backtest_dir = Path(__file__).parent
        backtest_dir.mkdir(exist_ok=True)

        results_path = backtest_dir / f'backtest_results_{strategy}.csv'
        trades_df.to_csv(results_path, index=False)

        print(f"✓ Results saved to {results_path}")

    except Exception as e:
        raise BacktestError(
            error_code="SAVE_FAILED",
            user_message="Failed to save backtest results",
            technical_message=str(e)
        )
