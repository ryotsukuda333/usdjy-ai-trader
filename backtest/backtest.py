"""Backtest Engine module for simulation and trade tracking.

Implements BUY/SELL signal logic, stop loss/take profit execution,
trade tracking with performance metrics. Supports both static and
dynamic risk management.

Tasks: 6.1, 6.2, 6.3, 5.1, 5.2
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from utils.errors import BacktestError
from trader.dynamic_risk_manager import DynamicRiskManager, initialize_risk_manager


def run_backtest(df_ohlcv: pd.DataFrame, df_features: pd.DataFrame,
                predictions: np.ndarray, use_dynamic_risk: bool = True) -> pd.DataFrame:
    """Run backtest simulation with signal logic and trade tracking.

    Implements time-series backtest with:
    - BUY signals: model_pred==1 AND rsi<50 AND ma20_slope>0
    - SELL signals: model_pred==0 AND rsi>50 AND ma20_slope<0
    - Stop Loss & Take Profit:
        - Static mode: SL = -0.3%, TP = +0.6%
        - Dynamic mode: Adjusted by volatility factor (volatility-aware)
    - Priority: SL > TP > SELL signal
    Requirement 5: Backtest simulation with trade tracking.
    Tasks: 5.1, 5.2 - Dynamic risk management

    Args:
        df_ohlcv: OHLCV DataFrame with DatetimeIndex
                 Columns: ['Open', 'High', 'Low', 'Close', 'Volume']
        df_features: Features DataFrame with same length as df_ohlcv
                    Must include: 'rsi14', 'ma20_slope', 'volatility_20'
        predictions: Array of binary predictions (0 or 1) with length = len(df_ohlcv)
        use_dynamic_risk: If True, uses volatility-adjusted TP/SL (default: True)

    Returns:
        pd.DataFrame: Trade records with columns:
                     entry_date, entry_price, exit_date, exit_price,
                     return_percent, win_loss, exit_reason, tp_level (dynamic),
                     sl_level (dynamic)

    Raises:
        BacktestError: If inputs invalid or data mismatched
    """
    # Validate inputs
    if df_ohlcv is None or df_features is None or predictions is None:
        raise BacktestError(
            error_code="INVALID_INPUT",
            user_message="Input data is None",
            technical_message="df_ohlcv, df_features, and predictions must not be None"
        )

    if len(df_ohlcv) != len(df_features) or len(df_ohlcv) != len(predictions):
        raise BacktestError(
            error_code="LENGTH_MISMATCH",
            user_message="Input data lengths do not match",
            technical_message=f"OHLCV: {len(df_ohlcv)}, Features: {len(df_features)}, "
                            f"Predictions: {len(predictions)}"
        )

    if 'Close' not in df_ohlcv.columns:
        raise BacktestError(
            error_code="MISSING_COLUMNS",
            user_message="OHLCV data missing 'Close' column",
            technical_message=f"Available columns: {list(df_ohlcv.columns)}"
        )

    if 'rsi14' not in df_features.columns or 'ma20_slope' not in df_features.columns:
        raise BacktestError(
            error_code="MISSING_FEATURES",
            user_message="Features missing required columns: rsi14, ma20_slope",
            technical_message=f"Available features: {list(df_features.columns)}"
        )

    # Initialize dynamic risk manager if requested
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

    trades = []
    position_open = False
    entry_date = None
    entry_price = None
    dynamic_tp_level = None
    dynamic_sl_level = None

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
                        # Dynamic risk management (Task 5.1, 5.2)
                        current_vol = df_features['volatility_20'].iloc[i]
                        dynamic_tp_level, dynamic_sl_level = risk_manager.get_dynamic_tp_sl(
                            entry_price, current_vol
                        )
                        tp_pct = ((dynamic_tp_level - entry_price) / entry_price) * 100
                        sl_pct = ((entry_price - dynamic_sl_level) / entry_price) * 100
                        print(f"📈 BUY signal at {entry_date}: price={entry_price:.2f}")
                        print(f"   Dynamic TP: {dynamic_tp_level:.5f} (+{tp_pct:.3f}%), SL: {dynamic_sl_level:.5f} (-{sl_pct:.3f}%)")
                    else:
                        # Static risk management (original)
                        dynamic_tp_level = entry_price * 1.006
                        dynamic_sl_level = entry_price * 0.997
                        print(f"📈 BUY signal at {entry_date}: price={entry_price:.2f} (static TP/SL)")

            else:
                # Position is open - check exit conditions
                # Priority: SL > TP > SELL signal
                # Uses dynamic_tp_level and dynamic_sl_level (set at BUY)

                # 1. Check Stop Loss: price <= dynamic_sl_level
                if current_close <= dynamic_sl_level:
                    exit_price = dynamic_sl_level
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    win_loss = 1 if return_pct >= 0 else 0
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': current_date,
                        'exit_price': exit_price,
                        'return_percent': return_pct,
                        'win_loss': win_loss,
                        'exit_reason': 'stop_loss',
                        'tp_level': dynamic_tp_level,
                        'sl_level': dynamic_sl_level
                    })
                    print(f"🛑 STOP LOSS at {current_date}: price={exit_price:.2f}, return={return_pct:.3f}%")
                    position_open = False

                # 2. Check Take Profit: price >= dynamic_tp_level
                elif current_close >= dynamic_tp_level:
                    exit_price = dynamic_tp_level
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    win_loss = 1  # Always a win
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': current_date,
                        'exit_price': exit_price,
                        'return_percent': return_pct,
                        'win_loss': win_loss,
                        'exit_reason': 'take_profit',
                        'tp_level': dynamic_tp_level,
                        'sl_level': dynamic_sl_level
                    })
                    print(f"💰 TAKE PROFIT at {current_date}: price={exit_price:.2f}, return={return_pct:.3f}%")
                    position_open = False

                # 3. Check SELL signal: pred==0 AND rsi>50 AND ma20_slope<0
                elif pred == 0 and rsi > 50 and ma20_slope < 0:
                    exit_price = current_close
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    win_loss = 1 if return_pct >= 0 else 0
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': current_date,
                        'exit_price': exit_price,
                        'return_percent': return_pct,
                        'win_loss': win_loss,
                        'exit_reason': 'signal',
                        'tp_level': dynamic_tp_level,
                        'sl_level': dynamic_sl_level
                    })
                    print(f"📉 SELL signal at {current_date}: price={exit_price:.2f}, return={return_pct:.3f}%")
                    position_open = False

        # Close any remaining open position at end of data
        if position_open and len(df_ohlcv) > 0:
            final_price = df_ohlcv['Close'].iloc[-1]
            return_pct = ((final_price - entry_price) / entry_price) * 100
            win_loss = 1 if return_pct >= 0 else 0
            trades.append({
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': df_ohlcv.index[-1],
                'exit_price': final_price,
                'return_percent': return_pct,
                'win_loss': win_loss,
                'exit_reason': 'end_of_data'
            })
            print(f"📊 END OF DATA: closed position at {final_price:.2f}, return={return_pct:.3f}%")

        # Create trades DataFrame
        trades_df = pd.DataFrame(trades)

        # Save results
        if len(trades_df) > 0:
            _save_backtest_results(trades_df)

        print(f"✓ Backtest complete: {len(trades_df)} trades")
        return trades_df

    except BacktestError:
        raise
    except Exception as e:
        raise BacktestError(
            error_code="BACKTEST_FAILED",
            user_message="Backtest execution failed",
            technical_message=f"Error during backtest: {str(e)}"
        )


def _save_backtest_results(trades_df: pd.DataFrame) -> None:
    """Save backtest results to CSV.

    Args:
        trades_df: DataFrame with trade records

    Raises:
        BacktestError: If save fails
    """
    try:
        backtest_dir = Path(__file__).parent
        backtest_dir.mkdir(exist_ok=True)

        results_path = backtest_dir / 'backtest_results.csv'
        trades_df.to_csv(results_path, index=False)

        print(f"✓ Backtest results saved to {results_path}")

    except Exception as e:
        raise BacktestError(
            error_code="SAVE_FAILED",
            user_message="Failed to save backtest results",
            technical_message=f"Error saving results: {str(e)}"
        )
