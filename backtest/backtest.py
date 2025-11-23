"""Backtest Engine module for simulation and trade tracking.

Implements BUY/SELL signal logic, stop loss/take profit execution,
trade tracking with performance metrics.

Tasks: 6.1, 6.2, 6.3
"""

import pandas as pd
import numpy as np
from pathlib import Path

from utils.errors import BacktestError


def run_backtest(df_ohlcv: pd.DataFrame, df_features: pd.DataFrame, 
                predictions: np.ndarray) -> pd.DataFrame:
    """Run backtest simulation with signal logic and trade tracking.

    Implements time-series backtest with:
    - BUY signals: model_pred==1 AND rsi<50 AND ma20_slope>0
    - SELL signals: model_pred==0 AND rsi>50 AND ma20_slope<0
    - Stop Loss: price <= entry_price * 0.997 (-0.3%)
    - Take Profit: price >= entry_price * 1.006 (+0.6%)
    - Priority: SL > TP > SELL signal
    Requirement 5: Backtest simulation with trade tracking.

    Args:
        df_ohlcv: OHLCV DataFrame with DatetimeIndex
                 Columns: ['Open', 'High', 'Low', 'Close', 'Volume']
        df_features: Features DataFrame with same length as df_ohlcv
                    Must include: 'rsi14', 'ma20_slope'
        predictions: Array of binary predictions (0 or 1) with length = len(df_ohlcv)

    Returns:
        pd.DataFrame: Trade records with columns:
                     entry_date, entry_price, exit_date, exit_price,
                     return_percent, win_loss, exit_reason

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

    trades = []
    position_open = False
    entry_date = None
    entry_price = None

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
                    print(f"📈 BUY signal at {entry_date}: price={entry_price:.2f}")

            else:
                # Position is open - check exit conditions
                # Priority: SL > TP > SELL signal

                # 1. Check Stop Loss: price <= entry_price * 0.997
                if current_close <= entry_price * 0.997:
                    exit_price = entry_price * 0.997
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    win_loss = 1 if return_pct >= 0 else 0
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': current_date,
                        'exit_price': exit_price,
                        'return_percent': return_pct,
                        'win_loss': win_loss,
                        'exit_reason': 'stop_loss'
                    })
                    print(f"🛑 STOP LOSS at {current_date}: price={exit_price:.2f}, return={return_pct:.3f}%")
                    position_open = False

                # 2. Check Take Profit: price >= entry_price * 1.006
                elif current_close >= entry_price * 1.006:
                    exit_price = entry_price * 1.006
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    win_loss = 1  # Always a win
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': current_date,
                        'exit_price': exit_price,
                        'return_percent': return_pct,
                        'win_loss': win_loss,
                        'exit_reason': 'take_profit'
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
                        'exit_reason': 'signal'
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
