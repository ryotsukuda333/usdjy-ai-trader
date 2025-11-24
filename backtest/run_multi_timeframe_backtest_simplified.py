"""マルチタイムフレーム バックテスト - 簡潔版

XGBoost確率加重が困難な場合の代替アプローチ：
- テクニカル指標のみで信号生成
- コンフルエンス閾値を0.70 → 0.50に緩和
- 季節性スコアを組み込み

期待値：
- 実行可能な信号: 30-50個 (0.2%から向上)
- トレード数: 3-8回
- リターン: +2-3% (Phase 5-A同等)
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.multi_timeframe_fetcher import fetch_multi_timeframe_usdjpy
from features.multi_timeframe_engineer import engineer_features_multi_timeframe


class SimplifiedMultiTimeframeBacktester:
    """テクニカル指標ベースのマルチタイムフレーム バックテスター"""

    TIMEFRAME_WEIGHTS = {
        '1d': 0.40, '4h': 0.25, '1h': 0.20, '15m': 0.10, '5m': 0.05
    }

    def __init__(self, initial_capital: float = 100000, risk_per_trade: float = 0.01,
                 take_profit_pct: float = 1.0, stop_loss_pct: float = 0.5,
                 confluence_threshold: float = 0.50):  # 緩和閾値
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.confluence_threshold = confluence_threshold

        self.current_equity = initial_capital
        self.trades = []
        self.equity_curve = []
        self.position = None

    def map_5m_to_higher_timeframes(self, idx_5m: int) -> Dict[str, int]:
        """5分足インデックスを高いタイムフレームにマップ"""
        return {'5m': idx_5m, '15m': idx_5m // 3, '1h': idx_5m // 12,
                '4h': idx_5m // 48, '1d': idx_5m // 288}

    def get_technical_signal(self, df_features: pd.DataFrame) -> int:
        """テクニカル指標ベースの信号 (1=BUY, -1=SELL, 0=HOLD)"""
        if df_features.empty or len(df_features) == 0:
            return 0

        last = df_features.iloc[-1]
        score = 0.0

        # MA crossover
        ma_cols = [col for col in df_features.columns if col.startswith('ma') and '_' not in col]
        try:
            periods = sorted([int(col[2:]) for col in ma_cols if col[2:].isdigit()])
            if len(periods) >= 2:
                short_ma = last.get(f'ma{periods[0]}', 0)
                long_ma = last.get(f'ma{periods[-1]}', 0)
                if short_ma > long_ma:
                    score += 0.4
                elif short_ma < long_ma:
                    score -= 0.4
        except:
            pass

        # RSI
        if 'rsi14' in df_features.columns:
            rsi = last.get('rsi14', 50)
            if rsi < 30:
                score += 0.3
            elif rsi > 70:
                score -= 0.3

        # MACD
        if 'macd' in df_features.columns and 'macd_signal' in df_features.columns:
            if last.get('macd', 0) > last.get('macd_signal', 0):
                score += 0.3
            else:
                score -= 0.3

        return 1 if score > 0.2 else (-1 if score < -0.2 else 0)

    def calculate_alignment_confidence(self, signals: Dict[str, int], timestamp: datetime) -> Tuple[float, str]:
        """アライメントスコアと信頼度を計算"""
        if not signals:
            return 0.0, "No signals"

        # 加重信号計算
        weighted_sum = sum(signals.get(tf, 0) * self.TIMEFRAME_WEIGHTS.get(tf, 0) for tf in signals)
        alignment = abs(weighted_sum)

        # 季節性スコア（曜日・時間帯）
        dow = timestamp.weekday()
        hour = timestamp.hour
        seasonality = {0: 0.55, 1: 0.50, 2: 0.52, 3: 0.50, 4: 0.48, 5: 0.45, 6: 0.47}.get(dow, 0.50)
        if 8 <= hour < 11:
            seasonality = max(seasonality, 0.58)
        elif 14 <= hour < 17:
            seasonality = max(seasonality, 0.55)

        # 信頼度 = 60% alignment + 40% seasonality
        confidence = 0.6 * alignment + 0.4 * seasonality

        reason = f"Align={alignment:.2f}, Season={seasonality:.2f}, Conf={confidence:.2f}"
        return confidence, reason

    def backtest(self, data_dict: Dict[str, pd.DataFrame],
                 features_dict: Dict[str, pd.DataFrame]) -> Tuple[float, Dict]:
        """バックテスト実行"""
        print("\n" + "=" * 80)
        print("マルチタイムフレーム バックテスト (簡潔版) ")
        print("=" * 80)

        df_5m = data_dict['5m']
        bars_count = len(df_5m)

        print(f"\n設定:")
        print(f"  初期資金:        ${self.initial_capital:,.2f}")
        print(f"  リスク:          {self.risk_per_trade*100:.1f}%/取引")
        print(f"  TP/SL:           {self.take_profit_pct:.2f}% / {self.stop_loss_pct:.2f}%")
        print(f"  コンフルエンス閾値: {self.confluence_threshold:.2f}")
        print(f"\nデータ:")
        print(f"  5m足:            {bars_count:,} 本")
        print(f"  期間:            {df_5m.index[0]} 〜 {df_5m.index[-1]}")

        start_time = time.time()
        signal_stats = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'EXECUTED': 0}

        for idx_5m in range(len(df_5m)):
            if idx_5m % 20000 == 0:
                print(f"  処理中: {idx_5m:,} / {bars_count:,} ({100*idx_5m/bars_count:.1f}%)...")

            current_price = df_5m.iloc[idx_5m]['Close']
            current_time = df_5m.index[idx_5m]
            idx_map = self.map_5m_to_higher_timeframes(idx_5m)

            # 各タイムフレームのシグナル取得
            signals = {}
            for tf, idx in idx_map.items():
                if tf in features_dict and idx < len(features_dict[tf]):
                    signals[tf] = self.get_technical_signal(features_dict[tf].iloc[:idx+1])

            # メインシグナル決定
            main_signal = signals.get('1d', 0) or signals.get('4h', 0)

            if main_signal == 1:
                signal_stats['BUY'] += 1
            elif main_signal == -1:
                signal_stats['SELL'] += 1
            else:
                signal_stats['HOLD'] += 1

            # 既存ポジション管理
            if self.position is not None:
                entry_price = self.position['entry_price']
                entry_signal = self.position['signal']

                should_exit = False
                exit_reason = ""

                if entry_signal == 1 and main_signal == -1:
                    should_exit = True
                    exit_reason = "シグナル反転"
                elif entry_signal == 1 and current_price >= entry_price * (1 + self.take_profit_pct / 100):
                    should_exit = True
                    exit_reason = f"利益確定 (+{self.take_profit_pct}%)"
                elif entry_signal == 1 and current_price <= entry_price * (1 - self.stop_loss_pct / 100):
                    should_exit = True
                    exit_reason = f"損切 (-{self.stop_loss_pct}%)"
                elif entry_signal == -1 and main_signal == 1:
                    should_exit = True
                    exit_reason = "シグナル反転"
                elif entry_signal == -1 and current_price <= entry_price * (1 - self.take_profit_pct / 100):
                    should_exit = True
                    exit_reason = f"利益確定 (+{self.take_profit_pct}%)"
                elif entry_signal == -1 and current_price >= entry_price * (1 + self.stop_loss_pct / 100):
                    should_exit = True
                    exit_reason = f"損切 (-{self.stop_loss_pct}%)"

                if should_exit:
                    profit_loss = (current_price - entry_price) if entry_signal == 1 else (entry_price - current_price)
                    profit_pct = (profit_loss / entry_price) * 100
                    profit_usd = profit_loss * self.position['size']

                    self.current_equity += profit_usd

                    self.trades.append({
                        'entry_time': self.position['entry_time'],
                        'entry_price': entry_price,
                        'exit_time': current_time,
                        'exit_price': current_price,
                        'signal': entry_signal,
                        'size': self.position['size'],
                        'profit': profit_usd,
                        'profit_pct': profit_pct,
                        'bars_held': idx_5m - self.position['entry_idx'],
                        'exit_reason': exit_reason,
                        'confidence': self.position['confidence']
                    })

                    signal_type = "LONG" if entry_signal == 1 else "SHORT"
                    print(f"  終了: {signal_type} at {current_price:.4f} "
                          f"(P&L: {profit_pct:+.3f}%, {exit_reason})")

                    self.position = None

            # 新規エントリー判定
            if self.position is None and main_signal != 0:
                confidence, reason = self.calculate_alignment_confidence(signals, current_time)

                if confidence >= self.confluence_threshold:
                    if main_signal == 1:
                        sl_price = current_price * (1 - self.stop_loss_pct / 100)
                    else:
                        sl_price = current_price * (1 + self.stop_loss_pct / 100)

                    risk_amount = self.current_equity * self.risk_per_trade
                    stop_loss_dist = abs(current_price - sl_price)
                    size = int(risk_amount / stop_loss_dist) if stop_loss_dist > 0 else 0

                    if size > 0:
                        signal_stats['EXECUTED'] += 1

                        self.position = {
                            'entry_time': current_time,
                            'entry_idx': idx_5m,
                            'entry_price': current_price,
                            'stop_loss': sl_price,
                            'signal': main_signal,
                            'size': size,
                            'confidence': confidence
                        }

                        sig_type = "買" if main_signal == 1 else "売"
                        print(f"  エントリー: {sig_type} at {current_price:.4f} (信頼度: {confidence:.3f})")

            self.equity_curve.append({'timestamp': current_time, 'equity': self.current_equity})

        elapsed = time.time() - start_time

        total_return = self.current_equity - self.initial_capital
        return_pct = (total_return / self.initial_capital) * 100

        print(f"\nバックテスト完了: {elapsed:.1f}秒")
        print(f"\nシグナル統計:")
        print(f"  買いシグナル:     {signal_stats['BUY']:,}")
        print(f"  売りシグナル:     {signal_stats['SELL']:,}")
        print(f"  ホールド:         {signal_stats['HOLD']:,}")
        print(f"  実行シグナル:     {signal_stats['EXECUTED']} ({100*signal_stats['EXECUTED']/bars_count:.2f}%)")

        # 結果統計
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            wins = len(df_trades[df_trades['profit'] > 0])
            losses = len(df_trades[df_trades['profit'] < 0])
            win_rate = (wins / len(df_trades)) * 100

            total_wins = df_trades[df_trades['profit'] > 0]['profit'].sum()
            total_losses = abs(df_trades[df_trades['profit'] < 0]['profit'].sum())
            pf = total_wins / total_losses if total_losses > 0 else 0

            returns = df_trades['profit_pct'].values
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 and len(returns) > 1 else 0

            equity_vals = [self.initial_capital] + [self.initial_capital + df_trades.iloc[:i+1]['profit'].sum() for i in range(len(df_trades))]
            running_max = np.maximum.accumulate(equity_vals)
            drawdown = (np.array(equity_vals) - running_max) / running_max
            max_dd = np.min(drawdown) * 100 if len(drawdown) > 0 else 0

            metrics = {
                'total_return': return_pct, 'num_trades': len(self.trades),
                'winning_trades': wins, 'losing_trades': losses, 'win_rate': win_rate,
                'profit_factor': pf, 'avg_bars_held': df_trades['bars_held'].mean(),
                'sharpe': sharpe, 'max_drawdown': max_dd, 'final_equity': self.current_equity
            }

            print(f"\n" + "=" * 80)
            print("結果: マルチタイムフレーム vs Phase 5-A")
            print("=" * 80)
            print(f"\n{'指標':<30} {'Phase 5-A':<20} {'マルチTF+改善':<20}")
            print("-" * 70)
            print(f"{'リターン':<30} {2.00:>7.2f}% {return_pct:>15.2f}%")
            print(f"{'トレード数':<30} {7:>7d} {len(self.trades):>15d}")
            print(f"{'勝率':<30} {57.1:>7.1f}% {win_rate:>15.1f}%")
            print(f"{'Sharpe比':<30} {7.87:>7.2f} {sharpe:>15.2f}")
            print(f"{'最大DD':<30} {-0.25:>7.2f}% {max_dd:>15.2f}%")
            print(f"{'最終資金':<30} ${102000:>17,.0f} ${self.current_equity:>15,.0f}")
            print("\n" + "=" * 80)

        else:
            metrics = {'total_return': 0.0, 'num_trades': 0, 'winning_trades': 0,
                      'losing_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
                      'avg_bars_held': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0,
                      'final_equity': self.initial_capital}
            print(f"\n⚠️ トレードなし (閾値: {self.confluence_threshold})")

        return return_pct, metrics


def run_backtest(years: int = 2) -> Tuple[float, Dict]:
    """バックテスト実行"""
    print("\n" + "=" * 80)
    print("マルチタイムフレーム + 改善版 バックテスト パイプライン")
    print("=" * 80)

    print("\n[1/3] データ取得...")
    start = time.time()
    data_dict = fetch_multi_timeframe_usdjpy(years=years)
    print(f"✓ {time.time() - start:.1f}秒")

    print("\n[2/3] フィーチャー エンジニアリング...")
    start = time.time()
    features_dict = engineer_features_multi_timeframe(data_dict)
    print(f"✓ {time.time() - start:.1f}秒")

    print("\n[3/3] バックテスト実行...")
    start = time.time()
    backtester = SimplifiedMultiTimeframeBacktester(confluence_threshold=0.50)  # 緩和
    return_pct, metrics = backtester.backtest(data_dict, features_dict)

    # 結果保存
    output_dir = Path(__file__).parent
    if backtester.trades:
        pd.DataFrame(backtester.trades).to_csv(output_dir / "multi_timeframe_simplified_trades.csv", index=False)
        pd.DataFrame(backtester.equity_curve).to_csv(output_dir / "multi_timeframe_simplified_equity.csv", index=False)

    print(f"✓ {time.time() - start:.1f}秒")
    return return_pct, metrics


if __name__ == "__main__":
    return_pct, metrics = run_backtest(years=2)

    results = {
        'metrics': metrics,
        'timestamp': str(pd.Timestamp.now()),
        'configuration': {
            'initial_capital': 100000,
            'risk_per_trade': 0.01,
            'confluence_threshold': 0.50  # 緩和
        }
    }

    results_file = Path(__file__).parent / "MULTI_TIMEFRAME_SIMPLIFIED_RESULTS.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ 結果保存: {results_file}")
    print(f"\n最終リターン: {return_pct:+.2f}%")
