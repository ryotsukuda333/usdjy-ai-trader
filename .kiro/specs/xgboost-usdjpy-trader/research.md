# 研究 & 設計決定ログ

## サマリー

- **フィーチャー**: xgboost-usdjpy-trader
- **発見スコープ**: 新規グリーンフィールド機能（複雑な統合 — 機械学習 + 金融バックテスト + 時系列データ処理）
- **主要な知見**:
  - XGBoostで時系列予測を行う場合、walk-forward検証と厳密な順序保持が必須
  - yfinanceはレート制限とデータ遅延の問題がある → リトライロジックとエラーハンドリング必須
  - テクニカル指標は「ta」ライブラリを採用（Pure Python、Pure Python、インストール容易、要件で明記）
  - バックテストエンジンは複数終了条件を管理するため、イベントループアーキテクチャを採用

---

## 研究ログ

### XGBoost時系列予測のベストプラクティス

**背景**: 金融時系列データ（USDJPY日足）でXGBoostを使用した二項分類（翌日上昇/下降）を実装する必要がある。時系列データには特有の落とし穴がある。

**参照資料**:
- MachineLearningMastery.com: "How to Use XGBoost for Time Series Forecasting"
- Analytics Vidhya: "Use XGBoost for Time-Series Forecasting"
- Towards Data Science: "Multi-step time series forecasting with XGBoost"

**知見**:
- **Shuffle=False の必須性**: 時系列データを訓練/テストに分割する際、`shuffle=False` により時間順序を維持する（要件で既に記載）
- **Supervised Learning 変換**: 時系列は遅延特徴量やローリング統計を使ってシューバッドラーニング問題に変換が必須
- **Walk-Forward 検証**: k-fold交差検証は往復バイアスを招くため、walk-forward法を推奨（実装フェーズで検討）
- **特徴量エンジニアリング**: lag特徴量（過去値）、ローリング統計（MA）、フーリエ変換が重要
- **XGBoostの限界**: XGBoostは相互作用を学習するが、傾向を外挿できない（ルールベースのMA傾きと組み合わせるのは適切）

**影響**: バックテストはホールドアウト検証（8:2分割、shuffle=False）を使用し、訓練セットと独立した期間でテストを実施。

---

### yfinance API の品質と信頼性（2024年）

**背景**: USDJPY日足データを `yfinance` で自動取得する。yfinanceはスクレイピングベースのAPIで、Yahoo Financeエンドポイントに依存している。

**参照資料**:
- Stack Overflow: "How do you get USD JPY currency Data using yfinance API"
- GitHub yfinance Issues: "#2340 Yahoo Finance Historical Data Access Failure"
- Medium: "Why yfinance Keeps Getting Blocked, and What to Use Instead"
- Saturn Cloud Blog: "How to Get USDJPY Currency Rates with Pandas and Yahoo Finance"

**知見**:
- **レート制限**: 2024年初頭、Yahoo Financeはレート制限を強化。同じIPからの急速な複数リクエストで429 "Too Many Requests"エラーが発生
- **データ品質**: 遅延データやデータの欠落が報告されている。公式なSLAがないため、信頼性は限定的
- **ブロック問題**: yfinanceはオフィシャルAPIではなく、HTMLスクレイピングに依存。Yahooが防御を強化すると機能停止のリスク
- **USDJPY固有**: USDJPY=X でのデータ取得は技術的に可能だが、general yfinance issues が適用される
- **推奨される対策**:
  1. リトライロジック（エクスポーネンシャルバックオフ）を実装
  2. タイムアウトとネットワークエラーへのハンドリング
  3. 例外ログと詳細なエラーメッセージ

**影響**: Data Fetcher は最大3回のリトライ、5〜30秒のバックオフ、詳細なエラーログを実装。システムはyfinanceの一時的な障害に対して耐性を持つ設計に。

---

### テクニカル指標ライブラリの選択

**背景**: RSI14、MACD、ボリンジャーバンドなど複数のテクニカル指標を計算する必要がある。

**参照資料**:
- TA-Lib Official: https://ta-lib.org/
- GitHub: TA-Lib/ta-lib-python, 0xAVX/pandas-ta
- Kaggle: "Introduction to using of TA-Lib"
- PyPI: pandas-ta

**候補の評価**:

| オプション | 説明 | 強み | 制限 | 選択判定 |
|-----------|------|------|------|---------|
| TA-Lib | C/C++コア、200+ indicator | 高速（2-4倍）、BSD License、実績 | C拡張インストール難（Windows等で複雑） | × |
| pandas-ta | Pure Python、130+ indicator | インストール容易、Pandas統合 | TA-Libより若干遅い | ○ 検討 |
| ta | Pure Python、80+ indicator | インストール容易、要件で明記 | 少ないindicator数 | ✓ **採用** |

**決定**: 要件で明記されている「ta」を採用。Pure Pythonのため、プラットフォーム依存性がなく、環境セットアップが簡単。

**影響**: Feature Engineer は `ta` ライブラリを使用。ta.RSI()、ta.MACD()、ta.BBANDS() を利用。

---

### バックテストエンジンのアーキテクチャ：複数終了条件の管理

**背景**: バックテストでは複数の終了条件が存在：（1）BUY/SELL ルールによる新規エントリー、（2）損切り（-0.3%）、（3）利確（+0.6%）。同一バーで複数条件が発火する場合の優先順位が必要。

**参照資料**:
- backtesting.py: Backtest framework (stop-loss/take-profit support)
- vectorbt: nd-array backtesting (sl_stop, tp_stop parameters)
- Stack Overflow: "Vectorize stop loss / take profit backtesting"
- Blog: "Stop Loss, Trailing Stop, or Take Profit? 2 Million Backtests Shed Light"

**知見**:
- **イベントループアーキテクチャ**: 各バーを順序通り処理し、まずスケール効果チェック（SL/TP）を行い、その後ルールベースシグナルを評価
- **優先順位**: 同一バーで SL と TP が発火した場合、backtesting.py は **SL を先に優先**
- **実装パターン**: 状態機械（ポジション: なし / オープン）と条件チェック順序の明確化

**設計上の決定**:
1. ポジション状態を管理（`position_open=False/True`, `entry_price`, `entry_date`）
2. 各バーで順序立てて検査：
   - ① SL条件チェック（ポジションがオープンなら）
   - ② TP条件チェック（ポジションがオープンなら）
   - ③ ルールベースシグナル（BUY/SELL）チェック
3. トレード記録に `exit_reason` フィールド（'rule', 'stop_loss', 'take_profit'）を含める

**影響**: Backtest Engine は上記の状態遷移ロジックを明確に実装。トレード記録にはエグジット理由も記録して分析可能に。

---

### NaN値処理戦略（時系列ギャップフィリング）

**背景**: テクニカル指標計算（特にMACD、BBands）は初期期間でNaN値を生成する。モデル訓練前にこれらの行を処理する必要がある。

**参照資料**:
- Towards Data Science: "Filling Gaps in Time Series Data"
- Medium: "Forward Fill vs Backward Fill"
- GeeksforGeeks: "How to deal with missing values in a Timeseries"
- pandas documentation: `fillna()`, `dropna()`

**知見**:
- **Forward Fill (ffill)**: 直前の有効値を使用。時系列には自然で、外挿より安全
- **Backward Fill (bfill)**: 次の有効値を使用。時系列では非推奨（未来情報を含む）
- **Drop (dropna)**: 初期NaN行をすべて削除。シンプルだが期間を失う
- **補間 (interpolate)**: 線形や spline 補間。複雑だが正確性が向上可能

**決定**: Feature Engineer は初期NaN行（テクニカル指標の lookback 期間）を **dropna()** で削除。シンプルで確実。

**影響**: Feature Engineer は最後に `.dropna()` を呼び出して、完全に有効な行のみをモデル訓練用に返す。

---

## アーキテクチャパターン評価

| オプション | 説明 | 強み | 制限 | 選択判定 |
|-----------|------|------|------|---------|
| **モジュラーパイプライン** | 各機能（Data, Features, Model, Backtest）が独立したモジュール | 並列実装可能、テスト容易、再利用性 | オーケストレーション必要 | ✓ **採用** |
| モノリシック | すべてが単一スクリプト | シンプル | スケーリング困難、テスト困難 | × |
| イベント駆動 | メッセージキューベース | 非同期処理可能 | 複雑性増加、このスケールでは過度 | × |

**選択**: **モジュラーパイプラインアーキテクチャ** — 各コンポーネント（DataFetcher, FeatureEngineer, ModelTrainer, Predictor, BacktestEngine, Plotter）が明確に独立し、チーム内での並列実装を支援。

---

## 設計決定

### 決定: テクニカル指標ライブラリの選定

- **背景**: 複数のテクニカル指標計算ライブラリが利用可能
- **検討した選択肢**:
  1. TA-Lib（C拡張）— 高速だがインストール複雑
  2. pandas-ta（Pure Python）— バランスの取れた選択肢
  3. ta（Pure Python）— 要件で明記、軽量
- **選択**: **ta ライブラリ**（要件合致、インストール容易、Pure Python）
- **根拠**: グリーンフィールドプロジェクトでは、環境セットアップの容易性を優先
- **トレードオフ**: TA-Libより若干遅い可能性があるが、日足ベースのため性能上の問題なし
- **検証予定**: 3年分データ（〜750行）での特徴量計算パフォーマンステスト

---

### 決定: バックテストのポジション管理とエグジット優先順位

- **背景**: 複数のエグジット条件（ルール、SL、TP）が存在
- **検討した選択肢**:
  1. TP > SL 優先 — 利益を最大化するが非現実的
  2. SL > TP 優先 — リスク管理を優先（業界標準）
  3. 同時に処理 — あいまいで非決定的
- **選択**: **SL > TP > ルール** の優先順位
- **根拠**: リスク管理の原則。損切りを先に検証することで、最大損失を確実に制限
- **実装**: Backtest Engine で状態遷移の順序を明確に記述

---

## リスク & ミティゲーション

| リスク | 確率 | 影響 | ミティゲーション |
|--------|------|------|-----------------|
| yfinance レート制限 | 高 | システム障害 | リトライロジック（3回、エクスポーネンシャルバックオフ） |
| データ品質（遅延/欠落） | 中 | バックテスト精度低下 | データ検証ステップ、警告ログ |
| XGBoost look-ahead bias | 高 | 結果の過楽観的評価 | shuffle=False 厳密化、walk-forward 検証（後期拡張） |
| 複数終了条件の混乱 | 中 | トレード記録の不正確性 | 明確な優先順位、exit_reason フィールド記録 |
| 時系列順序崩れ | 高 | モデル無効化 | ユニットテスト（日付昇順検証） |

---

## 参考資料

- [TA-Lib Official](https://ta-lib.org/) — Technical Analysis Library documentation
- [MachineLearningMastery](https://machinelearningmastery.com/xgboost-for-time-series-forecasting/) — XGBoost Time Series Guide
- [backtesting.py](https://kernc.github.io/backtesting.py/) — Event-loop backtest framework reference
- [pandas Documentation](https://pandas.pydata.org/docs/) — fillna(), dropna(), resample()
- [yfinance GitHub](https://github.com/ranaroussi/yfinance) — Known issues and API
- [ta Library](https://pypi.org/project/ta/) — Technical Indicators
