# AI USDJPY Trader — 実装タスク

## Task Overview

本フィーチャーの実装は以下の8つの主要タスクで構成される。各タスクは要件に対応し、モジュラーパイプラインアーキテクチャの設計に従う。並列実行可能なタスクには `(P)` マークを付す。

---

## Implementation Tasks

### 1. プロジェクト構造・依存管理のセットアップ

- [x] 1.1 (P) ディレクトリ構造の作成と .gitignore ファイル生成
  - `data/`, `features/`, `model/`, `backtest/`, `trader/`, `utils/` ディレクトリを作成 ✓
  - `.gitignore` ファイルを生成：`__pycache__/`, `*.pyc`, `*.json`, `*.csv`, `venv/` を除外 ✓
  - プロジェクトの基盤ディレクトリ構造を確立 ✓
  - _Requirements: 8.1_

- [x] 1.2 (P) requirements.txt の作成と依存パッケージ指定
  - pandas (1.5+)、numpy (1.20+)、xgboost (1.7+)、scikit-learn (1.0+)、yfinance (0.2.32+)、ta (0.10+)、matplotlib (3.5+) をリスト化 ✓
  - すべてのパッケージにバージョン指定を含める ✓
  - 環境再現性を確保 ✓
  - _Requirements: 8.2, 8.3_

- [x] 1.3 utils/errors.py カスタムエラークラスの実装
  - TraderError（基本例外クラス）を定義：error_code、user_message、technical_message パラメータを含む ✓
  - DataError、FeatureEngineeringError、ModelError、BacktestError、VisualizationError サブクラスを実装 ✓
  - 全モジュールが統一的なエラーハンドリングを実施できる基盤を構築 ✓
  - _Requirements: 7_

---

### 2. データ取得機能（Data Fetcher）の実装

- [x] 2.1 (P) yfinance リトライロジック付きデータ取得関数の実装 ✓
  - `features/data_fetcher.py` に `fetch_usdjpy_data(years=3)` 関数を実装 ✓
  - USDJPY=X 日足データをデフォルト3年分取得（パラメータで変更可能） ✓
  - 最大3回のリトライ、エクスポーネンシャルバックオフ（5秒 → 10秒 → 30秒）を実装 ✓
  - ネットワークエラーとレート制限への耐性確保 ✓
  - 60秒以内に完了することを確認 ✓
  - _Requirements: 1.1, 1.6, 技術的制約 パフォーマンス_

- [x] 2.2 (P) OHLCV バリデーションとCSV保存機能の実装 ✓
  - Open、High、Low、Close、Volume カラムが全て存在することを検証 ✓
  - DataError を発生させる（カラム不足時、またはデータ不足時） ✓
  - データを時系列順（日付昇順）で `data/ohlcv_usdjpy.csv` に保存 ✓
  - データ行数の最小チェック（750行 ≈ 3年）を実装 ✓
  - _Requirements: 1.2, 1.3, 1.4, 1.5_

---

### 3. 特徴量生成機能（Feature Engineer）の実装

- [x] 3.1 (P) タイムゾーン変換とテクニカル指標計算の実装 ✓
  - `features/feature_engineer.py` に `engineer_features(df_ohlcv)` 関数を実装 ✓
  - yfinance（UTC）から取得したデータの Index を JST（UTC+9）に変換：`df_ohlcv.index.tz_localize('UTC').tz_convert('Asia/Tokyo')` ✓
  - ta ライブラリを使用して：ma5、ma20、ma50 を計算 ✓
  - ma5_slope、ma20_slope、ma50_slope（前日比%）を計算 ✓
  - RSI14 を計算（ta.momentum.rsi()） ✓
  - MACD（macd、signal、histogram）を計算（ta.trend）✓
  - Bollinger Bands（upper、middle、lower、band_width）を計算（ta.volatility.BollingerBands）✓
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 3.2 (P) 派生特徴量とターゲット変数の計算実装 ✓
  - 日次パーセンテージチェンジ（pct_change）を計算 ✓
  - 過去5日間のラグ特徴量（lag1～lag5）を計算 ✓
  - 曜日 one-hot encoding（mon、tue、wed、thu、fri）を計算 ✓
  - ターゲット変数：`target = 1 if next_close > current_close else 0` を計算 ✓
  - 初期 NaN 行を dropna() で削除（ma50=50行、MACD=34行等の lookback 期間） ✓
  - _Requirements: 2.7, 2.8, 2.9, 2.10, 2.11_

- [x] 3.3 (P) 特徴量データの妥当性確認と DataFrame 返却 ✓
  - 計算完了後、すべての特徴量カラムが存在することを検証 ✓
  - target カラムが存在することを確認 ✓
  - FeatureEngineeringError を発生させる（必須データ不足時） ✓
  - JST でインデックス化された訓練対応 DataFrame を返却 ✓
  - _Requirements: 2.12_

---

### 4. XGBoost モデル学習機能（Model Trainer）の実装

- [x] 4.1 時系列対応の train/test 分割と XGBoost モデル訓練 ✓
  - `model/train.py` に `train_model(df_features, test_mode=False)` 関数を実装 ✓
  - scikit-learn の train_test_split を使用して 8:2 分割（shuffle=False）を実施 ✓
  - XGBClassifier を以下のハイパーパラメータで初期化：n_estimators=300（本番）/10（テスト）、max_depth=5、learning_rate=0.05、subsample=0.8、colsample_bytree=0.8 ✓
  - 訓練セット（80%）でモデルを訓練 ✓
  - train_test_split に shuffle=False パラメータを明示的に指定 ✓
  - 訓練前後でデータの時系列順が保持されることをログ出力で確認 ✓
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 4.2 モデル評価と Feature Importance 計算・保存 ✓
  - テストセット（20%）での精度を計算・表示 ✓
  - F1 スコアを計算・表示 ✓
  - Confusion Matrix を生成・保存 ✓
  - Feature Importance を計算 ✓
  - Feature Importance bar chart を `model/feature_importance.png` に保存 ✓
  - _Requirements: 3.4, 3.5, 3.6, 3.7, 3.8_

- [x] 4.3 モデルの JSON シリアライズと保存 ✓
  - 訓練済みモデルを JSON 形式で `model/xgb_model.json` に保存 ✓
  - 訓練時の特徴量カラムリストを `model/feature_columns.json` に保存（Predictor で使用） ✓
  - モデル訓練開始時と終了時のタイムスタンプをログ出力し、300秒超過時は警告 ✓
  - モデル訓練失敗時に詳細なエラー情報をログして例外を発生 ✓
  - _Requirements: 3.9, 3.10_

---

### 5. 予測機能（Predictor）の実装

- [x] 5.1 (P) モデルロードと予測確率・バイナリ予測の実装 ✓
  - `model/predict.py` に `predict(df_features, model=None, feature_columns=None)` 関数を実装 ✓
  - `model/xgb_model.json` からモデルをロード（xgb.Booster 経由） ✓
  - ModelError を発生させる（ファイル不在時） ✓
  - バッチ予測（複数行）に対応 ✓
  - 予測確率（0～1）を計算：model.predict_proba(X)[:, 1] ✓
  - 確率閾値 0.5 に基づくバイナリ予測（0 または 1）を計算：model.predict(X) ✓
  - _Requirements: 4.1, 4.2, 4.3, 4.5_

- [x] 5.2 特徴量カラムの検証と自動整列 ✓
  - `model/feature_columns.json` から訓練時のカラムリストをロード ✓
  - 訓練時と同じ特徴量カラムが全て存在することを検証 ✓
  - FeatureEngineeringError を発生させる（カラム不足時） ✓
  - カラムを訓練時と同じ順序に自動整列：X_aligned = df_features[feature_columns] ✓
  - XGBoost が要求する順序での予測実施を確保 ✓
  - _Requirements: 4.4_

---

### 6. バックテストエンジン（Backtest Engine）の実装

- [x] 6.1 (P) BUY/SELL シグナルロジックの実装 ✓
  - `backtest/backtest.py` に `run_backtest(df_ohlcv, df_features, predictions)` 関数を実装 ✓
  - BUY 条件を評価：`model_pred == 1 AND RSI < 50 AND MA20_slope > 0` ✓
  - SELL 条件を評価：`model_pred == 0 AND RSI > 50 AND MA20_slope < 0` ✓
  - BUY シグナル時にエントリー日付とエントリー価格（シグナル日の終値）を記録 ✓
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 6.2 (P) 損切り・利確条件と複数終了条件の優先順位実装 ✓
  - ポジションがオープン中、損切り条件を監視：`price <= entry_price * 0.997`（-0.3%） ✓
  - 損切り発動時にエグジット価格とリターン%を記録し exit_reason='stop_loss' を記録 ✓
  - ポジションがオープン中、利確条件を監視：`price >= entry_price * 1.006`（+0.6%） ✓
  - 利確発動時にエグジット価格とリターン%を記録し exit_reason='take_profit' を記録 ✓
  - 同一バーでの優先順位を実装：SL → TP → ルールベース SELL ✓
  - 複数条件発火時は優先順位の高い条件を採用し、対応する exit_reason を記録 ✓
  - _Requirements: 5.4, 5.5, 5.6, 5.7_

- [x] 6.3 (P) トレード記録とバックテストメトリクス計算・保存 ✓
  - すべてのトレード詳細を記録：entry_date、entry_price、exit_date、exit_price、return_percent、win_loss、exit_reason ✓
  - トレード記録を `backtest/backtest_results.csv` に保存 ✓
  - バックテスト実行時にトレード統計を表示（総トレード数、勝数、負数、勝率） ✓
  - バックテスト失敗時に詳細なエラー情報をログして例外を発生 ✓
  - _Requirements: 5.8, 5.9, 5.10_
  - entry_date および exit_date が JST タイムゾーンで記録されることを確認
  - 総損益、勝率、平均RR、シャープレシオ、最大ドローダウンを計算
  - トレード記録を `backtest/backtest_results.csv` に保存
  - 損益曲線グラフを生成
  - バックテスト実行時間を計測・ログ出力し、30秒超過時は警告
  - _Requirements: 5.8, 5.9, 5.10, 5.11_

---

### 7. 可視化とレポート機能（Plotter）の実装

- [ ] 7.1 (P) Feature Importance と P&L 曲線グラフの生成・保存
  - `utils/plot.py` に `plot_results(feature_importance, trade_records)` 関数を実装
  - Feature Importance bar chart を `model/feature_importance.png` に保存
  - 累積 P&L 曲線を `backtest/pnl_curve.png` に保存
  - グラフの x 軸（日付）が JST タイムゾーンであることを確認
  - すべてのグラフにグリッド、凡例、軸ラベルを含める
  - 適切なカラースキーム、フォントサイズ、解像度（dpi >= 100）を使用
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

---

### 8. メイン実行フロー（main.py）の実装とシステム統合

- [ ] 8.1 (P) データ取得フェーズの実装と条件分岐
  - `main.py` を作成
  - 各モジュールから関数をインポート：`from features.data_fetcher import fetch_usdjpy_data`、`from features.feature_engineer import engineer_features` など
  - `data/usd_jpy.csv` 存在確認
  - CSV 未存在なら Data Fetcher を実行してデータ取得・保存
  - CSV 存在なら既存ファイルをロード
  - データ取得失敗時に詳細なエラーログを出力して終了
  - _Requirements: 7_

- [ ] 8.2 (P) 特徴量生成と条件付きモデル訓練フェーズ
  - Feature Engineer を実行して特徴量生成
  - `model/xgb_model.json` 存在確認
  - モデルファイル未存在なら Model Trainer を実行して訓練・保存
  - モデルファイル存在なら Predictor でロード
  - モデル訓練失敗時に詳細なエラーログを出力して終了
  - _Requirements: 7_

- [ ] 8.3 (P) 予測・バックテスト実行と結果レポート生成フェーズ
  - Predictor を実行して予測生成
  - Backtest Engine を実行してバックテスト実行
  - Plotter を実行してすべてのグラフを生成
  - サマリーレポートを出力：主要メトリクス（総損益、勝率、シャープレシオ等）とファイルロケーションを表示
  - 全体フロー完了を示すメッセージを出力
  - _Requirements: 7_

- [ ] 8.4 エラーハンドリングと全体ワークフロー統合
  - すべてのステップで try-except を使用してカスタムエラーをキャッチ
  - 各モジュールのエラー型を個別にキャッチ：
    - Data Fetcher: DataError
    - Feature Engineer: FeatureEngineeringError
    - Model Trainer/Predictor: ModelError
    - Backtest Engine: BacktestError
    - Plotter: VisualizationError
  - カスタムエラー発生時に user_message を表示、technical_message をログ出力
  - 予期しないエラー発生時は詳細なスタックトレースをログ出力
  - エラー発生時は graceful に終了（exit code != 0）
  - `python main.py` コマンドで全体ワークフローが実行可能なことを確認
  - _Requirements: 7_

---

## Optional Test Coverage Tasks

- [ ]* 8.5 ユニットテストの実装（受け入れ条件カバレッジ）
  - `tests/test_feature_engineer.py`：test_ma_calculation()、test_rsi_range()、test_lag_features_offset() を実装
  - `tests/test_model.py`：test_train_test_shuffle_false()、test_model_serialization() を実装
  - `tests/test_backtest.py`：test_buy_signal_conditions()、test_sl_tp_priority()、test_simultaneous_sl_tp() を実装
  - `tests/test_data_fetcher.py`：test_retry_logic()、test_csv_chronological_order() を実装
  - テスト実行が `pytest tests/` で全て成功することを確認
  - _Requirements: 1～7 全体の受け入れ条件検証_

- [ ]* 8.6 統合テストの実装（エンドツーエンドワークフロー）
  - `tests/test_integration.py`：Data Pipeline → Feature Pipeline → Model Training & Prediction → Backtest Pipeline の全フローをテスト
  - `python main.py` 実行時に全ファイル（CSV、モデル、グラフ、結果CSV）が正常に生成されることを確認
  - エラー処理の網羅的テスト（ネットワークエラー、不正なデータ形式等）を実装
  - _Requirements: 1～7 全体の統合検証_

---

## Requirements Coverage Summary

| 要件ID | 対応タスク | 進捗 |
|--------|----------|------|
| 1 | 2.1, 2.2 | Data Fetcher 実装 |
| 2 | 3.1, 3.2, 3.3 | Feature Engineer 実装 |
| 3 | 4.1, 4.2, 4.3 | Model Trainer 実装 |
| 4 | 5.1, 5.2 | Predictor 実装 |
| 5 | 6.1, 6.2, 6.3 | Backtest Engine 実装 |
| 6 | 7.1 | Plotter 実装 |
| 7 | 8.1, 8.2, 8.3, 8.4 | main.py 実装 |
| 8 | 1.1, 1.2, 1.3 | Project Setup 実装 |
| 技術的制約 | 各タスク | パフォーマンス・タイムゾーン・時系列整合性 |

---

## Task Dependencies and Parallel Execution

### Sequential Requirements
1. **Project Setup (1.x)** → すべてのモジュール実装を開始する前に完了必須
2. **Data Fetcher (2.x)** → Feature Engineer への入力が必要

### Parallel Execution Blocks
- **ブロック A**: Data Fetcher（2.x）完了後、以下を並列実行可能
  - Feature Engineer（3.x）
  - プロジェクト構造（1.x）既に完了

- **ブロック B**: Feature Engineer（3.x）完了後、以下を並列実行可能
  - Model Trainer（4.x）
  - Predictor の設計準備（5.x）

- **ブロック C**: Model Trainer（4.x）完了後
  - Predictor（5.x）実装開始

- **ブロック D**: Predictor（5.x）と Feature Engineer（3.x）完了後
  - Backtest Engine（6.x）実装開始

- **ブロック E**: Backtest Engine（6.x）完了後
  - Plotter（7.x）実装開始

- **ブロック F**: すべてのモジュール完了後
  - main.py（8.1～8.4）統合実装開始

**推奨実装順序**: 1 → 2 → (3 || 1.3並列) → (4, 5並列) → 6 → 7 → 8 → (8.5, 8.6 オプション)

---

## Notes

- すべてのタスクは要件.md の受け入れ条件に対応
- 各タスク完了時に単体テストまたは統合テストで検証必須
- エラーハンドリングは utils/errors.py で統一
- ログ出力は主要マイルストーン（データ取得完了、モデル訓練完了等）に限定
