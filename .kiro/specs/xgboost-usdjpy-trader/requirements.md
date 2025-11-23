# AI USDJPY Trader — 要件仕様書

## はじめに

本ドキュメントは、XGBoostモデルを用いたUSJPY自動売買システムの要件を定義します。
システムは翌日ドル円上昇確率を予測し、ルールベース売買戦略のバックテストを実行します。

---

## 要件

### 1. データ取得機能

**目的：** データ分析者として、USDJPY為替データを自動取得したい。そうすることで、常に最新のデータに基づいてモデル学習とバックテストができる

#### 受け入れ条件

1. The Data Fetcher shall retrieve USDJPY=X daily candle data from yfinance with default period of 3 years
2. When data retrieval is initiated, the Data Fetcher shall validate that all required OHLCV (Open, High, Low, Close, Volume) columns are present
3. If any required data column is missing, the Data Fetcher shall raise an informative error and halt execution
4. When data is successfully retrieved, the Data Fetcher shall save the data to `data/usd_jpy.csv` with Date, Open, High, Low, Close, Volume columns
5. While saving to CSV, the Data Fetcher shall ensure chronological date ordering (earliest to latest)
6. The Data Fetcher shall support configurable date range parameters (デフォルト3年) for flexibility

---

### 2. 特徴量生成機能

**目的：** 機械学習エンジニアとして、テクニカル指標とファンダメンタル特徴量を自動生成したい。そうすることで、モデルに高品質な入力データを提供できる

#### 受け入れ条件

1. When raw OHLCV data is provided, the Feature Engineer shall compute Moving Averages (MA5, MA20, MA50) for each row
2. The Feature Engineer shall compute the slope (前日比) of each MA (MA5_slope, MA20_slope, MA50_slope) as percentage change
3. When computing technical indicators, the Feature Engineer shall compute RSI14 (14-period Relative Strength Index)
4. The Feature Engineer shall compute MACD components: macd, signal line, and histogram
5. While calculating MACD, the Feature Engineer shall use standard parameters (12, 26, 9 EMA periods)
6. The Feature Engineer shall compute Bollinger Bands with 20-period MA and 2 standard deviation offset (upper, middle, lower, band_width)
7. The Feature Engineer shall compute daily percentage change (pct_change) as (Close_today - Close_yesterday) / Close_yesterday
8. The Feature Engineer shall create lag features for past 5 days (lag1, lag2, lag3, lag4, lag5) of close price
9. When processing dates, the Feature Engineer shall generate one-hot encoded day-of-week features (Monday through Friday)
10. The Feature Engineer shall compute the target variable: target = 1 if Close_tomorrow > Close_today, else 0
11. If any required data is missing or insufficient for feature computation, the Feature Engineer shall handle missing values appropriately (forward fill or drop rows)
12. When feature generation is complete, the Feature Engineer shall return a DataFrame ready for model training

---

### 3. XGBoostモデル学習機能

**目的：** データサイエンティストとして、USDJPY方向性を予測するXGBoostモデルを学習したい。そうすることで、予測的な売買ルールを実装できる

#### 受け入れ条件

1. When feature data and target labels are provided, the Model Trainer shall split data chronologically (8:2 ratio, shuffle=False)
2. The Model Trainer shall use XGBClassifier with default hyperparameters: n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8
3. The Model Trainer shall train the XGBoost model on the training set (80% of data)
4. When training is complete, the Model Trainer shall compute and display Accuracy on the test set (20%)
5. The Model Trainer shall compute and display F1 Score on the test set
6. The Model Trainer shall generate and save a Confusion Matrix visualization
7. The Model Trainer shall compute Feature Importance from the trained model
8. The Model Trainer shall save Feature Importance as a plot to `model/feature_importance.png`
9. When model training is finished, the Model Trainer shall serialize and save the trained model to `model/xgb_model.json`
10. If model training fails or produces invalid results, the Model Trainer shall log detailed error information and raise an exception

---

### 4. 予測機能

**目的：** トレーダーとして、学習済みモデルから指定した日付の上昇確率と売買予測を取得したい。そうすることで、リアルタイムの売買判断ができる

#### 受け入れ条件

1. The Predictor shall load the trained XGBoost model from `model/xgb_model.json`
2. When feature data for a new date is provided, the Predictor shall compute prediction probabilities (0-1) for the positive class (上昇確率)
3. The Predictor shall also output binary predictions (0 or 1) based on probability threshold (default 0.5)
4. If the model file does not exist, the Predictor shall raise an informative error
5. The Predictor shall support batch prediction for multiple dates in a single call

---

### 5. バックテストエンジン

**目的：** ポートフォリオマネージャーとして、売買戦略の過去パフォーマンスを評価したい。そうすることで、戦略の妥当性を検証できる

#### 受け入れ条件

1. When OHLCV data and model predictions are provided, the Backtest Engine shall apply BUY entry rule: model_pred == 1 AND RSI < 50 AND MA20_slope > 0
2. The Backtest Engine shall apply SELL exit rule: model_pred == 0 AND RSI > 50 AND MA20_slope < 0
3. When a BUY signal is triggered, the Backtest Engine shall record entry date and entry price (Close price of the signal day)
4. While position is open, the Backtest Engine shall continuously monitor price for stop-loss condition: price <= entry_price * (1 - 0.003)
5. If stop-loss condition is met, the Backtest Engine shall exit position and record exit price and return percentage
6. While position is open, the Backtest Engine shall continuously monitor price for take-profit condition: price >= entry_price * (1 + 0.006)
7. If take-profit condition is met, the Backtest Engine shall exit position and record exit price and return percentage
8. When position is exited (by rule, stop-loss, or take-profit), the Backtest Engine shall record all trade details: date, entry_price, exit_price, return_percent, win_loss_flag, cumulative_pnl
9. The Backtest Engine shall compute and display summary metrics: 総損益, 勝率, 平均RR (Risk-Reward ratio), シャープレシオ, 最大ドローダウン
10. When backtest completes, the Backtest Engine shall generate a profit/loss curve graph saved to `backtest/pnl_curve.png`
11. The Backtest Engine shall export all trade records to `backtest/backtest_results.csv` with columns: Date, EntryPrice, ExitPrice, ReturnPercent, WinLoss, CumulativePnL

---

### 6. 可視化とレポート機能

**目的：** アナリストとして、モデルパフォーマンスと取引結果を可視化したい。そうすることで、戦略の強みと弱みを直感的に理解できる

#### 受け入れ条件

1. When Feature Importance data is computed, the Plotter shall generate and save a bar chart showing top N features to `model/feature_importance.png`
2. When backtest trade records are available, the Plotter shall generate and save a cumulative P&L curve chart to `backtest/pnl_curve.png`
3. The Plotter shall include grid, legend, and proper axis labels for all charts
4. When generating visualizations, the Plotter shall use appropriate color schemes and font sizes for readability
5. The Plotter shall support saving all plots in PNG format with appropriate resolution (dpi >= 100)

---

### 7. メイン実行フロー

**目的：** ユーザーとして、一つのコマンドでデータ取得からバックテスト・レポート生成まで全体のワークフローを実行したい。そうすることで、手動作業を最小化できる

#### 受け入れ条件

1. The main.py script shall orchestrate the following sequence: data fetch → feature generation → model training (or load) → backtest execution → report generation
2. When main.py is executed, the script shall first check if `data/usd_jpy.csv` exists
3. If data file does not exist, the main.py shall invoke Data Fetcher to retrieve and save USDJPY data
4. After data is available, main.py shall invoke Feature Engineer to generate training features
5. When features are ready, main.py shall check if `model/xgb_model.json` exists
6. If model file does not exist, main.py shall invoke Model Trainer to train a new model
7. If model file exists, main.py shall load the pre-trained model via Predictor
8. After model is available, main.py shall invoke Backtest Engine with feature data and predictions
9. When backtest completes, main.py shall invoke Plotter to generate all visualization plots
10. Upon completion, main.py shall print a summary report with key metrics and file locations
11. If any step fails, main.py shall catch the exception, log detailed error information, and exit gracefully

---

### 8. プロジェクト構造と依存パッケージ管理

**目的：** 開発者として、すべてのコンポーネントが正しく整理されたディレクトリ構造と依存パッケージが明確に定義されたいい。そうすることで、プロジェクト全体を保守・拡張しやすくなる

#### 受け入れ条件

1. The project shall be organized with the following directory structure:
   - `data/` → Data files (usd_jpy.csv)
   - `features/` → feature_engineer.py module
   - `model/` → train.py, predict.py, and saved model file (xgb_model.json)
   - `backtest/` → backtest.py module and results (backtest_results.csv)
   - `trader/` → MT5 integration module (mt5_trader.py, optional for future extension)
   - `utils/` → plot.py for visualization
   - `main.py` → Main orchestration script
   - `requirements.txt` → All Python dependencies with pinned versions
2. The project shall include a requirements.txt listing all mandatory dependencies: pandas, numpy, xgboost, scikit-learn, yfinance, ta, matplotlib
3. When dependencies are listed, requirements.txt shall include version specifications to ensure reproducibility
4. All Python modules shall use consistent import paths (e.g., `from features.feature_engineer import FeatureEngineer`)
5. The project shall include a .gitignore file to exclude: `__pycache__/`, `*.pyc`, `*.json`, `*.csv`, venv/
6. The project shall be executable with `python main.py` and shall complete without external manual intervention (except network calls to yfinance)

---

## 技術的制約

### タイムゾーン処理
- While processing date/time data, the system shall treat all dates in Japan Standard Time (JST, UTC+9)
- When retrieving data from yfinance (UTC), the system shall convert to JST if necessary

### 時系列データ整合性
- All data processing shall maintain chronological ordering
- The system shall never shuffle time-series data during model training (shuffle=False)

### パフォーマンス
- The Data Fetcher shall complete data retrieval within 60 seconds
- The Model Trainer shall complete training within 300 seconds for 3-year daily data
- The Backtest Engine shall complete backtest execution within 30 seconds

---

## 今後の拡張（スコープ外）
- MT5リアルトレード連携 (trader/mt5_trader.py は実装スケルトンのみ)
- ハイパーパラメータ自動チューニング
- 複数通貨ペア対応
- リスク管理パラメータの動的調整
