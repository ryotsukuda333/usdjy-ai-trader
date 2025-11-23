# USDJPY AI Trader

XGBoostを使用したUSDJPY（米ドル/日本円）自動トレードシステム。3年間の過去データを用いて、技術的指標に基づいた買い売り信号を生成し、バックテストを実行します。

## プロジェクト概要

このプロジェクトは、以下の8つのタスクで構成されています：

- **Task 1**: プロジェクト初期化（環境構築）
- **Task 2**: データ取得・検証（yfinance から USDJPY 3年分を取得）
- **Task 3**: 特徴量エンジニアリング（30個の技術的指標を生成）
- **Task 4**: モデルトレーニング（XGBoost による分類モデル）
- **Task 5**: 予測生成（学習済みモデルによる買い売い予測）
- **Task 6**: バックテスト（過去データに対する取引シミュレーション）
- **Task 7**: 可視化（エクイティカーブの生成）
- **Task 8**: パイプライン統合（全タスクの自動実行）

## プロジェクト構成

```
usdjpy-ai-trader/
├── main.py                 # 統合パイプライン（全処理の自動実行）
├── requirements.txt        # Python依存パッケージ
│
├── features/              # データ取得・特徴量エンジニアリング
│   ├── data_fetcher.py    # USDJPY データ取得（Task 2）
│   └── feature_engineer.py # 技術的指標生成（Task 3）
│
├── model/                 # モデル学習・予測
│   ├── train.py           # XGBoost トレーニング（Task 4）
│   ├── predict.py         # 予測生成（Task 5）
│   ├── xgb_model.json     # 学習済みモデル（自動生成）
│   ├── feature_columns.json # 学習時の特徴量リスト（自動生成）
│   └── feature_importance.png # 特徴量重要度チャート（自動生成）
│
├── backtest/              # バックテスト実行
│   ├── backtest.py        # バックテストエンジン（Task 6）
│   └── backtest_results.csv # 取引結果（自動生成）
│
├── trader/                # 可視化
│   ├── plotter.py         # エクイティカーブ生成（Task 7）
│   └── backtest_equity_curve.png # 結果グラフ（自動生成）
│
├── data/                  # データ保存
│   └── ohlcv_usdjpy.csv   # OHLCV データ（自動生成）
│
├── tests/                 # テストスイート
│   ├── test_project_setup.py
│   ├── test_data_fetcher.py
│   ├── test_feature_engineer.py
│   ├── test_model_trainer.py
│   ├── test_backtest.py
│   └── test_plotter.py
│
├── utils/                 # ユーティリティ
│   └── errors.py          # カスタム例外定義
│
└── .kiro/                 # 仕様書（Kiro-style spec driven development）
    └── specs/
        └── xgboost-usdjpy-trader/
            ├── spec.json
            ├── requirements.md
            ├── design.md
            ├── tasks.md
            └── research.md
```

## 使用方法

### 1. 環境構築

```bash
# 仮想環境の作成と有効化
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# Windows の場合: venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. 完全なパイプラインの実行

全タスク（データ取得→特徴量生成→モデル学習→予測→バックテスト→可視化）を一括実行：

```bash
python3 main.py
```

**出力例:**
```
============================================================
USDJPY AI TRADER - COMPLETE PIPELINE
============================================================

[8.1] Fetching USDJPY data (3 years)...
✓ Fetched 732 candles
  Date range: 2022-11-21 00:00:00 to 2025-11-21 00:00:00

[8.2] Engineering features from OHLCV...
✓ Engineered 31 features
  Features: Close, High, Low, Open, Volume... (31 total)

[8.3] Training model and generating predictions...
✓ Model trained successfully
✓ Generated 732 predictions
    Buy signals: 454
    Sell signals: 278

[8.4] Executing backtest and visualizing results...
✓ Backtest complete: 19 trades
  BACKTEST STATISTICS:
  - Total Trades: 19
  - Wins: 12, Losses: 7
  - Win Rate: 63.16%
  - Total Return: +5.10%
  - Avg Return/Trade: +0.268%
✓ Visualization saved: ./trader/backtest_equity_curve.png

============================================================
✓ PIPELINE COMPLETE
============================================================
```

### 3. 個別タスクの実行

各モジュールを個別に使用することも可能です：

#### 3.1 データ取得のみ
```python
from features.data_fetcher import fetch_usdjpy_data

df_ohlcv = fetch_usdjpy_data(years=3)
print(f"取得行数: {len(df_ohlcv)}")
```

#### 3.2 特徴量エンジニアリング
```python
from features.feature_engineer import engineer_features

df_features = engineer_features(df_ohlcv)
print(f"生成された特徴量数: {len(df_features.columns)}")
```

#### 3.3 モデル学習
```python
from model.train import train_model, save_model

model = train_model(df_features, test_mode=False)
save_model(model, df_features)
```

#### 3.4 予測生成
```python
from model.predict import predict

predictions = predict(df_features)
print(f"買いシグナル: {(predictions == 1).sum()}")
print(f"売りシグナル: {(predictions == 0).sum()}")
```

#### 3.5 バックテスト実行
```python
from backtest.backtest import run_backtest

trades = run_backtest(df_ohlcv, df_features, predictions)
print(f"取引数: {len(trades)}")
print(f"勝率: {(trades['win_loss'] == 1).sum() / len(trades) * 100:.2f}%")
```

#### 3.6 結果可視化
```python
from trader.plotter import plot_backtest_results

plot_path = plot_backtest_results(trades)
print(f"グラフ保存位置: {plot_path}")
```

## テストの実行

```bash
# 全テストを実行
pytest tests/ -v

# 特定のテストモジュールを実行
pytest tests/test_data_fetcher.py -v
pytest tests/test_feature_engineer.py -v
pytest tests/test_model_trainer.py -v
pytest tests/test_backtest.py -v
pytest tests/test_plotter.py -v

# カバレッジレポート付きで実行
pytest tests/ --cov=features --cov=model --cov=backtest --cov=trader
```

## 主な特徴量（Task 3）

### 基本OHLCV
- Open, High, Low, Close, Volume

### トレンド指標
- MA5, MA20, MA50（移動平均）
- MA5_slope, MA20_slope, MA50_slope（傾斜）
- MACD, MACD Signal, MACD Histogram

### モメンタム指標
- RSI14（相対力指数）

### ボラティリティ指標
- Bollinger Bands Upper/Middle/Lower
- Bollinger Bands Width

### その他
- Pct_change（前日比変動率）
- Lag1～Lag5（過去5日間の遅延値）
- Mon, Tue, Wed, Thu, Fri（曜日ダミー変数）

**合計31個の特徴量**

## モデル構成（Task 4）

### XGBoost 分類器

```
n_estimators: 300
max_depth: 5
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
objective: binary:logistic
```

### 訓練データスプリット
- 8:2の時系列対応スプリット（シャッフルなし）
- 訓練データ: 585行
- テストデータ: 147行

### パフォーマンス
- 精度（Accuracy）: 48.98%
- F1スコア: 0.6269

## バックテスト仕様（Task 6）

### トレード信号ロジック

**買いシグナル（BUY）:**
```
予測値 == 1 AND RSI14 < 50 AND MA20の傾斜 > 0
```

**売りシグナル（SELL）:**
```
予測値 == 0 AND RSI14 > 50 AND MA20の傾斜 < 0
```

### 利益確定・損切り
- 利益確定（Take Profit）: +0.60%
- 損切り（Stop Loss）: -0.30%
- 優先度: TP > SL > 通常決済

### 最終成績

| 指標 | 値 |
|------|-----|
| 総取引数 | 19 |
| 勝ち | 12 |
| 負け | 7 |
| **勝率** | **63.16%** |
| **総リターン** | **+5.10%** |
| 平均リターン/取引 | +0.268% |

## ファイル出力一覧

| ファイル | 説明 | 自動生成 |
|---------|------|---------|
| data/ohlcv_usdjpy.csv | OHLCV データ | Task 2 |
| model/xgb_model.json | 学習済みモデル | Task 4 |
| model/feature_columns.json | 学習時の特徴量リスト | Task 4 |
| model/feature_importance.png | 特徴量重要度チャート | Task 4 |
| backtest/backtest_results.csv | 取引結果詳細 | Task 6 |
| trader/backtest_equity_curve.png | エクイティカーブ | Task 7 |

## トラブルシューティング

### yfinance の警告メッセージ
```
FutureWarning: YF.download() has changed argument auto_adjust default to True
```
これは yfinance のバージョン変更による警告で、機能には影響しません。

### モデルがない場合
```
ModelError: Model file not found
```
**解決策:** `python3 main.py` を実行してモデルを学習・保存してください。

### データ長のミスマッチ
```
BacktestError: Input data lengths do not match
```
**解決策:** 特徴量エンジニアリング時に初期50行が dropna() で削除されるため、OHLCV データが自動で同じ長さに揃えられます（main.py の行 54-59 を参照）。

## 開発ガイドライン

### エラーハンドリング
全てのモジュールは `utils/errors.py` で定義されたカスタム例外を使用：
- `DataError`: データ取得・検証エラー
- `FeatureEngineeringError`: 特徴量生成エラー
- `ModelError`: モデル学習・予測エラー
- `BacktestError`: バックテストエラー
- `VisualizationError`: 可視化エラー

### テスト駆動開発
各モジュールは 100% テスト対応：
- `tests/` ディレクトリ内のテストを実行して機能検証
- テストは Given-When-Then フォーマットで記述

### ログ出力
進捗状況は各関数で `print()` で出力されます：
- ✓ 成功時はチェックマーク
- ⚠ 警告時は警告マーク
- ❌ エラー時はバツマーク

## 次のステップ

### 短期（パフォーマンス改善）
1. **ハイパーパラメータチューニング** - GridSearch で最適パラメータを探索
2. **特徴量追加** - 新しい技術的指標の追加テスト
3. **ウィンドウサイズ最適化** - MA, RSI, MACD のウィンドウサイズ調整

### 中期（機能拡張）
1. **複数通貨ペア対応** - EUR/USD, GBP/JPY など
2. **リアルタイム予測** - ライブデータへの対応
3. **ポジション管理** - マルチポジション・ポートフォリオ管理

### 長期（プロダクション化）
1. **Web ダッシュボード** - Flask/Streamlit でリアルタイム表示
2. **Docker 化** - コンテナ化による配布
3. **CI/CD パイプライン** - GitHub Actions による自動テスト・デプロイ
4. **API 実装** - REST API によるモデル公開

## ライセンス

MIT License

## 作者

Tsukuda

## 参考資料

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [TA-Lib Python](https://github.com/bukosabino/ta)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
