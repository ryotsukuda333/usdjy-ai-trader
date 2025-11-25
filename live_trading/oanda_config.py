"""
OANDA Japan API Configuration & Authentication

OANDA Japan FX自動売買用の設定ファイル
APIトークンと接続情報を管理
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .envファイルから設定読み込み
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

class OandaConfig:
    """OANDA Japan API Configuration"""

    # APIエンドポイント
    OANDA_DEMO_URL = "https://api-fxpractice.oanda.com"  # デモ環境 (Paper Trading)
    OANDA_LIVE_URL = "https://api-fxtrade.oanda.com"      # 本番環境 (Real Trading)

    # 現在のモード (デモ/本番)
    ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "demo")

    # APIトークン (本番)
    API_TOKEN_LIVE = os.getenv("OANDA_API_TOKEN_LIVE", "")

    # APIトークン (デモ/Paper Trading)
    API_TOKEN_DEMO = os.getenv("OANDA_API_TOKEN_DEMO", "")

    # アカウントID (本番)
    ACCOUNT_ID_LIVE = os.getenv("OANDA_ACCOUNT_ID_LIVE", "")

    # アカウントID (デモ)
    ACCOUNT_ID_DEMO = os.getenv("OANDA_ACCOUNT_ID_DEMO", "")

    # 現在使用する認証情報
    @classmethod
    def get_base_url(cls):
        """現在のモードに応じたベースURLを取得"""
        return cls.OANDA_LIVE_URL if cls.ENVIRONMENT == "live" else cls.OANDA_DEMO_URL

    @classmethod
    def get_api_token(cls):
        """現在のモードに応じたAPIトークンを取得"""
        return cls.API_TOKEN_LIVE if cls.ENVIRONMENT == "live" else cls.API_TOKEN_DEMO

    @classmethod
    def get_account_id(cls):
        """現在のモードに応じたアカウントIDを取得"""
        return cls.ACCOUNT_ID_LIVE if cls.ENVIRONMENT == "live" else cls.ACCOUNT_ID_DEMO

    # トレード設定
    INSTRUMENT = "USD_JPY"  # 取引通貨ペア

    # ポジションサイジング設定
    RISK_PER_TRADE = 0.01  # 1トレードあたりのリスク (1%)
    ACCOUNT_BALANCE_REFERENCE = 100000.0  # 参照資金 (ポジションサイズ計算用)

    # Entry Threshold (Phase 5-B最適値)
    ENTRY_THRESHOLD = 0.70

    # TP/SL設定 (Phase 5-Dで検証)
    TAKE_PROFIT_PCT = 0.012  # 1.20%
    STOP_LOSS_PCT = 0.003    # 0.30%

    # トレーリングストップ設定
    TRAILING_STOP_PCT = 0.02  # 2.0%

    # 日次損失制限
    DAILY_LOSS_LIMIT_PCT = 0.01  # 1.0%

    # リクエストタイムアウト
    REQUEST_TIMEOUT = 10  # 秒

    # データストリーム設定
    STREAM_HEARTBEAT_INTERVAL = 35  # ハートビート間隔 (秒)

    # ロギング設定
    LOG_LEVEL = "INFO"
    LOG_FILE = Path(__file__).parent.parent / "logs" / "oanda_trading.log"

    @classmethod
    def validate(cls):
        """設定の妥当性チェック"""
        errors = []

        if not cls.get_api_token():
            errors.append(f"❌ API Token not set for {cls.ENVIRONMENT} environment")

        if not cls.get_account_id():
            errors.append(f"❌ Account ID not set for {cls.ENVIRONMENT} environment")

        if cls.RISK_PER_TRADE <= 0 or cls.RISK_PER_TRADE > 0.1:
            errors.append(f"❌ Invalid risk per trade: {cls.RISK_PER_TRADE}")

        if cls.ENTRY_THRESHOLD < 0 or cls.ENTRY_THRESHOLD > 1.0:
            errors.append(f"❌ Invalid entry threshold: {cls.ENTRY_THRESHOLD}")

        if errors:
            print("\n".join(errors))
            return False

        return True

    @classmethod
    def print_config(cls):
        """設定情報を表示"""
        print("\n" + "="*80)
        print("OANDA Japan Configuration")
        print("="*80)
        print(f"Environment: {cls.ENVIRONMENT.upper()}")
        print(f"Base URL: {cls.get_base_url()}")
        print(f"Account ID: {cls.get_account_id()}")
        print(f"Instrument: {cls.INSTRUMENT}")
        print(f"\nTrade Settings:")
        print(f"  Entry Threshold: {cls.ENTRY_THRESHOLD}")
        print(f"  Take-Profit: {cls.TAKE_PROFIT_PCT*100:.2f}%")
        print(f"  Stop-Loss: {cls.STOP_LOSS_PCT*100:.2f}%")
        print(f"  Trailing Stop: {cls.TRAILING_STOP_PCT*100:.2f}%")
        print(f"  Daily Loss Limit: {cls.DAILY_LOSS_LIMIT_PCT*100:.2f}%")
        print(f"  Risk per Trade: {cls.RISK_PER_TRADE*100:.2f}%")
        print("="*80 + "\n")