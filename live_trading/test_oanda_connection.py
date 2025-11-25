"""
OANDA Japan API Connection Test

OANDA APIへの接続とバリデーションをテストします
APIトークンとアカウントID設定後に実行してください
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# 環境変数読み込み
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from oanda_config import OandaConfig
from oanda_api_client import OandaAPIClient


def test_configuration():
    """設定をテスト"""
    print("\n" + "="*80)
    print("OANDA Configuration Test")
    print("="*80)

    # 設定を表示
    OandaConfig.print_config()

    # 設定を検証
    print("[1/4] Validating configuration...")
    if not OandaConfig.validate():
        print("✗ Configuration validation failed")
        return False
    print("✓ Configuration is valid")

    return True


def test_api_connection():
    """API接続をテスト"""
    print("\n[2/4] Testing API connection...")

    try:
        client = OandaAPIClient(OandaConfig)

        # 接続テスト
        if not client.validate_connection():
            print("✗ API connection failed")
            return False

        print("✓ API connection successful")
        return True

    except Exception as e:
        print(f"✗ API connection error: {e}")
        return False


def test_account_info(client: OandaAPIClient):
    """アカウント情報を取得・表示"""
    print("\n[3/4] Fetching account information...")

    try:
        info = client.get_account_info()

        print(f"✓ Account information retrieved:")
        print(f"  Account ID: {info.account_id}")
        print(f"  Balance: ${info.balance:,.2f}")
        print(f"  Unrealized P&L: ${info.unrealized_pl:,.2f}")
        print(f"  Used Margin: ${info.used_margin:,.2f}")
        print(f"  Available Margin: ${info.available_margin:,.2f}")
        print(f"  Margin Level: {info.margin_level:.2%}")

        return True

    except Exception as e:
        print(f"✗ Failed to fetch account info: {e}")
        return False


def test_current_price(client: OandaAPIClient):
    """現在の価格を取得・表示"""
    print("\n[4/4] Fetching current price...")

    try:
        price = client.get_current_price(OandaConfig.INSTRUMENT)

        if price:
            print(f"✓ Current price retrieved:")
            print(f"  {OandaConfig.INSTRUMENT}: {price:.4f}")
            return True
        else:
            print(f"✗ Failed to get current price")
            return False

    except Exception as e:
        print(f"✗ Failed to fetch current price: {e}")
        return False


def main():
    """メイン実行"""
    print("\n" + "="*80)
    print("OANDA Japan FX Automated Trading System")
    print("Connection & Configuration Test")
    print("="*80)

    # 設定テスト
    if not test_configuration():
        print("\n✗ Configuration test failed")
        print("\n対策:")
        print("1. .env.example を .env にコピー")
        print("2. .env に OANDA_ENVIRONMENT, API_TOKEN, ACCOUNT_ID を設定")
        print("3. 再度実行")
        return 1

    # API接続テスト
    if not test_api_connection():
        print("\n✗ API connection test failed")
        print("\n対策:")
        print("1. APIトークンが正しいか確認")
        print("2. アカウントIDが正しいか確認")
        print("3. ネットワーク接続を確認")
        print("4. OANDA APIサーバーが起動しているか確認")
        return 1

    # クライアント作成
    try:
        client = OandaAPIClient(OandaConfig)
    except Exception as e:
        print(f"\n✗ Failed to create API client: {e}")
        return 1

    # アカウント情報取得
    if not test_account_info(client):
        print("\n✗ Account info test failed")
        return 1

    # 現在の価格取得
    if not test_current_price(client):
        print("\n✗ Current price test failed")
        return 1

    # 全テスト成功
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED")
    print("="*80)
    print("\nOANDA API is ready for automated trading!")
    print("\n次のステップ:")
    print("1. live_trading/test_oanda_connection.py で接続確認 ✓ (完了)")
    print("2. live_trading/paper_trading_bot.py でPaper Trading検証 → 次")
    print("3. live_trading/automated_trader.py で本番運用 (準備完了時)")

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())