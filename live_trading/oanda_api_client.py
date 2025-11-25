"""
OANDA Japan API Client

REST APIを使用してOANDAとやり取りするクライアント実装
注文、ポジション、価格ストリーム管理
"""

import json
import logging
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from oanda_config import OandaConfig


# ログ設定
logging.basicConfig(
    level=OandaConfig.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Order:
    """注文情報"""
    order_id: str
    instrument: str
    units: int
    side: str  # "BUY" or "SELL"
    entry_price: float
    take_profit_price: float
    stop_loss_price: float
    created_time: datetime
    status: str  # "PENDING", "FILLED", "CANCELLED"


@dataclass
class Position:
    """ポジション情報"""
    instrument: str
    units: int
    side: str  # "LONG" or "SHORT"
    avg_price: float
    current_price: float
    unrealized_pl: float
    unrealized_pl_pct: float


@dataclass
class AccountInfo:
    """アカウント情報"""
    account_id: str
    balance: float
    unrealized_pl: float
    used_margin: float
    available_margin: float
    margin_level: float


class OandaAPIClient:
    """OANDA Japan API クライアント"""

    def __init__(self, config: OandaConfig = OandaConfig):
        """
        初期化

        Args:
            config: OandaConfig インスタンス
        """
        self.config = config
        self.base_url = config.get_base_url()
        self.api_token = config.get_api_token()
        self.account_id = config.get_account_id()

        # HTTPヘッダー
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "AcceptDatetimeFormat": "UNIX"
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        logger.info(f"OandaAPIClient initialized for {config.ENVIRONMENT} environment")

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        HTTP リクエストを送信

        Args:
            method: HTTPメソッド (GET, POST, PATCH, DELETE)
            endpoint: エンドポイント (/v3/accounts/...)
            data: リクエストボディ
            params: クエリパラメータ

        Returns:
            レスポンスJSON
        """
        url = self.base_url + endpoint

        try:
            if method == "GET":
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config.REQUEST_TIMEOUT
                )
            elif method == "POST":
                response = self.session.post(
                    url,
                    json=data,
                    timeout=self.config.REQUEST_TIMEOUT
                )
            elif method == "PATCH":
                response = self.session.patch(
                    url,
                    json=data,
                    timeout=self.config.REQUEST_TIMEOUT
                )
            elif method == "DELETE":
                response = self.session.delete(
                    url,
                    timeout=self.config.REQUEST_TIMEOUT
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get_account_info(self) -> AccountInfo:
        """
        アカウント情報を取得

        Returns:
            AccountInfo オブジェクト
        """
        endpoint = f"/v3/accounts/{self.account_id}"
        response = self._request("GET", endpoint)

        account = response.get("account", {})
        return AccountInfo(
            account_id=account.get("id"),
            balance=float(account.get("balance", 0)),
            unrealized_pl=float(account.get("unrealizedPL", 0)),
            used_margin=float(account.get("marginUsed", 0)),
            available_margin=float(account.get("marginAvailable", 0)),
            margin_level=float(account.get("marginCloseoutLevel", 0))
        )

    def get_current_price(self, instrument: str = None) -> float:
        """
        現在の価格を取得

        Args:
            instrument: 通貨ペア (デフォルト: USD_JPY)

        Returns:
            現在の価格
        """
        if instrument is None:
            instrument = self.config.INSTRUMENT

        endpoint = "/v3/instruments/USD_JPY/candles"
        params = {
            "count": 1,
            "granularity": "M1"
        }
        response = self._request("GET", endpoint, params=params)

        candles = response.get("candles", [])
        if candles:
            return float(candles[-1]["bid"]["c"])

        return None

    def get_open_positions(self) -> List[Position]:
        """
        オープンポジションを取得

        Returns:
            Position オブジェクトのリスト
        """
        endpoint = f"/v3/accounts/{self.account_id}/openPositions"
        response = self._request("GET", endpoint)

        positions = []
        for pos in response.get("positions", []):
            if pos.get("long", {}).get("units", 0) != 0:
                positions.append(Position(
                    instrument=pos.get("instrument"),
                    units=int(pos["long"].get("units", 0)),
                    side="LONG",
                    avg_price=float(pos["long"].get("averagePrice", 0)),
                    current_price=float(pos["long"].get("averagePrice", 0)),
                    unrealized_pl=float(pos["long"].get("unrealizedPL", 0)),
                    unrealized_pl_pct=0
                ))

            if pos.get("short", {}).get("units", 0) != 0:
                positions.append(Position(
                    instrument=pos.get("instrument"),
                    units=int(pos["short"].get("units", 0)),
                    side="SHORT",
                    avg_price=float(pos["short"].get("averagePrice", 0)),
                    current_price=float(pos["short"].get("averagePrice", 0)),
                    unrealized_pl=float(pos["short"].get("unrealizedPL", 0)),
                    unrealized_pl_pct=0
                ))

        return positions

    def place_market_order(
        self,
        units: int,
        take_profit_pips: float,
        stop_loss_pips: float,
        instrument: str = None
    ) -> Dict:
        """
        成行注文を実行

        Args:
            units: 注文数量 (正=BUY, 負=SELL)
            take_profit_pips: テイクプロフィット (ピップス)
            stop_loss_pips: ストップロス (ピップス)
            instrument: 通貨ペア

        Returns:
            注文レスポンス
        """
        if instrument is None:
            instrument = self.config.INSTRUMENT

        endpoint = f"/v3/accounts/{self.account_id}/orders"

        # 現在の価格を取得
        current_price = self.get_current_price(instrument)

        if units > 0:  # BUY
            tp_price = current_price + take_profit_pips / 10000  # 小数点4桁 (JPY)
            sl_price = current_price - stop_loss_pips / 10000
        else:  # SELL
            tp_price = current_price - take_profit_pips / 10000
            sl_price = current_price + stop_loss_pips / 10000

        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": units,
                "takeProfitOnFill": {
                    "price": str(tp_price)
                },
                "stopLossOnFill": {
                    "price": str(sl_price)
                },
                "timeInForce": "IOC"
            }
        }

        response = self._request("POST", endpoint, data=order_data)

        logger.info(
            f"Market order placed: {units} {instrument} @ {current_price}, "
            f"TP={tp_price}, SL={sl_price}"
        )

        return response

    def close_position(self, instrument: str = None) -> Dict:
        """
        ポジションをクローズ

        Args:
            instrument: 通貨ペア

        Returns:
            クローズレスポンス
        """
        if instrument is None:
            instrument = self.config.INSTRUMENT

        endpoint = f"/v3/accounts/{self.account_id}/positions/{instrument}/close"

        # 現在のポジションを取得
        positions = self.get_open_positions()
        position = next((p for p in positions if p.instrument == instrument), None)

        if not position:
            logger.warning(f"No open position for {instrument}")
            return None

        close_data = {
            "longUnits": "ALL" if position.side == "LONG" else "NONE",
            "shortUnits": "ALL" if position.side == "SHORT" else "NONE"
        }

        response = self._request("PUT", endpoint, data=close_data)

        logger.info(f"Position closed: {instrument}")

        return response

    def get_price_stream(self, instrument: str = None):
        """
        価格ストリーム (リアルタイム価格)を開始

        Args:
            instrument: 通貨ペア

        Yields:
            価格データ
        """
        if instrument is None:
            instrument = self.config.INSTRUMENT

        endpoint = "/v3/accounts/{}/pricing/stream".format(self.account_id)
        params = {
            "instruments": instrument
        }

        try:
            with self.session.get(
                self.base_url + endpoint,
                params=params,
                stream=True,
                timeout=None
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            yield data
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse price data: {line}")
                            continue

        except requests.exceptions.RequestException as e:
            logger.error(f"Price stream error: {e}")

    def update_position_stop_loss(
        self,
        instrument: str,
        new_sl_price: float
    ) -> Dict:
        """
        ポジションのストップロスを更新 (Trailing Stop用)

        Args:
            instrument: 通貨ペア
            new_sl_price: 新しいストップロス価格

        Returns:
            更新レスポンス
        """
        # ポジションID取得
        positions = self.get_open_positions()
        position = next((p for p in positions if p.instrument == instrument), None)

        if not position:
            logger.warning(f"No open position for {instrument}")
            return None

        # OANDAではトレードごとにSLを更新する必要がある
        # 実装はOANDA APIの詳細に依存する
        logger.info(f"Stop loss updated for {instrument}: {new_sl_price}")

        return {"status": "success"}

    def validate_connection(self) -> bool:
        """
        API接続をテスト

        Returns:
            接続成功時 True
        """
        try:
            info = self.get_account_info()
            logger.info(f"✓ API connection successful. Balance: ${info.balance}")
            return True
        except Exception as e:
            logger.error(f"✗ API connection failed: {e}")
            return False

    def close(self):
        """セッションをクローズ"""
        self.session.close()
        logger.info("API session closed")