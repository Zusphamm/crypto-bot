"""
===============================================================================
 TOOL 1: BINANCE MULTI-TIMEFRAME PRICE FETCHER
===============================================================================
 Kéo biểu đồ giá (OHLCV) từ Binance cho tất cả các khung từ M1 đến H12.
 Hỗ trợ:
   - Tối đa 10,000 nến mỗi khung (auto-paginate qua giới hạn 1000/request)
   - Lookback lên đến 5 năm
   - Cả Spot và Futures (USDT-M Perpetual)
   - Xuất DataFrame chuẩn hóa với index datetime UTC
===============================================================================
"""

import time
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Tất cả các khung từ M1 đến H12 mà Binance hỗ trợ
SUPPORTED_INTERVALS: List[str] = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
]

# Số phút/giờ mỗi khung — dùng để tính startTime
_INTERVAL_MINUTES: Dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
}

BINANCE_LIMIT_PER_REQUEST = 1000      # Binance cap per-request
DEFAULT_MAX_CANDLES = 10_000          # Giới hạn user yêu cầu
DEFAULT_MAX_YEARS = 5                 # Lookback tối đa 5 năm

SPOT_BASE_URL = "https://api.binance.com/api/v3/klines"
FUT_BASE_URL = "https://fapi.binance.com/fapi/v1/klines"


# -----------------------------------------------------------------------------
# Core fetcher
# -----------------------------------------------------------------------------

def _fetch_klines_single(
    symbol: str,
    interval: str,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: int = 1000,
    market: str = "spot",
    session: Optional[requests.Session] = None,
) -> List[list]:
    """Gọi 1 request đến endpoint /klines (tối đa 1000 nến)."""
    url = FUT_BASE_URL if market == "futures" else SPOT_BASE_URL
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    sess = session or requests
    resp = sess.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_klines(
    symbol: str,
    interval: str,
    max_candles: int = DEFAULT_MAX_CANDLES,
    max_years: float = DEFAULT_MAX_YEARS,
    market: str = "spot",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Kéo đến `max_candles` nến của 1 khung thời gian, không vượt quá `max_years`.

    Args:
        symbol:      Cặp giao dịch (vd 'BTCUSDT').
        interval:    Khung thời gian (vd '1m', '15m', '4h').
        max_candles: Số nến tối đa muốn có (cap = 10,000).
        max_years:   Lookback tối đa (năm). Mặc định 5.
        market:      'spot' hoặc 'futures'.
        verbose:     In log tiến trình.

    Returns:
        DataFrame với index datetime UTC, cột:
        [open, high, low, close, volume, quote_volume, trades,
         taker_buy_base, taker_buy_quote].
    """
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(
            f"Interval '{interval}' không hỗ trợ. "
            f"Các khung hợp lệ: {SUPPORTED_INTERVALS}"
        )

    max_candles = min(max_candles, DEFAULT_MAX_CANDLES)

    # Khoảng thời gian theo max_years
    end_dt = datetime.now(timezone.utc)
    earliest_dt = end_dt - timedelta(days=365 * max_years)

    # Khoảng thời gian theo max_candles
    minutes_per_candle = _INTERVAL_MINUTES[interval]
    candles_dt = end_dt - timedelta(minutes=minutes_per_candle * max_candles)

    # Start = max(2 ràng buộc) → đảm bảo KHÔNG vượt 5 năm VÀ KHÔNG vượt 10k nến
    start_dt = max(earliest_dt, candles_dt)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    if verbose:
        est_candles = min(
            max_candles,
            int((end_ms - start_ms) / (minutes_per_candle * 60 * 1000)),
        )
        print(
            f"  [{interval:>3}] Kéo ~{est_candles} nến từ "
            f"{start_dt:%Y-%m-%d %H:%M} → {end_dt:%Y-%m-%d %H:%M} UTC"
        )

    # Paginate
    all_rows: List[list] = []
    cursor_ms = start_ms
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (price-analyzer)"})
    max_retries = 3
    fail_count = 0

    while cursor_ms < end_ms and len(all_rows) < max_candles:
        remaining = max_candles - len(all_rows)
        req_limit = min(BINANCE_LIMIT_PER_REQUEST, remaining)

        try:
            batch = _fetch_klines_single(
                symbol=symbol,
                interval=interval,
                start_ms=cursor_ms,
                end_ms=end_ms,
                limit=req_limit,
                market=market,
                session=session,
            )
            fail_count = 0  # reset sau khi thành công
        except (requests.HTTPError, requests.ConnectionError) as e:
            fail_count += 1
            if verbose:
                print(f"      ! Lỗi mạng ({fail_count}/{max_retries}): {e}")
            if fail_count >= max_retries:
                raise RuntimeError(
                    f"Không thể kết nối Binance sau {max_retries} lần thử. "
                    f"Nguyên nhân có thể: (1) IP bị Binance chặn "
                    f"(một số quốc gia/cloud IP bị block — thử dùng VPN), "
                    f"(2) mạng lỗi, (3) symbol sai. "
                    f"Lỗi cuối: {e}"
                )
            time.sleep(2)
            continue

        if not batch:
            break

        all_rows.extend(batch)

        # Cursor = thời gian open của nến cuối + 1 đơn vị khung
        last_open_ms = batch[-1][0]
        next_cursor = last_open_ms + minutes_per_candle * 60 * 1000
        if next_cursor <= cursor_ms:
            break  # không tiến được — thoát
        cursor_ms = next_cursor

        # Nếu batch nhỏ hơn limit → đã hết data
        if len(batch) < req_limit:
            break

        time.sleep(0.15)  # tôn trọng rate limit của Binance

    if not all_rows:
        raise RuntimeError(f"Không kéo được dữ liệu cho {symbol} {interval}")

    # Cắt nếu lỡ vượt max_candles (do batch cuối có thể trả thêm)
    all_rows = all_rows[:max_candles]

    return _klines_to_dataframe(all_rows)


def _klines_to_dataframe(rows: List[list]) -> pd.DataFrame:
    """Chuyển raw klines → DataFrame chuẩn hóa."""
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)

    # Ép kiểu số
    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_volume", "taker_buy_base", "taker_buy_quote",
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["trades"] = df["trades"].astype(int)

    # Index = open_time UTC
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")

    df = df[numeric_cols + ["trades"]]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


# -----------------------------------------------------------------------------
# Multi-timeframe convenience wrapper
# -----------------------------------------------------------------------------

def fetch_all_timeframes(
    symbol: str,
    intervals: Optional[List[str]] = None,
    max_candles: int = DEFAULT_MAX_CANDLES,
    max_years: float = DEFAULT_MAX_YEARS,
    market: str = "spot",
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Kéo dữ liệu cho nhiều khung một lúc.

    Returns:
        dict { interval: DataFrame }.
    """
    intervals = intervals or SUPPORTED_INTERVALS
    result: Dict[str, pd.DataFrame] = {}

    if verbose:
        print(f"\n🔻 Đang kéo dữ liệu {symbol} ({market.upper()}) "
              f"cho {len(intervals)} khung thời gian\n")

    for iv in intervals:
        try:
            df = fetch_klines(
                symbol=symbol,
                interval=iv,
                max_candles=max_candles,
                max_years=max_years,
                market=market,
                verbose=verbose,
            )
            result[iv] = df
            if verbose:
                print(
                    f"      ✓ {len(df):>5} nến  "
                    f"({df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d})"
                )
        except Exception as e:
            print(f"      ✗ {iv}: {e}")
            result[iv] = pd.DataFrame()

    return result


# -----------------------------------------------------------------------------
# Demo / self-test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Demo nhỏ: kéo 500 nến 1h cho BTCUSDT
    df = fetch_klines("BTCUSDT", "1h", max_candles=500, verbose=True)
    print(f"\nShape: {df.shape}")
    print(df.head(3))
    print("...")
    print(df.tail(3))
