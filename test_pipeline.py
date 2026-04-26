"""
Test end-to-end pipeline với synthetic BTC data 5 năm.
Mô phỏng: có bull runs, bear markets, sideway periods — giống crypto thực.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")

from multi_timeframe import multi_timeframe_analysis, print_report
from binance_fetcher import SUPPORTED_INTERVALS


def synthesize_btc_like_data(interval: str, n_candles: int) -> pd.DataFrame:
    """Mô phỏng data crypto giống BTC với regime shifts."""
    np.random.seed(hash(interval) % 2**32)

    # Base drift + volatility theo interval (khung lớn → volatility trên mỗi bar lớn hơn)
    base_vol_map = {
        "1m": 0.0005, "3m": 0.0008, "5m": 0.001, "15m": 0.002, "30m": 0.003,
        "1h": 0.005, "2h": 0.007, "4h": 0.012, "6h": 0.015, "8h": 0.018, "12h": 0.022,
    }
    vol = base_vol_map.get(interval, 0.005)

    rets = np.random.randn(n_candles) * vol

    # Inject vài regime
    segs = np.array_split(np.arange(n_candles), 6)
    drifts = [+0.5, -0.3, +0.2, -0.5, +0.4, +0.1]  # 6 regime khác nhau
    for seg, d in zip(segs, drifts):
        rets[seg] += d * vol * 0.3

    # Generate price
    start_price = 50000.0
    close = start_price * np.exp(np.cumsum(rets))

    # OHLC synthesis
    high = close * (1 + np.abs(np.random.randn(n_candles)) * vol * 0.5)
    low = close * (1 - np.abs(np.random.randn(n_candles)) * vol * 0.5)
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    minutes_map = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
                    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720}
    freq = f"{minutes_map[interval]}min"

    end = pd.Timestamp("2026-04-20", tz="UTC")
    idx = pd.date_range(end=end, periods=n_candles, freq=freq)

    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.abs(np.random.randn(n_candles) * 1000 + 5000),
        "quote_volume": np.abs(np.random.randn(n_candles) * 1e7 + 5e7),
        "trades": (np.random.rand(n_candles) * 1000 + 100).astype(int),
        "taker_buy_base": np.abs(np.random.randn(n_candles) * 500 + 2500),
        "taker_buy_quote": np.abs(np.random.randn(n_candles) * 1e6 + 2e7),
    }, index=idx)


if __name__ == "__main__":
    print("🧪 Sinh dữ liệu synthetic giống BTC cho tất cả khung M1 → H12...\n")

    data = {}
    for iv in SUPPORTED_INTERVALS:
        # Lấy 5000 nến mỗi khung để test nhanh
        df = synthesize_btc_like_data(iv, 5000)
        data[iv] = df
        print(f"  [{iv:>4}] {len(df):>5} nến, last_close = {df['close'].iloc[-1]:,.2f}")

    print("\n🚀 Chạy full pipeline analysis...")
    result = multi_timeframe_analysis(data, run_backtest=True)

    print_report(result)
