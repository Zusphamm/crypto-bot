"""
===============================================================================
 QUICK DEMO — Chạy với 1 symbol + subset khung để test nhanh
===============================================================================
 Dùng khi bạn chỉ muốn kiểm tra nhanh (< 1 phút) trước khi chạy full 11 khung.

   python quick_demo.py                    # BTCUSDT, 3000 candles, 5 khung
   python quick_demo.py ETHUSDT 5000       # ETH, 5000 candles
===============================================================================
"""
import sys
from main import run

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    candles = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

    # Subset: 5 khung đại diện — nhanh hơn ~2x so với full 11 khung
    quick_intervals = ["15m", "1h", "4h", "8h", "12h"]

    print(f"⚡ Quick demo: {symbol}, {candles} nến, {len(quick_intervals)} khung")

    run(
        symbol=symbol,
        max_candles=candles,
        market="spot",
        intervals=quick_intervals,
        run_backtest=True,
    )
