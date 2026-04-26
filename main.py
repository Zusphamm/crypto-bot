"""
===============================================================================
 MAIN — ENTRY POINT
===============================================================================
 Chạy end-to-end:
   1. Kéo data từ Binance cho tất cả khung M1 → H12
   2. Chạy multi-timeframe analysis
   3. In báo cáo tổng hợp

 CLI:
   python main.py                         # default: BTCUSDT, 5000 candles, spot
   python main.py ETHUSDT                 # đổi symbol
   python main.py BTCUSDT 10000           # 10k candles mỗi khung
   python main.py BTCUSDT 10000 futures   # dùng futures
===============================================================================
"""
import argparse
import json
import sys

from binance_fetcher import (
    fetch_all_timeframes, SUPPORTED_INTERVALS, DEFAULT_MAX_CANDLES,
    DEFAULT_MAX_YEARS,
)
from multi_timeframe import multi_timeframe_analysis, print_report


def run(
    symbol: str = "BTCUSDT",
    max_candles: int = DEFAULT_MAX_CANDLES,
    max_years: float = DEFAULT_MAX_YEARS,
    market: str = "spot",
    intervals=None,
    run_backtest: bool = True,
    save_json: str = None,
):
    """
    Chạy toàn bộ pipeline.

    Args:
        symbol:       Cặp (BTCUSDT, ETHUSDT, ...)
        max_candles:  Số nến mỗi khung (≤ 10,000)
        max_years:    Lookback max (năm)
        market:       'spot' hoặc 'futures'
        intervals:    Danh sách khung (default = tất cả từ 1m đến 12h)
        run_backtest: Có chạy empirical hit rate không
        save_json:    Path để lưu kết quả chi tiết
    """
    intervals = intervals or SUPPORTED_INTERVALS

    # Step 1: Fetch data
    data = fetch_all_timeframes(
        symbol=symbol,
        intervals=intervals,
        max_candles=max_candles,
        max_years=max_years,
        market=market,
    )

    # Step 2: Run analysis
    result = multi_timeframe_analysis(data, run_backtest=run_backtest)

    # Step 3: Print report
    print_report(result)

    # Step 4: Optionally save JSON
    if save_json:
        # Convert numpy/pandas types → JSON-safe
        import numpy as np

        def default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

        with open(save_json, "w") as f:
            json.dump(result, f, indent=2, default=default)
        print(f"\n  💾 Kết quả chi tiết → {save_json}")

    return result


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Multi-timeframe trend analyzer (Binance + advanced algos)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("symbol", nargs="?", default="BTCUSDT",
                   help="Trading pair (default: BTCUSDT)")
    p.add_argument("candles", nargs="?", type=int, default=DEFAULT_MAX_CANDLES,
                   help=f"Max candles per TF (default: {DEFAULT_MAX_CANDLES})")
    p.add_argument("market", nargs="?", default="spot",
                   choices=["spot", "futures"],
                   help="Market type (default: spot)")
    p.add_argument("--years", type=float, default=DEFAULT_MAX_YEARS,
                   help=f"Max lookback years (default: {DEFAULT_MAX_YEARS})")
    p.add_argument("--intervals", nargs="+", default=None,
                   help=f"TFs (default all: {SUPPORTED_INTERVALS})")
    p.add_argument("--no-backtest", action="store_true",
                   help="Skip empirical backtest (faster)")
    p.add_argument("--save", default=None, help="Save JSON path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        symbol=args.symbol,
        max_candles=args.candles,
        max_years=args.years,
        market=args.market,
        intervals=args.intervals,
        run_backtest=not args.no_backtest,
        save_json=args.save,
    )
