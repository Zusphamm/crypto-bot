"""
===============================================================================
 INDICATORS & MARKET STRUCTURE
===============================================================================
 Các hàm tính toán cơ sở dùng trong các thuật toán phân tích xu hướng:
   - Moving Averages (SMA, EMA)
   - RSI, MACD, Bollinger Bands, ATR
   - Ichimoku Kinko Hyo (Section 1.2 của file md)
   - Market Structure: HH/HL/LH/LL và BOS (Section 1.2)
   - Swing points detection
===============================================================================
"""
from typing import Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Moving averages
# -----------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


# -----------------------------------------------------------------------------
# RSI, MACD, ATR, Bollinger
# -----------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def bollinger(
    close: pd.Series, period: int = 20, n_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, period)
    std = close.rolling(period).std()
    return mid - n_std * std, mid, mid + n_std * std


# -----------------------------------------------------------------------------
# Ichimoku Kinko Hyo  (file md — Section 1.2)
# -----------------------------------------------------------------------------

def ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> pd.DataFrame:
    """Trả về DataFrame với tenkan, kijun, senkou_a, senkou_b, chikou."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    out = pd.DataFrame(index=df.index)
    out["tenkan"] = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    out["kijun"] = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    out["senkou_a"] = ((out["tenkan"] + out["kijun"]) / 2).shift(kijun)
    out["senkou_b"] = (
        (high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2
    ).shift(kijun)
    out["chikou"] = close.shift(-kijun)
    return out


def ichimoku_signal(df: pd.DataFrame, icm: pd.DataFrame) -> int:
    """
    Trả về +1 (bullish), -1 (bearish), 0 (neutral) dựa trên quy tắc Ichimoku
    được mô tả trong file md — Section 1.2:
    'Giá trên Kumo + Tenkan > Kijun → tăng'.
    """
    i = -1
    # lấy điểm kumo tại hiện tại (vì senkou đã shift forward 26, cần lấy tại -1)
    try:
        close = df["close"].iloc[i]
        t = icm["tenkan"].iloc[i]
        k = icm["kijun"].iloc[i]
        sa = icm["senkou_a"].iloc[i]
        sb = icm["senkou_b"].iloc[i]
    except IndexError:
        return 0
    if any(pd.isna(v) for v in [close, t, k, sa, sb]):
        return 0

    above_cloud = close > max(sa, sb)
    below_cloud = close < min(sa, sb)

    if above_cloud and t > k:
        return 1
    if below_cloud and t < k:
        return -1
    return 0


# -----------------------------------------------------------------------------
# Swing points & Market Structure (file md — Section 1.2)
# -----------------------------------------------------------------------------

def find_swing_points(
    df: pd.DataFrame, lookback: int = 5,
) -> Tuple[pd.Series, pd.Series]:
    """
    Phát hiện swing high / swing low: điểm là swing high nếu cao nhất
    trong 2*lookback+1 nến xung quanh.

    Returns:
        (is_swing_high, is_swing_low) — hai Series boolean cùng index với df.
    """
    high = df["high"]
    low = df["low"]

    roll_max = high.rolling(2 * lookback + 1, center=True).max()
    roll_min = low.rolling(2 * lookback + 1, center=True).min()

    is_high = high == roll_max
    is_low = low == roll_min

    return is_high.fillna(False), is_low.fillna(False)


def market_structure(df: pd.DataFrame, lookback: int = 5) -> dict:
    """
    Phân tích market structure theo Dow Theory (file md Section 1.2):
       HH + HL  → uptrend
       LH + LL  → downtrend
       BOS (Break of Structure) xác nhận đảo chiều.

    Returns:
        dict {trend, last_structure, bos, swing_count}
    """
    is_high, is_low = find_swing_points(df, lookback)

    swing_highs = df["high"][is_high].dropna()
    swing_lows = df["low"][is_low].dropna()

    # Cần tối thiểu 2 swing high + 2 swing low để xác định xu hướng
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {
            "trend": "unknown",
            "last_structure": None,
            "bos": False,
            "swing_count": len(swing_highs) + len(swing_lows),
        }

    # Lấy 2 swing gần nhất mỗi loại
    hh = swing_highs.iloc[-1] > swing_highs.iloc[-2]  # Higher High
    hl = swing_lows.iloc[-1] > swing_lows.iloc[-2]    # Higher Low
    lh = swing_highs.iloc[-1] < swing_highs.iloc[-2]  # Lower High
    ll = swing_lows.iloc[-1] < swing_lows.iloc[-2]    # Lower Low

    if hh and hl:
        trend = "uptrend"
        structure = "HH+HL"
    elif lh and ll:
        trend = "downtrend"
        structure = "LH+LL"
    elif hh and ll:
        trend = "expansion"  # biên độ mở rộng
        structure = "HH+LL"
    elif lh and hl:
        trend = "contraction"  # biên độ thu hẹp
        structure = "LH+HL"
    else:
        trend = "sideways"
        structure = "mixed"

    # BOS: close hiện tại phá swing high gần nhất (bullish BOS)
    # hoặc phá swing low gần nhất (bearish BOS)
    last_close = df["close"].iloc[-1]
    bos_up = last_close > swing_highs.iloc[-1]
    bos_down = last_close < swing_lows.iloc[-1]
    bos = bos_up or bos_down

    return {
        "trend": trend,
        "last_structure": structure,
        "bos": bos,
        "bos_direction": "bullish" if bos_up else ("bearish" if bos_down else None),
        "last_swing_high": float(swing_highs.iloc[-1]),
        "last_swing_low": float(swing_lows.iloc[-1]),
        "swing_count": len(swing_highs) + len(swing_lows),
    }


# -----------------------------------------------------------------------------
# Moving Average alignment (file md — Section 1.2)
# -----------------------------------------------------------------------------

def ma_alignment_score(close: pd.Series) -> dict:
    """
    Tính MA Alignment Score (file md Section 1.2):
       EMA 20 > EMA 50 > EMA 200  → bullish alignment (+1)
       EMA 20 < EMA 50 < EMA 200  → bearish alignment (-1)
       Khác  → mixed (0)

    Cộng thêm: close vs EMA20, EMA50, EMA200 → tinh tế hơn
    """
    e20 = ema(close, 20).iloc[-1]
    e50 = ema(close, 50).iloc[-1]
    e200 = ema(close, 200).iloc[-1] if len(close) > 200 else np.nan
    last = close.iloc[-1]

    if pd.isna(e200):
        # Dùng fallback khi không đủ data cho EMA200
        if e20 > e50 and last > e20:
            return {"alignment": "bullish", "score": 1, "ema20": e20, "ema50": e50, "ema200": None}
        if e20 < e50 and last < e20:
            return {"alignment": "bearish", "score": -1, "ema20": e20, "ema50": e50, "ema200": None}
        return {"alignment": "mixed", "score": 0, "ema20": e20, "ema50": e50, "ema200": None}

    if e20 > e50 > e200 and last > e20:
        return {"alignment": "strong_bullish", "score": 2, "ema20": e20, "ema50": e50, "ema200": e200}
    if e20 < e50 < e200 and last < e20:
        return {"alignment": "strong_bearish", "score": -2, "ema20": e20, "ema50": e50, "ema200": e200}
    if e20 > e50 > e200:
        return {"alignment": "bullish", "score": 1, "ema20": e20, "ema50": e50, "ema200": e200}
    if e20 < e50 < e200:
        return {"alignment": "bearish", "score": -1, "ema20": e20, "ema50": e50, "ema200": e200}

    return {"alignment": "mixed", "score": 0, "ema20": e20, "ema50": e50, "ema200": e200}
