"""
===============================================================================
 SINGLE-TIMEFRAME TREND ANALYZER
===============================================================================
 Phân tích xu hướng cho 1 khung thời gian duy nhất bằng cách ENSEMBLE
 nhiều model, theo triết lý "Ensemble luôn thắng single model" của file md
 (Key Takeaway #4).

 Mỗi model sinh ra 1 "vote" ∈ {-1, 0, +1} và 1 confidence weight.
 Xác suất cuối cùng được tính bằng weighted softmax.

 Các model được dùng (mapping tới section file md):
   - MA Alignment        (Section 1.2)
   - Market Structure    (Section 1.2)
   - Ichimoku            (Section 1.2)
   - RSI / MACD          (classical momentum)
   - HMM regime          (Section 7)
   - Kalman velocity     (Section 9)
   - Hurst + trend sign  (Section 8)
   - Bollinger position  (mean-reversion filter)
===============================================================================
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from indicators import (
    ema, rsi, macd, atr, bollinger, ichimoku, ichimoku_signal,
    market_structure, ma_alignment_score,
)
from advanced_algorithms import (
    hmm_regime_detection, kalman_signal, rolling_hurst_signal,
)


# -----------------------------------------------------------------------------
# Helper: vote + weight
# -----------------------------------------------------------------------------

def _rsi_vote(close: pd.Series) -> Dict:
    r = rsi(close, 14).iloc[-1]
    if pd.isna(r):
        return {"vote": 0, "weight": 0.0, "rsi": None}
    # Crypto-tuned thresholds: 75/25 for overbought/oversold (wider than equities)
    if r > 75:
        return {"vote": 1, "weight": 0.5, "rsi": float(r), "note": "overbought"}
    if r < 25:
        return {"vote": -1, "weight": 0.5, "rsi": float(r), "note": "oversold"}
    if r > 55:
        return {"vote": 1, "weight": 1.0, "rsi": float(r)}
    if r < 45:
        return {"vote": -1, "weight": 1.0, "rsi": float(r)}
    return {"vote": 0, "weight": 0.5, "rsi": float(r)}


def _macd_vote(close: pd.Series) -> Dict:
    m, s, h = macd(close)
    if len(h) < 2 or pd.isna(h.iloc[-1]):
        return {"vote": 0, "weight": 0.0}

    cross_up = h.iloc[-1] > 0 and h.iloc[-2] <= 0
    cross_dn = h.iloc[-1] < 0 and h.iloc[-2] >= 0

    if cross_up:
        return {"vote": 1, "weight": 1.5, "macd_hist": float(h.iloc[-1]),
                "note": "bullish_cross"}
    if cross_dn:
        return {"vote": -1, "weight": 1.5, "macd_hist": float(h.iloc[-1]),
                "note": "bearish_cross"}
    if h.iloc[-1] > 0:
        return {"vote": 1, "weight": 0.8, "macd_hist": float(h.iloc[-1])}
    return {"vote": -1, "weight": 0.8, "macd_hist": float(h.iloc[-1])}


def _bollinger_vote(df: pd.DataFrame) -> Dict:
    lo, mid, up = bollinger(df["close"])
    if pd.isna(up.iloc[-1]):
        return {"vote": 0, "weight": 0.0}

    close = df["close"].iloc[-1]
    if close > up.iloc[-1]:
        # Phá biên trên → 2 khả năng: breakout bullish HOẶC overextended
        # Dùng trọng số nhỏ và cho vote trung tính
        return {"vote": 0, "weight": 0.3, "position": "above_upper"}
    if close < lo.iloc[-1]:
        return {"vote": 0, "weight": 0.3, "position": "below_lower"}
    if close > mid.iloc[-1]:
        return {"vote": 1, "weight": 0.5, "position": "upper_half"}
    return {"vote": -1, "weight": 0.5, "position": "lower_half"}


def _ma_alignment_vote(close: pd.Series) -> Dict:
    info = ma_alignment_score(close)
    score = info["score"]
    # Reduced weight (was 2.0/1.2) to prevent technical dominance
    if score >= 2:
        return {"vote": 1, "weight": 1.5, **info}
    if score == 1:
        return {"vote": 1, "weight": 1.0, **info}
    if score <= -2:
        return {"vote": -1, "weight": 1.5, **info}
    if score == -1:
        return {"vote": -1, "weight": 1.0, **info}
    return {"vote": 0, "weight": 0.3, **info}


def _structure_vote(df: pd.DataFrame) -> Dict:
    info = market_structure(df, lookback=5)
    trend = info["trend"]
    # Reduced weight (was 1.5) to prevent technical dominance
    if trend == "uptrend":
        vote, weight = 1, 1.2
    elif trend == "downtrend":
        vote, weight = -1, 1.2
    else:
        vote, weight = 0, 0.4

    # BOS — dampened boost (was +0.8, now +0.5)
    if info.get("bos") and info.get("bos_direction") == "bullish":
        vote, weight = 1, weight + 0.5
    elif info.get("bos") and info.get("bos_direction") == "bearish":
        vote, weight = -1, weight + 0.5

    return {"vote": vote, "weight": weight, **info}


def _ichimoku_vote(df: pd.DataFrame) -> Dict:
    icm = ichimoku(df)
    sig = ichimoku_signal(df, icm)
    # Reduced weight (was 1.5) to prevent technical dominance
    if sig == 1:
        return {"vote": 1, "weight": 1.2, "status": "above_cloud_bullish"}
    if sig == -1:
        return {"vote": -1, "weight": 1.2, "status": "below_cloud_bearish"}
    return {"vote": 0, "weight": 0.5, "status": "inside_cloud"}


def _hmm_vote(df: pd.DataFrame) -> Dict:
    info = hmm_regime_detection(df)
    if "error" in info:
        return {"vote": 0, "weight": 0.0, **info}
    regime = info["current_regime"]
    prob = info["regime_probability"]

    # Trọng số tỉ lệ với xác suất posterior (càng chắc chắn càng nặng)
    weight = 2.0 * prob  # 2.0 là trần khi prob = 1.0

    if regime == "Bull":
        vote = 1
    elif regime == "Bear":
        vote = -1
    else:
        vote = 0
        weight *= 0.3  # sideway → giảm trọng số vì không có hướng rõ

    return {
        "vote": vote,
        "weight": weight,
        "regime": regime,
        "probability": prob,
    }


def _kalman_vote(df: pd.DataFrame) -> Dict:
    info = kalman_signal(df)
    sig = info["signal"]
    strength = info.get("trend_strength", 0.0)

    # Trọng số tỉ lệ với trend strength, cap ở 2.0
    weight = min(2.0, 50.0 * strength + 0.5)

    return {"vote": sig, "weight": weight, **info}


def _hurst_vote(df: pd.DataFrame) -> Dict:
    """
    Hurst bản thân nó không nói hướng — nó nói xu hướng có tiếp diễn hay đảo chiều.
    Ta kết hợp Hurst với dấu của return gần đây:
       H > 0.55 + return dương  → mạnh tin tiếp tục tăng
       H > 0.55 + return âm     → mạnh tin tiếp tục giảm
       H < 0.45                 → tín hiệu đảo chiều (giảm trọng số xu hướng)
    """
    info = rolling_hurst_signal(df)
    h = info["hurst"]

    recent_ret = df["close"].pct_change(20).iloc[-1]
    if pd.isna(recent_ret) or abs(recent_ret) < 1e-6:
        return {"vote": 0, "weight": 0.1, **info}

    if h > 0.55:
        vote = 1 if recent_ret > 0 else -1
        weight = 1.5 * min(1.0, (h - 0.5) * 4)
    elif h < 0.45:
        # Anti-persistent → vote NGƯỢC hướng hiện tại (mean revert)
        vote = -1 if recent_ret > 0 else 1
        weight = 0.8 * min(1.0, (0.5 - h) * 4)
    else:
        vote = 0
        weight = 0.1

    return {"vote": vote, "weight": weight, **info}


# -----------------------------------------------------------------------------
# Counter-trend indicators (break positive feedback loop)
# -----------------------------------------------------------------------------

def _rsi_divergence_vote(df: pd.DataFrame) -> Dict:
    """
    RSI divergence: price makes new high but RSI doesn't (bearish divergence)
    or price makes new low but RSI doesn't (bullish divergence).
    Counter-trend signal to offset trend-following dominance.
    """
    close = df["close"]
    r = rsi(close, 14)
    if len(close) < 30 or pd.isna(r.iloc[-1]):
        return {"vote": 0, "weight": 0.0, "divergence": "none"}

    lookback = 20
    price_window = close.iloc[-lookback:]
    rsi_window = r.iloc[-lookback:]

    # Bearish divergence: price at/near recent high, RSI lower
    price_near_high = close.iloc[-1] >= price_window.quantile(0.9)
    rsi_declining = r.iloc[-1] < rsi_window.quantile(0.7)

    # Bullish divergence: price at/near recent low, RSI higher
    price_near_low = close.iloc[-1] <= price_window.quantile(0.1)
    rsi_rising = r.iloc[-1] > rsi_window.quantile(0.3)

    if price_near_high and rsi_declining:
        return {"vote": -1, "weight": 1.0, "divergence": "bearish",
                "rsi": float(r.iloc[-1])}
    if price_near_low and rsi_rising:
        return {"vote": 1, "weight": 1.0, "divergence": "bullish",
                "rsi": float(r.iloc[-1])}
    return {"vote": 0, "weight": 0.2, "divergence": "none",
            "rsi": float(r.iloc[-1])}


def _bollinger_pctb_vote(df: pd.DataFrame) -> Dict:
    """
    Bollinger %B: measures where price is relative to the bands.
    %B > 1.0 = above upper band (overextended), %B < 0 = below lower (oversold).
    Counter-trend: extreme %B votes against the trend.
    """
    lo, mid, up = bollinger(df["close"])
    if pd.isna(up.iloc[-1]) or pd.isna(lo.iloc[-1]):
        return {"vote": 0, "weight": 0.0, "pct_b": None}

    close = df["close"].iloc[-1]
    band_width = up.iloc[-1] - lo.iloc[-1]
    if band_width < 1e-10:
        return {"vote": 0, "weight": 0.1, "pct_b": 0.5}

    pct_b = (close - lo.iloc[-1]) / band_width

    # Counter-trend: extreme %B signals mean-reversion
    if pct_b > 1.0:
        return {"vote": -1, "weight": 0.8, "pct_b": float(pct_b),
                "note": "above_upper_band"}
    if pct_b < 0.0:
        return {"vote": 1, "weight": 0.8, "pct_b": float(pct_b),
                "note": "below_lower_band"}
    if pct_b > 0.8:
        return {"vote": -1, "weight": 0.4, "pct_b": float(pct_b),
                "note": "near_upper"}
    if pct_b < 0.2:
        return {"vote": 1, "weight": 0.4, "pct_b": float(pct_b),
                "note": "near_lower"}
    return {"vote": 0, "weight": 0.2, "pct_b": float(pct_b)}


# -----------------------------------------------------------------------------
# Main single-timeframe analyzer
# -----------------------------------------------------------------------------

def analyze_timeframe(
    df: pd.DataFrame, timeframe_label: str = "",
) -> Dict:
    """
    Chạy tất cả model trên 1 khung và trả về:
        - dict các vote chi tiết
        - xác suất {up, down, neutral}
        - conclusion string
        - key levels (swing high/low, EMA, ATR)
    """
    if len(df) < 50:
        return {
            "timeframe": timeframe_label,
            "error": f"Cần ≥ 50 nến, chỉ có {len(df)}",
            "probabilities": {"up": 0.33, "down": 0.33, "neutral": 0.34},
        }

    votes: Dict[str, Dict] = {
        "ma_alignment": _ma_alignment_vote(df["close"]),
        "market_structure": _structure_vote(df),
        "ichimoku": _ichimoku_vote(df),
        "rsi": _rsi_vote(df["close"]),
        "macd": _macd_vote(df["close"]),
        "bollinger": _bollinger_vote(df),
        "hmm_regime": _hmm_vote(df),
        "kalman_trend": _kalman_vote(df),
        "hurst_memory": _hurst_vote(df),
        # Counter-trend indicators to break positive feedback loop
        "rsi_divergence": _rsi_divergence_vote(df),
        "bollinger_pctb": _bollinger_pctb_vote(df),
    }

    # Weighted sum
    score_up = 0.0
    score_down = 0.0
    score_neutral = 0.0
    total_weight = 0.0

    for name, v in votes.items():
        w = v.get("weight", 0.0)
        total_weight += w
        if v["vote"] > 0:
            score_up += w
        elif v["vote"] < 0:
            score_down += w
        else:
            score_neutral += w

    if total_weight == 0:
        probs = {"up": 1/3, "down": 1/3, "neutral": 1/3}
    else:
        # --- Agreement-based probability (fixes 100% bug) ---
        # Count how many models agree on each direction
        n_models = len(votes)
        n_up = sum(1 for v in votes.values() if v["vote"] > 0)
        n_down = sum(1 for v in votes.values() if v["vote"] < 0)
        n_neutral = sum(1 for v in votes.values() if v["vote"] == 0)

        # Agreement ratio: what fraction of models agree on majority direction
        agreement_ratio = max(n_up, n_down, n_neutral) / n_models

        # Weight-based score (clipped to [-1, 1])
        raw_score = (score_up - score_down) / total_weight
        raw_score = max(-1.0, min(1.0, raw_score))

        # Blend: 60% agreement ratio + 40% score magnitude
        score_magnitude = abs(raw_score)
        blended = 0.6 * agreement_ratio + 0.4 * score_magnitude

        # Convert to directional probabilities
        if raw_score > 0:
            p_up = 0.33 + blended * 0.52   # range: 0.33 → 0.85
            p_down = (1.0 - p_up) * 0.4
            p_neutral = 1.0 - p_up - p_down
        elif raw_score < 0:
            p_down = 0.33 + blended * 0.52
            p_up = (1.0 - p_down) * 0.4
            p_neutral = 1.0 - p_up - p_down
        else:
            p_up = score_up / total_weight
            p_down = score_down / total_weight
            p_neutral = 1.0 - p_up - p_down

        # Hard ceiling at 85% — no trading model should output 100%
        MAX_PROB = 0.85
        p_up = min(p_up, MAX_PROB)
        p_down = min(p_down, MAX_PROB)
        p_neutral = max(p_neutral, 0.0)

        # Renormalize
        total_p = p_up + p_down + p_neutral
        probs = {
            "up": p_up / total_p,
            "down": p_down / total_p,
            "neutral": p_neutral / total_p,
        }

    # Kết luận
    top_dir = max(probs, key=probs.get)
    top_prob = probs[top_dir]
    if top_prob > 0.60:
        strength = "strong"
    elif top_prob > 0.45:
        strength = "moderate"
    else:
        strength = "weak"

    # Key levels
    close = df["close"].iloc[-1]
    atr14 = atr(df["high"], df["low"], df["close"], 14).iloc[-1]
    ema20 = ema(df["close"], 20).iloc[-1]
    ema50 = ema(df["close"], 50).iloc[-1]
    ema200 = ema(df["close"], 200).iloc[-1] if len(df) > 200 else None

    return {
        "timeframe": timeframe_label,
        "n_candles": len(df),
        "last_close": float(close),
        "last_time": str(df.index[-1]),
        "probabilities": probs,
        "direction": top_dir,
        "strength": strength,
        "conclusion": f"{strength}_{top_dir}",
        "votes": votes,
        "key_levels": {
            "ema_20": float(ema20),
            "ema_50": float(ema50),
            "ema_200": float(ema200) if ema200 else None,
            "atr_14": float(atr14) if not pd.isna(atr14) else None,
            "last_swing_high": votes["market_structure"].get("last_swing_high"),
            "last_swing_low": votes["market_structure"].get("last_swing_low"),
        },
    }
