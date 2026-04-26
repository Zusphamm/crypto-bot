"""
===============================================================================
 HISTORICAL BACKTEST: EMPIRICAL PROBABILITY OF DIRECTION
===============================================================================
 File md — Section 14 (Walk-Forward) và Key Takeaway #8:
   "Backtest ≠ Live trading. Walk-forward là bắt buộc."

 Module này chạy ensemble-vote trên TOÀN BỘ lịch sử (điểm-theo-điểm),
 rồi so sánh với hướng giá THỰC TẾ (N nến sau đó) để tính:

     P(up | model_says_up, khung=X) = hit_rate empirical

 Điều này trả lời câu hỏi bạn đặt ra: "Đưa các mô hình phân tích hướng
 vào càng nhiều điểm mua bán trong quá khứ."

 Thay vì chạy lại toàn bộ HMM/Kalman/indicator trên mỗi bar (rất chậm),
 chúng ta dùng approach hiệu quả:
   1. Tính tất cả indicator 1 lần trên toàn bộ series (vectorized)
   2. Tại mỗi bar trong lịch sử, lấy giá trị indicator tại thời điểm đó
   3. Đánh nhãn bar: "model dự đoán UP/DOWN/NEUTRAL"
   4. Tính forward return N nến sau → so với dự đoán → hit/miss
===============================================================================
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from indicators import ema, rsi, macd, bollinger


# -----------------------------------------------------------------------------
# Vectorized per-bar signals (các model đơn giản — chạy trên toàn series)
# -----------------------------------------------------------------------------

def _vectorized_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính bộ signals có thể vector hóa cho mọi bar trong history:
      - MA alignment (EMA20/50/200 quan hệ)
      - MACD histogram sign
      - RSI > 55 / < 45
      - Bollinger position
      - Hurst proxy (momentum sign tại window khác nhau)

    Không dùng HMM/Kalman ở đây vì chúng đắt tính; 5 signals này đã đủ
    thống kê ý nghĩa trên hàng nghìn điểm.
    """
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    # MA alignment
    e20, e50 = ema(close, 20), ema(close, 50)
    e200 = ema(close, 200)
    out["ma_vote"] = 0
    out.loc[(e20 > e50) & (e50 > e200) & (close > e20), "ma_vote"] = 1
    out.loc[(e20 < e50) & (e50 < e200) & (close < e20), "ma_vote"] = -1

    # MACD hist
    m, s, h = macd(close)
    out["macd_vote"] = np.sign(h).fillna(0).astype(int)

    # RSI
    r = rsi(close, 14)
    out["rsi_vote"] = 0
    out.loc[r > 55, "rsi_vote"] = 1
    out.loc[r < 45, "rsi_vote"] = -1

    # Bollinger
    lo, mid, up = bollinger(close)
    out["bb_vote"] = 0
    out.loc[close > mid, "bb_vote"] = 1
    out.loc[close < mid, "bb_vote"] = -1

    # Momentum 20-bar — proxy cho hướng tức thời
    out["mom_vote"] = np.sign(close.pct_change(20)).fillna(0).astype(int)

    # Ensemble score = trung bình trọng số
    weights = {"ma_vote": 2.0, "macd_vote": 1.2, "rsi_vote": 1.0,
               "bb_vote": 0.6, "mom_vote": 1.0}
    total_w = sum(weights.values())
    out["score"] = sum(out[c] * w for c, w in weights.items()) / total_w

    # Label dự đoán
    out["prediction"] = 0
    out.loc[out["score"] > 0.15, "prediction"] = 1
    out.loc[out["score"] < -0.15, "prediction"] = -1

    return out


# -----------------------------------------------------------------------------
# Forward return + hit-rate
# -----------------------------------------------------------------------------

def compute_empirical_hit_rate(
    df: pd.DataFrame,
    forward_bars: int = 10,
    min_move_pct: float = None,
    use_mfe: bool = True,
) -> Dict:
    """
    Chạy vectorized signals trên toàn lịch sử, đánh dấu dự đoán,
    rồi so với hướng giá thực tế sau `forward_bars` nến.

    Args:
        df:            DataFrame OHLCV.
        forward_bars:  Số nến forward để đo hướng thực tế.
        min_move_pct:  Ngưỡng biên neutral. None = auto-adaptive theo volatility
                       (ATR/close × sqrt(forward_bars)/2).
        use_mfe:       True = dùng Max-Favorable-Excursion (tính cả đỉnh/đáy
                       trong khoảng forward, không chỉ close cuối). Sát với
                       thực tế trade có take-profit hơn.

    Returns:
        dict với hit_rate_up/down/overall, edge_vs_baseline, baselines, ...
    """
    if len(df) < forward_bars + 250:
        return {"error": f"Cần ≥ {forward_bars + 250} nến cho backtest, "
                          f"chỉ có {len(df)}"}

    sigs = _vectorized_signals(df)

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # --- Adaptive threshold ---
    if min_move_pct is None:
        # ATR 14 / close — volatility tương đối
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        # Ngưỡng = ATR/close × sqrt(fwd) × 0.25
        # (cũ là 0.5 — quá cao làm baseline ~25%. 0.25 cho baseline ~35-45%)
        adaptive_thresh = (atr14 / close) * (forward_bars ** 0.5) * 0.25
        # Cap dưới 0.1% và trên 2% để tránh extremes
        adaptive_thresh = adaptive_thresh.clip(lower=0.001, upper=0.02)
        adaptive_thresh = adaptive_thresh.fillna(0.005)
    else:
        adaptive_thresh = pd.Series(min_move_pct, index=close.index)

    # --- Forward return ---
    if use_mfe:
        # Max-favorable-excursion: trong forward_bars tiếp theo,
        # giá cao nhất / thấp nhất so với close hiện tại.
        # rolling(fwd).max() at bar j = max(high[j-fwd+1..j])
        # shift(-fwd) maps bar i → value at j=i+fwd → max(high[i+1..i+fwd])
        fwd_max = high.rolling(forward_bars).max().shift(-forward_bars)
        fwd_min = low.rolling(forward_bars).min().shift(-forward_bars)
        fwd_up_move = (fwd_max / close) - 1
        fwd_down_move = 1 - (fwd_min / close)

        # actual: UP nếu có lúc giá lên > threshold TRƯỚC khi xuống > threshold
        # Approximation: chỉ xét ai > ai. Không biết thứ tự hit → dùng
        # simple rule: up nếu up_move > down_move + threshold
        actual = pd.Series(0, index=df.index, dtype=int)
        actual[fwd_up_move > fwd_down_move + adaptive_thresh] = 1
        actual[fwd_down_move > fwd_up_move + adaptive_thresh] = -1
        fwd_ret = (close.shift(-forward_bars) / close) - 1  # vẫn lưu close-to-close
    else:
        fwd_ret = (close.shift(-forward_bars) / close) - 1
        actual = pd.Series(0, index=df.index, dtype=int)
        actual[fwd_ret > adaptive_thresh] = 1
        actual[fwd_ret < -adaptive_thresh] = -1

    # Bỏ NaN
    valid = sigs["prediction"].notna() & fwd_ret.notna() & actual.notna()
    pred = sigs["prediction"][valid]
    act = actual[valid]
    fwd = fwd_ret[valid]

    if len(pred) == 0:
        return {"error": "Không có mẫu hợp lệ"}

    up_mask = pred == 1
    down_mask = pred == -1
    neutral_mask = pred == 0

    n_up = int(up_mask.sum())
    n_down = int(down_mask.sum())
    n_neutral = int(neutral_mask.sum())

    hit_up = float((act[up_mask] == 1).mean()) if n_up > 0 else None
    hit_down = float((act[down_mask] == -1).mean()) if n_down > 0 else None

    directional_mask = up_mask | down_mask
    if directional_mask.sum() > 0:
        overall = float((pred[directional_mask] == act[directional_mask]).mean())
    else:
        overall = None

    avg_ret_up = float(fwd[up_mask].mean()) if n_up > 0 else None
    avg_ret_down = float(fwd[down_mask].mean()) if n_down > 0 else None

    # --- BASELINES: tỉ lệ UP/DOWN trong toàn bộ lịch sử (không cần model) ---
    baseline_up = float((act == 1).mean())
    baseline_down = float((act == -1).mean())

    # --- EDGE vs baseline ---
    edge_up = (hit_up - baseline_up) if hit_up is not None else None
    edge_down = (hit_down - baseline_down) if hit_down is not None else None

    return {
        "n_samples": int(len(pred)),
        "forward_bars": forward_bars,
        "use_mfe": use_mfe,
        "avg_threshold_pct": float(adaptive_thresh[valid].mean()),
        "n_up_signals": n_up,
        "n_down_signals": n_down,
        "n_neutral": n_neutral,
        "hit_rate_up": hit_up,
        "hit_rate_down": hit_down,
        "hit_rate_overall": overall,
        "baseline_up": baseline_up,
        "baseline_down": baseline_down,
        "edge_up": edge_up,              # > 0 = model thực sự có skill cho UP
        "edge_down": edge_down,          # > 0 = model thực sự có skill cho DOWN
        "avg_forward_return_on_up": avg_ret_up,
        "avg_forward_return_on_down": avg_ret_down,
        # Giữ tên cũ để backward compat
        "empirical_up_rate": baseline_up,
        "empirical_down_rate": baseline_down,
    }


def bayesian_adjusted_probability(
    current_vote: int,
    hit_stats: Dict,
) -> Dict:
    """
    Dùng empirical hit_rate + baseline để tính xác suất calibrated của
    dự đoán hiện tại, KÈM THEO edge (model giỏi hơn baseline bao nhiêu).

    EDGE > 0: model thực sự có kỹ năng.
    EDGE ≤ 0: model tệ hơn hoặc bằng random trên khung đó — KHÔNG NÊN TRADE.
    """
    if "error" in hit_stats:
        return {
            "calibrated_probability": None,
            "reason": "không đủ dữ liệu lịch sử để calibrate",
        }

    baseline_up = hit_stats.get("baseline_up", 0.33)
    baseline_down = hit_stats.get("baseline_down", 0.33)

    if current_vote == 1:
        p = hit_stats.get("hit_rate_up")
        if p is None:
            p = baseline_up
        edge = p - baseline_up
        return {
            "direction": "up",
            "calibrated_probability": float(p),
            "baseline": float(baseline_up),
            "edge_vs_baseline": float(edge),
            "has_edge": edge > 0.02,  # yêu cầu edge ≥ 2% mới gọi là "có skill"
            "sample_size": hit_stats.get("n_up_signals"),
        }
    if current_vote == -1:
        p = hit_stats.get("hit_rate_down")
        if p is None:
            p = baseline_down
        edge = p - baseline_down
        return {
            "direction": "down",
            "calibrated_probability": float(p),
            "baseline": float(baseline_down),
            "edge_vs_baseline": float(edge),
            "has_edge": edge > 0.02,
            "sample_size": hit_stats.get("n_down_signals"),
        }

    baseline_neutral = 1 - baseline_up - baseline_down
    return {
        "direction": "neutral",
        "calibrated_probability": float(baseline_neutral),
        "baseline": float(baseline_neutral),
        "edge_vs_baseline": 0.0,
        "has_edge": False,
    }


# -----------------------------------------------------------------------------
# Default forward_bars theo interval
# -----------------------------------------------------------------------------

DEFAULT_FORWARD_BARS = {
    "1m": 15,    # nhìn 15 phút
    "3m": 10,
    "5m": 12,    # 1 giờ
    "15m": 8,    # 2 giờ
    "30m": 8,    # 4 giờ
    "1h": 12,    # 12 giờ
    "2h": 12,
    "4h": 6,     # 1 ngày
    "6h": 4,
    "8h": 3,
    "12h": 4,    # 2 ngày
}


def backtest_timeframe(df: pd.DataFrame, interval: str) -> Dict:
    """Wrapper tiện dụng gắn forward_bars mặc định theo interval."""
    fwd = DEFAULT_FORWARD_BARS.get(interval, 10)
    return compute_empirical_hit_rate(df, forward_bars=fwd)


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic data có trend
    np.random.seed(1)
    n = 2000
    # Random walk với drift + vài regime shift
    rets = np.random.randn(n) * 0.01
    rets[500:1000] += 0.002
    rets[1500:] -= 0.002
    close = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({
        "open": close, "high": close * 1.002, "low": close * 0.998,
        "close": close, "volume": np.ones(n) * 1000,
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

    stats = compute_empirical_hit_rate(df, forward_bars=10)
    print("Backtest stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
