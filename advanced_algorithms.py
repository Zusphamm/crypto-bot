"""
===============================================================================
 ADVANCED ALGORITHMS: HMM + Kalman + Hurst
===============================================================================
 Triển khai 3 thuật toán "must-have" theo file md:
   - Section 7: Hidden Markov Model để detect Bull/Bear/Sideway regime
   - Section 9: Two-state Kalman Filter (position + velocity) để extract trend
   - Section 8: Rolling Hurst Exponent cho long-term memory
===============================================================================
"""
from typing import Tuple

import numpy as np
import pandas as pd
import warnings


# -----------------------------------------------------------------------------
# SECTION 7 — HMM Regime Detection (Gaussian HMM 3 trạng thái)
# -----------------------------------------------------------------------------

def hmm_regime_detection(
    df: pd.DataFrame,
    n_states: int = 3,
    min_samples: int = 200,
) -> dict:
    """
    Fit Gaussian HMM trên [log_return, rolling_volatility] để phân loại
    hidden state thành Bull / Sideway / Bear (theo mean return).

    File md — Section 7.1, 7.3: "observable = returns, volatility;
    hidden states ∈ {Bull, Bear, Sideway}; Gaussian emission;
    inference via Viterbi; learning via Baum-Welch."

    Returns:
        dict {
            "current_regime":       str,        # 'Bull' | 'Sideway' | 'Bear'
            "regime_probability":   float,      # posterior prob của regime hiện tại
            "regime_sequence":      list[str],  # chuỗi regime lịch sử
            "transition_matrix":    list[list], # A matrix
            "state_means":          dict,       # mean return per regime
        }
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        return {"error": "hmmlearn chưa được cài đặt"}

    close = df["close"]
    log_ret = np.log(close / close.shift(1)).dropna()
    vol = log_ret.rolling(10).std()

    features = pd.concat([log_ret, vol], axis=1).dropna()
    if len(features) < min_samples:
        return {
            "error": f"Cần ≥ {min_samples} mẫu, chỉ có {len(features)}",
            "current_regime": "unknown",
            "regime_probability": 0.0,
        }

    X = features.values

    # Giảm log xuất từ hmmlearn khi convergence
    import logging, io, contextlib
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)

    def _try_fit(cov_type: str):
        mdl = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=cov_type,
            n_iter=200,
            random_state=42,
            tol=1e-3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _buf = io.StringIO()
            with contextlib.redirect_stdout(_buf):
                mdl.fit(X)
        return mdl

    model = None
    last_err = None
    # Thử 'full' trước, rồi fallback sang 'diag', cuối cùng 'spherical'
    for cov_type in ("full", "diag", "spherical"):
        try:
            model = _try_fit(cov_type)
            break
        except Exception as e:
            last_err = str(e)
            continue

    if model is None:
        return {"error": f"HMM fit thất bại trên mọi covariance_type: {last_err}",
                "current_regime": "unknown", "regime_probability": 0.0}

    # Decode full sequence (Viterbi) — cũng có thể fail với ill-conditioned covars
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            states = model.predict(X)
            posteriors = model.predict_proba(X)
    except Exception as e:
        return {"error": f"HMM predict thất bại: {e}",
                "current_regime": "unknown", "regime_probability": 0.0}

    # Map hidden state → {Bear, Sideway, Bull} theo mean return
    state_means = model.means_[:, 0]  # chỉ cột return
    sorted_idx = np.argsort(state_means)
    labels = ["Bear", "Sideway", "Bull"]
    if n_states == 2:
        labels = ["Bear", "Bull"]
    elif n_states == 4:
        labels = ["Crisis", "Bear", "Sideway", "Bull"]

    state_to_label = {sorted_idx[i]: labels[i] for i in range(n_states)}

    current_state = states[-1]
    current_regime = state_to_label[current_state]
    current_prob = float(posteriors[-1, current_state])

    regime_seq = [state_to_label[s] for s in states]

    state_means_dict = {
        state_to_label[i]: float(state_means[i]) for i in range(n_states)
    }

    return {
        "current_regime": current_regime,
        "regime_probability": current_prob,
        "regime_sequence_tail": regime_seq[-50:],  # chỉ lưu 50 nến gần nhất
        "transition_matrix": model.transmat_.tolist(),
        "state_means_return": state_means_dict,
        "n_samples": len(X),
    }


# -----------------------------------------------------------------------------
# SECTION 9 — Two-State Kalman Filter (position + velocity)
# -----------------------------------------------------------------------------

def kalman_trend_filter(
    prices: np.ndarray,
    obs_cov: float = 1.0,
    trans_cov_pos: float = 0.01,
    trans_cov_vel: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-state Kalman: state = [position, velocity].
    File md — Section 9.3:
        p_t = p_{t-1} + v_{t-1}·Δt + noise
        v_t = v_{t-1}            + noise

    Returns:
        (smoothed_trend, velocity)   — đều là np.ndarray cùng length prices
    """
    try:
        from pykalman import KalmanFilter
    except ImportError:
        # Fallback: naive EMA
        s = pd.Series(prices)
        trend = s.ewm(span=20, adjust=False).mean().values
        velocity = pd.Series(trend).diff().fillna(0).values
        return trend, velocity

    kf = KalmanFilter(
        transition_matrices=[[1, 1], [0, 1]],
        observation_matrices=[[1, 0]],
        initial_state_mean=[prices[0], 0.0],
        initial_state_covariance=[[1.0, 0.0], [0.0, 1.0]],
        observation_covariance=obs_cov,
        transition_covariance=[[trans_cov_pos, 0.0], [0.0, trans_cov_vel]],
    )
    state_means, _ = kf.filter(prices)
    return state_means[:, 0], state_means[:, 1]


def kalman_signal(df: pd.DataFrame) -> dict:
    """
    Tín hiệu từ Kalman filter:
      velocity > 0 và đang tăng  →  bullish (+1)
      velocity < 0 và đang giảm  →  bearish (-1)
      còn lại                     →  neutral (0)
    """
    if len(df) < 30:
        return {"signal": 0, "velocity": 0.0, "trend_strength": 0.0}

    prices = df["close"].values.astype(float)
    trend, velocity = kalman_trend_filter(prices)

    v_last = velocity[-1]
    v_prev = velocity[-5] if len(velocity) > 5 else velocity[0]
    accelerating = v_last * v_prev >= 0 and abs(v_last) > abs(v_prev) * 0.5

    # Chuẩn hóa velocity về % giá hiện tại
    v_pct = v_last / prices[-1] if prices[-1] != 0 else 0.0

    if v_last > 0 and accelerating:
        sig = 1
    elif v_last < 0 and accelerating:
        sig = -1
    elif v_last > 0:
        sig = 1  # vẫn bullish nhưng yếu hơn
    elif v_last < 0:
        sig = -1
    else:
        sig = 0

    return {
        "signal": sig,
        "velocity": float(v_last),
        "velocity_pct": float(v_pct),
        "trend_strength": float(abs(v_pct)),
        "trend_last": float(trend[-1]),
        "accelerating": bool(accelerating),
    }


# -----------------------------------------------------------------------------
# SECTION 8 — Hurst Exponent (R/S method)
# -----------------------------------------------------------------------------

def hurst_exponent(ts: np.ndarray, max_lag: int = 100) -> float:
    """
    Tính Hurst exponent bằng R/S method (file md Section 8.3, 13.4):
        H > 0.5  → persistent/trending
        H = 0.5  → random walk
        H < 0.5  → mean-reverting
    """
    ts = np.asarray(ts, dtype=float)
    if len(ts) < max_lag + 10:
        max_lag = max(2, len(ts) // 4)

    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        std = np.std(diff)
        tau.append(std if std > 0 else 1e-10)

    log_lags = np.log(list(lags))
    log_tau = np.log(tau)
    poly = np.polyfit(log_lags, log_tau, 1)
    # Slope of log(std) vs log(lag) IS the Hurst exponent directly
    # (std scales as lag^H for fBM). Clamp to valid range [0, 1].
    h = float(np.clip(poly[0], 0.0, 1.0))
    return h


def rolling_hurst_signal(df: pd.DataFrame, window: int = 200) -> dict:
    """
    File md Section 8.4 — Rolling Hurst + interpretation.
    """
    close = df["close"]
    if len(close) < window + 20:
        # dùng toàn bộ data nếu không đủ cho rolling
        if len(close) < 50:
            return {"hurst": 0.5, "regime": "unknown", "signal": 0}
        h = hurst_exponent(close.values)
    else:
        h = hurst_exponent(close.iloc[-window:].values)

    if h > 0.55:
        regime = "strong_trending"
        signal = "use_trend_following"
    elif h > 0.5:
        regime = "weakly_trending"
        signal = "cautious_trend"
    elif h < 0.45:
        regime = "mean_reverting"
        signal = "use_mean_reversion"
    else:
        regime = "random_walk"
        signal = "avoid_trading"

    return {
        "hurst": float(h),
        "regime": regime,
        "signal": signal,
        "window": window,
    }


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Tạo synthetic data có trend rõ
    np.random.seed(0)
    n = 500
    trend = np.linspace(100, 200, n)
    noise = np.random.randn(n) * 3
    close = trend + noise
    df = pd.DataFrame({
        "open": close, "high": close + 2, "low": close - 2,
        "close": close, "volume": np.ones(n) * 1000,
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

    print("HMM:", hmm_regime_detection(df))
    print("Kalman:", kalman_signal(df))
    print("Hurst:", rolling_hurst_signal(df))
