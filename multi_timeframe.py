"""
===============================================================================
 MULTI-TIMEFRAME TREND ANALYZER — ORCHESTRATOR
===============================================================================
 Đây là "mặt tiền" của Tool 2. Nó:
   1. Chạy analyze_timeframe trên TỪNG khung (M1 → H12)
   2. Chạy backtest trên TỪNG khung để lấy calibrated probability
   3. Áp dụng nguyên lý Top-Down (file md Section 1.1):
        - Khung lớn quyết định context (trend)
        - Khung nhỏ quyết định trigger (entry)
   4. Tính confluence score — bao nhiêu khung cùng hướng
   5. Xuất báo cáo cuối: xác suất hướng giá cho từng khung + overall

 Nguyên tắc:
   - Khung lớn có trọng số cao hơn khung nhỏ (higher timeframe = higher context)
   - Xác suất cuối = weighted average của per-timeframe probability
   - CHỈ RA alignment: các khung cùng nói UP / DOWN / mâu thuẫn
===============================================================================
"""
from typing import Dict, List, Optional

import pandas as pd

from single_timeframe import analyze_timeframe
from backtest import backtest_timeframe, bayesian_adjusted_probability


# -----------------------------------------------------------------------------
# Trọng số khung thời gian — file md Section 1.1:
#   "Khung lớn quyết định xu hướng, khung nhỏ quyết định entry"
# -----------------------------------------------------------------------------

TIMEFRAME_WEIGHTS = {
    "1m":   0.5,
    "3m":   0.6,
    "5m":   0.7,
    "15m":  0.9,
    "30m":  1.0,
    "1h":   1.3,
    "2h":   1.5,
    "4h":   1.8,
    "6h":   2.0,
    "8h":   2.1,
    "12h":  2.3,
}


# -----------------------------------------------------------------------------
# Main orchestrator
# -----------------------------------------------------------------------------

def multi_timeframe_analysis(
    data_by_tf: Dict[str, pd.DataFrame],
    run_backtest: bool = True,
) -> Dict:
    """
    Chạy phân tích đầy đủ trên tất cả khung.

    Args:
        data_by_tf: dict { 'interval' : DataFrame OHLCV }
        run_backtest: có chạy empirical hit rate hay không (chậm hơn nhưng
                      cho xác suất calibrated).

    Returns:
        dict chứa phân tích chi tiết từng khung + overall confluence.
    """
    per_tf: Dict[str, Dict] = {}

    print("\n📊 Phân tích từng khung thời gian...\n")

    for tf, df in data_by_tf.items():
        if df is None or df.empty:
            per_tf[tf] = {"error": "không có data"}
            continue

        # Step 1: phân tích hiện tại
        analysis = analyze_timeframe(df, timeframe_label=tf)

        # Step 2: backtest empirical hit rate
        if run_backtest:
            backtest_result = backtest_timeframe(df, tf)
            analysis["backtest"] = backtest_result

            # Step 3: calibrate xác suất hiện tại bằng hit rate thực tế
            current_dir = analysis.get("direction", "neutral")
            current_vote = 1 if current_dir == "up" else (
                -1 if current_dir == "down" else 0
            )
            analysis["calibrated"] = bayesian_adjusted_probability(
                current_vote, backtest_result
            )

        per_tf[tf] = analysis

        # In 1 dòng tóm tắt
        if "error" not in analysis:
            p = analysis["probabilities"]
            direction = analysis["direction"]
            strength = analysis["strength"]
            cal = analysis.get("calibrated", {})
            cal_txt = ""
            if cal.get("calibrated_probability") is not None:
                edge = cal.get("edge_vs_baseline", 0)
                base = cal.get("baseline", 0)
                marker = "✓" if cal.get("has_edge") else "✗"
                cal_txt = (
                    f" | hit={cal['calibrated_probability']*100:4.1f}%  "
                    f"base={base*100:4.1f}%  "
                    f"edge={edge*100:+5.1f}% {marker}  "
                    f"(n={cal.get('sample_size', 'n/a')})"
                )
            print(
                f"  [{tf:>4}] {strength:>8}_{direction:<7} | "
                f"up={p['up']*100:4.1f}%  dn={p['down']*100:4.1f}%  "
                f"nu={p['neutral']*100:4.1f}%{cal_txt}"
            )
        else:
            print(f"  [{tf:>4}] ERROR: {analysis['error']}")

    # --- Overall Confluence ---
    overall = _compute_confluence(per_tf)

    return {
        "per_timeframe": per_tf,
        "overall": overall,
    }


def _compute_confluence(per_tf: Dict[str, Dict]) -> Dict:
    """
    Tính confluence trên tất cả khung — file md Section 12.2:
    'Trade only khi ≥ 2/3 layer agree'.
    """
    weighted_up = 0.0
    weighted_down = 0.0
    weighted_neutral = 0.0
    total_w = 0.0

    directions_by_group = {"higher": [], "middle": [], "lower": []}
    HIGHER = {"4h", "6h", "8h", "12h"}
    MIDDLE = {"1h", "2h"}
    LOWER = {"1m", "3m", "5m", "15m", "30m"}

    details = []
    for tf, info in per_tf.items():
        if "error" in info:
            continue
        w = TIMEFRAME_WEIGHTS.get(tf, 1.0)

        # Ưu tiên calibrated probability nếu có
        cal = info.get("calibrated", {})
        cal_prob = cal.get("calibrated_probability")
        direction = info.get("direction", "neutral")

        p = info["probabilities"]
        # Dùng xác suất từ analyze_timeframe (ensemble votes)
        weighted_up += p["up"] * w
        weighted_down += p["down"] * w
        weighted_neutral += p["neutral"] * w
        total_w += w

        details.append({
            "tf": tf,
            "direction": direction,
            "ensemble_prob": p[direction],
            "empirical_prob": cal_prob,
            "edge": cal.get("edge_vs_baseline"),
            "has_edge": cal.get("has_edge", False),
            "weight": w,
        })

        if tf in HIGHER:
            directions_by_group["higher"].append(direction)
        elif tf in MIDDLE:
            directions_by_group["middle"].append(direction)
        elif tf in LOWER:
            directions_by_group["lower"].append(direction)

    if total_w == 0:
        return {"error": "không có khung nào phân tích được"}

    # Đếm số khung có edge thực sự
    n_with_edge = sum(1 for d in details if d.get("has_edge"))
    n_total = len(details)

    overall_probs = {
        "up": weighted_up / total_w,
        "down": weighted_down / total_w,
        "neutral": weighted_neutral / total_w,
    }
    overall_direction = max(overall_probs, key=overall_probs.get)

    # Confluence theo group — % khung cùng hướng
    def majority(lst):
        if not lst:
            return None, 0.0
        from collections import Counter
        c = Counter(lst)
        dir_, cnt = c.most_common(1)[0]
        return dir_, cnt / len(lst)

    higher_dir, higher_agree = majority(directions_by_group["higher"])
    middle_dir, middle_agree = majority(directions_by_group["middle"])
    lower_dir, lower_agree = majority(directions_by_group["lower"])

    # Tín hiệu chất lượng cao: khung lớn + khung trung cùng hướng
    # VÀ có ít nhất 1 khung có edge
    edge_ratio = n_with_edge / n_total if n_total > 0 else 0
    high_quality = (
        higher_dir == middle_dir == overall_direction
        and higher_dir is not None
        and higher_dir != "neutral"
        and edge_ratio >= 0.5
    )

    # Nếu KHÔNG có khung nào có edge — override recommendation
    if n_with_edge == 0:
        recommendation = (
            f"⛔ NO-EDGE — không khung nào có hit_rate > baseline. "
            f"Ensemble có thể đang recognize pattern đẹp, nhưng lịch sử "
            f"cho thấy pattern này không sinh edge. KHÔNG TRADE."
        )
    else:
        recommendation = _top_down_recommendation(
            higher_dir, middle_dir, lower_dir, overall_probs,
        )

    return {
        "weighted_probabilities": overall_probs,
        "overall_direction": overall_direction,
        "overall_probability": overall_probs[overall_direction],
        "higher_tf_direction": higher_dir,
        "higher_tf_agreement": higher_agree,
        "middle_tf_direction": middle_dir,
        "middle_tf_agreement": middle_agree,
        "lower_tf_direction": lower_dir,
        "lower_tf_agreement": lower_agree,
        "high_quality_setup": high_quality,
        "n_timeframes_with_edge": n_with_edge,
        "n_timeframes_total": n_total,
        "top_down_recommendation": recommendation,
        "details": details,
    }


def _top_down_recommendation(
    higher: Optional[str], middle: Optional[str], lower: Optional[str],
    probs: Dict[str, float],
) -> str:
    """
    File md Section 1.1 + 12.2:
      - Khung lớn = xu hướng (trade direction)
      - Khung trung = setup (context confirm)
      - Khung nhỏ = trigger (entry timing)
    """
    if higher is None:
        return "Không đủ dữ liệu khung lớn"

    p_top = probs[max(probs, key=probs.get)]

    # Ưu tiên đi theo khung lớn — đó là triết lý top-down
    if higher in ("up", "down") and middle == higher:
        if lower == higher and p_top > 0.55:
            return (
                f"🟢 HIGH QUALITY {higher.upper()} SETUP — "
                f"cả 3 nhóm khung đồng thuận. Entry trigger đã bắt đầu."
            )
        if lower == "neutral":
            return (
                f"🟡 {higher.upper()} context xác lập, đang chờ entry trigger "
                f"khung nhỏ."
            )
        if lower != higher:
            return (
                f"🟡 Khung lớn + trung nói {higher.upper()} nhưng khung nhỏ "
                f"đang đi ngược → chờ pullback theo hướng khung lớn."
            )
    if higher != middle:
        return (
            f"🔴 MÂU THUẪN: khung lớn={higher}, khung trung={middle}. "
            f"Không trade — đang trong transition."
        )

    return (
        f"⚪ Sideway / không có edge rõ ràng. Overall prob top = "
        f"{p_top*100:.1f}%. Nên đứng ngoài."
    )


# -----------------------------------------------------------------------------
# Pretty print
# -----------------------------------------------------------------------------

def print_report(result: Dict) -> None:
    """In báo cáo tổng hợp dễ đọc."""
    print("\n" + "=" * 72)
    print("                   MULTI-TIMEFRAME TREND REPORT")
    print("=" * 72)

    overall = result["overall"]
    if "error" in overall:
        print(f"  ❌ {overall['error']}")
        return

    probs = overall["weighted_probabilities"]
    print(f"\n  📈 Overall direction : {overall['overall_direction'].upper()}")
    print(f"  📊 Overall prob     : UP={probs['up']*100:.1f}%  "
          f"DOWN={probs['down']*100:.1f}%  NEUTRAL={probs['neutral']*100:.1f}%")
    print(f"\n  Higher TF ({'4h-12h'}): "
          f"{overall['higher_tf_direction']}  "
          f"(agree={overall['higher_tf_agreement']*100:.0f}%)")
    print(f"  Middle TF (1h-2h)  : "
          f"{overall['middle_tf_direction']}  "
          f"(agree={overall['middle_tf_agreement']*100:.0f}%)")
    print(f"  Lower  TF (1m-30m) : "
          f"{overall['lower_tf_direction']}  "
          f"(agree={overall['lower_tf_agreement']*100:.0f}%)")

    hq = "✅" if overall["high_quality_setup"] else "❌"
    n_edge = overall.get("n_timeframes_with_edge", 0)
    n_total = overall.get("n_timeframes_total", 0)
    print(f"\n  High quality setup : {hq}")
    print(f"  Khung có edge > baseline: {n_edge}/{n_total}")
    print(f"\n  💬 {overall['top_down_recommendation']}")
    print("\n" + "=" * 72)
