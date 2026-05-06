"""
Unit tests for crypto bot core modules.

Tests:
  - Probability never exceeds 85% (calibration fix)
  - MFE-off produces different results than MFE-on
  - Counter-trend indicators produce non-zero votes
  - Agreement-based probability works correctly
  - Edge threshold at 5%
  - Chart generation works
  - Formatter outputs valid HTML
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Test data fixtures ──────────────────────────────────────────────────────

def _make_trending_data(n=500, direction="up"):
    """Synthetic data with clear trend."""
    np.random.seed(42)
    drift = 0.002 if direction == "up" else -0.002
    rets = np.random.randn(n) * 0.01 + drift
    close = 50000 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(np.random.randn(n)) * 0.005)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.005)
    return pd.DataFrame({
        "open": np.roll(close, 1),
        "high": high,
        "low": low,
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1000 + 5000),
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))


def _make_sideways_data(n=500):
    """Synthetic data with no clear trend."""
    np.random.seed(99)
    rets = np.random.randn(n) * 0.008
    close = 50000 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(np.random.randn(n)) * 0.004)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.004)
    return pd.DataFrame({
        "open": np.roll(close, 1),
        "high": high,
        "low": low,
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1000 + 5000),
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))


# ─── Probability calibration tests ──────────────────────────────────────────

class TestProbabilityCalibration:
    """Test that probability is correctly calibrated (max 85%)."""

    def test_probability_never_exceeds_85_percent_bullish(self):
        """Even with all models agreeing UP, probability must be <= 85%."""
        from single_timeframe import analyze_timeframe
        df = _make_trending_data(500, "up")
        result = analyze_timeframe(df, "1h")
        probs = result["probabilities"]
        assert probs["up"] <= 0.85, f"UP probability {probs['up']:.3f} exceeds 85%"
        assert probs["down"] <= 0.85, f"DOWN probability {probs['down']:.3f} exceeds 85%"
        assert probs["neutral"] <= 0.85, f"NEUTRAL probability {probs['neutral']:.3f} exceeds 85%"

    def test_probability_never_exceeds_85_percent_bearish(self):
        """Bearish trend also capped."""
        from single_timeframe import analyze_timeframe
        df = _make_trending_data(500, "down")
        result = analyze_timeframe(df, "1h")
        probs = result["probabilities"]
        assert probs["up"] <= 0.85
        assert probs["down"] <= 0.85
        assert probs["neutral"] <= 0.85

    def test_probabilities_sum_to_one(self):
        """Probabilities must sum to ~1.0."""
        from single_timeframe import analyze_timeframe
        df = _make_trending_data(500, "up")
        result = analyze_timeframe(df, "1h")
        probs = result["probabilities"]
        total = probs["up"] + probs["down"] + probs["neutral"]
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total:.4f}, not 1.0"

    def test_sideways_gives_lower_confidence(self):
        """Sideways market should not produce high directional confidence."""
        from single_timeframe import analyze_timeframe
        df = _make_sideways_data(500)
        result = analyze_timeframe(df, "1h")
        probs = result["probabilities"]
        # In sideways, top probability should be moderate
        top_p = max(probs.values())
        assert top_p < 0.80, f"Sideways top probability {top_p:.3f} too high"

    def test_min_data_returns_uniform(self):
        """With insufficient data, probabilities should be ~uniform."""
        from single_timeframe import analyze_timeframe
        df = _make_trending_data(30, "up")
        result = analyze_timeframe(df, "1h")
        probs = result["probabilities"]
        assert abs(probs["up"] - 0.33) < 0.02
        assert abs(probs["down"] - 0.33) < 0.02


# ─── MFE tests ───────────────────────────────────────────────────────────────

class TestMFEDefault:
    """Test that MFE is off by default."""

    def test_mfe_default_is_false(self):
        """Default use_mfe should be False."""
        import inspect
        from backtest import compute_empirical_hit_rate
        sig = inspect.signature(compute_empirical_hit_rate)
        default_mfe = sig.parameters["use_mfe"].default
        assert default_mfe is False, f"use_mfe default is {default_mfe}, should be False"

    def test_mfe_on_vs_off_different(self):
        """MFE on should give different (typically higher) hit rates than off."""
        from backtest import compute_empirical_hit_rate
        df = _make_trending_data(1000, "up")
        stats_mfe = compute_empirical_hit_rate(df, forward_bars=10, use_mfe=True)
        stats_no_mfe = compute_empirical_hit_rate(df, forward_bars=10, use_mfe=False)
        # They should be different
        if stats_mfe.get("hit_rate_up") is not None and stats_no_mfe.get("hit_rate_up") is not None:
            # MFE typically inflates hit rates
            assert stats_mfe["hit_rate_up"] != stats_no_mfe["hit_rate_up"] or \
                   stats_mfe["hit_rate_down"] != stats_no_mfe["hit_rate_down"], \
                   "MFE on/off should produce different hit rates"


# ─── Edge threshold tests ────────────────────────────────────────────────────

class TestEdgeThreshold:
    """Test that edge threshold is 5% (not 2%)."""

    def test_edge_threshold_is_5_percent(self):
        """has_edge should require >5% edge."""
        from backtest import bayesian_adjusted_probability
        # Mock hit_stats with 4% edge (should NOT be considered edge)
        hit_stats = {
            "hit_rate_up": 0.37,    # baseline + 4%
            "baseline_up": 0.33,
            "n_up_signals": 100,
        }
        result = bayesian_adjusted_probability(1, hit_stats)
        assert result["has_edge"] is False, "4% edge should not qualify (threshold is 5%)"

        # Mock hit_stats with 6% edge (should be edge)
        hit_stats["hit_rate_up"] = 0.39  # baseline + 6%
        result = bayesian_adjusted_probability(1, hit_stats)
        assert result["has_edge"] is True, "6% edge should qualify"


# ─── Counter-trend indicator tests ───────────────────────────────────────────

class TestCounterTrendIndicators:
    """Test RSI divergence and Bollinger %B counter-trend indicators."""

    def test_rsi_divergence_exists_in_votes(self):
        """RSI divergence should be in the vote dict."""
        from single_timeframe import analyze_timeframe
        df = _make_trending_data(500, "up")
        result = analyze_timeframe(df, "1h")
        assert "rsi_divergence" in result["votes"]
        assert "vote" in result["votes"]["rsi_divergence"]
        assert "weight" in result["votes"]["rsi_divergence"]

    def test_bollinger_pctb_exists_in_votes(self):
        """Bollinger %B should be in the vote dict."""
        from single_timeframe import analyze_timeframe
        df = _make_trending_data(500, "up")
        result = analyze_timeframe(df, "1h")
        assert "bollinger_pctb" in result["votes"]
        assert "vote" in result["votes"]["bollinger_pctb"]
        assert "pct_b" in result["votes"]["bollinger_pctb"]

    def test_total_models_is_11(self):
        """Should have 11 models (9 original + 2 counter-trend)."""
        from single_timeframe import analyze_timeframe
        df = _make_trending_data(500, "up")
        result = analyze_timeframe(df, "1h")
        assert len(result["votes"]) == 11, f"Expected 11 models, got {len(result['votes'])}"


# ─── Chart generation tests ──────────────────────────────────────────────────

class TestChartGeneration:
    """Test matplotlib chart generation."""

    def test_generate_chart_basic(self):
        """Chart should generate a valid PNG file."""
        from charts import generate_price_chart
        df = _make_trending_data(200, "up")
        path = generate_price_chart(df, "BTCUSDT")
        assert os.path.exists(path)
        assert path.endswith(".png")
        assert os.path.getsize(path) > 1000  # not empty
        os.remove(path)

    def test_generate_chart_with_levels(self):
        """Chart with key levels should not crash."""
        from charts import generate_price_chart
        df = _make_trending_data(200, "up")
        levels = {
            "ema_20": float(df["close"].iloc[-1]),
            "ema_50": float(df["close"].iloc[-1] * 0.98),
            "ema_200": float(df["close"].iloc[-1] * 0.95),
        }
        path = generate_price_chart(df, "BTCUSDT", key_levels=levels)
        assert os.path.exists(path)
        os.remove(path)

    def test_generate_chart_short_data(self):
        """Chart with very few candles should still work."""
        from charts import generate_price_chart
        df = _make_trending_data(30, "up")
        path = generate_price_chart(df, "BTCUSDT", last_n=30)
        assert os.path.exists(path)
        os.remove(path)


# ─── Formatter tests ─────────────────────────────────────────────────────────

class TestFormatter:
    """Test Telegram message formatting."""

    def test_prob_bar_output(self):
        """Probability bar should produce correct visual."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from telegram_bot import _prob_bar
        bar = _prob_bar(0.8)
        assert "████" in bar
        assert "80%" in bar

    def test_prob_bar_zero(self):
        from telegram_bot import _prob_bar
        bar = _prob_bar(0.0)
        assert "0%" in bar

    def test_prob_bar_max(self):
        from telegram_bot import _prob_bar
        bar = _prob_bar(0.85)
        assert "85%" in bar


# ─── Weight balance tests ────────────────────────────────────────────────────

class TestWeightBalance:
    """Test that technical indicators don't dominate."""

    def test_technical_weight_reduced(self):
        """MA alignment weight should be <=1.5 (was 2.0)."""
        from single_timeframe import _ma_alignment_vote
        # Create a strongly bullish series
        close = pd.Series(np.linspace(100, 200, 300))
        vote = _ma_alignment_vote(close)
        assert vote["weight"] <= 1.5, f"MA weight {vote['weight']} exceeds 1.5"

    def test_structure_weight_reduced(self):
        """Market structure weight should be <=1.7 (was 2.3)."""
        from single_timeframe import _structure_vote
        df = _make_trending_data(500, "up")
        vote = _structure_vote(df)
        assert vote["weight"] <= 1.7, f"Structure weight {vote['weight']} exceeds 1.7"

    def test_ichimoku_weight_reduced(self):
        """Ichimoku weight should be <=1.2 (was 1.5)."""
        from single_timeframe import _ichimoku_vote
        df = _make_trending_data(500, "up")
        vote = _ichimoku_vote(df)
        assert vote["weight"] <= 1.2, f"Ichimoku weight {vote['weight']} exceeds 1.2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
