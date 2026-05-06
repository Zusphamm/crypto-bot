"""
===============================================================================
 PRICE CHART GENERATOR — Dark-theme Matplotlib charts for Telegram
===============================================================================
 Generates:
   - Candlestick-style price chart with EMA overlays
   - Entry / SL / TP level lines (if provided)
   - Volume subplot
   - Dark theme optimized for mobile Telegram viewing

 Usage:
   from charts import generate_price_chart
   path = generate_price_chart(df, "BTCUSDT", key_levels={...})
   # Send path as photo via Telegram, then os.remove(path)
===============================================================================
"""
import os
import tempfile
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


# ─── Dark theme ──────────────────────────────────────────────────────────────

DARK_BG = "#1a1a2e"
DARK_PANEL = "#16213e"
GREEN = "#00d474"
RED = "#ff4757"
BLUE = "#4ecdc4"
ORANGE = "#ff9f43"
PURPLE = "#a55eea"
GRAY = "#636e72"
WHITE = "#dfe6e9"
YELLOW = "#ffeaa7"


def _apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": DARK_PANEL,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": WHITE,
        "text.color": WHITE,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "grid.color": "#2d3436",
        "grid.alpha": 0.3,
        "font.size": 10,
    })


def generate_price_chart(
    df: pd.DataFrame,
    symbol: str,
    key_levels: Optional[Dict] = None,
    last_n: int = 100,
) -> str:
    """
    Generate a dark-theme price chart with EMA overlays and volume.

    Args:
        df:          OHLCV DataFrame with DatetimeIndex
        symbol:      Pair name for title
        key_levels:  Optional dict with ema_20, ema_50, ema_200, atr_14
        last_n:      Number of recent candles to display

    Returns:
        Path to temporary PNG file. Caller must delete after use.
    """
    _apply_dark_theme()

    # Trim to last N candles
    plot_df = df.iloc[-last_n:].copy()
    if plot_df.empty:
        raise ValueError("No data to plot")

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1,
        figsize=(12, 7),
        height_ratios=[3, 1],
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.05)

    # ─── Price chart (OHLC bars) ─────────────────────────────────────────
    dates = np.arange(len(plot_df))
    opens = plot_df["open"].values
    highs = plot_df["high"].values
    lows = plot_df["low"].values
    closes = plot_df["close"].values

    colors = [GREEN if c >= o else RED for o, c in zip(opens, closes)]

    # Candlestick bodies
    body_width = 0.6
    for i in range(len(dates)):
        body_bottom = min(opens[i], closes[i])
        body_height = abs(closes[i] - opens[i])
        rect = Rectangle(
            (dates[i] - body_width / 2, body_bottom),
            body_width, max(body_height, closes[i] * 0.0001),
            facecolor=colors[i], edgecolor=colors[i], linewidth=0.5,
        )
        ax_price.add_patch(rect)
        # Wicks
        ax_price.plot(
            [dates[i], dates[i]], [lows[i], highs[i]],
            color=colors[i], linewidth=0.7,
        )

    ax_price.set_xlim(-1, len(dates))
    ax_price.set_ylim(lows.min() * 0.998, highs.max() * 1.002)

    # ─── EMA overlays ────────────────────────────────────────────────────
    close_s = plot_df["close"]

    if len(close_s) >= 20:
        ema20 = close_s.ewm(span=20, adjust=False).mean()
        ax_price.plot(dates, ema20.values, color=BLUE, linewidth=1.2,
                      label="EMA 20", alpha=0.9)

    if len(close_s) >= 50:
        ema50 = close_s.ewm(span=50, adjust=False).mean()
        ax_price.plot(dates, ema50.values, color=ORANGE, linewidth=1.2,
                      label="EMA 50", alpha=0.9)

    # ─── Key level lines ─────────────────────────────────────────────────
    if key_levels:
        price_range = highs.max() - lows.min()
        for label, key, color, style in [
            ("EMA200", "ema_200", PURPLE, "--"),
            ("SL", "stop_loss", RED, ":"),
            ("TP", "take_profit", GREEN, ":"),
            ("Entry", "entry", YELLOW, "-."),
        ]:
            val = key_levels.get(key)
            if val is not None:
                # Only draw if within visible price range (±20%)
                if lows.min() * 0.8 < val < highs.max() * 1.2:
                    ax_price.axhline(
                        y=val, color=color, linestyle=style,
                        linewidth=1.0, alpha=0.7, label=label,
                    )

    ax_price.legend(loc="upper left", fontsize=8, facecolor=DARK_PANEL,
                    edgecolor=GRAY, labelcolor=WHITE)
    ax_price.set_ylabel("Price", fontsize=10)
    ax_price.set_title(f"{symbol}  —  Last {len(plot_df)} candles",
                       fontsize=13, fontweight="bold", pad=10)
    ax_price.grid(True, alpha=0.2)

    # ─── Volume subplot ──────────────────────────────────────────────────
    volumes = plot_df["volume"].values
    vol_colors = [GREEN if c >= o else RED for o, c in zip(opens, closes)]
    ax_vol.bar(dates, volumes, width=body_width, color=vol_colors, alpha=0.6)
    ax_vol.set_ylabel("Volume", fontsize=10)
    ax_vol.grid(True, alpha=0.2)

    # X-axis: show dates at intervals
    n_labels = min(8, len(plot_df))
    step = max(1, len(plot_df) // n_labels)
    tick_positions = list(range(0, len(plot_df), step))
    tick_labels = [plot_df.index[i].strftime("%m/%d %H:%M")
                   if hasattr(plot_df.index[i], "strftime")
                   else str(plot_df.index[i])[:10]
                   for i in tick_positions]
    ax_vol.set_xticks(tick_positions)
    ax_vol.set_xticklabels(tick_labels, rotation=30, fontsize=8)

    # ─── Save to temp file ───────────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(
        suffix=".png", prefix=f"chart_{symbol}_",
        delete=False, dir=tempfile.gettempdir(),
    )
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    return tmp.name
