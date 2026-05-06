"""
===============================================================================
 CRYPTO MARKET INSIGHT — TELEGRAM BOT (24/7)
===============================================================================
 Integrates with the multi-timeframe trend prediction pipeline to:
   - Scan top 20+ crypto pairs every 2 hours
   - Send alerts for high-quality setups (all TF groups aligned + edge)
   - Provide on-demand full analysis per coin
   - Daily market summary

 Commands:
   /scan       — Quick scan all tracked coins
   /analyze    — Full analysis: /analyze BTCUSDT
   /market     — Quick market overview (top 10)
   /universe   — Show tracked coins
   /add        — Add coin: /add SOLUSDT
   /remove     — Remove coin
   /alert      — Toggle auto-alerts: /alert on|off
   /interval   — Set scan interval: /interval 120
   /status     — Bot status
   /help       — Show commands
===============================================================================
"""

import os
import sys
import asyncio
import logging
import signal
from datetime import datetime, timezone
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from dotenv import load_dotenv
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binance_fetcher import fetch_all_timeframes, SUPPORTED_INTERVALS
from multi_timeframe import multi_timeframe_analysis

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ─── Config ───────────────────────────────────────────────────────────────────

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Thread pool for blocking Binance API + analysis
executor = ThreadPoolExecutor(max_workers=3)

# Top crypto pairs
DEFAULT_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "MATICUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "SUIUSDT", "SEIUSDT", "TIAUSDT",
]

# Timeframe sets
QUICK_TFS = ["15m", "1h", "4h", "12h"]
FULL_TFS = SUPPORTED_INTERVALS

# Default auto-scan every 3 minutes
SCAN_INTERVAL_SEC = 3 * 60

# Timeout per symbol analysis (seconds)
ANALYSIS_TIMEOUT = 180


# ─── Analysis wrapper (runs in thread) ───────────────────────────────────────

def _run_analysis(
    symbol: str,
    intervals: List[str],
    run_backtest: bool = False,
    max_candles: int = 2000,
) -> Dict:
    """Blocking: fetch data + run multi-TF pipeline."""
    try:
        data = fetch_all_timeframes(
            symbol=symbol,
            intervals=intervals,
            max_candles=max_candles,
            max_years=1,
            market="spot",
            verbose=False,
        )
        result = multi_timeframe_analysis(data, run_backtest=run_backtest)
        return {"symbol": symbol, "result": result, "error": None}
    except Exception as e:
        return {"symbol": symbol, "result": None, "error": str(e)}


async def _run_analysis_async(
    symbol: str,
    intervals: List[str],
    run_backtest: bool = False,
    timeout: float = ANALYSIS_TIMEOUT,
) -> Dict:
    """Run analysis in executor with timeout protection."""
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                executor, partial(_run_analysis, symbol, intervals, run_backtest)
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return {"symbol": symbol, "result": None, "error": f"Timeout ({timeout}s)"}


# ─── Message formatters (Rich UX) ────────────────────────────────────────────

_DIR_EMOJI = {"up": "🟢", "down": "🔴", "neutral": "⚪"}
_CONFIDENCE_EMOJI = {
    "high": "🔥", "medium": "💡", "low": "❄️",
}
_STRENGTH_EMOJI = {
    "strong": "💪", "moderate": "👍", "weak": "🤏",
}


def _get_price(result: Dict) -> str:
    per_tf = result.get("per_timeframe", {})
    for tf in ["1h", "4h", "15m", "12h"]:
        if tf in per_tf and "last_close" in per_tf[tf]:
            return f"${per_tf[tf]['last_close']:,.2f}"
    return "?"


def _prob_bar(pct: float, width: int = 10) -> str:
    """Visual probability bar: ████████░░ 80%"""
    filled = round(pct * width)
    empty = width - filled
    return "█" * filled + "░" * empty + f" {pct*100:.0f}%"


def _confidence_level(prob: float) -> str:
    if prob >= 0.65:
        return "high"
    if prob >= 0.50:
        return "medium"
    return "low"


def format_symbol_line(symbol: str, result: Dict) -> str:
    """Rich one-liner for scan summary."""
    overall = result.get("overall", {})
    if "error" in overall:
        return f"⚠️ {symbol}: {overall['error']}"

    d = overall.get("overall_direction", "neutral")
    prob = overall.get("weighted_probabilities", {})
    top_p = prob.get(d, 0)
    hq = " ⭐" if overall.get("high_quality_setup") else ""
    conf = _confidence_level(top_p)
    conf_emoji = _CONFIDENCE_EMOJI.get(conf, "")
    htf = _DIR_EMOJI.get(overall.get("higher_tf_direction", "?"), "❓")
    mtf = _DIR_EMOJI.get(overall.get("middle_tf_direction", "?"), "❓")
    ltf = _DIR_EMOJI.get(overall.get("lower_tf_direction", "?"), "❓")
    price = _get_price(result)

    return (
        f"{_DIR_EMOJI.get(d, '❓')} <b>{symbol}</b> {price}{hq} {conf_emoji}\n"
        f"   {_prob_bar(prob.get('up',0))} UP\n"
        f"   {_prob_bar(prob.get('down',0))} DN\n"
        f"   TFs: {htf}{mtf}{ltf}"
    )


def format_full_report(symbol: str, result: Dict) -> str:
    """Rich detailed report for /analyze."""
    overall = result.get("overall", {})
    if "error" in overall:
        return f"⚠️ {symbol}: {overall['error']}"

    d = overall.get("overall_direction", "?")
    d_upper = d.upper()
    prob = overall.get("weighted_probabilities", {})
    top_p = prob.get(d, 0)
    hq = "⭐ <b>HIGH QUALITY SETUP</b>" if overall.get("high_quality_setup") else ""
    n_edge = overall.get("n_timeframes_with_edge", 0)
    n_total = overall.get("n_timeframes_total", 0)
    conf = _confidence_level(top_p)
    conf_emoji = _CONFIDENCE_EMOJI.get(conf, "")

    lines = [
        f"{'━' * 30}",
        f"{_DIR_EMOJI.get(d, '❓')} <b>{symbol}</b> — {d_upper} {conf_emoji}",
        f"{'━' * 30}",
    ]
    if hq:
        lines.append(hq)
    lines.append("")

    # Probability bars
    lines.append("📊 <b>Probability</b>")
    lines.append(f"  🟢 UP   {_prob_bar(prob.get('up',0))}")
    lines.append(f"  🔴 DOWN {_prob_bar(prob.get('down',0))}")
    lines.append(f"  ⚪ FLAT {_prob_bar(prob.get('neutral',0))}")
    lines.append("")

    # Confluence
    h_dir = overall.get("higher_tf_direction", "?")
    m_dir = overall.get("middle_tf_direction", "?")
    l_dir = overall.get("lower_tf_direction", "?")
    lines.append("🔍 <b>Top-Down Confluence</b>")
    lines.append(
        f"  {_DIR_EMOJI.get(h_dir,'❓')} Higher (4h-12h): <b>{h_dir}</b>"
        f" ({overall.get('higher_tf_agreement',0)*100:.0f}%)"
    )
    lines.append(
        f"  {_DIR_EMOJI.get(m_dir,'❓')} Middle  (1h-2h): <b>{m_dir}</b>"
        f" ({overall.get('middle_tf_agreement',0)*100:.0f}%)"
    )
    lines.append(
        f"  {_DIR_EMOJI.get(l_dir,'❓')} Lower   (≤30m): <b>{l_dir}</b>"
        f" ({overall.get('lower_tf_agreement',0)*100:.0f}%)"
    )
    lines.append(f"  📈 Edge TFs: <b>{n_edge}/{n_total}</b>")
    lines.append("")

    # Per-timeframe
    per_tf = result.get("per_timeframe", {})
    lines.append("📋 <b>Per Timeframe</b>")
    for tf, info in per_tf.items():
        if "error" in info:
            lines.append(f"  ❌ <code>{tf:>4}</code> {info['error']}")
            continue
        p = info.get("probabilities", {})
        td = info.get("direction", "?")
        ts = info.get("strength", "?")
        s_emoji = _STRENGTH_EMOJI.get(ts, "")
        cal = info.get("calibrated", {})
        edge_txt = ""
        if cal.get("edge_vs_baseline") is not None:
            e = cal["edge_vs_baseline"]
            marker = "✅" if cal.get("has_edge") else "❌"
            edge_txt = f" {marker}{e*100:+.1f}%"
        lines.append(
            f"  {_DIR_EMOJI.get(td,'❓')} <code>{tf:>4}</code> {s_emoji}"
            f" UP:{p.get('up',0)*100:.0f}% DN:{p.get('down',0)*100:.0f}%{edge_txt}"
        )

    lines.append("")
    rec = overall.get("top_down_recommendation", "")
    lines.append(f"💬 {rec}")

    # Key levels from 4h or 1h
    for tf_key in ["4h", "1h"]:
        if tf_key in per_tf and "key_levels" in per_tf[tf_key]:
            kl = per_tf[tf_key]["key_levels"]
            lines.append(f"\n📍 <b>Key Levels ({tf_key})</b>")
            if kl.get("ema_20") is not None:
                lines.append(f"  EMA20:  <code>${kl['ema_20']:,.2f}</code>")
            if kl.get("ema_50") is not None:
                lines.append(f"  EMA50:  <code>${kl['ema_50']:,.2f}</code>")
            if kl.get("ema_200") is not None:
                lines.append(f"  EMA200: <code>${kl['ema_200']:,.2f}</code>")
            if kl.get("atr_14") is not None:
                lines.append(f"  ATR14:  <code>${kl['atr_14']:,.2f}</code>")
            break

    lines.append("")
    lines.append(f"<i>⚠️ Not financial advice. Model max confidence capped at 85%.</i>")

    return "\n".join(lines)


# ─── Inline keyboards ────────────────────────────────────────────────────────

def _analyze_keyboard(symbol: str) -> InlineKeyboardMarkup:
    """Inline buttons after /analyze result."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📈 Chart", callback_data=f"chart:{symbol}"),
            InlineKeyboardButton("🔄 Refresh", callback_data=f"refresh:{symbol}"),
        ],
        [
            InlineKeyboardButton("📋 Details", callback_data=f"details:{symbol}"),
            InlineKeyboardButton("🔔 Alert", callback_data=f"alert_set:{symbol}"),
        ],
    ])


# ─── Telegram helpers ─────────────────────────────────────────────────────────

async def _send_long(target, text: str, parse_mode=ParseMode.HTML, keyboard=None):
    """Send a message, splitting into chunks if > 4096 chars. Last chunk gets keyboard."""
    chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)]
    for idx, chunk in enumerate(chunks):
        is_last = idx == len(chunks) - 1
        await target.reply_text(
            chunk,
            parse_mode=parse_mode,
            reply_markup=keyboard if is_last else None,
        )
        if not is_last:
            await asyncio.sleep(0.5)


async def _send_to_chat(bot, chat_id, text: str):
    chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)]
    for idx, chunk in enumerate(chunks):
        await bot.send_message(
            chat_id=chat_id, text=chunk, parse_mode=ParseMode.HTML
        )
        if idx < len(chunks) - 1:
            await asyncio.sleep(0.5)


# ─── Command handlers ────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 <b>Crypto Market Insight Bot</b>\n\n"
        "<b>Commands:</b>\n"
        "  /scan — Quick scan top 20 coins\n"
        "  /analyze BTCUSDT — Full analysis\n"
        "  /analyze BTCUSDT -b — With backtest\n"
        "  /market — Quick overview top 10\n"
        "  /universe — Show tracked coins\n"
        "  /add XYZUSDT — Add coin\n"
        "  /remove XYZUSDT — Remove coin\n"
        "  /alert on|off — Toggle auto-alerts\n"
        "  /interval 120 — Scan interval (min)\n"
        "  /status — Bot status\n\n"
        "Auto-scan runs every 3 min and sends full market report."
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick scan all tracked coins."""
    universe = context.bot_data.get("universe", DEFAULT_UNIVERSE.copy())
    await update.message.reply_text(
        f"🔍 Scanning <b>{len(universe)}</b> coins on {QUICK_TFS}… (a few minutes)",
        parse_mode=ParseMode.HTML,
    )

    ups, downs, neutrals, errors, alerts = [], [], [], [], []

    for symbol in universe:
        try:
            res = await _run_analysis_async(symbol, QUICK_TFS, False)
            if res["error"]:
                errors.append(symbol)
                continue

            overall = res["result"].get("overall", {})
            d = overall.get("overall_direction", "neutral")
            bucket = {"up": ups, "down": downs}.get(d, neutrals)
            bucket.append(res)

            if overall.get("high_quality_setup"):
                alerts.append(res)
        except Exception as e:
            logger.error(f"Scan {symbol}: {e}")
            errors.append(symbol)

    # Build message
    now = datetime.now(timezone.utc)
    lines = [f"📊 <b>Market Scan</b> — {now:%Y-%m-%d %H:%M} UTC\n"]

    if ups:
        lines.append(f"🟢 <b>BULLISH ({len(ups)}):</b>")
        for r in ups:
            lines.append(format_symbol_line(r["symbol"], r["result"]))
        lines.append("")
    if downs:
        lines.append(f"🔴 <b>BEARISH ({len(downs)}):</b>")
        for r in downs:
            lines.append(format_symbol_line(r["symbol"], r["result"]))
        lines.append("")
    if neutrals:
        lines.append(f"⚪ <b>NEUTRAL ({len(neutrals)}):</b>")
        for r in neutrals:
            lines.append(format_symbol_line(r["symbol"], r["result"]))
        lines.append("")
    if errors:
        lines.append(f"⚠️ Errors: {', '.join(errors)}")
    if alerts:
        lines.append("\n⭐ <b>HIGH QUALITY SETUPS:</b>")
        for r in alerts:
            ad = r["result"]["overall"]["overall_direction"].upper()
            lines.append(f"  ⭐ <b>{r['symbol']}</b> → {ad}")

    await _send_long(update.message, "\n".join(lines))


async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Full analysis for one coin with progress indicators + inline keyboard."""
    if not context.args:
        await update.message.reply_text(
            "Usage: /analyze BTCUSDT [-b for backtest]",
            parse_mode=ParseMode.HTML,
        )
        return

    symbol = context.args[0].upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    run_bt = "-b" in context.args or "--backtest" in context.args
    bt_label = " + backtest" if run_bt else ""

    # Progress message — edit in-place as each step completes
    progress_msg = await update.message.reply_text(
        f"🔍 <b>Analyzing {symbol}</b>{bt_label}\n\n"
        f"⏳ Step 1/3 — Fetching market data…",
        parse_mode=ParseMode.HTML,
    )

    # Step 1: Start analysis
    try:
        await progress_msg.edit_text(
            f"🔍 <b>Analyzing {symbol}</b>{bt_label}\n\n"
            f"✅ Step 1/3 — Data fetched\n"
            f"⏳ Step 2/3 — Running {len(FULL_TFS)} timeframe analysis…",
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        pass  # edit may fail if too fast

    res = await _run_analysis_async(symbol, FULL_TFS, run_bt, timeout=300)

    if res["error"]:
        try:
            await progress_msg.edit_text(
                f"❌ <b>{symbol}</b>: {res['error']}",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            await update.message.reply_text(
                f"❌ {res['error']}", parse_mode=ParseMode.HTML,
            )
        return

    # Step 3: Format and send
    try:
        await progress_msg.edit_text(
            f"🔍 <b>Analyzing {symbol}</b>{bt_label}\n\n"
            f"✅ Step 1/3 — Data fetched\n"
            f"✅ Step 2/3 — Analysis complete\n"
            f"⏳ Step 3/3 — Generating report…",
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        pass

    # Store result for callback buttons
    context.bot_data[f"last_result:{symbol}"] = res

    report = format_full_report(symbol, res["result"])
    keyboard = _analyze_keyboard(symbol)

    # Delete progress message, send final report with keyboard
    try:
        await progress_msg.delete()
    except Exception:
        pass

    await _send_long(
        update.message, report, keyboard=keyboard,
    )


async def cmd_market(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick overview of top 10 coins (1h + 4h only)."""
    universe = context.bot_data.get("universe", DEFAULT_UNIVERSE.copy())[:10]
    await update.message.reply_text(f"📊 Checking top {len(universe)} coins…")

    lines = [f"📊 <b>Market Overview</b> — {datetime.now(timezone.utc):%H:%M} UTC\n"]

    for symbol in universe:
        try:
            res = await _run_analysis_async(symbol, ["1h", "4h"], False, timeout=120)
            if res["error"]:
                lines.append(f"⚠️ {symbol}: error")
                continue
            overall = res["result"]["overall"]
            d = overall.get("overall_direction", "?")
            prob = overall.get("weighted_probabilities", {})
            price = _get_price(res["result"])
            lines.append(
                f"{_DIR_EMOJI.get(d,'❓')} <b>{symbol}</b>  {price}"
                f"  UP:{prob.get('up',0)*100:.0f}% DN:{prob.get('down',0)*100:.0f}%"
            )
        except Exception as e:
            lines.append(f"⚠️ {symbol}: {e}")

    await _send_long(update.message, "\n".join(lines))


async def cmd_universe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    universe = context.bot_data.get("universe", DEFAULT_UNIVERSE.copy())
    msg = f"🌐 <b>Tracked ({len(universe)}):</b>\n\n" + "  ".join(universe)
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /add SOLUSDT")
        return
    symbol = context.args[0].upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    universe = context.bot_data.setdefault("universe", DEFAULT_UNIVERSE.copy())
    if symbol in universe:
        await update.message.reply_text(f"{symbol} already tracked")
        return
    universe.append(symbol)
    await update.message.reply_text(f"✅ Added {symbol} ({len(universe)} total)")


async def cmd_remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /remove SOLUSDT")
        return
    symbol = context.args[0].upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    universe = context.bot_data.setdefault("universe", DEFAULT_UNIVERSE.copy())
    if symbol not in universe:
        await update.message.reply_text(f"{symbol} not tracked")
        return
    universe.remove(symbol)
    await update.message.reply_text(f"✅ Removed {symbol} ({len(universe)} left)")


async def cmd_alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        on = context.bot_data.get("alerts_on", True)
        await update.message.reply_text(
            f"Auto alerts: {'ON ✅' if on else 'OFF 🔇'}\nUsage: /alert on|off"
        )
        return
    if context.args[0].lower() == "on":
        context.bot_data["alerts_on"] = True
        await update.message.reply_text("✅ Auto-alerts ON")
    else:
        context.bot_data["alerts_on"] = False
        await update.message.reply_text("🔇 Auto-alerts OFF")


async def cmd_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        mins = context.bot_data.get("scan_sec", SCAN_INTERVAL_SEC) // 60
        await update.message.reply_text(
            f"Current interval: {mins} min\nUsage: /interval 120"
        )
        return
    try:
        mins = int(context.args[0])
        if mins < 1:
            await update.message.reply_text("Minimum 1 minute")
            return
        context.bot_data["scan_sec"] = mins * 60
        # Reschedule
        for job in context.job_queue.get_jobs_by_name("auto_scan"):
            job.schedule_removal()
        context.job_queue.run_repeating(
            auto_scan_job, interval=mins * 60, first=mins * 60, name="auto_scan"
        )
        await update.message.reply_text(f"✅ Interval → {mins} min")
    except ValueError:
        await update.message.reply_text("Invalid number")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    universe = context.bot_data.get("universe", DEFAULT_UNIVERSE.copy())
    on = context.bot_data.get("alerts_on", True)
    secs = context.bot_data.get("scan_sec", SCAN_INTERVAL_SEC)
    last = context.bot_data.get("last_scan", "Never")
    msg = (
        f"🤖 <b>Bot Status</b>\n\n"
        f"  Coins: {len(universe)}\n"
        f"  Alerts: {'ON' if on else 'OFF'}\n"
        f"  Interval: {secs // 60} min\n"
        f"  Last scan: {last}\n"
        f"  Status: Running ✅"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ─── Callback handler for inline buttons ─────────────────────────────────────

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard button presses."""
    query = update.callback_query
    await query.answer()

    data = query.data
    action, symbol = data.split(":", 1)

    if action == "chart":
        await query.edit_message_reply_markup(reply_markup=None)
        progress = await query.message.reply_text(
            f"📈 Generating chart for <b>{symbol}</b>…",
            parse_mode=ParseMode.HTML,
        )
        try:
            from charts import generate_price_chart
            res = context.bot_data.get(f"last_result:{symbol}")
            if res and res.get("result"):
                # Re-fetch data for chart
                loop = asyncio.get_event_loop()
                chart_data = await loop.run_in_executor(
                    executor, partial(
                        _run_analysis, symbol, ["1h"], False, 500
                    )
                )
                if chart_data and chart_data.get("result"):
                    per_tf = chart_data["result"].get("per_timeframe", {})
                    tf_key = next(iter(per_tf), None)
                    if tf_key:
                        # Get key levels from cached result
                        cached_overall = res["result"].get("overall", {})
                        cached_per_tf = res["result"].get("per_timeframe", {})
                        key_levels = None
                        for ktf in ["4h", "1h"]:
                            if ktf in cached_per_tf and "key_levels" in cached_per_tf[ktf]:
                                key_levels = cached_per_tf[ktf]["key_levels"]
                                break

                        # generate_price_chart needs OHLCV data
                        # We fetch it separately for chart
                        from binance_fetcher import fetch_all_timeframes
                        ohlcv_data = await loop.run_in_executor(
                            executor, partial(
                                fetch_all_timeframes,
                                symbol=symbol,
                                intervals=["1h"],
                                max_candles=200,
                                max_years=0.1,
                                market="spot",
                                verbose=False,
                            )
                        )
                        if "1h" in ohlcv_data and ohlcv_data["1h"] is not None:
                            chart_path = generate_price_chart(
                                ohlcv_data["1h"], symbol, key_levels=key_levels,
                            )
                            await progress.delete()
                            with open(chart_path, "rb") as f:
                                await query.message.reply_photo(
                                    photo=f,
                                    caption=f"📈 {symbol} 1H Chart",
                                )
                            os.remove(chart_path)
                            return
            await progress.edit_text("❌ No data available for chart")
        except ImportError:
            await progress.edit_text("❌ Chart module not available (install matplotlib)")
        except Exception as e:
            logger.error(f"Chart error: {e}")
            await progress.edit_text(f"❌ Chart error: {e}")

    elif action == "refresh":
        await query.edit_message_reply_markup(reply_markup=None)
        progress = await query.message.reply_text(
            f"🔄 Refreshing <b>{symbol}</b>…",
            parse_mode=ParseMode.HTML,
        )
        res = await _run_analysis_async(symbol, FULL_TFS, False, timeout=300)
        await progress.delete()
        if res["error"]:
            await query.message.reply_text(
                f"❌ {res['error']}", parse_mode=ParseMode.HTML,
            )
        else:
            context.bot_data[f"last_result:{symbol}"] = res
            report = format_full_report(symbol, res["result"])
            keyboard = _analyze_keyboard(symbol)
            await _send_long(query.message, report, keyboard=keyboard)

    elif action == "details":
        res = context.bot_data.get(f"last_result:{symbol}")
        if not res or not res.get("result"):
            await query.message.reply_text("❌ No cached data. Run /analyze again.")
            return
        # Show per-model vote details
        per_tf = res["result"].get("per_timeframe", {})
        lines = [f"📋 <b>{symbol} — Model Votes Detail</b>\n"]
        for tf, info in per_tf.items():
            if "error" in info or "votes" not in info:
                continue
            lines.append(f"\n<b>[{tf}]</b>")
            for model, v in info["votes"].items():
                vote = v.get("vote", 0)
                weight = v.get("weight", 0.0)
                v_emoji = "🟢" if vote > 0 else ("🔴" if vote < 0 else "⚪")
                lines.append(f"  {v_emoji} {model}: vote={vote:+d} w={weight:.1f}")
            # Only show first 3 TFs to avoid message limit
            if len(lines) > 60:
                lines.append("\n<i>… truncated (showing top TFs)</i>")
                break
        await _send_long(query.message, "\n".join(lines))

    elif action == "alert_set":
        await query.message.reply_text(
            f"🔔 Alert for <b>{symbol}</b> will trigger on next high-quality setup.\n"
            f"Auto-scan is {'ON ✅' if context.bot_data.get('alerts_on', True) else 'OFF 🔇'}",
            parse_mode=ParseMode.HTML,
        )


# ─── Scheduled auto-scan ─────────────────────────────────────────────────────

async def auto_scan_job(context: ContextTypes.DEFAULT_TYPE):
    """Runs on schedule: full market report like /market, sent automatically."""
    if not context.bot_data.get("alerts_on", True):
        return

    chat_id = context.bot_data.get("chat_id", CHAT_ID)
    if not chat_id:
        return

    universe = context.bot_data.get("universe", DEFAULT_UNIVERSE.copy())
    logger.info(f"Auto-scan: {len(universe)} coins")
    now = datetime.now(timezone.utc)
    context.bot_data["last_scan"] = f"{now:%Y-%m-%d %H:%M} UTC"

    ups, downs, neutrals, errors, alerts = [], [], [], [], []

    for symbol in universe:
        try:
            res = await _run_analysis_async(symbol, QUICK_TFS, False)
            if res["error"]:
                errors.append(symbol)
                continue

            overall = res["result"].get("overall", {})
            d = overall.get("overall_direction", "neutral")
            bucket = {"up": ups, "down": downs}.get(d, neutrals)
            bucket.append(res)

            if overall.get("high_quality_setup"):
                alerts.append(res)
        except Exception as e:
            logger.error(f"Auto-scan {symbol}: {e}")
            errors.append(symbol)

    # ── Build full market report (same style as /scan) ──
    lines = [
        f"📊 <b>Auto Market Report</b> — {now:%Y-%m-%d %H:%M} UTC",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🟢 {len(ups)} Bullish  |  🔴 {len(downs)} Bearish  |  ⚪ {len(neutrals)} Neutral",
        "",
    ]

    if ups:
        lines.append(f"🟢 <b>BULLISH ({len(ups)}):</b>")
        for r in ups:
            lines.append(format_symbol_line(r["symbol"], r["result"]))
        lines.append("")

    if downs:
        lines.append(f"🔴 <b>BEARISH ({len(downs)}):</b>")
        for r in downs:
            lines.append(format_symbol_line(r["symbol"], r["result"]))
        lines.append("")

    if neutrals:
        lines.append(f"⚪ <b>NEUTRAL ({len(neutrals)}):</b>")
        for r in neutrals:
            lines.append(format_symbol_line(r["symbol"], r["result"]))
        lines.append("")

    if errors:
        lines.append(f"⚠️ Errors: {', '.join(errors)}")

    # ── High-quality alerts section ──
    if alerts:
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("⭐ <b>HIGH QUALITY SETUPS:</b>")
        for r in alerts:
            ad = r["result"]["overall"]["overall_direction"].upper()
            prob = r["result"]["overall"]["weighted_probabilities"]
            rec = r["result"]["overall"].get("top_down_recommendation", "")
            price = _get_price(r["result"])
            lines.append(
                f"\n  ⭐ <b>{r['symbol']}</b> → {ad}  ({price})\n"
                f"     UP:{prob.get('up',0)*100:.0f}% DN:{prob.get('down',0)*100:.0f}%\n"
                f"     {rec}"
            )

    secs = context.bot_data.get("scan_sec", SCAN_INTERVAL_SEC)
    lines.append(f"\n🔄 Next scan in {secs // 60} min")

    try:
        await _send_to_chat(context.bot, chat_id, "\n".join(lines))
    except Exception as e:
        logger.error(f"Send failed: {e}")


# ─── Post-init & startup ─────────────────────────────────────────────────────

async def post_init(application: Application):
    commands = [
        BotCommand("scan", "Quick scan all tracked coins"),
        BotCommand("analyze", "Full analysis: /analyze BTCUSDT"),
        BotCommand("market", "Quick market overview"),
        BotCommand("universe", "Show tracked coins"),
        BotCommand("add", "Add coin: /add SOLUSDT"),
        BotCommand("remove", "Remove coin: /remove SOLUSDT"),
        BotCommand("alert", "Toggle alerts: /alert on|off"),
        BotCommand("interval", "Set scan interval (min)"),
        BotCommand("status", "Bot status"),
        BotCommand("help", "Show commands"),
    ]
    await application.bot.set_my_commands(commands)

    application.bot_data.update({
        "chat_id": CHAT_ID,
        "universe": DEFAULT_UNIVERSE.copy(),
        "alerts_on": True,
        "scan_sec": SCAN_INTERVAL_SEC,
    })

    # Schedule auto-scan (first run after 60s to let bot warm up)
    application.job_queue.run_repeating(
        auto_scan_job,
        interval=SCAN_INTERVAL_SEC,
        first=60,
        name="auto_scan",
    )

    if CHAT_ID:
        try:
            await application.bot.send_message(
                chat_id=CHAT_ID,
                text=(
                    "🤖 <b>Crypto Insight Bot Started!</b>\n\n"
                    f"Tracking {len(DEFAULT_UNIVERSE)} coins\n"
                    f"Auto-scan every {SCAN_INTERVAL_SEC // 60} min\n\n"
                    "Use /help to see commands"
                ),
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.warning(f"Startup msg failed: {e}")


def main():
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ BOT_TOKEN not set!")
        print("   1. Open Telegram → search @BotFather → /newbot")
        print("   2. Copy the token")
        print("   3. Paste into .env:  BOT_TOKEN=<token>")
        sys.exit(1)

    if not CHAT_ID or CHAT_ID == "YOUR_CHAT_ID_HERE":
        print("⚠️  CHAT_ID not set — auto-alerts won't work.")
        print("   Add CHAT_ID=<your_id> to .env")

    print(f"🚀 Starting Crypto Insight Bot…")
    print(f"   Universe: {len(DEFAULT_UNIVERSE)} coins")
    print(f"   Auto-scan: every {SCAN_INTERVAL_SEC // 60} min")
    print(f"   TFs (quick): {QUICK_TFS}")
    print(f"   TFs (full):  {FULL_TFS}")

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    handlers = [
        ("start", cmd_start), ("help", cmd_help), ("scan", cmd_scan),
        ("analyze", cmd_analyze), ("market", cmd_market),
        ("universe", cmd_universe), ("add", cmd_add), ("remove", cmd_remove),
        ("alert", cmd_alert), ("interval", cmd_interval), ("status", cmd_status),
    ]
    for name, fn in handlers:
        app.add_handler(CommandHandler(name, fn))

    # Inline keyboard callback handler
    app.add_handler(CallbackQueryHandler(button_callback))

    try:
        app.run_polling(drop_pending_updates=True)
    finally:
        executor.shutdown(wait=False)
        logger.info("Bot stopped, executor shut down.")


if __name__ == "__main__":
    main()
