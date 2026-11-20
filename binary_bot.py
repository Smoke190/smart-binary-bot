import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import io

import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()

# ====================== НАСТРОЙКИ ======================
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID", "0"))

DEFAULT_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]
HTF = "5m"
LTF = "1m"
MIN_SCORE = 6
SCAN_INTERVAL = 55

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
exchange = ccxt.binance({"enableRateLimit": True})

# ====================== БАЗОВЫЕ ФУНКЦИИ ======================

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 120) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["rsi"] = ta.rsi(df["close"], length=7)

    stoch = ta.stoch(df["high"], df["low"], df["close"], k=5, d=3, smooth_k=3)
    df = pd.concat([df, stoch], axis=1)

    bb = ta.bbands(df["close"], length=20, std=2.0)
    df = pd.concat([df, bb], axis=1)

    df["vol_sma"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma"]
    return df

def find_swing_levels(df: pd.DataFrame, lookback: int = 22) -> Tuple[List[float], List[float]]:
    highs, lows = [], []
    for i in range(3, min(lookback, len(df) - 3)):
        h = df["high"].iloc[-i]
        l = df["low"].iloc[-i]
        if (h > df["high"].iloc[-i - 1] and h > df["high"].iloc[-i - 2] and
            h > df["high"].iloc[-i + 1] and h > df["high"].iloc[-i + 2]):
            highs.append(h)
        if (l < df["low"].iloc[-i - 1] and l < df["low"].iloc[-i - 2] and
            l < df["low"].iloc[-i + 1] and l < df["low"].iloc[-i + 2]):
            lows.append(l)
    highs = sorted(set([round(x, 5) for x in highs]), reverse=True)[:5]
    lows = sorted(set([round(x, 5) for x in lows]))[:5]
    return highs, lows

def get_nearest(price, levels, side="support"):
    if not levels:
        return None
    if side == "support":
        candidates = [l for l in levels if l < price * 1.001]
        return max(candidates) if candidates else None
    candidates = [l for l in levels if l > price * 0.999]
    return min(candidates) if candidates else None

# ====================== РЕЖИМ РЫНКА ======================

def detect_market_regime(df: pd.DataFrame) -> Dict:
    try:
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        adx = adx_df["ADX_14"].iloc[-1]
    except Exception:
        adx = 25.0

    recent = df.iloc[-18:]
    range_high = recent["high"].max()
    range_low = recent["low"].min()

    touches = ((recent["high"] >= range_high * 0.998).sum() +
               (recent["low"] <= range_low * 1.002).sum())

    regime = "transition"
    if adx < 23 and touches >= 3:
        regime = "ranging"
    elif adx > 26:
        regime = "trending"

    return {
        "regime": regime,
        "adx": round(float(adx), 1),
        "range_high": range_high,
        "range_low": range_low
    }

# ====================== СТРУКТУРА ======================

def detect_market_structure(df: pd.DataFrame) -> Dict:
    highs = df["high"].iloc[-12:]
    lows = df["low"].iloc[-12:]
    recent_high = highs.max()
    prev_high = highs.iloc[:-3].max()
    recent_low = lows.min()
    prev_low = lows.iloc[:-3].min()

    structure = "neutral"
    if recent_high > prev_high and recent_low > prev_low:
        structure = "bullish"
    elif recent_high < prev_high and recent_low < prev_low:
        structure = "bearish"

    return {"structure": structure}

# ====================== СВЕЧНЫЕ ПАТТЕРНЫ ======================

def detect_candlestick_patterns(df: pd.DataFrame) -> Dict:
    last = df.iloc[-1]
    prev = df.iloc[-2]

    body = abs(last["close"] - last["open"])
    full = last["high"] - last["low"] or 0.0001
    upper = last["high"] - max(last["open"], last["close"])
    lower = min(last["open"], last["close"]) - last["low"]

    patterns = {"bullish": False, "bearish": False, "details": []}

    # Pin Bar
    if lower > body * 1.7 and upper < body * 0.45 and last["close"] > last["open"]:
        patterns["bullish"] = True
        patterns["details"].append("Bullish Pin Bar")
    if upper > body * 1.7 and lower < body * 0.45 and last["close"] < last["open"]:
        patterns["bearish"] = True
        patterns["details"].append("Bearish Pin Bar")

    # Engulfing
    if (prev["close"] < prev["open"] and last["close"] > last["open"] and
        last["close"] > prev["open"] and last["open"] < prev["close"]):
        patterns["bullish"] = True
        patterns["details"].append("Bullish Engulfing")
    if (prev["close"] > prev["open"] and last["close"] < last["open"] and
        last["close"] < prev["open"] and last["open"] > prev["close"]):
        patterns["bearish"] = True
        patterns["details"].append("Bearish Engulfing")

    return patterns

# ====================== ПРОБОИ + ОБЪЁМ + БОКОВИК ======================

def analyze_breakout_with_volume(df, support, resistance, regime) -> Dict:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    vol_ratio = last["vol_ratio"]

    result = {"event": None, "side": None, "quality": 0, "details": []}

    vol_q = "high" if vol_ratio >= 1.65 else ("low" if vol_ratio <= 0.75 else "normal")
    body = abs(last["close"] - last["open"])
    upper_wick = last["high"] - max(last["open"], last["close"])
    lower_wick = min(last["open"], last["close"]) - last["low"]

    # Ложный пробой вверх (Bull Trap)
    if resistance and prev["high"] > resistance + atr * 0.07 and last["close"] < resistance:
        if upper_wick > body * 1.2:
            if regime == "ranging" and vol_q in ["high", "normal"]:
                result.update({
                    "event": "false_breakout",
                    "side": "PUT",
                    "quality": 4 if vol_q == "high" else 3
                })
                result["details"].append(f"Качественный ложный пробой сопротивления (боковик) ×{vol_ratio:.1f}")
            elif vol_q == "high":
                result.update({"event": "false_breakout", "side": "PUT", "quality": 2})
                result["details"].append(f"Ложный пробой сопротивления ×{vol_ratio:.1f}")

    # Ложный пробой вниз (Bear Trap)
    if support and result["event"] is None and prev["low"] < support - atr * 0.07 and last["close"] > support:
        if lower_wick > body * 1.2:
            if regime == "ranging" and vol_q in ["high", "normal"]:
                result.update({
                    "event": "false_breakout",
                    "side": "CALL",
                    "quality": 4 if vol_q == "high" else 3
                })
                result["details"].append(f"Качественный ложный пробой поддержки (боковик) ×{vol_ratio:.1f}")
            elif vol_q == "high":
                result.update({"event": "false_breakout", "side": "CALL", "quality": 2})
                result["details"].append(f"Ложный пробой поддержки ×{vol_ratio:.1f}")

    # Сильный пробой вверх
    if resistance and result["event"] is None and prev["close"] < resistance and last["close"] > resistance:
        if vol_q == "high" and body > atr * 0.45 and regime != "ranging":
            result.update({"event": "strong_breakout", "side": "CALL", "quality": 4})
            result["details"].append(f"Сильный пробой сопротивления на объёме ×{vol_ratio:.1f}")
        elif vol_q == "low" or regime == "ranging":
            result["details"].append("Слабый/опасный пробой — пропуск")

    # Сильный пробой вниз
    if support and result["event"] is None and prev["close"] > support and last["close"] < support:
        if vol_q == "high" and body > atr * 0.45 and regime != "ranging":
            result.update({"event": "strong_breakout", "side": "PUT", "quality": 4})
            result["details"].append(f"Сильный пробой поддержки на объёме ×{vol_ratio:.1f}")
        elif vol_q == "low" or regime == "ranging":
            result["details"].append("Слабый/опасный пробой — пропуск")

    return result

# ====================== ГЛАВНЫЙ АНАЛИЗ ======================

def analyze_pair(symbol: str) -> Optional[Dict]:
    try:
        df_5m = add_indicators(fetch_ohlcv(symbol, HTF, 60))
        df_1m = add_indicators(fetch_ohlcv(symbol, LTF, 100))
        if len(df_1m) < 40:
            return None

        last = df_1m.iloc[-1]
        highs, lows = find_swing_levels(df_1m)
        h5, l5 = find_swing_levels(df_5m, 12)
        highs = list(set(highs + h5))[:5]
        lows = list(set(lows + l5))[:5]

        support = get_nearest(last["close"], lows, "support")
        resistance = get_nearest(last["close"], highs, "resistance")

        regime_info = detect_market_regime(df_1m)
        structure = detect_market_structure(df_1m)
        candles = detect_candlestick_patterns(df_1m)
        brk = analyze_breakout_with_volume(df_1m, support, resistance, regime_info["regime"])

        call_score = 0
        put_score = 0
        reasons = []

        # Режим
        if regime_info["regime"] == "ranging":
            reasons.append(f"Боковик (ADX {regime_info['adx']})")
        elif regime_info["regime"] == "trending":
            reasons.append(f"Тренд (ADX {regime_info['adx']})")

        # Структура
        if structure["structure"] == "bullish":
            call_score += 2
            reasons.append("Бычья структура")
        elif structure["structure"] == "bearish":
            put_score += 2
            reasons.append("Медвежья структура")

        # Пробои / ложные пробои
        if brk["event"] in ["false_breakout", "strong_breakout"] and brk["quality"] >= 2:
            if brk["side"] == "CALL":
                call_score += brk["quality"]
            else:
                put_score += brk["quality"]
            reasons.extend(brk["details"])
        elif any("пропуск" in str(d) for d in brk["details"]):
            return None

        # Свечные паттерны
        if candles["bullish"]:
            call_score += 2
            reasons.extend(candles["details"])
        if candles["bearish"]:
            put_score += 2
            reasons.extend(candles["details"])

        # Объём
        if last["vol_ratio"] > 1.6:
            if call_score > put_score:
                call_score += 1
            elif put_score > call_score:
                put_score += 1
            reasons.append(f"Объём ×{last['vol_ratio']:.1f}")

        # Stochastic
        k_col, d_col = "STOCHk_5_3_3", "STOCHd_5_3_3"
        if (df_1m[k_col].iloc[-2] < df_1m[d_col].iloc[-2] and
            last[k_col] > last[d_col] and last[k_col] < 40):
            call_score += 1
        if (df_1m[k_col].iloc[-2] > df_1m[d_col].iloc[-2] and
            last[k_col] < last[d_col] and last[k_col] > 60):
            put_score += 1

        # Финальное решение
        if call_score >= MIN_SCORE and call_score >= put_score + 2:
            signal = "CALL"
            score = call_score
        elif put_score >= MIN_SCORE and put_score >= call_score + 2:
            signal = "PUT"
            score = put_score
        else:
            return None

        now = datetime.utcnow()
        return {
            "symbol": symbol,
            "signal": signal,
            "score": score,
            "price": last["close"],
            "regime": regime_info["regime"],
            "structure": structure["structure"],
            "support": support,
            "resistance": resistance,
            "vol_ratio": last["vol_ratio"],
            "reasons": list(dict.fromkeys(reasons))[:6],
            "df_1m": df_1m.tail(50),
            "time": now.strftime("%H:%M:%S"),
            "expiry_4": (now + timedelta(minutes=4)).strftime("%H:%M"),
            "expiry_5": (now + timedelta(minutes=5)).strftime("%H:%M")
        }
    except Exception as e:
        logger.error(f"{symbol}: {e}")
        return None

# ====================== ГРАФИК ======================

def create_chart(data: Dict) -> io.BytesIO:
    df = data["df_1m"]
    mc = mpf.make_marketcolors(up="#26a69a", down="#ef5350", wick="inherit", volume="in", inherit=True)
    style = mpf.make_mpf_style(
        marketcolors=mc, gridstyle=":", facecolor="#131722", figcolor="#131722",
        rc={"axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"}
    )

    apds = [
        mpf.make_addplot(df["ema_9"], color="#ff9800", width=1.1),
        mpf.make_addplot(df["ema_21"], color="#42a5f5", width=1.2)
    ]

    hlines = {"hlines": [], "colors": [], "linewidths": [], "linestyle": "--"}
    if data.get("support"):
        hlines["hlines"].append(data["support"])
        hlines["colors"].append("#00e676")
        hlines["linewidths"].append(1.3)
    if data.get("resistance"):
        hlines["hlines"].append(data["resistance"])
        hlines["colors"].append("#ff1744")
        hlines["linewidths"].append(1.3)

    buf = io.BytesIO()
    fig, axes = mpf.plot(
        df, type="candle", style=style, addplot=apds,
        hlines=hlines if hlines["hlines"] else None,
        volume=True, figsize=(11, 6.5), returnfig=True,
        title=f"{data['symbol']} | {data['signal']} | {data['regime']}",
        tight_layout=True
    )

    ax = axes[0]
    color = "#00e676" if data["signal"] == "CALL" else "#ff1744"
    ax.text(0.97, 0.95, data["signal"], transform=ax.transAxes, fontsize=16,
            fontweight="bold", color=color, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="#131722", edgecolor=color))

    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#131722")
    plt.close(fig)
    buf.seek(0)
    return buf

def format_text(data: Dict) -> str:
    emoji = "🟢 CALL ↑" if data["signal"] == "CALL" else "🔴 PUT ↓"
    reasons = "\n".join([f"• {r}" for r in data["reasons"]])

    levels = ""
    if data.get("support"):
        levels += f"🛡 Поддержка: <code>{data['support']:.5f}</code>\n"
    if data.get("resistance"):
        levels += f"🚧 Сопротивление: <code>{data['resistance']:.5f}</code>\n"

    return f"""
{emoji} <b>{data['symbol']}</b>

📊 Сила: {data['score']} | Режим: {data['regime']}
Структура: {data['structure']}
💰 Цена: <code>{data['price']:.5f}</code>
📦 Объём: ×{data['vol_ratio']:.1f}

{levels}
⏱ <b>Экспирация:</b>
4 мин → <code>{data['expiry_4']}</code>
5 мин → <code>{data['expiry_5']}</code>

<b>Почему:</b>
{reasons}

⏰ {data['time']} UTC
""".strip()

# ====================== TELEGRAM ======================

async def send_signal(context, chat_id, data):
    try:
        chart = await asyncio.to_thread(create_chart, data)
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=InputFile(chart, filename="signal.png"),
            caption=format_text(data),
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(e)
        await context.bot.send_message(chat_id=chat_id, text=format_text(data), parse_mode="HTML")

async def auto_scan(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id or CHAT_ID
    for symbol in DEFAULT_PAIRS:
        res = await asyncio.to_thread(analyze_pair, symbol)
        if res:
            await send_signal(context, chat_id, res)
            await asyncio.sleep(1.3)
        await asyncio.sleep(0.4)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 <b>Smart Binary Bot</b>\n\n"
        "Умеет:\n"
        "• Боковик / Тренд\n"
        "• Качественные ложные и настоящие пробои\n"
        "• Объём на пробое\n"
        "• Структура + свечные паттерны\n"
        "• Скриншот + время экспирации\n\n"
        "/scan — проверить\n"
        "/auto_on — автоскан\n"
        "/auto_off — стоп",
        parse_mode="HTML"
    )

async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔍 Анализ...")
    pairs = DEFAULT_PAIRS
    if context.args:
        s = context.args[0].upper()
        pairs = [s if "/" in s else s + "/USDT"]

    found = 0
    for p in pairs:
        res = await asyncio.to_thread(analyze_pair, p)
        if res:
            found += 1
            await send_signal(context, update.effective_chat.id, res)
            await asyncio.sleep(1.1)
    await msg.edit_text(f"✅ Найдено: {found}" if found else "😴 Нет качественных сигналов")

async def auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for j in context.job_queue.get_jobs_by_name("auto_scan"):
        j.schedule_removal()
    context.job_queue.run_repeating(
        auto_scan,
        interval=SCAN_INTERVAL,
        first=8,
        chat_id=update.effective_chat.id,
        name="auto_scan"
    )
    await update.message.reply_text(f"🟢 Автоскан включён (каждые {SCAN_INTERVAL} сек)")

async def auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for j in context.job_queue.get_jobs_by_name("auto_scan"):
        j.schedule_removal()
    await update.message.reply_text("🔴 Автоскан выключен")

def main():
    if not BOT_TOKEN:
        print("Ошибка: не указан BOT_TOKEN")
        return
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("scan", scan_cmd))
    app.add_handler(CommandHandler("auto_on", auto_on))
    app.add_handler(CommandHandler("auto_off", auto_off))
    print("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()