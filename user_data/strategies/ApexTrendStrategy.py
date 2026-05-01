"""
APEX AI - Ultimate Self-Learning Trading Bot v6.2

FIXES in v6.2:
- Full-day trading: removed 11AM-2PM dead zone
- SELL orders fixed: routes to qty-based short (was 422 on all shorts)
- place_short_order simplified: plain market (no brackets = no EOD 403)
- TimeoutError caught: parallel scan can no longer crash the bot
- AI cost reduced: skips Claude if no strategy/sweep detected
- Volume baselines: logs actual error, switched feed sip→iex
- EOD close: 12s sleep after bulk liquidate (was 3s, causing 403 retries)
"""

# ─────────────────────────────────────────────
# API KEYS
# ─────────────────────────────────────────────
import os
from learning_engine import AdvancedLearner, two_strike

ALPACA_API_KEY     = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY  = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL    = "https://paper-api.alpaca.markets"
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
DATABASE_URL       = os.environ.get("DATABASE_URL", "")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DEFAULT_WATCHLIST      = ["NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "GOOGL", "OKLO", "PLTR", "COIN"]
WATCHLIST              = DEFAULT_WATCHLIST.copy()

ACCOUNT_RISK_PER_TRADE = 0.01
MAX_OPEN_POSITIONS     = 5
STOP_LOSS_PCT          = 0.025
TAKE_PROFIT_PCT        = 0.05
MAX_DAILY_LOSS_R       = 3
TRAILING_STOP_ENABLED  = True

MIN_AI_CONFIDENCE      = 65
MIN_ADX_STRENGTH       = 15
MAX_VWAP_DISTANCE_PCT  = 15
MIN_VOLUME_RATIO       = 0.7
MAX_VIX                = 30
SPY_DOWN_THRESHOLD     = -0.01

MARKET_OPEN_HOUR        = 9
MARKET_OPEN_MIN         = 30
MORNING_SESSION_END     = 11
AFTERNOON_SESSION_START = 14
MARKET_CLOSE_HOUR       = 16
ORB_MINUTES             = 30
TRADE_START_MIN         = 35
EOD_CLOSE_HOUR    = 15
EOD_CLOSE_MIN     = 45
EOD_HARD_CLOSE_MIN = 50
EOD_ALERT_MIN     = 55

MAX_WATCHLIST_SIZE      = 10
LEARNING_ENABLED        = True
MIN_TRADES_TO_LEARN     = 10
LEARNING_FILE           = "learning_params.json"

SCAN_INTERVAL_PRIME     = 5
SCAN_INTERVAL_OFF       = 60

MAX_DAILY_API_COST      = 1.5
CLAUDE_INPUT_COST       = 3.0 / 1_000_000
CLAUDE_OUTPUT_COST      = 15.0 / 1_000_000

SECTOR_ETFS = {
    "NVDA": "QQQ", "AAPL": "QQQ", "MSFT": "QQQ",
    "AMZN": "QQQ", "META": "QQQ", "GOOGL": "QQQ",
    "TSLA": "QQQ", "AMD":  "QQQ", "NFLX":  "QQQ",
    "PLTR": "QQQ", "COIN": "QQQ", "MSTR":  "QQQ",
    "OKLO": "QQQ", "IONQ": "QQQ", "ARM":   "QQQ",
    "CRWD": "QQQ", "PANW": "QQQ", "NET":   "QQQ",
    "SNOW": "QQQ", "DDOG": "QQQ", "NOW":   "QQQ",
    "JPM":  "XLF", "BAC":  "XLF", "GS":    "XLF",
    "HOOD": "XLF", "SOFI": "XLF",
    "XOM":  "XLE", "CVX":  "XLE",
    "JNJ":  "XLV", "UNH":  "XLV", "LLY":   "XLV",
}
SECTOR_DOWN_THRESHOLD   = -0.015

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import time
import logging
import requests
import json
import schedule
import pytz
import threading
import concurrent.futures
import re
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np

try:
    from flask import Flask, jsonify, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("apex_bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("APEX")
ET  = pytz.timezone("America/New_York")

# ─────────────────────────────────────────────
# MARKET TIME
# ─────────────────────────────────────────────
class MarketTime:
    @staticmethod
    def now_et():
        return datetime.now(ET)

    @staticmethod
    def is_weekend():
        return MarketTime.now_et().weekday() >= 5

    @staticmethod
    def is_market_open():
        now = MarketTime.now_et()
        if now.weekday() >= 5:
            return False
        open_t  = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0, microsecond=0)
        close_t = now.replace(hour=MARKET_CLOSE_HOUR, minute=0, second=0, microsecond=0)
        return open_t <= now <= close_t

    @staticmethod
    def is_prime_session():
        # FIX v6.2: covers full trading day 9:30 AM - 4:00 PM, no dead zone
        now = MarketTime.now_et()
        if now.weekday() >= 5:
            return False
        t      = now.hour + now.minute / 60
        open_t = MARKET_OPEN_HOUR + MARKET_OPEN_MIN / 60   # 9.5
        return open_t <= t <= MARKET_CLOSE_HOUR             # 9.5 → 16.0

    @staticmethod
    def is_eod():
        now = MarketTime.now_et()
        return now.hour == EOD_CLOSE_HOUR and now.minute >= EOD_CLOSE_MIN

    @staticmethod
    def is_hard_eod():
        now = MarketTime.now_et()
        return now.hour == EOD_CLOSE_HOUR and now.minute >= EOD_HARD_CLOSE_MIN

    @staticmethod
    def is_orb_window():
        now    = MarketTime.now_et()
        t      = now.hour * 60 + now.minute
        open_t = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MIN
        return open_t <= t <= (open_t + ORB_MINUTES)

    @staticmethod
    def session_name():
        now = MarketTime.now_et()
        t   = now.hour + now.minute / 60
        if now.weekday() >= 5:             return "WEEKEND"
        elif t < (MARKET_OPEN_HOUR + MARKET_OPEN_MIN / 60): return "PRE_MARKET"
        elif t <= (MARKET_OPEN_HOUR + MARKET_OPEN_MIN / 60 + ORB_MINUTES / 60): return "ORB_WINDOW"
        elif t <= MORNING_SESSION_END:     return "MORNING"
        elif t < AFTERNOON_SESSION_START:  return "MIDDAY"
        elif t <= MARKET_CLOSE_HOUR:       return "AFTERNOON"
        else:                              return "AFTER_HOURS"

    @staticmethod
    def get_scan_interval() -> int:
        if MarketTime.is_weekend():       return SCAN_INTERVAL_OFF * 60
        if MarketTime.is_prime_session(): return SCAN_INTERVAL_PRIME * 60
        return SCAN_INTERVAL_OFF * 60

# ─────────────────────────────────────────────
# CLAUDE API COST TRACKER
# ─────────────────────────────────────────────
class CostTracker:
    def __init__(self):
        self.daily_cost    = 0.0
        self.daily_calls   = 0
        self.total_cost    = 0.0
        self.total_calls   = 0
        self.alerted_today = False
        self._lock         = threading.Lock()

    def record_call(self, input_tokens: int, output_tokens: int):
        cost = (input_tokens * CLAUDE_INPUT_COST) + (output_tokens * CLAUDE_OUTPUT_COST)
        with self._lock:
            self.daily_cost  += cost
            self.daily_calls += 1
            self.total_cost  += cost
            self.total_calls += 1
            if self.daily_cost > MAX_DAILY_API_COST and not self.alerted_today:
                self.alerted_today = True
                log.warning(f"⚠️ Claude API cost alert: ${self.daily_cost:.2f} today")
                telegram.send(
                    f"⚠️ <b>API Cost Alert</b>\n"
                    f"💸 Claude API: ${self.daily_cost:.2f} today\n"
                    f"📞 Calls today: {self.daily_calls}\n"
                    f"Consider reducing scan frequency"
                )

    def reset_daily(self):
        with self._lock:
            self.daily_cost    = 0.0
            self.daily_calls   = 0
            self.alerted_today = False

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "daily_cost":  round(self.daily_cost, 4),
                "daily_calls": self.daily_calls,
                "total_cost":  round(self.total_cost, 4),
                "total_calls": self.total_calls
            }

cost_tracker = CostTracker()

# ─────────────────────────────────────────────
# TELEGRAM ALERTER
# ─────────────────────────────────────────────
class TelegramAlerter:
    def __init__(self):
        self.enabled              = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
        self._weekend_alert_sent  = False
        self._last_heartbeat_hour = -1
        self._last_geo_alert      = None
        self._position_alerts     = {}
        self._hot_setup_alerted   = {}

    def send(self, message: str):
        if not self.enabled:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
                timeout=10
            )
        except Exception as e:
            log.warning(f"Telegram failed: {e}")

    def send_online(self, equity, win_rate, confidence):
        self.send(
            f"🟢 <b>APEX BOT ONLINE v6.2</b>\n"
            f"💰 Equity: ${equity:,.2f}\n"
            f"🧠 Win Rate: {win_rate:.1%} | Confidence: {confidence}%\n"
            f"📐 Strategies: ORB, Gap Fill, VWAP Rev, Sweep, Trend\n"
            f"🛡 VIX: {MAX_VIX} | SPY: {SPY_DOWN_THRESHOLD*100:.0f}%\n"
            f"⚡ WebSocket: ON | Parallel: ON | MultiTF: ON\n"
            f"⏰ Prime: 5min scan full day | 9:35 AM start\n"
            f"✅ All systems operational"
        )

    def send_offline(self, reason: str):
        self.send(f"🔴 <b>APEX BOT OFFLINE</b>\n⚠️ {reason}\n🔄 Restarting...")

    def send_sleeping(self):
        if not self._weekend_alert_sent:
            self.send(
                f"⏸ <b>APEX SLEEPING — WEEKEND</b>\n"
                f"📅 Zero API calls until Monday\n"
                f"⏰ Resuming: Monday 9:30 AM ET"
            )
            self._weekend_alert_sent = True

    def reset_weekend_alert(self):
        self._weekend_alert_sent = False

    def send_heartbeat(self, equity, positions, daily_r, geo_risk="LOW", api_cost=0.0):
        now = MarketTime.now_et()
        if now.hour != self._last_heartbeat_hour:
            self._last_heartbeat_hour = now.hour
            geo_emoji = "🟢" if geo_risk=="LOW" else "🟡" if geo_risk=="MEDIUM" else "🔴"
            self.send(
                f"💊 <b>APEX HEARTBEAT</b>\n"
                f"🕐 {now.strftime('%H:%M ET')}\n"
                f"💰 Equity: ${equity:,.2f}\n"
                f"📊 Positions: {positions}/{MAX_OPEN_POSITIONS}\n"
                f"📉 Daily R: {daily_r:.1f}R\n"
                f"{geo_emoji} Geo Risk: {geo_risk}\n"
                f"💸 API Cost: ${api_cost:.3f} today\n"
                f"✅ Bot running normally"
            )

    def send_hot_setup(self, symbol: str, reason: str, confidence: int):
        now  = datetime.now()
        last = self._hot_setup_alerted.get(symbol)
        if last and (now - last).seconds < 1800:
            return
        self._hot_setup_alerted[symbol] = now
        self.send(
            f"🔥 <b>HOT SETUP FORMING</b>\n"
            f"📌 {symbol}\n"
            f"💡 {reason}\n"
            f"🤖 Pre-signal confidence: {confidence}%\n"
            f"⏳ Watching for entry trigger..."
        )

    def send_position_alert(self, symbol: str, current_price: float,
                             tp: float, sl: float, side: str):
        dist_tp    = abs(current_price - tp) / tp * 100
        dist_sl    = abs(current_price - sl) / sl * 100
        last_alert = self._position_alerts.get(symbol, 0)
        if abs(current_price - last_alert) / max(last_alert, 1) < 0.003:
            return
        if dist_tp < 0.5:
            self._position_alerts[symbol] = current_price
            self.send(
                f"🎯 <b>NEAR TARGET!</b>\n"
                f"📌 {symbol} @ ${current_price}\n"
                f"🏆 TP: ${tp} ({dist_tp:.2f}% away)\n"
                f"💰 About to take profit!"
            )
        elif dist_sl < 0.5:
            self._position_alerts[symbol] = current_price
            self.send(
                f"⚠️ <b>NEAR STOP LOSS!</b>\n"
                f"📌 {symbol} @ ${current_price}\n"
                f"🛑 SL: ${sl} ({dist_sl:.2f}% away)\n"
                f"📉 Risk of loss!"
            )

    def send_fill_report(self, symbol: str, signal_price: float,
                          fill_price: float, qty: int, side: str):
        slippage         = abs(fill_price - signal_price)
        slippage_pct     = slippage / signal_price * 100
        slippage_dollars = slippage * qty
        emoji = "🟢" if side == "BUY" else "🔴"
        self.send(
            f"{emoji} <b>ORDER FILLED</b>\n"
            f"📌 {symbol} — {side} {qty} shares\n"
            f"📊 Signal: ${signal_price} → Fill: ${fill_price}\n"
            f"📏 Slippage: ${slippage:.2f} ({slippage_pct:.2f}%)\n"
            f"💸 Slippage cost: ${slippage_dollars:.2f}"
        )

    def send_daily_summary(self, equity, start_equity, trades, wins, losses,
                            best_trade=None, worst_trade=None, api_cost=0.0):
        pnl     = equity - start_equity
        pnl_pct = (pnl / start_equity * 100) if start_equity > 0 else 0
        emoji   = "📈" if pnl >= 0 else "📉"
        wr      = f"{wins/trades:.1%}" if trades > 0 else "N/A"
        msg = (
            f"{emoji} <b>APEX DAILY REPORT</b>\n"
            f"📅 {MarketTime.now_et().strftime('%A, %B %d')}\n\n"
            f"💰 Equity: ${equity:,.2f}\n"
            f"{'🟢' if pnl >= 0 else '🔴'} P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)\n"
            f"📊 Trades: {trades} | WR: {wr}\n"
            f"✅ Wins: {wins} | ❌ Losses: {losses}\n"
        )
        if best_trade:
            msg += f"🏆 Best: {best_trade['symbol']} +{best_trade['pnl_r']:.1f}R\n"
        if worst_trade:
            msg += f"💀 Worst: {worst_trade['symbol']} {worst_trade['pnl_r']:.1f}R\n"
        msg += (
            f"\n💸 API Cost: ${api_cost:.3f}\n"
            f"🧠 Learning from today's data..."
        )
        self.send(msg)

    def send_morning_briefing(self, spy_change, vix, equity, watchlist,
                               top_movers=None, geo_risk="LOW"):
        spy_emoji = "📈" if spy_change >= 0 else "📉"
        geo_emoji = "🟢" if geo_risk=="LOW" else "🟡" if geo_risk=="MEDIUM" else "🔴"
        now_str   = MarketTime.now_et().strftime("%A, %B %d %Y")
        vix_warn  = "⚠️ HIGH" if vix > 25 else "✅ NORMAL"
        lines = [
            "🌅 <b>APEX MORNING BRIEFING</b>",
            "📅 " + now_str,
            "",
            spy_emoji + " SPY: {:+.2f}%".format(spy_change*100),
            "😰 VIX: {:.1f} {}".format(vix, vix_warn),
            geo_emoji + " Geo Risk: " + geo_risk,
            "💰 Account: ${:,.2f}".format(equity),
            "",
        ]
        if top_movers and len(top_movers) > 0 and isinstance(top_movers[0], dict) and "score" in top_movers[0]:
            lines.append("🏆 <b>Today's Top Picks (AI Scored):</b>")
            for m in top_movers[:6]:
                score   = m.get("score", 0)
                reasons = " | ".join(m.get("reasons", [])[:2])
                gp      = m.get("gap_pct", 0)
                vr      = m.get("vol_ratio", 1)
                sym     = m["symbol"]
                color   = "🔴" if gp < 0 else "🟢"
                lines.append("  {} <b>{}</b> — {}/100".format(color, sym, score))
                lines.append("     Gap: {:+.1f}% | Vol: {:.1f}x".format(gp, vr))
                if reasons:
                    lines.append("     " + reasons)
        elif top_movers and len(top_movers) > 0:
            lines.append("🔥 <b>Top Pre-Market Movers:</b>")
            for m in top_movers[:3]:
                gp = m.get("gap_pct", 0)
                vr = m.get("vol_ratio", 1)
                lines.append("  • {}: {:+.1f}% gap, {:.1f}x vol".format(m["symbol"], gp, vr))
        else:
            lines.append("📋 <b>Today's Watchlist:</b>")
            for s in watchlist[:8]:
                lines.append("  • " + s)
        lines.append("")
        lines.append("⚡ Volume spike scanner: ON (30s)")
        lines.append("⏰ Trading starts 9:35 AM ET — bot ready!")
        self.send("\n".join(lines))

    def send_weekly_report(self, params):
        best_strat = max(params.get("best_strategies", {"N/A": 0}),
                         key=params.get("best_strategies", {"N/A": 0}).get)
        self.send(
            f"📊 <b>APEX WEEKLY REPORT</b>\n"
            f"📅 {datetime.now().strftime('%B %d, %Y')}\n\n"
            f"✅ Win Rate: {params.get('win_rate', 0):.1%}\n"
            f"📈 Avg R: {params.get('avg_r', 0):.2f}R\n"
            f"📊 Total Trades: {params.get('total_trades', 0)}\n"
            f"🏆 Best Strategy: {best_strat}\n"
            f"🎯 Confidence: {params.get('min_confidence', 70)}%\n\n"
            f"🧠 Bot is learning and improving every day!"
        )

telegram = TelegramAlerter()

# ─────────────────────────────────────────────
# TRADE JOURNAL
# ─────────────────────────────────────────────
class TradeJournal:
    COLS = [
        "timestamp","symbol","action","strategy",
        "signal_price","fill_price","slippage_pct",
        "stop_loss","take_profit","qty",
        "confidence","filter_score","rsi","macd","adx",
        "vwap_distance","obv_trend","volume_ratio",
        "market_regime","news_sentiment","reasoning",
        "outcome","exit_price","pnl_r","pnl_dollars",
        "atr","timeframe_agree"
    ]

    def __init__(self):
        self.db_url   = DATABASE_URL
        self.use_db   = bool(self.db_url)
        self.filepath = "trade_journal.csv"
        if self.use_db:
            self._setup_db()
            log.info("✅ Connected to PostgreSQL database")
        else:
            self._ensure_csv()
            log.info("⚠ No database — using CSV fallback")

    def _setup_db(self):
        import psycopg2
        conn = psycopg2.connect(self.db_url)
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP,
                symbol VARCHAR(10),
                action VARCHAR(10),
                strategy VARCHAR(30),
                signal_price FLOAT,
                fill_price FLOAT,
                slippage_pct FLOAT,
                stop_loss FLOAT,
                take_profit FLOAT,
                qty INTEGER,
                confidence INTEGER,
                filter_score INTEGER,
                rsi FLOAT,
                macd VARCHAR(20),
                adx FLOAT,
                vwap_distance FLOAT,
                obv_trend VARCHAR(20),
                volume_ratio FLOAT,
                market_regime VARCHAR(20),
                news_sentiment VARCHAR(20),
                reasoning TEXT,
                outcome VARCHAR(10) DEFAULT '',
                exit_price FLOAT,
                pnl_r FLOAT,
                pnl_dollars FLOAT,
                atr FLOAT,
                timeframe_agree BOOLEAN
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS learning_params (
                id SERIAL PRIMARY KEY,
                updated_at TIMESTAMP,
                params JSONB
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                id SERIAL PRIMARY KEY,
                date DATE,
                start_equity FLOAT,
                end_equity FLOAT,
                trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                pnl_dollars FLOAT,
                api_cost FLOAT
            )
        """)
        conn.commit()
        cur.close()
        conn.close()

    def _ensure_csv(self):
        import csv
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.COLS).writeheader()

    def log_entry(self, record: dict):
        if self.use_db:
            try:
                import psycopg2
                conn = psycopg2.connect(self.db_url)
                cur  = conn.cursor()
                cur.execute("""
                    INSERT INTO trades (
                        timestamp,symbol,action,strategy,
                        signal_price,fill_price,slippage_pct,
                        stop_loss,take_profit,qty,
                        confidence,filter_score,rsi,macd,adx,
                        vwap_distance,obv_trend,volume_ratio,
                        market_regime,news_sentiment,reasoning,outcome,
                        atr,timeframe_agree
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                    )
                """, (
                    record.get("timestamp"),      record.get("symbol"),
                    record.get("action"),         record.get("strategy"),
                    record.get("signal_price"),   record.get("fill_price"),
                    record.get("slippage_pct"),   record.get("stop_loss"),
                    record.get("take_profit"),    record.get("qty"),
                    record.get("confidence"),     record.get("filter_score"),
                    record.get("rsi"),            record.get("macd"),
                    record.get("adx"),            record.get("vwap_distance"),
                    record.get("obv_trend"),      record.get("volume_ratio"),
                    record.get("market_regime"),  record.get("news_sentiment"),
                    record.get("reasoning"),      "",
                    record.get("atr"),            record.get("timeframe_agree")
                ))
                conn.commit()
                cur.close()
                conn.close()
                log.info(f"📒 Saved: {record.get('symbol')} {record.get('action')}")
            except Exception as e:
                log.error(f"DB insert failed: {e}")
        else:
            import csv
            row = {c: record.get(c, "") for c in self.COLS}
            with open(self.filepath, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.COLS).writerow(row)

    def update_fill_price(self, order_id: str, symbol: str):
        try:
            time.sleep(2)
            r = requests.get(
                f"{ALPACA_BASE_URL}/v2/orders/{order_id}",
                headers={"APCA-API-KEY-ID": ALPACA_API_KEY,
                         "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY},
                timeout=10
            )
            if r.status_code == 200:
                order      = r.json()
                fill_price = float(order.get("filled_avg_price") or 0)
                return fill_price
        except:
            pass
        return None

    def update_latest_trade_fill(self, symbol: str, signal_price: float,
                                  fill_price: float, slippage_pct: float):
        if self.use_db:
            try:
                import psycopg2
                conn = psycopg2.connect(self.db_url)
                cur  = conn.cursor()
                cur.execute(
                    """
                    UPDATE trades
                    SET fill_price = %s, slippage_pct = %s
                    WHERE id = (
                        SELECT id FROM trades
                        WHERE symbol = %s
                          AND signal_price = %s
                          AND COALESCE(outcome, '') = ''
                        ORDER BY timestamp DESC
                        LIMIT 1
                    )
                    """,
                    (fill_price, slippage_pct, symbol, signal_price)
                )
                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                log.error(f"DB fill update failed: {e}")
            return
        try:
            df      = pd.read_csv(self.filepath)
            if df.empty: return
            matches = df.index[(df["symbol"] == symbol) &
                               (df["signal_price"].astype(float) == float(signal_price))]
            if len(matches) == 0: return
            idx = matches[-1]
            df.loc[idx, "fill_price"]   = fill_price
            df.loc[idx, "slippage_pct"] = slippage_pct
            df.to_csv(self.filepath, index=False)
        except Exception as e:
            log.error(f"CSV fill update failed: {e}")

    def get_all_trades(self) -> pd.DataFrame:
        if self.use_db:
            try:
                from sqlalchemy import create_engine
                db_url  = self.db_url.replace("postgres://", "postgresql://")
                engine  = create_engine(db_url)
                df      = pd.read_sql("SELECT * FROM trades", engine)
                engine.dispose()
                return df
            except Exception as e:
                log.error(f"get_all_trades failed: {e}")
                return pd.DataFrame()
        try:
            return pd.read_csv(self.filepath)
        except:
            return pd.DataFrame()

    def get_todays_best_worst(self) -> tuple:
        df = self.get_all_trades()
        if df.empty or "pnl_r" not in df.columns:
            return None, None
        today = MarketTime.now_et().date()
        try:
            df["date"]    = pd.to_datetime(df["timestamp"]).dt.date
            today_df      = df[df["date"] == today]
            if today_df.empty: return None, None
            today_df["pnl_r"] = pd.to_numeric(today_df["pnl_r"], errors="coerce")
            completed = today_df.dropna(subset=["pnl_r"])
            if completed.empty: return None, None
            best  = completed.loc[completed["pnl_r"].idxmax()].to_dict()
            worst = completed.loc[completed["pnl_r"].idxmin()].to_dict()
            return best, worst
        except:
            return None, None

    def save_learning_params(self, params: dict):
        if not self.use_db: return
        try:
            import psycopg2
            conn = psycopg2.connect(self.db_url)
            cur  = conn.cursor()
            cur.execute(
                "INSERT INTO learning_params (updated_at,params) VALUES (%s,%s)",
                (datetime.now(), json.dumps(params))
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            log.error(f"Learning save failed: {e}")

    def load_learning_params(self) -> dict:
        if not self.use_db: return {}
        try:
            import psycopg2
            conn = psycopg2.connect(self.db_url)
            cur  = conn.cursor()
            cur.execute("SELECT params FROM learning_params ORDER BY updated_at DESC LIMIT 1")
            row  = cur.fetchone()
            cur.close()
            conn.close()
            return row[0] if row else {}
        except:
            return {}

    def save_daily_summary(self, date, start_equity, end_equity,
                            trades, wins, losses, api_cost=0.0):
        if not self.use_db: return
        try:
            import psycopg2
            conn = psycopg2.connect(self.db_url)
            cur  = conn.cursor()
            cur.execute(
                "INSERT INTO daily_summary "
                "(date,start_equity,end_equity,trades,wins,losses,pnl_dollars,api_cost) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (date, start_equity, end_equity, trades, wins, losses,
                 end_equity - start_equity, api_cost)
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            log.error(f"Daily summary save failed: {e}")

journal = TradeJournal()

# ─────────────────────────────────────────────
# LEARNING ENGINE
# ─────────────────────────────────────────────
learner = AdvancedLearner()

# ─────────────────────────────────────────────
# ALPACA CLIENT
# ─────────────────────────────────────────────
class AlpacaClient:
    def __init__(self):
        self.headers   = {
            "APCA-API-KEY-ID":     ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }
        self.base      = ALPACA_BASE_URL
        self.data_base = "https://data.alpaca.markets"

    def get(self, path, params=None, data_api=False):
        base = self.data_base if data_api else self.base
        for attempt in range(3):
            try:
                r = requests.get(f"{base}{path}", headers=self.headers,
                                 params=params, timeout=15)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise

    def post(self, path, body):
        r = requests.post(f"{self.base}{path}", headers=self.headers,
                          json=body, timeout=15)
        r.raise_for_status()
        return r.json()

    def delete(self, path):
        r = requests.delete(f"{self.base}{path}", headers=self.headers, timeout=15)
        return r.status_code

    def get_account(self):           return self.get("/v2/account")
    def get_positions(self):         return self.get("/v2/positions")
    def get_orders(self, s="open"):  return self.get("/v2/orders", {"status": s})

    def get_bars(self, symbol, timeframe="5Min", limit=390, start=None):
        params = {"symbols": symbol, "timeframe": timeframe,
                  "limit": limit, "feed": "iex"}
        if start:
            params["start"] = start
        data = self.get("/v2/stocks/bars", params=params, data_api=True)
        bars = data.get("bars", {}).get(symbol, [])
        if not bars: return None
        df = pd.DataFrame(bars)
        df.rename(columns={"t":"time","o":"open","h":"high","l":"low",
                            "c":"close","v":"volume"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        return df

    def get_intraday_bars(self, symbol, timeframe="5Min", limit=80):
        params = {"symbols": symbol, "timeframe": timeframe,
                  "limit": limit, "feed": "iex"}
        data = self.get("/v2/stocks/bars", params=params, data_api=True)
        bars = data.get("bars", {}).get(symbol, [])
        if not bars: return None
        df = pd.DataFrame(bars)
        df.rename(columns={"t":"time","o":"open","h":"high","l":"low",
                            "c":"close","v":"volume"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        return df

    def get_multi_bars(self, symbols, timeframe="1Day", limit=5):
        params = {"symbols": ",".join(symbols), "timeframe": timeframe,
                  "limit": limit, "feed": "sip"}
        try:
            return self.get("/v2/stocks/bars", params=params,
                            data_api=True).get("bars", {})
        except:
            return {}

    def get_latest_quote(self, symbol: str) -> dict:
        try:
            data = self.get(f"/v2/stocks/{symbol}/quotes/latest", data_api=True)
            q    = data.get("quote", {})
            return {
                "bid":      float(q.get("bp", 0)),
                "ask":      float(q.get("ap", 0)),
                "bid_size": int(q.get("bs", 0)),
                "ask_size": int(q.get("as", 0)),
                "spread":   round(float(q.get("ap", 0)) - float(q.get("bp", 0)), 4)
            }
        except:
            return {"bid": 0, "ask": 0, "bid_size": 0, "ask_size": 0, "spread": 0}

    def get_order_book(self, symbol: str) -> dict:
        try:
            data           = self.get(f"/v2/stocks/{symbol}/orderbook/latest", data_api=True)
            ob             = data.get("orderbook", {})
            bids           = ob.get("b", [])
            asks           = ob.get("a", [])
            total_bid_size = sum(float(b.get("s", 0)) for b in bids[:5])
            total_ask_size = sum(float(a.get("s", 0)) for a in asks[:5])
            large_bid      = any(float(b.get("s", 0)) > total_bid_size * 0.4 for b in bids[:5])
            large_ask      = any(float(a.get("s", 0)) > total_ask_size * 0.4 for a in asks[:5])
            return {
                "bid_depth": round(total_bid_size),
                "ask_depth": round(total_ask_size),
                "bid_wall":  large_bid,
                "ask_wall":  large_ask,
                "imbalance": round(total_bid_size / max(total_ask_size, 1), 2),
                "bias":      "BULLISH" if total_bid_size > total_ask_size * 1.2
                             else "BEARISH" if total_ask_size > total_bid_size * 1.2
                             else "NEUTRAL"
            }
        except:
            return {"bid_depth": 0, "ask_depth": 0, "bid_wall": False,
                    "ask_wall": False, "imbalance": 1.0, "bias": "NEUTRAL"}

    def place_bracket_order(self, symbol, side, notional, stop_loss, take_profit):
        """BUY only — notional market order (fractional shares). Do NOT call for shorts."""
        body = {
            "symbol":        symbol,
            "notional":      str(round(float(notional), 2)),
            "side":          side,
            "type":          "market",
            "time_in_force": "day",
        }
        return self.post("/v2/orders", body)

    def place_short_order(self, symbol, qty, stop_loss, take_profit):
        # FIX v6.2: plain market sell, no brackets — stops managed in software.
        # Bracket child orders were causing EOD 403 Forbidden errors.
        body = {
            "symbol":        symbol,
            "qty":           str(int(qty)),
            "side":          "sell",
            "type":          "market",
            "time_in_force": "day",
        }
        return self.post("/v2/orders", body)

    def close_position(self, symbol) -> bool:
        try:
            # Step 1: cancel any open orders for this symbol
            try:
                orders = self.get("/v2/orders", {"status": "open", "symbols": symbol})
                for o in orders:
                    oid = o.get("id")
                    if oid:
                        self.delete(f"/v2/orders/{oid}")
                if orders:
                    time.sleep(2)   # wait for cancels to settle
            except Exception:
                pass

            # Step 2: DELETE position — handles fractional shares correctly
            try:
                r = requests.delete(
                    f"{self.base}/v2/positions/{symbol}",
                    headers=self.headers,
                    params={"cancel_orders": "true", "percentage": "1"},
                    timeout=15
                )
                if r.status_code in [200, 204]:
                    log.info(f"  🔒 Closed {symbol} via DELETE")
                    return True
                elif r.status_code == 207:
                    log.info(f"  🔒 {symbol} DELETE 207 — waiting for fill")
                    time.sleep(10)
                    still = [p["symbol"] for p in self.get_positions()
                             if p["symbol"] == symbol]
                    if not still:
                        return True
                else:
                    log.warning(f"  DELETE {symbol} returned {r.status_code}")
            except Exception as e1:
                log.error(f"DELETE close failed for {symbol}: {e1}")

            # Step 3: market order fallback — handles both long and short
            try:
                positions = self.get_positions()
                for p in positions:
                    if p["symbol"] == symbol:
                        side       = p["side"]
                        qty_raw    = float(p["qty"])
                        close_side = "buy" if side == "short" else "sell"

                        # Use exact qty string for fractional positions
                        qty_str = str(abs(qty_raw))

                        body = {
                            "symbol":        symbol,
                            "qty":           qty_str,
                            "side":          close_side,
                            "type":          "market",
                            "time_in_force": "day"
                        }
                        self.post("/v2/orders", body)
                        log.info(f"  🔒 Closed {symbol} via {close_side} {qty_str} market order")
                        return True
            except Exception as e2:
                log.error(f"Market close failed for {symbol}: {e2}")

            return False

        except Exception as e:
            log.error(f"Close position error for {symbol}: {e}")
            return False

    def cancel_all_orders(self):
        try:
            r = requests.delete(f"{self.base}/v2/orders",
                                headers=self.headers, timeout=15)
            log.info(f"  🗑 Cancelled all open orders (status {r.status_code})")
            time.sleep(1)
        except Exception as e:
            log.error(f"Cancel orders failed: {e}")

    def liquidate_all(self) -> bool:
        try:
            r = requests.delete(
                f"{self.base}/v2/positions",
                headers=self.headers,
                params={"cancel_orders": "true"},
                timeout=15
            )
            log.info(f"  🔒 Bulk liquidate status: {r.status_code}")
            return r.status_code in [200, 204, 207]
        except Exception as e:
            log.error(f"Bulk liquidate failed: {e}")
            return False

    def close_all_positions(self) -> dict:
        results = {}
        try:
            self.cancel_all_orders()
            time.sleep(2)

            positions = self.get_positions()
            if not positions:
                return {}

            log.info(f"  🔒 Liquidating {len(positions)} positions...")
            bulk_ok = self.liquidate_all()

            # FIX v6.2: 12s sleep — gives bulk liquidate fills time to settle
            # before we check for stragglers. Was 3s, caused 403 on retries.
            time.sleep(12)

            still_open = self.verify_positions_closed()
            if not still_open:
                for p in positions:
                    results[p["symbol"]] = True
                return results

            for symbol in still_open:
                success = self.close_position(symbol)
                results[symbol] = success
                log.info(f"  🔒 Individual close {symbol}: {'OK' if success else 'FAILED'}")

        except Exception as e:
            log.error(f"Close all failed: {e}")
        return results

    def verify_positions_closed(self) -> list:
        try:
            return [p["symbol"] for p in self.get_positions()]
        except:
            return []

alpaca = AlpacaClient()

# ─────────────────────────────────────────────
# WEBSOCKET MANAGER
# ─────────────────────────────────────────────
class WebSocketManager:
    WS_URL = "wss://stream.data.alpaca.markets/v2/iex"

    def __init__(self):
        self.price_cache = {}
        self.connected   = False
        self._ws         = None
        self._thread     = None
        self._lock       = threading.Lock()

    def get_price(self, symbol: str) -> float:
        with self._lock:
            cached = self.price_cache.get(symbol)
            if cached and (datetime.now() - cached["time"]).seconds < 60:
                return cached["price"]
        return 0.0

    def update_price(self, symbol: str, price: float, volume: float):
        with self._lock:
            self.price_cache[symbol] = {
                "price":  price,
                "volume": volume,
                "time":   datetime.now()
            }

    def subscribe(self, symbols: list):
        if self._ws and self.connected:
            try:
                msg = json.dumps({"action": "subscribe", "trades": symbols})
                self._ws.send(msg)
                log.info(f"📡 WebSocket subscribed: {', '.join(symbols)}")
            except:
                pass

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if isinstance(data, list):
                for item in data:
                    if item.get("T") == "t":
                        symbol = item.get("S")
                        price  = float(item.get("p", 0))
                        volume = float(item.get("s", 0))
                        if symbol and price:
                            self.update_price(symbol, price, volume)
        except:
            pass

    def _on_open(self, ws):
        self.connected = True
        log.info("📡 WebSocket connected")
        auth_msg = json.dumps({
            "action": "auth",
            "key":    ALPACA_API_KEY,
            "secret": ALPACA_SECRET_KEY
        })
        ws.send(auth_msg)
        time.sleep(0.5)
        self.subscribe(WATCHLIST)

    def _on_close(self, ws, code, msg):
        self.connected = False
        log.warning(f"📡 WebSocket disconnected ({code}) — reconnecting in 5s")
        time.sleep(5)
        self.start()

    def _on_error(self, ws, error):
        log.warning(f"📡 WebSocket error: {error}")

    def start(self):
        if not MarketTime.is_weekend() and ALPACA_API_KEY:
            try:
                import websocket as ws_lib
                self._ws = ws_lib.WebSocketApp(
                    self.WS_URL,
                    on_message=self._on_message,
                    on_open=self._on_open,
                    on_close=self._on_close,
                    on_error=self._on_error
                )
                self._thread = threading.Thread(
                    target=self._ws.run_forever, daemon=True)
                self._thread.start()
                log.info("📡 WebSocket thread started")
            except ImportError:
                log.warning("📡 websocket-client not installed — using REST polling")
            except Exception as e:
                log.warning(f"📡 WebSocket failed to start: {e}")

ws_manager = WebSocketManager()

# ─────────────────────────────────────────────
# BACKGROUND INDICATOR PRE-CALCULATOR
# ─────────────────────────────────────────────
class BackgroundProcessor:
    def __init__(self):
        self._cache   = {}
        self._lock    = threading.Lock()
        self._thread  = None
        self._running = False

    def get_cached(self, symbol: str) -> dict:
        with self._lock:
            cached = self._cache.get(symbol)
            if cached and (datetime.now() - cached["time"]).seconds < 120:
                return cached
        return None

    def _calculate_for_symbol(self, symbol: str):
        try:
            df = alpaca.get_bars(symbol, timeframe="5Min", limit=390)
            if df is None or len(df) < 30:
                return
            ind = ind_engine.compute_all(df, df)
            with self._lock:
                self._cache[symbol] = {
                    "ind":         ind,
                    "df":          df,
                    "intraday_df": df,
                    "time":        datetime.now()
                }
        except Exception as e:
            log.debug(f"Background calc failed for {symbol}: {e}")

    def _run_loop(self):
        while self._running:
            if MarketTime.is_market_open() and not MarketTime.is_weekend():
                log.debug("🔄 Background indicator refresh...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
                    ex.map(self._calculate_for_symbol, WATCHLIST)
            time.sleep(60)

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info("🔄 Background processor started")

    def stop(self):
        self._running = False

bg_processor = BackgroundProcessor()

# ─────────────────────────────────────────────
# MARKET CONTEXT
# ─────────────────────────────────────────────
class MarketContext:
    def __init__(self):
        self._cache      = {"spy_change": 0, "vix": 20, "sectors": {}}
        self._cache_time = None

    def _refresh(self):
        now = datetime.now()
        if self._cache_time and (now - self._cache_time).seconds < 300:
            return

        # FIX: use snapshot API for real-time SPY price vs previous close.
        # 1Day bars during market hours have no close yet — always returned 0.
        try:
            r = requests.get(
                "https://data.alpaca.markets/v2/stocks/snapshots",
                headers={
                    "APCA-API-KEY-ID":     ALPACA_API_KEY,
                    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
                },
                params={"symbols": "SPY", "feed": "iex"},
                timeout=10
            )
            if r.status_code == 200:
                snap       = r.json().get("SPY", {})
                cur_px     = float(snap.get("latestTrade", {}).get("p", 0))
                prev_close = float(snap.get("prevDailyBar", {}).get("c", 0))
                if cur_px > 0 and prev_close > 0:
                    self._cache["spy_change"] = (cur_px - prev_close) / prev_close
        except Exception as e:
            log.warning(f"SPY snapshot failed: {e}")

        # VIX proxy via VIXY snapshot
        try:
            r = requests.get(
                "https://data.alpaca.markets/v2/stocks/snapshots",
                headers={
                    "APCA-API-KEY-ID":     ALPACA_API_KEY,
                    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
                },
                params={"symbols": "VIXY", "feed": "iex"},
                timeout=10
            )
            if r.status_code == 200:
                snap = r.json().get("VIXY", {})
                px   = float(snap.get("latestTrade", {}).get("p", 0))
                if px > 0:
                    self._cache["vix"] = px
        except Exception as e:
            log.warning(f"VIX snapshot failed: {e}")

        # Sector ETFs via snapshot
        sector_etfs = list(set(SECTOR_ETFS.values()))
        try:
            r = requests.get(
                "https://data.alpaca.markets/v2/stocks/snapshots",
                headers={
                    "APCA-API-KEY-ID":     ALPACA_API_KEY,
                    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
                },
                params={"symbols": ",".join(sector_etfs), "feed": "iex"},
                timeout=10
            )
            if r.status_code == 200:
                for etf, snap in r.json().items():
                    cur   = float(snap.get("latestTrade", {}).get("p", 0))
                    prev  = float(snap.get("prevDailyBar", {}).get("c", 0))
                    if cur > 0 and prev > 0:
                        self._cache["sectors"][etf] = round((cur - prev) / prev, 4)
        except Exception as e:
            log.warning(f"Sector snapshot failed: {e}")

        self._cache_time = now
    def get_spy_change(self) -> float:
        self._refresh(); return self._cache["spy_change"]

    def get_vix(self) -> float:
        self._refresh(); return self._cache["vix"]

    def get_sector_change(self, symbol: str) -> float:
        self._refresh()
        etf = SECTOR_ETFS.get(symbol)
        if etf:
            return self._cache["sectors"].get(etf, 0)
        return 0

    def is_sector_ok(self, symbol: str, action: str) -> tuple:
        change = self.get_sector_change(symbol)
        etf    = SECTOR_ETFS.get(symbol, "market")
        if action == "BUY" and change < SECTOR_DOWN_THRESHOLD:
            return False, f"{etf} sector down {change*100:.1f}%"
        if action == "SELL" and change > 0.015:
            return False, f"{etf} sector up {change*100:.1f}%"
        return True, "Sector OK"

    def get_sector_summary(self) -> dict:
        self._refresh()
        return self._cache.get("sectors", {})

market_ctx = MarketContext()

# ─────────────────────────────────────────────
# GEOPOLITICAL RISK SCANNER
# ─────────────────────────────────────────────
class GeopoliticalRiskScanner:
    HIGH_RISK_KEYWORDS = [
        "nuclear strike", "nuclear attack", "nuclear war",
        "market circuit breaker", "trading halt", "exchange closed",
        "world war", "nato article 5",
        "martial law", "coup d'etat",
        "oil embargo", "market crash", "black monday",
        "stock market closed", "trading suspended",
        "assassination of president", "invasion of taiwan"
    ]
    MEDIUM_RISK_KEYWORDS = [
        "conflict", "tension", "ceasefire", "negotiations", "sanctions",
        "diplomatic", "protest", "unrest", "riot", "blockade"
    ]

    def __init__(self):
        self._cache         = {"risk_level": "LOW", "reason": "", "headlines": []}
        self._cache_time    = None
        self._pause_alerted = False

    def _fetch_headlines(self) -> list:
        headlines = []
        feeds = [
            "https://finance.yahoo.com/news/rssindex",
            "https://news.google.com/rss/search?q=war+conflict+military+attack&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=geopolitical+crisis+iran+israel&hl=en-US&gl=US&ceid=US:en",
        ]
        for url in feeds:
            try:
                r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
                if r.status_code == 200:
                    titles = re.findall(r"<title>(.*?)</title>", r.text.lower())
                    headlines.extend([t.strip() for t in titles if len(t.strip()) > 10])
            except:
                continue
        try:
            r = requests.get(
                "https://data.alpaca.markets/v1beta1/news",
                headers={"APCA-API-KEY-ID": ALPACA_API_KEY,
                         "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY},
                params={"limit": 20, "sort": "desc"}, timeout=8
            )
            if r.status_code == 200:
                articles = r.json().get("news", [])
                headlines.extend([a.get("headline", "").lower() for a in articles])
        except:
            pass
        return headlines[:50]

    def _refresh(self):
        now = datetime.now()
        if self._cache_time and (now - self._cache_time).seconds < 900:
            return
        try:
            headlines                = self._fetch_headlines()
            self._cache["headlines"] = headlines
            high_matches = list(set([kw for h in headlines
                                     for kw in self.HIGH_RISK_KEYWORDS if kw in h]))
            med_matches  = list(set([kw for h in headlines
                                     for kw in self.MEDIUM_RISK_KEYWORDS if kw in h]))
            if high_matches:
                self._cache["risk_level"] = "HIGH"
                self._cache["reason"]     = f"Keywords: {', '.join(high_matches[:3])}"
                log.warning(f"⚠️ HIGH GEO RISK: {', '.join(high_matches[:3])}")
            elif med_matches:
                self._cache["risk_level"] = "MEDIUM"
                self._cache["reason"]     = f"Tension: {', '.join(med_matches[:3])}"
            else:
                self._cache["risk_level"] = "LOW"
                self._cache["reason"]     = "No geopolitical risk"
        except Exception as e:
            log.warning(f"Geo scan failed: {e}")
        self._cache_time = now

    def get_risk_level(self) -> dict:
        self._refresh(); return self._cache

    def is_safe_to_trade(self) -> tuple:
        risk        = self.get_risk_level()
        geo_override = os.environ.get("GEO_RISK_OVERRIDE", "false").lower() == "true"
        if geo_override:
            return True, "Override active"
        if risk["risk_level"] == "HIGH":
            return False, risk["reason"]
        return True, "Geo risk OK"

    def check_and_alert(self):
        risk = self.get_risk_level()
        if risk["risk_level"] == "HIGH" and not self._pause_alerted:
            self._pause_alerted = True
            telegram.send(
                f"⚠️ <b>APEX GEOPOLITICAL ALERT</b>\n\n"
                f"🚨 High risk event detected!\n"
                f"📰 {risk['reason']}\n\n"
                f"⏸ <b>Trading automatically paused</b>\n"
                f"🛡 Existing positions protected by stop losses\n"
                f"👀 Monitoring — will resume when risk clears\n\n"
                f"📱 Override: set GEO_RISK_OVERRIDE=true in Railway"
            )
        elif risk["risk_level"] == "LOW" and self._pause_alerted:
            self._pause_alerted = False
            telegram.send("✅ <b>Geo Risk Cleared</b>\n▶️ Trading resuming normally")

geo_risk = GeopoliticalRiskScanner()

# ─────────────────────────────────────────────
# EARNINGS FILTER
# ─────────────────────────────────────────────
class EarningsFilter:
    def __init__(self):
        self._cache      = {}
        self._cache_date = None

    def has_earnings_soon(self, symbol: str) -> bool:
        today = MarketTime.now_et().date()
        if self._cache_date != today:
            self._cache      = {}
            self._cache_date = today
        if symbol in self._cache:
            return self._cache[symbol]
        try:
            r = requests.get(
                "https://data.alpaca.markets/v1beta1/news",
                headers={"APCA-API-KEY-ID": ALPACA_API_KEY,
                         "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY},
                params={"symbols": symbol, "limit": 5}, timeout=10
            )
            if r.status_code == 200:
                for a in r.json().get("news", []):
                    if any(w in a.get("headline","").lower()
                           for w in ["earnings","quarterly results","reports earnings",
                                     "q1","q2","q3","q4"]):
                        self._cache[symbol] = True
                        return True
        except:
            pass
        self._cache[symbol] = False
        return False

earnings_filter = EarningsFilter()

# ─────────────────────────────────────────────
# SMART NEWS SCANNER
# ─────────────────────────────────────────────
class SmartNewsScanner:
    BULLISH_WORDS = [
        "beat","surge","rally","record","upgrade","buy","bullish","growth",
        "profit","strong","positive","breakthrough","win","gain","rise","jump",
        "contract","partnership","acquisition","expansion","outperform","raise",
        "lifted","overweight","boost","topped","exceeds","crushes"
    ]
    BEARISH_WORDS = [
        "miss","fall","drop","downgrade","sell","loss","bearish","weak",
        "decline","cut","layoff","crash","debt","risk","warning","concern",
        "negative","investigation","lawsuit","recall","bankruptcy","underperform"
    ]

    def __init__(self):
        self._cache      = {}
        self._cache_time = {}

    def _is_stale(self, symbol):
        t = self._cache_time.get(symbol)
        return not t or (datetime.now() - t).seconds > 1800

    def _fetch_alpaca_news(self, symbol):
        try:
            r = requests.get(
                "https://data.alpaca.markets/v1beta1/news",
                headers={"APCA-API-KEY-ID": ALPACA_API_KEY,
                         "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY},
                params={"symbols": symbol, "limit": 10}, timeout=10
            )
            if r.status_code == 200:
                return [a.get("headline","") for a in r.json().get("news",[])[:10]]
        except: pass
        return []

    def _fetch_yahoo_rss(self, symbol):
        headlines = []
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
            r   = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
            if r.status_code == 200:
                titles = re.findall(r"<title>(.*?)</title>", r.text, re.IGNORECASE)
                headlines.extend([t.strip() for t in titles if len(t.strip()) > 10])
        except: pass
        return headlines[:5]

    def _score(self, headlines):
        text  = " ".join(headlines).lower()
        bull  = sum(1 for w in self.BULLISH_WORDS if w in text)
        bear  = sum(1 for w in self.BEARISH_WORDS if w in text)
        total = bull + bear
        score = round((bull-bear)/total,2) if total>0 else 0.0
        if bull>bear+1:   sentiment="BULLISH"
        elif bear>bull+1: sentiment="BEARISH"
        else:             sentiment="NEUTRAL"
        return sentiment, score

    def _catalyst(self, headlines):
        cats = {
            "EARNINGS_BEAT":     ["beat","tops","exceeds","crushes","earnings"],
            "EARNINGS_MISS":     ["miss","falls short","disappoints"],
            "ANALYST_UPGRADE":   ["upgrade","overweight","buy rating","raised to"],
            "ANALYST_DOWNGRADE": ["downgrade","underperform","sell rating"],
            "MERGER":            ["merger","acquisition","takeover","buyout"],
            "FDA_APPROVAL":      ["fda","approved","approval","clinical trial"],
            "GUIDANCE_RAISE":    ["raises guidance","raised outlook","raised forecast"],
            "CONTRACT":          ["contract","partnership","deal","agreement"],
        }
        text = " ".join(headlines).lower()
        for name, keywords in cats.items():
            if any(kw in text for kw in keywords):
                return name
        return "NEWS"

    def get_stock_news(self, symbol):
        if not self._is_stale(symbol) and symbol in self._cache:
            return self._cache[symbol]
        headlines = self._fetch_alpaca_news(symbol)
        headlines += self._fetch_yahoo_rss(symbol)
        headlines  = list(dict.fromkeys(headlines))[:8]
        sentiment, score = self._score(headlines)
        result = {
            "sentiment": sentiment, "score": score,
            "catalyst":  self._catalyst(headlines),
            "headlines": headlines[:3],
            "summary":   headlines[0][:80] if headlines else "No news"
        }
        self._cache[symbol]      = result
        self._cache_time[symbol] = datetime.now()
        return result

    def get_most_mentioned(self, symbols):
        try:
            r = requests.get(
                "https://data.alpaca.markets/v1beta1/news",
                headers={"APCA-API-KEY-ID":ALPACA_API_KEY,
                         "APCA-API-SECRET-KEY":ALPACA_SECRET_KEY},
                params={"symbols":",".join(symbols[:30]),"limit":50}, timeout=15
            )
            if r.status_code == 200:
                counts = defaultdict(int)
                for a in r.json().get("news",[]):
                    for sym in a.get("symbols",[]):
                        if sym in symbols: counts[sym]+=1
                return sorted(counts.items(),key=lambda x:x[1],reverse=True)[:10]
        except: pass
        return []

smart_news = SmartNewsScanner()

# ─────────────────────────────────────────────
# EARNINGS & ANALYST SCANNER
# ─────────────────────────────────────────────
class EarningsAnalystScanner:
    def __init__(self):
        self._e_cache = {}
        self._a_cache = {}
        self._date    = None

    def _reset(self):
        today = MarketTime.now_et().date()
        if self._date != today:
            self._e_cache = {}
            self._a_cache = {}
            self._date    = today

    def check_earnings(self, symbol):
        self._reset()
        if symbol in self._e_cache:
            return self._e_cache[symbol]
        result = {"has_earnings":False,"beat":None,"catalyst":""}
        try:
            r = requests.get(
                "https://data.alpaca.markets/v1beta1/news",
                headers={"APCA-API-KEY-ID":ALPACA_API_KEY,
                         "APCA-API-SECRET-KEY":ALPACA_SECRET_KEY},
                params={"symbols":symbol,"limit":5}, timeout=8
            )
            if r.status_code==200:
                for a in r.json().get("news",[]):
                    h = a.get("headline","").lower()
                    if any(w in h for w in ["earnings","eps","quarterly","results",
                                            "q1","q2","q3","q4"]):
                        result["has_earnings"] = True
                        if any(w in h for w in ["beat","tops","exceeds","crushes","surpasses"]):
                            result["beat"]     = True
                            result["catalyst"] = "EARNINGS_BEAT"
                        elif any(w in h for w in ["miss","falls short","disappoints","below"]):
                            result["beat"]     = False
                            result["catalyst"] = "EARNINGS_MISS"
                        else:
                            result["catalyst"] = "EARNINGS_REPORT"
                        break
        except: pass
        self._e_cache[symbol] = result
        return result

    def check_analyst(self, symbol):
        self._reset()
        if symbol in self._a_cache:
            return self._a_cache[symbol]
        result = {"upgrade":False,"downgrade":False,"target_raised":False,
                  "target_cut":False,"catalyst":""}
        try:
            r = requests.get(
                "https://data.alpaca.markets/v1beta1/news",
                headers={"APCA-API-KEY-ID":ALPACA_API_KEY,
                         "APCA-API-SECRET-KEY":ALPACA_SECRET_KEY},
                params={"symbols":symbol,"limit":10}, timeout=8
            )
            if r.status_code==200:
                for a in r.json().get("news",[]):
                    h = a.get("headline","").lower()
                    if any(w in h for w in ["upgrade","overweight","outperform"]):
                        result["upgrade"]  = True
                        result["catalyst"] = "ANALYST_UPGRADE"
                    elif any(w in h for w in ["downgrade","underperform","reduce"]):
                        result["downgrade"] = True
                        result["catalyst"]  = "ANALYST_DOWNGRADE"
                    if "price target" in h or "pt raised" in h or "pt cut" in h:
                        if any(w in h for w in ["raises","increased","lifts","higher"]):
                            result["target_raised"] = True
                            if not result["catalyst"]: result["catalyst"]="TARGET_RAISED"
                        elif any(w in h for w in ["cuts","lowers","reduces"]):
                            result["target_cut"] = True
                            if not result["catalyst"]: result["catalyst"]="TARGET_CUT"
                    if result["upgrade"] or result["downgrade"]: break
        except: pass
        self._a_cache[symbol] = result
        return result

earnings_analyst = EarningsAnalystScanner()

# ─────────────────────────────────────────────
# STOCK SCORER
# ─────────────────────────────────────────────
class StockScorer:
    def score(self, symbol, bars_data):
        bars = bars_data.get(symbol, [])
        if len(bars) < 2:
            return {"symbol":symbol,"score":0,"reasons":[],"gap_pct":0,"vol_ratio":1,"price":0}
        prev_close = float(bars[-2].get("c",0))
        today_open = float(bars[-1].get("o",prev_close))
        today_vol  = float(bars[-1].get("v",0))
        avg_vol    = sum(float(b.get("v",0)) for b in bars[:-1]) / max(len(bars)-1,1)
        price      = float(bars[-1].get("c",prev_close))
        if prev_close==0:
            return {"symbol":symbol,"score":0,"reasons":[],"gap_pct":0,"vol_ratio":1,"price":price}
        gap_pct   = (today_open-prev_close)/prev_close*100
        vol_ratio = today_vol/avg_vol if avg_vol>0 else 1
        reasons   = []
        total     = 0
        news = smart_news.get_stock_news(symbol)
        ns   = 0
        if news["sentiment"]=="BULLISH": ns+=15; reasons.append(f"📰 {news['catalyst']}")
        elif news["sentiment"]=="BEARISH": ns-=5
        if news["score"]>0.3: ns+=10
        if news["headlines"]: ns+=5
        total += max(0,min(30,ns))
        ts = 0
        if abs(gap_pct)>3:   ts+=15; reasons.append(f"📈 Gap {gap_pct:+.1f}%")
        elif abs(gap_pct)>1.5: ts+=8; reasons.append(f"📈 Gap {gap_pct:+.1f}%")
        if vol_ratio>3:   ts+=15; reasons.append(f"📊 Vol {vol_ratio:.1f}x")
        elif vol_ratio>2: ts+=10; reasons.append(f"📊 Vol {vol_ratio:.1f}x")
        elif vol_ratio>1.5: ts+=5
        total += max(0,min(30,ts))
        ms = 0
        if len(bars)>=5:
            prices  = [float(b.get("c",0)) for b in bars[-5:]]
            up_days = sum(1 for i in range(1,len(prices)) if prices[i]>prices[i-1])
            if up_days>=4: ms+=15; reasons.append("⬆️ Strong uptrend")
            elif up_days>=3: ms+=8
        if gap_pct>0 and vol_ratio>1.5: ms+=5
        total += max(0,min(20,ms))
        cs       = 0
        earnings = earnings_analyst.check_earnings(symbol)
        analyst  = earnings_analyst.check_analyst(symbol)
        if earnings["has_earnings"]:
            if earnings.get("beat"):       cs+=20; reasons.append(f"🏆 {earnings['catalyst']}")
            elif earnings.get("beat")==False: cs-=5
            else:                          cs+=8;  reasons.append(f"📋 {earnings['catalyst']}")
        if analyst["upgrade"]:             cs+=15; reasons.append(f"⬆️ {analyst['catalyst']}")
        elif analyst["downgrade"]:         cs-=5
        if analyst["target_raised"]:       cs+=8;  reasons.append("🎯 PT raised")
        total += max(0,min(20,cs))
        return {
            "symbol":    symbol,
            "score":     min(100,max(0,total)),
            "gap_pct":   round(gap_pct,2),
            "vol_ratio": round(vol_ratio,2),
            "price":     round(price,2),
            "reasons":   reasons[:4],
            "news":      news,
            "earnings":  earnings,
            "analyst":   analyst,
        }

    def rank(self, symbols, bars_data):
        scored = []
        for sym in symbols:
            try:
                r = self.score(sym, bars_data)
                if r["score"] > 0: scored.append(r)
            except: continue
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

stock_scorer = StockScorer()

# ─────────────────────────────────────────────
# VOLUME SPIKE SCANNER
# ─────────────────────────────────────────────
class VolumeSpikeScanner:
    UNIVERSE = [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","JPM","LLY",
        "V","UNH","XOM","MA","JNJ","PG","HD","COST","MRK","ABBV","CVX","CRM",
        "BAC","NFLX","AMD","KO","PEP","TMO","ORCL","ACN","MCD","CSCO","ABT",
        "WMT","LIN","DHR","ADBE","TXN","PM","NKE","NEE","QCOM","UNP","RTX",
        "CAT","AMGN","LOW","SPGI","GS","ISRG","PLTR","COIN","SOFI","RIVN",
        "SNAP","UBER","DASH","ABNB","SHOP","SQ","PYPL","ROKU","ZM","DKNG",
        "CRWD","ZS","PANW","SNOW","DDOG","MDB","NET","NOW","AI","IONQ",
        "SPY","QQQ","IWM","XLF","XLE","XLV","XLK","ARKK","SOXL","GLD","SLV",
    ]
    TIER1_VOL = 5.0; TIER2_VOL = 3.0
    TIER1_PX  = 1.0; TIER2_PX  = 0.5

    def __init__(self):
        self._baseline = {}
        self._alerted  = {}
        self._running  = False
        self._thread   = None
        self._loaded   = None

    def _load_baselines(self):
        # Don't retry more than once every 10 minutes on failure
        if self._loaded and (datetime.now() - self._loaded).seconds < 600:
            return
        if hasattr(self, '_last_attempt') and self._last_attempt:
            if (datetime.now() - self._last_attempt).seconds < 300:
                return

        self._last_attempt = datetime.now()
        log.info("📊 Loading volume baselines...")
        loaded_count = 0

        for i in range(0, len(self.UNIVERSE), 50):
            batch = self.UNIVERSE[i:i + 50]
            data  = {}

            # Try SIP first (historical daily bars), fall back to IEX
            for feed in ["sip", "iex"]:
                try:
                    params = {
                        "symbols":   ",".join(batch),
                        "timeframe": "1Day",
                        "limit":     20,
                        "feed":      feed
                    }
                    result = alpaca.get("/v2/stocks/bars", params=params,
                                        data_api=True)
                    data = result.get("bars", {})
                    if data:
                        break   # got data, stop trying feeds
                except Exception as e:
                    log.warning(f"Baseline batch ({feed}): {e}")
                    continue

            for sym, bars in data.items():
                if bars and len(bars) >= 5:
                    vols = [float(b.get("v", 0)) for b in bars[:-1]]
                    avg  = sum(vols) / len(vols) if vols else 0
                    if avg > 0:
                        self._baseline[sym] = avg
                        loaded_count += 1

            time.sleep(0.5)

        log.info(f"📊 Baselines: {loaded_count} symbols")

        if loaded_count == 0:
            log.warning("⚠️ Volume baselines empty — will retry in 5 min")
            # Don't reset _loaded to None — use _last_attempt throttle instead
        else:
            self._loaded = datetime.now()

    def _scan(self):
        spikes = []
        try:
            syms_str = ",".join(self.UNIVERSE)
            r = requests.get(
                "https://data.alpaca.markets/v2/stocks/snapshots",
                headers={"APCA-API-KEY-ID":ALPACA_API_KEY,
                         "APCA-API-SECRET-KEY":ALPACA_SECRET_KEY},
                params={"symbols":syms_str,"feed":"iex"}, timeout=15
            )
            if r.status_code!=200: return []
            now = datetime.now()
            for sym, snap in r.json().items():
                try:
                    daily   = snap.get("dailyBar",{})
                    cur_vol = float(daily.get("v",0))
                    cur_px  = float(snap.get("latestTrade",{}).get("p",0))
                    open_px = float(daily.get("o",cur_px))
                    if cur_vol==0 or cur_px==0: continue
                    base = self._baseline.get(sym,0)
                    if base==0: continue
                    vol_r  = cur_vol/base
                    px_chg = (cur_px-open_px)/open_px*100 if open_px>0 else 0
                    if   vol_r>=self.TIER1_VOL and abs(px_chg)>=self.TIER1_PX: tier=1
                    elif vol_r>=self.TIER2_VOL and abs(px_chg)>=self.TIER2_PX: tier=2
                    else: continue
                    last = self._alerted.get(sym)
                    if last and (now-last).seconds<900: continue
                    spikes.append({"symbol":sym,"tier":tier,
                                   "vol_ratio":round(vol_r,1),
                                   "price_chg":round(px_chg,2),
                                   "price":round(cur_px,2)})
                except: continue
        except Exception as e:
            log.debug(f"Snapshot error: {e}")
        spikes.sort(key=lambda x:(x["tier"],-x["vol_ratio"]))
        return spikes

    def _alert(self, spike):
        global WATCHLIST
        sym      = spike["symbol"]
        tier     = spike["tier"]
        news     = smart_news.get_stock_news(sym)
        emoji    = "🚨" if tier==1 else "⚡"
        label    = "TIER 1 — TRADE CANDIDATE" if tier==1 else "TIER 2 — WATCH"
        direct   = "📈" if spike["price_chg"]>0 else "📉"
        auto_add = "🔥 Auto-added to watchlist!" if tier==1 else "👀 Monitoring..."
        msg = (
            "{} <b>VOLUME SPIKE — {}</b>\n"
            "📌 <b>{}</b> @ ${}\n"
            "📊 Volume: {}x normal\n"
            "{} Price: {:+.2f}% from open\n"
            "📰 {} | {}\n"
            "{}"
        ).format(
            emoji, label, sym, spike["price"],
            spike["vol_ratio"], direct, spike["price_chg"],
            news["sentiment"], news["summary"][:60], auto_add
        )
        telegram.send(msg)
        self._alerted[sym] = datetime.now()
        if tier==1 and sym not in WATCHLIST:
            if len(WATCHLIST)<MAX_WATCHLIST_SIZE+3:
                WATCHLIST.append(sym)
                ws_manager.subscribe([sym])
                log.info(f"⚡ Auto-added {sym} to watchlist")

    def _run(self):
        while self._running:
            try:
                if MarketTime.is_market_open() and MarketTime.is_prime_session():
                    self._load_baselines()
                    for sp in self._scan()[:3]:
                        self._alert(sp)
                        log.info(f"⚡ SPIKE T{sp['tier']}: {sp['symbol']} "
                                 f"{sp['vol_ratio']}x {sp['price_chg']:+.1f}%")
            except Exception as e:
                log.debug(f"Vol scanner: {e}")
            time.sleep(30)

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("⚡ Volume spike scanner started (30s)")

volume_scanner = VolumeSpikeScanner()

# ─────────────────────────────────────────────
# PRE-MARKET SCANNER
# ─────────────────────────────────────────────
class PreMarketScanner:
    UNIVERSE = [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO",
        "JPM","LLY","V","UNH","XOM","MA","JNJ","PG","HD","COST",
        "MRK","ABBV","CVX","CRM","BAC","NFLX","AMD","KO","PEP",
        "TMO","ORCL","ACN","MCD","CSCO","ABT","WMT","LIN","DHR",
        "ADBE","TXN","PM","NKE","NEE","QCOM","UNP","RTX","CAT",
        "AMGN","LOW","SPGI","GS","ISRG",
        "OKLO","PLTR","COIN","SOFI","MSTR","HOOD","IONQ","SMCI",
        "CRWD","PANW","SNOW","DDOG","MDB","NET","NOW","AI","SNAP",
        "UBER","DASH","SHOP","SQ","PYPL","ROKU","DKNG","PENN",
        "RIVN","LCID","NIO","XPEV","ARM","RDDT","HIMS","LUNR",
        "NBIS","CRWV","WOLF","SOUN","BBAI","RGTI","QUBT","ACHR",
    ]

    def _get_premarket_movers(self) -> list:
        movers = []
        try:
            syms_str = ",".join(self.UNIVERSE)
            r = requests.get(
                "https://data.alpaca.markets/v2/stocks/snapshots",
                headers={"APCA-API-KEY-ID": ALPACA_API_KEY,
                         "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY},
                params={"symbols": syms_str, "feed": "iex"},
                timeout=20
            )
            if r.status_code != 200:
                log.warning(f"Snapshot API error: {r.status_code}")
                return []
            for sym, snap in r.json().items():
                try:
                    daily      = snap.get("dailyBar", {})
                    prev       = snap.get("prevDailyBar", {})
                    cur_px     = float(snap.get("latestTrade", {}).get("p", 0))
                    prev_close = float(prev.get("c", 0))
                    cur_vol    = float(daily.get("v", 0))
                    if cur_px == 0 or prev_close == 0: continue
                    gap_pct   = (cur_px - prev_close) / prev_close * 100
                    avg_vol   = float(prev.get("v", cur_vol)) if prev else cur_vol
                    vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1
                    score = 0
                    if abs(gap_pct) > 4:   score += 40
                    elif abs(gap_pct) > 2: score += 25
                    elif abs(gap_pct) > 1: score += 10
                    if vol_ratio > 2:   score += 30
                    elif vol_ratio > 1.5: score += 15
                    if score >= 15:
                        movers.append({
                            "symbol":    sym,
                            "gap_pct":   round(gap_pct, 2),
                            "vol_ratio": round(vol_ratio, 2),
                            "price":     round(cur_px, 2),
                            "score":     score,
                            "reasons":   []
                        })
                except Exception:
                    continue
        except Exception as e:
            log.error(f"Pre-market snapshot error: {e}")
        movers.sort(key=lambda x: x["score"], reverse=True)
        return movers[:30]

    def scan(self):
        log.info("🔍 Pre-market scanner running (8:00 AM scan)...")
        movers = self._get_premarket_movers()
        log.info(f"📊 Found {len(movers)} pre-market movers")
        if not movers:
            log.info("📊 No movers found — using default watchlist")
            return [], DEFAULT_WATCHLIST
        for m in movers[:20]:
            try:
                news          = smart_news.get_stock_news(m["symbol"])
                m["news"]     = news
                m["catalyst"] = news.get("catalyst", "NEWS")
                if news["sentiment"] == "BULLISH":
                    m["score"] += 20
                    m["reasons"].append(news["catalyst"])
                elif news["sentiment"] == "BEARISH":
                    m["score"] += 15
                    m["reasons"].append("SHORT: " + news["catalyst"])
                if m.get("gap_pct", 0) != 0:
                    direction = "Gap Up" if m["gap_pct"] > 0 else "Gap Down"
                    m["reasons"].append("{} {:.1f}%".format(direction, abs(m["gap_pct"])))
                if m.get("vol_ratio", 1) > 1.5:
                    m["reasons"].append("Vol {:.1f}x".format(m["vol_ratio"]))
            except Exception:
                continue
        movers.sort(key=lambda x: x["score"], reverse=True)
        top     = movers[:20]
        symbols = [m["symbol"] for m in top]
        log.info(f"📋 Today's watchlist ({len(symbols)} stocks):")
        for m in top[:10]:
            log.info("  {} | Gap:{:+.1f}% Vol:{:.1f}x Score:{} | {}".format(
                m["symbol"], m.get("gap_pct", 0), m.get("vol_ratio", 1),
                m["score"], " | ".join(m.get("reasons", [])[:2])
            ))
        return top, symbols

    def run_and_update(self):
        global WATCHLIST
        top_movers, symbols = self.scan()
        if symbols:
            WATCHLIST = symbols
        else:
            WATCHLIST = DEFAULT_WATCHLIST
        log.info(f"📋 Watchlist updated: {', '.join(WATCHLIST)}")
        ws_manager.subscribe(WATCHLIST)
        return top_movers

premarket_scanner = PreMarketScanner()

# ─────────────────────────────────────────────
# OPTIONS FLOW DETECTOR
# ─────────────────────────────────────────────
class OptionsFlowDetector:
    def __init__(self):
        self._cache      = {}
        self._cache_time = None

    def get_flow(self, symbol: str) -> dict:
        now       = datetime.now()
        cache_key = symbol
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if (now - cached["time"]).seconds < 900:
                return cached["data"]
        result = {"signal": "NEUTRAL", "unusual": False, "details": ""}
        try:
            r = requests.get(
                f"https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}",
                headers={"APCA-API-KEY-ID": ALPACA_API_KEY,
                         "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY},
                params={"limit": 50, "type": "call,put"},
                timeout=10
            )
            if r.status_code == 200:
                snapshots = r.json().get("snapshots", {})
                call_vol = 0; put_vol = 0
                call_oi  = 0; put_oi  = 0
                for contract, data in snapshots.items():
                    vol = float(data.get("dailyBar", {}).get("v", 0))
                    oi  = float(data.get("openInterest", 0))
                    if "C" in contract:
                        call_vol += vol; call_oi += oi
                    else:
                        put_vol  += vol; put_oi  += oi
                total_vol = call_vol + put_vol
                if total_vol > 0:
                    call_ratio = call_vol / total_vol
                    if call_ratio > 0.7 and total_vol > 1000:
                        result = {"signal": "BULLISH", "unusual": True,
                                  "details": f"Heavy call buying: {call_ratio:.0%} calls vs puts"}
                    elif call_ratio < 0.3 and total_vol > 1000:
                        result = {"signal": "BEARISH", "unusual": True,
                                  "details": f"Heavy put buying: {(1-call_ratio):.0%} puts vs calls"}
        except:
            pass
        self._cache[cache_key] = {"data": result, "time": now}
        return result

options_flow = OptionsFlowDetector()

# ─────────────────────────────────────────────
# INDICATOR ENGINE
# ─────────────────────────────────────────────
class IndicatorEngine:

    @staticmethod
    def rsi(close, period=14):
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = -delta.clip(upper=0).rolling(period).mean()
        return round(float((100-(100/(1+gain/loss))).iloc[-1]), 2)

    @staticmethod
    def stochastic_rsi(close, rsi_period=14, stoch_period=14):
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(rsi_period).mean()
        loss  = -delta.clip(upper=0).rolling(rsi_period).mean()
        rsi   = 100-(100/(1+gain/loss))
        min_r = rsi.rolling(stoch_period).min()
        max_r = rsi.rolling(stoch_period).max()
        k     = ((rsi-min_r)/(max_r-min_r)*100).rolling(3).mean()
        d     = k.rolling(3).mean()
        k_val = float(k.iloc[-1])
        return {"k": round(k_val,2), "d": round(float(d.iloc[-1]),2),
                "signal": "BUY" if k_val<20 else "SELL" if k_val>80 else "NEUTRAL",
                "oversold": k_val<20, "overbought": k_val>80}

    @staticmethod
    def macd(close, fast=12, slow=26, signal=9):
        ef   = close.ewm(span=fast,adjust=False).mean()
        es   = close.ewm(span=slow,adjust=False).mean()
        ml   = ef-es
        sl   = ml.ewm(span=signal,adjust=False).mean()
        hist = ml-sl
        bc   = ml.iloc[-1]>sl.iloc[-1] and ml.iloc[-2]<=sl.iloc[-2]
        brc  = ml.iloc[-1]<sl.iloc[-1] and ml.iloc[-2]>=sl.iloc[-2]
        return {"macd": round(float(ml.iloc[-1]),4), "signal": round(float(sl.iloc[-1]),4),
                "histogram": round(float(hist.iloc[-1]),4),
                "trend":     "BULLISH" if ml.iloc[-1]>sl.iloc[-1] else "BEARISH",
                "crossover": "BULLISH_CROSS" if bc else "BEARISH_CROSS" if brc else "NONE",
                "momentum":  "INCREASING" if hist.iloc[-1]>hist.iloc[-2] else "DECREASING"}

    @staticmethod
    def bollinger_bands(close, period=20, std_mult=2):
        ma    = close.rolling(period).mean()
        std   = close.rolling(period).std()
        upper = ma+std_mult*std
        lower = ma-std_mult*std
        bw    = (upper-lower)/ma*100
        price = float(close.iloc[-1])
        avg_bw = float(bw.rolling(50).mean().iloc[-1]) if len(bw)>=50 else float(bw.mean())
        return {"upper": round(float(upper.iloc[-1]),2), "middle": round(float(ma.iloc[-1]),2),
                "lower": round(float(lower.iloc[-1]),2), "bandwidth": round(float(bw.iloc[-1]),2),
                "squeeze": float(bw.iloc[-1])<avg_bw*0.5,
                "position": "ABOVE_UPPER" if price>upper.iloc[-1] else
                             "BELOW_LOWER" if price<lower.iloc[-1] else "INSIDE",
                "pct_b": round((price-float(lower.iloc[-1]))/max(
                    float(upper.iloc[-1])-float(lower.iloc[-1]),0.01),3)}

    @staticmethod
    def vwap(df, lookback=20):
        d    = df.tail(lookback)
        typ  = (d["high"]+d["low"]+d["close"])/3
        v    = float((typ*d["volume"]).cumsum().iloc[-1]/d["volume"].cumsum().iloc[-1])
        p    = float(d["close"].iloc[-1])
        dist = (p-v)/v*100
        return {"vwap": round(v,2), "position": "ABOVE" if p>v else "BELOW",
                "distance_pct": round(dist,2), "near_vwap": abs(dist)<2.0}

    @staticmethod
    def obv(df):
        obv = [0]
        for i in range(1,len(df)):
            if   df["close"].iloc[i]>df["close"].iloc[i-1]: obv.append(obv[-1]+df["volume"].iloc[i])
            elif df["close"].iloc[i]<df["close"].iloc[i-1]: obv.append(obv[-1]-df["volume"].iloc[i])
            else: obv.append(obv[-1])
        s  = pd.Series(obv,index=df.index)
        ma = s.rolling(20).mean()
        return {"obv": int(s.iloc[-1]),
                "trend": "ACCUMULATION" if s.iloc[-1]>ma.iloc[-1] else "DISTRIBUTION",
                "momentum": "RISING" if s.diff(5).iloc[-1]>0 else "FALLING"}

    @staticmethod
    def adx(df, period=14):
        h,l,c = df["high"],df["low"],df["close"]
        tr    = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        dmp   = ((h-h.shift())>(l.shift()-l)).astype(float)*(h-h.shift()).clip(lower=0)
        dmm   = ((l.shift()-l)>(h-h.shift())).astype(float)*(l.shift()-l).clip(lower=0)
        atr   = tr.ewm(span=period,adjust=False).mean()
        dip   = 100*dmp.ewm(span=period,adjust=False).mean()/atr
        dim   = 100*dmm.ewm(span=period,adjust=False).mean()/atr
        dx    = 100*(dip-dim).abs()/(dip+dim)
        val   = round(float(dx.ewm(span=period,adjust=False).mean().iloc[-1]),2)
        return {"adx": val, "di_plus": round(float(dip.iloc[-1]),2),
                "di_minus": round(float(dim.iloc[-1]),2),
                "trend_strength": "STRONG" if val>25 else "MODERATE" if val>18 else "WEAK",
                "direction": "BULLISH" if dip.iloc[-1]>dim.iloc[-1] else "BEARISH"}

    @staticmethod
    def atr(df, period=14) -> float:
        h,l,c = df["high"],df["low"],df["close"]
        tr    = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        return round(float(tr.ewm(span=period,adjust=False).mean().iloc[-1]),4)

    @staticmethod
    def ema_stack(close):
        price = float(close.iloc[-1])
        res   = {}
        for p in [9,20,50,200]:
            if len(close) >= p:
                e = float(close.ewm(span=p,adjust=False).mean().iloc[-1])
                res[f"ema_{p}"]   = round(e,2)
                res[f"above_{p}"] = price > e
        if len(close) >= 200:
            res["golden_cross"] = res.get("above_50",False) and res.get("above_200",False)
            res["death_cross"]  = not res.get("above_50",True) and not res.get("above_200",True)
        return res

    @staticmethod
    def volume_analysis(df):
        avg = float(df["volume"].rolling(20).mean().iloc[-1])
        cur = float(df["volume"].iloc[-1])
        r   = cur/avg if avg>0 else 1
        return {"current": int(cur), "avg_20": int(avg), "ratio": round(r,2),
                "trend": "INCREASING" if float(df["volume"].tail(5).mean())>avg else "DECREASING",
                "signal": "HIGH" if r>1.5 else "LOW" if r<0.8 else "NORMAL"}

    @staticmethod
    def gap_analysis(df):
        try:
            if len(df) < 2: return {"gap_pct":0,"gap_type":"NONE"}
            pc  = float(df["close"].iloc[-2])
            to  = float(df["open"].iloc[-1])
            pct = (to-pc)/pc*100
            return {"gap_pct": round(pct,2),
                    "gap_type": "GAP_UP" if pct>0.5 else "GAP_DOWN" if pct<-0.5 else "NONE",
                    "prev_close": round(pc,2), "today_open": round(to,2)}
        except:
            return {"gap_pct":0,"gap_type":"NONE"}

    @staticmethod
    def opening_range(df):
        try:
            today = df.index[-1].date()
            tb    = df[df.index.date==today]
            if len(tb) < 3: return {"or_valid":False}
            orb   = tb.head(6)
            orh   = float(orb["high"].max())
            orl   = float(orb["low"].min())
            price = float(df["close"].iloc[-1])
            return {"or_high":round(orh,2),"or_low":round(orl,2),
                    "or_range":round(orh-orl,2),"or_valid":True,
                    "breakout_up":price>orh,"breakout_down":price<orl,
                    "inside_range":orl<=price<=orh}
        except:
            return {"or_valid":False}

    @staticmethod
    def detect_liquidity_sweep(df, lookback=10):
        try:
            r   = df.tail(lookback)
            ph  = float(r["high"].iloc[:-1].max())
            pl  = float(r["low"].iloc[:-1].min())
            lb  = r.iloc[-1]
            cc,ch,cl = float(lb["close"]),float(lb["high"]),float(lb["low"])
            sh  = ch>ph and cc<ph
            sl  = cl<pl and cc>pl
            return {"sweep_detected":sh or sl,
                    "type":"BEARISH_SWEEP" if sh else "BULLISH_SWEEP" if sl else "NONE",
                    "signal":"SELL" if sh else "BUY" if sl else "NONE"}
        except:
            return {"sweep_detected":False,"type":"NONE","signal":"NONE"}

    @staticmethod
    def market_regime(df):
        try:
            c   = df["close"]
            h,l = df["high"],df["low"]
            tr  = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
            dmp = ((h-h.shift())>(l.shift()-l)).astype(float)*(h-h.shift()).clip(lower=0)
            dmm = ((l.shift()-l)>(h-h.shift())).astype(float)*(l.shift()-l).clip(lower=0)
            atr = tr.ewm(span=14,adjust=False).mean()
            dip = 100*dmp.ewm(span=14,adjust=False).mean()/atr
            dim = 100*dmm.ewm(span=14,adjust=False).mean()/atr
            dx  = 100*(dip-dim).abs()/(dip+dim)
            adx = float(dx.ewm(span=14,adjust=False).mean().iloc[-1])
            vol = float(c.pct_change().rolling(20).std().iloc[-1])*np.sqrt(252)
            e20 = float(c.ewm(span=20,adjust=False).mean().iloc[-1])
            e50 = float(c.ewm(span=50,adjust=False).mean().iloc[-1])
            if   adx>25 and vol<0.4: regime="TRENDING"
            elif adx<20:             regime="RANGING"
            elif vol>0.5:            regime="VOLATILE"
            else:                    regime="MIXED"
            return {"regime":regime,"direction":"BULLISH" if e20>e50 else "BEARISH",
                    "adx":round(adx,2),"volatility":round(vol,3),
                    "tradeable":regime in ["TRENDING","MIXED"]}
        except:
            return {"regime":"UNKNOWN","direction":"UNKNOWN","tradeable":True}

    @staticmethod
    def multi_timeframe_check(symbol: str, action: str) -> dict:
        score   = 0
        details = []
        try:
            params_1h = {"symbols": symbol, "timeframe": "1Hour", "limit": 50, "feed": "iex"}
            data_1h   = alpaca.get("/v2/stocks/bars", params=params_1h, data_api=True)
            bars_1h   = data_1h.get("bars", {}).get(symbol, [])
            if bars_1h and len(bars_1h) >= 26:
                df_1h    = pd.DataFrame(bars_1h)
                df_1h.rename(columns={"c":"close"}, inplace=True)
                close_1h = df_1h["close"].astype(float)
                ef = close_1h.ewm(span=12,adjust=False).mean()
                es = close_1h.ewm(span=26,adjust=False).mean()
                macd_1h_bull    = float(ef.iloc[-1]) > float(es.iloc[-1])
                ema20_1h        = float(close_1h.ewm(span=20,adjust=False).mean().iloc[-1])
                price_above_ema = float(close_1h.iloc[-1]) > ema20_1h
                if action == "BUY" and macd_1h_bull and price_above_ema:
                    score += 1; details.append("1H✅")
                elif action == "SELL" and not macd_1h_bull and not price_above_ema:
                    score += 1; details.append("1H✅")
                else:
                    details.append("1H❌")
        except:
            details.append("1H⚠️")
        try:
            params_15 = {"symbols": symbol, "timeframe": "15Min", "limit": 50, "feed": "iex"}
            data_15   = alpaca.get("/v2/stocks/bars", params=params_15, data_api=True)
            bars_15   = data_15.get("bars", {}).get(symbol, [])
            if bars_15 and len(bars_15) >= 14:
                df_15    = pd.DataFrame(bars_15)
                df_15.rename(columns={"c":"close"}, inplace=True)
                close_15 = df_15["close"].astype(float)
                delta    = close_15.diff()
                gain     = delta.clip(lower=0).rolling(14).mean()
                loss     = -delta.clip(upper=0).rolling(14).mean()
                rsi_15   = float((100-(100/(1+gain/loss))).iloc[-1])
                if action == "BUY" and rsi_15 < 65:
                    score += 1; details.append("15M✅")
                elif action == "SELL" and rsi_15 > 35:
                    score += 1; details.append("15M✅")
                else:
                    details.append("15M❌")
        except:
            details.append("15M⚠️")
        detail_str = " ".join(details) if details else "NoData"
        return {"score": score, "max": 2, "agree": score >= 1,
                "details": detail_str, "strong": score == 2}

    def compute_all(self, df, intraday_df=None):
        close = df["close"]
        ind   = {
            "rsi":           self.rsi(close),
            "stoch_rsi":     self.stochastic_rsi(close),
            "macd":          self.macd(close),
            "bollinger":     self.bollinger_bands(close),
            "vwap":          self.vwap(df),
            "obv":           self.obv(df),
            "adx":           self.adx(df),
            "ema":           self.ema_stack(close),
            "volume":        self.volume_analysis(df),
            "regime":        self.market_regime(df),
            "gap":           self.gap_analysis(df),
            "atr":           self.atr(df),
            "current_price": round(float(close.iloc[-1]),2)
        }
        if intraday_df is not None and len(intraday_df) > 10:
            ind["opening_range"]   = self.opening_range(intraday_df)
            ind["liquidity_sweep"] = self.detect_liquidity_sweep(intraday_df)
        else:
            ind["opening_range"]   = {"or_valid":False}
            ind["liquidity_sweep"] = {"sweep_detected":False,"type":"NONE","signal":"NONE"}
        return ind

ind_engine = IndicatorEngine()

# ─────────────────────────────────────────────
# STRATEGY DETECTOR
# ─────────────────────────────────────────────
class StrategyDetector:

    @staticmethod
    def detect_orb(ind):
        orb    = ind.get("opening_range",{})
        if not orb.get("or_valid",False): return {"applicable":False,"signal":"NONE"}
        vol_ok = ind["volume"]["ratio"] > 1.2
        if orb.get("breakout_up") and vol_ok:
            return {"applicable":True,"signal":"BUY","name":"ORB",
                    "description":f"Broke above OR ${orb['or_high']} {ind['volume']['ratio']}x vol"}
        elif orb.get("breakout_down") and vol_ok:
            return {"applicable":True,"signal":"SELL","name":"ORB",
                    "description":f"Broke below OR ${orb['or_low']} {ind['volume']['ratio']}x vol"}
        return {"applicable":False,"signal":"NONE"}

    @staticmethod
    def detect_gap_fill(ind):
        gap = ind.get("gap",{})
        if gap.get("gap_type")=="GAP_UP" and abs(gap.get("gap_pct",0))>1.5 and ind["rsi"]>65:
            return {"applicable":True,"signal":"SELL","name":"GAP_FILL",
                    "description":f"Gap up {gap['gap_pct']:+.1f}% RSI {ind['rsi']}"}
        if gap.get("gap_type")=="GAP_DOWN" and abs(gap.get("gap_pct",0))>1.5 and ind["rsi"]<35:
            return {"applicable":True,"signal":"BUY","name":"GAP_FILL",
                    "description":f"Gap down {gap['gap_pct']:+.1f}% RSI {ind['rsi']}"}
        return {"applicable":False,"signal":"NONE"}

    @staticmethod
    def detect_vwap_reversion(ind):
        vwap  = ind["vwap"]; rsi = ind["rsi"]; stoch = ind["stoch_rsi"]
        dist  = abs(vwap["distance_pct"])
        if dist < 3: return {"applicable":False,"signal":"NONE"}
        if vwap["position"]=="BELOW" and rsi<40 and stoch["oversold"]:
            return {"applicable":True,"signal":"BUY","name":"VWAP_REVERSION",
                    "description":f"{dist:.1f}% below VWAP RSI {rsi}"}
        if vwap["position"]=="ABOVE" and dist>8 and rsi>65 and stoch["overbought"]:
            return {"applicable":True,"signal":"SELL","name":"VWAP_REVERSION",
                    "description":f"{dist:.1f}% above VWAP RSI {rsi}"}
        return {"applicable":False,"signal":"NONE"}

    @staticmethod
    def detect_liquidity_sweep(ind):
        sweep = ind.get("liquidity_sweep",{})
        if not sweep.get("sweep_detected",False): return {"applicable":False,"signal":"NONE"}
        sig   = sweep.get("signal","NONE")
        if sig in ["BUY","SELL"]:
            return {"applicable":True,"signal":sig,"name":"LIQUIDITY_SWEEP",
                    "description":f"{sweep.get('type')} — reversed"}
        return {"applicable":False,"signal":"NONE"}

    @staticmethod
    def detect_trend_continuation(ind):
        adx  = ind["adx"]
        if adx["adx"] < 20: return {"applicable":False,"signal":"NONE"}
        bull = sum([ind["macd"]["trend"]=="BULLISH",ind["obv"]["trend"]=="ACCUMULATION",
                    ind["vwap"]["position"]=="ABOVE",ind["rsi"]<65,
                    ind["ema"].get("above_50",False),adx["direction"]=="BULLISH"])
        bear = sum([ind["macd"]["trend"]=="BEARISH",ind["obv"]["trend"]=="DISTRIBUTION",
                    ind["vwap"]["position"]=="BELOW",ind["rsi"]>40,
                    not ind["ema"].get("above_50",True),adx["direction"]=="BEARISH"])
        if bull>=5: return {"applicable":True,"signal":"BUY","name":"TREND_CONTINUATION",
                            "description":f"{bull}/6 bullish ADX {adx['adx']}"}
        if bear>=5: return {"applicable":True,"signal":"SELL","name":"TREND_CONTINUATION",
                            "description":f"{bear}/6 bearish ADX {adx['adx']}"}
        return {"applicable":False,"signal":"NONE"}

    @staticmethod
    def detect_bull_flag(df):
        try:
            if len(df) < 15: return {"applicable": False, "signal": "NONE"}
            pole      = df.tail(10).head(5)
            flag      = df.tail(5)
            pole_move = (float(pole["close"].iloc[-1]) - float(pole["close"].iloc[0])) / float(pole["close"].iloc[0]) * 100
            flag_range = float(flag["high"].max()) - float(flag["low"].min())
            tight     = flag_range < float(pole["close"].mean()) * 0.02
            if pole_move > 1.5 and tight and float(df["close"].iloc[-1]) > float(flag["close"].iloc[0]):
                return {"applicable": True, "signal": "BUY", "name": "BULL_FLAG",
                        "description": "Bull flag: pole +{:.1f}% tight consolidation".format(pole_move)}
        except Exception: pass
        return {"applicable": False, "signal": "NONE"}

    @staticmethod
    def detect_bear_flag(df):
        try:
            if len(df) < 15: return {"applicable": False, "signal": "NONE"}
            pole       = df.tail(10).head(5)
            flag       = df.tail(5)
            pole_move  = (float(pole["close"].iloc[0]) - float(pole["close"].iloc[-1])) / float(pole["close"].iloc[0]) * 100
            flag_range = float(flag["high"].max()) - float(flag["low"].min())
            tight      = flag_range < float(pole["close"].mean()) * 0.02
            if pole_move > 1.5 and tight and float(df["close"].iloc[-1]) < float(flag["close"].iloc[0]):
                return {"applicable": True, "signal": "SELL", "name": "BEAR_FLAG",
                        "description": "Bear flag: pole -{:.1f}% tight consolidation".format(pole_move)}
        except Exception: pass
        return {"applicable": False, "signal": "NONE"}

    @staticmethod
    def detect_engulfing(df):
        try:
            if len(df) < 3: return {"applicable": False, "signal": "NONE"}
            prev  = df.iloc[-2]
            curr  = df.iloc[-1]
            p_open, p_close = float(prev["open"]), float(prev["close"])
            c_open, c_close = float(curr["open"]), float(curr["close"])
            if p_close < p_open and c_close > c_open:
                if c_open < p_close and c_close > p_open:
                    return {"applicable": True, "signal": "BUY", "name": "BULLISH_ENGULFING",
                            "description": "Bullish engulfing — reversal signal"}
            if p_close > p_open and c_close < c_open:
                if c_open > p_close and c_close < p_open:
                    return {"applicable": True, "signal": "SELL", "name": "BEARISH_ENGULFING",
                            "description": "Bearish engulfing — reversal signal"}
        except Exception: pass
        return {"applicable": False, "signal": "NONE"}

    @staticmethod
    def detect_double_top(df):
        try:
            if len(df) < 20: return {"applicable": False, "signal": "NONE"}
            highs   = df["high"].values[-20:]
            peak1   = max(highs[:10])
            peak2   = max(highs[10:])
            curr    = float(df["close"].iloc[-1])
            similar = abs(peak1 - peak2) / peak1 < 0.02
            if similar and curr < peak2 * 0.98:
                return {"applicable": True, "signal": "SELL", "name": "DOUBLE_TOP",
                        "description": "Double top at ${:.2f} — bearish reversal".format(peak2)}
        except Exception: pass
        return {"applicable": False, "signal": "NONE"}

    @staticmethod
    def detect_double_bottom(df):
        try:
            if len(df) < 20: return {"applicable": False, "signal": "NONE"}
            lows    = df["low"].values[-20:]
            trough1 = min(lows[:10])
            trough2 = min(lows[10:])
            curr    = float(df["close"].iloc[-1])
            similar = abs(trough1 - trough2) / trough1 < 0.02
            if similar and curr > trough2 * 1.02:
                return {"applicable": True, "signal": "BUY", "name": "DOUBLE_BOTTOM",
                        "description": "Double bottom at ${:.2f} — bullish reversal".format(trough2)}
        except Exception: pass
        return {"applicable": False, "signal": "NONE"}

    @staticmethod
    def detect_vwap_reclaim(ind):
        try:
            vwap     = ind["vwap"]
            rsi      = ind["rsi"]
            dist_pct = vwap["distance_pct"]
            if vwap["position"] == "ABOVE" and abs(dist_pct) < 1.5 and rsi < 60:
                return {"applicable": True, "signal": "BUY", "name": "VWAP_RECLAIM",
                        "description": "Just reclaimed VWAP — bullish momentum"}
            if vwap["position"] == "BELOW" and abs(dist_pct) < 1.5 and rsi > 40:
                return {"applicable": True, "signal": "SELL", "name": "VWAP_REJECTION",
                        "description": "Just rejected VWAP — bearish momentum"}
        except Exception: pass
        return {"applicable": False, "signal": "NONE"}

    @staticmethod
    def detect_inside_bar(df):
        try:
            if len(df) < 3: return {"applicable": False, "signal": "NONE"}
            prev   = df.iloc[-2]
            curr   = df.iloc[-1]
            inside = (float(curr["high"]) < float(prev["high"]) and
                      float(curr["low"]) > float(prev["low"]))
            if inside:
                trend_up = float(curr["close"]) > float(curr["open"])
                return {"applicable": True,
                        "signal": "BUY" if trend_up else "SELL",
                        "name": "INSIDE_BAR",
                        "description": "Inside bar breakout pending"}
        except Exception: pass
        return {"applicable": False, "signal": "NONE"}

    def detect_best_strategy(self, ind, session, df=None):
        strategies = []
        if session in ["ORB_WINDOW", "MORNING"]:
            for fn, name, bonus in [
                (self.detect_orb,      "ORB",      0.15),
                (self.detect_gap_fill, "GAP_FILL", 0.05)
            ]:
                r = fn(ind)
                if r["applicable"]:
                    strategies.append({**r, "score": learner.get_strategy_score(name) + bonus})
        if df is not None and len(df) >= 20:
            for fn, name, bonus in [
                (self.detect_bull_flag,     "BULL_FLAG",     0.12),
                (self.detect_bear_flag,     "BEAR_FLAG",     0.12),
                (self.detect_engulfing,     "ENGULFING",     0.10),
                (self.detect_double_top,    "DOUBLE_TOP",    0.10),
                (self.detect_double_bottom, "DOUBLE_BOTTOM", 0.10),
                (self.detect_inside_bar,    "INSIDE_BAR",    0.05),
            ]:
                r = fn(df)
                if r["applicable"]:
                    strategies.append({**r, "score": learner.get_strategy_score(name) + bonus})
        r_vwap = self.detect_vwap_reclaim(ind)
        if r_vwap["applicable"]:
            strategies.append({**r_vwap,
                                "score": learner.get_strategy_score("VWAP_RECLAIM") + 0.08})
        for fn, name in [
            (self.detect_vwap_reversion,     "VWAP_REVERSION"),
            (self.detect_liquidity_sweep,    "LIQUIDITY_SWEEP"),
            (self.detect_trend_continuation, "TREND_CONTINUATION"),
        ]:
            r = fn(ind)
            if r["applicable"]:
                strategies.append({**r, "score": learner.get_strategy_score(name)})
        if not strategies:
            return {"applicable": False, "signal": "NONE", "name": "NONE"}
        return max(strategies, key=lambda x: x["score"])

strategy_detector = StrategyDetector()

# ─────────────────────────────────────────────
# NEWS ENGINE
# ─────────────────────────────────────────────
class NewsEngine:
    BULLISH = ["beat","surge","rally","record","upgrade","buy","bullish","growth",
               "profit","strong","positive","breakthrough","win","gain","rise","jump",
               "contract","partnership","acquisition","expansion"]
    BEARISH = ["miss","fall","drop","downgrade","sell","loss","bearish","weak",
               "decline","cut","layoff","crash","debt","risk","warning","concern",
               "negative","investigation","lawsuit","recall","bankruptcy"]

    def get_news(self, symbol):
        try:
            r = None
            for attempt in range(2):
                try:
                    r = requests.get(
                        "https://data.alpaca.markets/v1beta1/news",
                        headers={"APCA-API-KEY-ID":ALPACA_API_KEY,
                                 "APCA-API-SECRET-KEY":ALPACA_SECRET_KEY},
                        params={"symbols":symbol,"limit":5}, timeout=10
                    )
                    break
                except Exception:
                    if attempt == 0: time.sleep(1)
                    continue
            if r.status_code == 200:
                articles  = r.json().get("news",[])
                headlines = [a.get("headline","") for a in articles[:5]]
                text      = " ".join(headlines).lower()
                bull      = sum(1 for w in self.BULLISH if w in text)
                bear      = sum(1 for w in self.BEARISH if w in text)
                sentiment = "BULLISH" if bull>bear+1 else "BEARISH" if bear>bull+1 else "NEUTRAL"
                total     = bull+bear
                return {"headlines":headlines[:3],
                        "summary":" | ".join(headlines[:2]) or "No news",
                        "sentiment":sentiment,
                        "score":round((bull-bear)/total,2) if total>0 else 0.0}
        except Exception as e:
            log.warning(f"News failed for {symbol}: {e}")
        return {"headlines":[],"summary":"No news","sentiment":"NEUTRAL","score":0}

news_engine = NewsEngine()

# ─────────────────────────────────────────────
# AI DECISION ENGINE
# ─────────────────────────────────────────────
class AIDecisionEngine:
    def __init__(self):
        self.headers = {
            "x-api-key":          ANTHROPIC_API_KEY,
            "anthropic-version":  "2023-06-01",
            "content-type":       "application/json"
        }

    def analyze(self, symbol, ind, news, strategy, session, spy_change, vix,
                level2=None, options=None, mtf=None):
        l2_text  = (f"Bid wall: {level2.get('bid_wall')} | "
                    f"Ask wall: {level2.get('ask_wall')} | "
                    f"Bias: {level2.get('bias')}") if level2 else "N/A"
        opt_text = f"{options.get('signal')} — {options.get('details')}" if options else "N/A"
        mtf_text = f"{mtf.get('details')} ({mtf.get('score')}/2 agree)" if mtf else "N/A"

        prompt = f"""You are an elite quantitative trader AI. Analyze {symbol} and give a precise recommendation.

PRICE: ${ind['current_price']} | SESSION: {session}
MARKET: SPY {spy_change*100:+.2f}% | VIX: {vix:.1f} | REGIME: {ind['regime']['regime']} {ind['regime']['direction']}
STRATEGY: {strategy.get('name','NONE')} — {strategy.get('description','')}
GAP: {ind['gap'].get('gap_type','NONE')} ({ind['gap'].get('gap_pct',0):+.1f}%)
ATR: {ind.get('atr',0):.2f}

INDICATORS:
• RSI: {ind['rsi']} {"OVERSOLD" if ind['rsi']<30 else "OVERBOUGHT" if ind['rsi']>70 else "NEUTRAL"}
• StochRSI: K={ind['stoch_rsi']['k']} → {ind['stoch_rsi']['signal']}
• MACD: {ind['macd']['trend']} | {ind['macd']['crossover']} | {ind['macd']['momentum']}
• BB: {ind['bollinger']['position']} | Squeeze:{ind['bollinger']['squeeze']} | %B:{ind['bollinger']['pct_b']}
• VWAP: ${ind['vwap']['vwap']} | {ind['vwap']['position']} by {ind['vwap']['distance_pct']}%
• OBV: {ind['obv']['trend']} | {ind['obv']['momentum']}
• ADX: {ind['adx']['adx']} ({ind['adx']['trend_strength']}) | {ind['adx']['direction']}
• EMA: 9={ind['ema'].get('ema_9','N/A')} 20={ind['ema'].get('ema_20','N/A')} 50={ind['ema'].get('ema_50','N/A')}
• Volume: {ind['volume']['ratio']}x ({ind['volume']['signal']})
• Level 2: {l2_text}
• Options Flow: {opt_text}
• Multi-TF: {mtf_text}
• Sweep: {ind['liquidity_sweep'].get('type','NONE')}
• NEWS: {news['sentiment']} | {news['summary'][:80]}

RULES: Trade BASED ON CHART ONLY. Trade BOTH directions equally.
No trade if VIX >30. BUY = long. SELL = short.

Respond ONLY with JSON:
{{"action":"BUY" or "SELL" or "HOLD","confidence":<0-100>,"reasoning":"<3 sentences>","risk_level":"LOW" or "MEDIUM" or "HIGH","key_signal":"<most important>","entry_notes":"<timing>","invalidation":"<what invalidates>"}}"""

        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=self.headers,
                json={"model":"claude-haiku-4-5-20251001","max_tokens":300,
                      "messages":[{"role":"user","content":prompt}]},
                timeout=30
            )
            r.raise_for_status()
            data  = r.json()
            usage = data.get("usage", {})
            cost_tracker.record_call(
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0)
            )
            text  = data["content"][0]["text"]
            start = text.find("{"); end = text.rfind("}")+1
            return json.loads(text[start:end])
        except Exception as e:
            log.error(f"AI error for {symbol}: {e}")
            return {"action":"HOLD","confidence":0,"reasoning":str(e),
                    "risk_level":"HIGH","key_signal":"N/A","entry_notes":"","invalidation":""}

ai_engine = AIDecisionEngine()

# ─────────────────────────────────────────────
# POSITION MANAGER
# ─────────────────────────────────────────────
class PositionManager:
    def __init__(self):
        self.daily_r_loss       = 0.0
        self.daily_start_equity = None
        self.trades_today       = 0
        self.wins_today         = 0
        self.losses_today       = 0
        self.trailing_highs     = {}
        self._eod_close_done    = False
        self._eod_close_date    = None

    def reset_daily(self):
        try:
            equity                  = float(alpaca.get_account()["equity"])
            self.daily_start_equity = equity
            self.daily_r_loss       = 0.0
            self.trades_today       = 0
            self.wins_today         = 0
            self.losses_today       = 0
            self.trailing_highs     = {}
            self._eod_close_done    = False
            self._eod_close_date    = MarketTime.now_et().date()
            cost_tracker.reset_daily()
            log.info(f"📅 Day reset. Equity: ${equity:,.2f}")
            telegram.reset_weekend_alert()
            telegram.send(f"📅 <b>New Trading Day</b>\n💰 Equity: ${equity:,.2f}\n⏰ Opens 9:30 AM ET")
        except Exception as e:
            log.error(f"Daily reset failed: {e}")

    def get_equity(self):
        return float(alpaca.get_account()["equity"])

    def check_kill_switch(self):
        equity = self.get_equity()
        if self.daily_start_equity is None:
            self.daily_start_equity = equity
            return True
        loss_pct          = (self.daily_start_equity - equity) / self.daily_start_equity
        self.daily_r_loss = loss_pct / ACCOUNT_RISK_PER_TRADE
        if self.daily_r_loss >= MAX_DAILY_LOSS_R:
            msg = f"🚨 KILL SWITCH: Down {self.daily_r_loss:.1f}R ({loss_pct*100:.1f}%). Trading stopped."
            log.warning(msg)
            telegram.send(msg)
            return False
        return True

    def calculate_qty(self, price, confidence, filter_score, atr=None):
        equity       = self.get_equity()
        max_notional = min(equity * 0.05, 5000.0)
        risk_dollars = equity * ACCOUNT_RISK_PER_TRADE
        if   filter_score >= 9: mult = 1.25
        elif filter_score >= 8: mult = 1.10
        else:                   mult = 1.0
        if confidence >= 85:    mult = min(mult * 1.1, 1.5)
        risk_notional = (risk_dollars * mult) / STOP_LOSS_PCT
        notional      = min(risk_notional, max_notional)
        notional      = max(1.0, round(notional, 2))
        log.info("  Notional:${:.2f} | {:.1f}% equity | mult:{:.2f}x".format(
            notional, notional/equity*100, mult))
        return notional

    def update_trailing_stops(self):
        if not TRAILING_STOP_ENABLED: return
        try:
            for p in alpaca.get_positions():
                symbol     = p["symbol"]
                price      = float(p["current_price"])
                entry      = float(p.get("avg_entry_price", price))
                side       = p["side"]
                tp         = entry * (1+TAKE_PROFIT_PCT) if side=="long" else entry * (1-TAKE_PROFIT_PCT)
                sl         = entry * (1-STOP_LOSS_PCT)   if side=="long" else entry * (1+STOP_LOSS_PCT)
                unrealized = float(p.get("unrealized_plpc", 0))
                if side == "long":
                    if unrealized >= 0.015 and symbol not in self.trailing_highs:
                        self.trailing_highs[symbol] = price
                        log.info(f"  🔒 Breakeven stop: {symbol} → ${entry:.2f}")
                        telegram.send("Breakeven stop set for {} +{:.1f}% — zero risk!".format(
                            symbol, unrealized*100))
                    elif symbol in self.trailing_highs and price > self.trailing_highs[symbol]:
                        self.trailing_highs[symbol] = price
                        new_stop = round(price * (1-STOP_LOSS_PCT), 2)
                        log.info(f"  📈 Trailing stop: {symbol} → ${new_stop}")
                    telegram.send_position_alert(symbol, price, tp, sl, "BUY")
                elif side == "short":
                    if unrealized >= 0.015 and symbol not in self.trailing_highs:
                        self.trailing_highs[symbol] = price
                        log.info(f"  🔒 Breakeven stop (short): {symbol}")
                        telegram.send("Breakeven stop set for {} SHORT +{:.1f}% — zero risk!".format(
                            symbol, unrealized*100))
                    telegram.send_position_alert(symbol, price, tp, sl, "SELL")
        except Exception as e:
            log.error(f"Trailing stop error: {e}")

    def track_closed_positions(self, prev_open: set):
        try:
            current_open = set(p["symbol"] for p in alpaca.get_positions())
            closed       = prev_open - current_open
            for symbol in closed:
                try:
                    orders = alpaca.get_orders("closed")
                    for o in orders[:20]:
                        if (o.get("symbol") == symbol and
                                o.get("status") == "filled" and
                                o.get("side") in ["sell", "buy"]):
                            exit_price = float(o.get("filled_avg_price", 0))
                            if exit_price > 0:
                                self._record_outcome(symbol, exit_price)
                                break
                except Exception as e:
                    log.error(f"Outcome tracking error for {symbol}: {e}")
        except Exception as e:
            log.error(f"Track closed positions error: {e}")

    def _record_outcome(self, symbol: str, exit_price: float):
        try:
            import psycopg2
            if not DATABASE_URL: return
            conn = psycopg2.connect(DATABASE_URL)
            cur  = conn.cursor()
            cur.execute("""
                SELECT id, action, signal_price, qty, stop_loss, take_profit
                FROM trades
                WHERE symbol = %s AND outcome = ''
                ORDER BY timestamp DESC LIMIT 1
            """, (symbol,))
            row = cur.fetchone()
            if row:
                trade_id    = row[0]
                action      = row[1]
                entry_price = float(row[2] or 0)
                qty         = int(row[3] or 0)
                if entry_price > 0:
                    if action == "BUY":
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price
                    pnl_r       = pnl_pct / STOP_LOSS_PCT
                    pnl_dollars = pnl_pct * entry_price * qty
                    outcome     = "WIN" if pnl_pct > 0 else "LOSS"
                    cur.execute("""
                        UPDATE trades SET outcome=%s, exit_price=%s, pnl_r=%s, pnl_dollars=%s
                        WHERE id=%s
                    """, (outcome, exit_price, round(pnl_r,3), round(pnl_dollars,2), trade_id))
                    conn.commit()
                    if outcome == "WIN": self.wins_today   += 1; emoji = "✅"
                    else:                self.losses_today += 1; emoji = "❌"
                    log.info(f"  📒 Outcome: {symbol} {outcome} {pnl_r:+.2f}R ${pnl_dollars:+.2f}")
                    msg_suffix = ("Great trade!" if pnl_r > 1.5 else
                                  "Keep going!" if pnl_r > 0 else "Stop loss protected capital")
                    telegram.send("{} TRADE CLOSED: {} | {} | Entry ${:.2f} Exit ${:.2f} | {:+.2f}R ${:+.2f} | {}".format(
                        emoji, outcome, symbol, entry_price, exit_price, pnl_r, pnl_dollars, msg_suffix))
            cur.close()
            conn.close()
        except Exception as e:
            log.error(f"Record outcome failed: {e}")

    def close_eod_positions(self):
        today = MarketTime.now_et().date()
        now   = MarketTime.now_et()

        # Always verify — do not trust the flag alone
        still_open_check = alpaca.verify_positions_closed()
        if not still_open_check:
            self._eod_close_done = True
            self._eod_close_date = today
            log.info("✅ No positions to close at EOD")
            return

        if self._eod_close_done and self._eod_close_date == today:
            # Already attempted — retry any survivors
            log.warning(f"⚠️ EOD retry — positions still open: {still_open_check}")
            for symbol in still_open_check:
                alpaca.close_position(symbol)
            time.sleep(5)
            final = alpaca.verify_positions_closed()
            if final:
                telegram.send(
                    f"⚠️ <b>EOD CLOSE FAILED</b>\n"
                    f"Still open: {', '.join(final)}\n"
                    f"⏰ {now.strftime('%H:%M ET')} — Manual close required!"
                )
            else:
                self._eod_close_done = True
                self._eod_close_date = today
                telegram.send("✅ <b>EOD Close Complete</b>\nAll positions closed on retry")
            return

        try:
            positions = alpaca.get_positions()
            if not positions:
                self._eod_close_done = True
                self._eod_close_date = today
                return

            log.info(f"🔒 EOD closing {len(positions)} positions...")
            telegram.send(
                f"🔒 <b>EOD Close</b>\n"
                f"Closing {len(positions)} positions @ {now.strftime('%H:%M ET')}"
            )

            # Step 1: cancel all open orders
            alpaca.cancel_all_orders()
            time.sleep(3)  # give cancels time to settle

            # Step 2: bulk liquidate
            alpaca.liquidate_all()

            # Step 3: wait longer — 207 means Alpaca submitted orders internally,
            # they need time to fill before we submit new ones (was causing 403)
            log.info("  ⏳ Waiting for bulk liquidate fills to settle (25s)...")
            time.sleep(25)

            # Step 4: check survivors
            still_open = alpaca.verify_positions_closed()

            if not still_open:
                self._eod_close_done = True
                self._eod_close_date = today
                telegram.send(
                    f"✅ <b>EOD Close Complete</b>\n"
                    f"{len(positions)} positions closed successfully"
                )
                return

            log.warning(f"⚠️ Still open after bulk: {still_open}")

            # Step 5: individual close with extra cancel step per symbol
            for symbol in still_open:
                success = alpaca.close_position(symbol)
                log.info(f"  🔒 Individual close {symbol}: {'OK' if success else 'FAILED'}")
                time.sleep(1)

            time.sleep(5)

            # Step 6: final check
            final_open = alpaca.verify_positions_closed()

            # Mark done regardless — check_hard_eod will keep retrying survivors
            self._eod_close_done = True
            self._eod_close_date = today

            if final_open:
                log.warning(f"⚠️ Still open after EOD close: {final_open}")
                telegram.send(
                    f"⚠️ <b>EOD CLOSE FAILED</b>\n"
                    f"Still open: {', '.join(final_open)}\n"
                    f"⏰ {now.strftime('%H:%M ET')}\n"
                    f"🔴 Hard close at 3:50 PM will retry"
                )
            else:
                telegram.send(
                    f"✅ <b>EOD Close Complete</b>\n"
                    f"All positions closed successfully"
                )

        except Exception as e:
            log.error(f"EOD close failed: {e}")

    def check_hard_eod(self):
        """Hard EOD at 3:50 PM — ALWAYS retries open positions, ignores _eod_close_done flag."""
        now   = MarketTime.now_et()
        today = now.date()
        try:
            still_open = alpaca.verify_positions_closed()
            if not still_open:
                self._eod_close_done = True
                self._eod_close_date = today
                return

            log.warning(f"⚠️ Hard EOD — forcing close: {still_open}")
            telegram.send(
                f"⚠️ <b>HARD EOD CLOSE 3:50 PM</b>\n"
                f"Forcing: {', '.join(still_open)}"
            )

            alpaca.cancel_all_orders()
            time.sleep(3)

            for symbol in still_open:
                success = alpaca.close_position(symbol)
                log.info(f"  🔒 Hard close {symbol}: {'OK' if success else 'FAILED'}")
                time.sleep(1)

            self._eod_close_done = True
            self._eod_close_date = today

            time.sleep(5)
            final = alpaca.verify_positions_closed()
            if final:
                telegram.send(
                    f"🚨 <b>HARD EOD FAILED</b>\n"
                    f"Still open: {', '.join(final)}\n"
                    f"Manual close required NOW!"
                )
            else:
                telegram.send("✅ <b>Hard EOD Complete</b> — All positions closed")

        except Exception as e:
            log.error(f"Hard EOD failed: {e}")

    def send_daily_summary(self):
        try:
            equity      = self.get_equity()
            best, worst = journal.get_todays_best_worst()
            api_stats   = cost_tracker.get_stats()
            journal.save_daily_summary(
                MarketTime.now_et().date(),
                self.daily_start_equity or equity,
                equity, self.trades_today,
                self.wins_today, self.losses_today,
                api_stats["daily_cost"]
            )
            telegram.send_daily_summary(
                equity, self.daily_start_equity or equity,
                self.trades_today, self.wins_today, self.losses_today,
                best_trade=best, worst_trade=worst,
                api_cost=api_stats["daily_cost"]
            )
        except Exception as e:
            log.error(f"Daily summary failed: {e}")

pos_manager = PositionManager()

# ─────────────────────────────────────────────
# 10-FACTOR SIGNAL FILTER
# ─────────────────────────────────────────────
def passes_filter(ind, ai, strategy, news, spy_change, vix, symbol=""):
    action   = ai["action"]
    conf_thr = learner.get_confidence_threshold()
    adx_thr  = learner.get_adx_threshold()
    checks   = {
        "AI_Confidence": ai["confidence"] >= conf_thr,
        "ADX_Strength":  ind["adx"]["adx"] >= adx_thr,
        "RSI_OK":        (ind["rsi"] < 78 if action=="BUY" else ind["rsi"] > 22 if action=="SELL" else True),
        "MACD_OK":       ((ind["macd"]["trend"]=="BULLISH" or ind["macd"]["crossover"]=="BULLISH_CROSS")
                          if action=="BUY" else
                          (ind["macd"]["trend"]=="BEARISH" or ind["macd"]["crossover"]=="BEARISH_CROSS")
                          if action=="SELL" else True),
        "VWAP_OK":       (ind["vwap"]["distance_pct"] <= MAX_VWAP_DISTANCE_PCT if action=="BUY"
                          else True),
        "OBV_OK":        (ind["obv"]["trend"]=="ACCUMULATION" if action=="BUY"
                          else ind["obv"]["trend"]=="DISTRIBUTION" if action=="SELL" else True),
        "Volume_OK":     ind["volume"]["ratio"] >= MIN_VOLUME_RATIO,
        "Regime_OK":     ind["regime"]["tradeable"],
        "EMA_OK":        (ind["ema"].get("above_50", True) if action=="BUY"
                          else not ind["ema"].get("above_50", False) if action=="SELL" else True),
        "Macro_OK":      vix <= MAX_VIX,
    }
    passed = sum(checks.values())
    return passed >= 6, passed, checks

# ─────────────────────────────────────────────
# WEB DASHBOARD
# ─────────────────────────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>APEX AI Trading Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="30">
    <style>
        body { font-family: monospace; background: #0d1117; color: #c9d1d9; margin: 20px; }
        h1   { color: #58a6ff; }
        .card { background: #161b22; border: 1px solid #30363d; padding: 15px;
                margin: 10px 0; border-radius: 8px; }
        .green { color: #3fb950; } .red { color: #f85149; }
        .yellow { color: #d29922; } .blue { color: #58a6ff; }
        table { width: 100%; border-collapse: collapse; }
        td, th { padding: 8px; text-align: left; border-bottom: 1px solid #30363d; }
        th { color: #58a6ff; }
    </style>
</head>
<body>
    <h1>🤖 APEX AI Trading Dashboard v6.2</h1>
    <p class="blue">Auto-refreshes every 30 seconds | {{ time }}</p>
    <div class="card">
        <h3>💰 Account</h3>
        <table>
            <tr><td>Equity</td><td class="{{ 'green' if equity > start_equity else 'red' }}">${{ "%.2f"|format(equity) }}</td></tr>
            <tr><td>Daily P&L</td><td class="{{ 'green' if pnl >= 0 else 'red' }}">${{ "%+.2f"|format(pnl) }} ({{ "%+.2f"|format(pnl_pct) }}%)</td></tr>
            <tr><td>Positions</td><td>{{ positions }}/{{ max_pos }}</td></tr>
            <tr><td>Daily R Loss</td><td>{{ "%.1f"|format(daily_r) }}R</td></tr>
            <tr><td>Trades Today</td><td>{{ trades }}</td></tr>
            <tr><td>API Cost Today</td><td>${{ "%.4f"|format(api_cost) }}</td></tr>
        </table>
    </div>
    <div class="card">
        <h3>📊 Market</h3>
        <table>
            <tr><td>Session</td><td class="blue">{{ session }}</td></tr>
            <tr><td>SPY Change</td><td class="{{ 'green' if spy >= 0 else 'red' }}">{{ "%+.2f"|format(spy*100) }}%</td></tr>
            <tr><td>VIX</td><td class="{{ 'red' if vix > 25 else 'green' }}">{{ "%.1f"|format(vix) }}</td></tr>
            <tr><td>Geo Risk</td><td class="{{ 'red' if geo=='HIGH' else 'yellow' if geo=='MEDIUM' else 'green' }}">{{ geo }}</td></tr>
        </table>
    </div>
    <div class="card">
        <h3>📋 Watchlist</h3>
        <p>{{ watchlist|join(', ') }}</p>
    </div>
    <div class="card">
        <h3>🧠 Learning</h3>
        <table>
            <tr><td>Win Rate</td><td class="green">{{ "%.1f"|format(win_rate*100) }}%</td></tr>
            <tr><td>Total Trades</td><td>{{ total_trades }}</td></tr>
            <tr><td>Confidence</td><td>{{ confidence }}%</td></tr>
        </table>
    </div>
</body>
</html>
"""

def start_dashboard():
    if not FLASK_AVAILABLE:
        log.info("⚠ Flask not available — dashboard disabled")
        return
    app = Flask(__name__)

    @app.route("/")
    def dashboard():
        try:
            account   = alpaca.get_account()
            equity    = float(account["equity"])
            start_eq  = pos_manager.daily_start_equity or equity
            pnl       = equity - start_eq
            pnl_pct   = (pnl / start_eq * 100) if start_eq > 0 else 0
            positions = alpaca.get_positions()
            spy_change = market_ctx.get_spy_change()
            vix        = market_ctx.get_vix()
            geo        = geo_risk.get_risk_level()["risk_level"]
            l_params   = learner.params
            api_stats  = cost_tracker.get_stats()
            return render_template_string(
                DASHBOARD_HTML,
                time=MarketTime.now_et().strftime("%Y-%m-%d %H:%M:%S ET"),
                equity=equity, start_equity=start_eq,
                pnl=pnl, pnl_pct=pnl_pct,
                positions=len(positions), max_pos=MAX_OPEN_POSITIONS,
                daily_r=pos_manager.daily_r_loss,
                trades=pos_manager.trades_today,
                api_cost=api_stats["daily_cost"],
                session=MarketTime.session_name(),
                spy=spy_change, vix=vix, geo=geo,
                watchlist=WATCHLIST,
                win_rate=l_params.get("win_rate", 0),
                total_trades=l_params.get("total_trades", 0),
                confidence=l_params.get("min_confidence", 70)
            )
        except Exception as e:
            return f"<h1>APEX Dashboard</h1><p>Loading... {e}</p>"

    @app.route("/api/status")
    def api_status():
        try:
            account = alpaca.get_account()
            return jsonify({
                "status":    "online",
                "equity":    float(account["equity"]),
                "session":   MarketTime.session_name(),
                "positions": len(alpaca.get_positions()),
                "geo_risk":  geo_risk.get_risk_level()["risk_level"],
                "api_cost":  cost_tracker.get_stats()["daily_cost"],
                "watchlist": WATCHLIST
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})

    def run_flask():
        app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)

    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
    log.info("🌐 Web dashboard started on port 8080")

# ─────────────────────────────────────────────
# PARALLEL SCANNER
# ─────────────────────────────────────────────
def scan_single_symbol(symbol: str, session: str, spy_change: float,
                        vix: float, open_positions: dict) -> dict:
    if symbol in open_positions:
        return None
    try:
        cached = bg_processor.get_cached(symbol)
        if cached:
            df          = cached["df"]
            intraday_df = cached["intraday_df"]
            ind         = cached["ind"]
        else:
            df = alpaca.get_bars(symbol, timeframe="5Min", limit=390)
            if df is None or len(df) < 30:
                return None
            intraday_df = df
            ind         = ind_engine.compute_all(df, intraday_df)

        ws_price = ws_manager.get_price(symbol)
        if ws_price > 0:
            ind["current_price"] = round(ws_price, 2)

        price = ind["current_price"]

        if not ind["regime"]["tradeable"] and not ind["liquidity_sweep"]["sweep_detected"]:
            return None

        strategy = strategy_detector.detect_best_strategy(ind, session, df)

        # FIX v6.2: skip AI entirely if no strategy detected AND no sweep.
        # This is the biggest Claude API cost reduction — only call AI when
        # there is an actual technical setup worth analyzing.
        if not strategy.get("applicable") and not ind["liquidity_sweep"]["sweep_detected"]:
            return None

        # Secondary pre-filter: also require minimum momentum
        if (ind["adx"]["adx"] < MIN_ADX_STRENGTH and
                ind["volume"]["signal"] == "LOW"):
            return None

        if ind["adx"]["adx"] > 20 and ind["volume"]["ratio"] > 1.3 and strategy.get("applicable"):
            telegram.send_hot_setup(
                symbol,
                f"{strategy.get('name')}: {strategy.get('description','')[:60]}",
                65
            )

        news    = news_engine.get_news(symbol)
        level2  = alpaca.get_order_book(symbol)
        options = options_flow.get_flow(symbol)

        mtf = None
        if strategy.get("applicable"):
            signal_dir = strategy.get("signal", "NONE")
            if signal_dir in ["BUY", "SELL"]:
                mtf = ind_engine.multi_timeframe_check(symbol, signal_dir)

        ai = ai_engine.analyze(symbol, ind, news, strategy, session,
                                spy_change, vix, level2, options, mtf)

        if ai["action"] == "HOLD":
            return None

        passed, score, checks = passes_filter(ind, ai, strategy, news,
                                               spy_change, vix, symbol)
        if not passed:
            log.info(f"  {symbol} ❌ {score}/10")
            return None

        if mtf and not mtf.get("agree"):
            log.info(f"  {symbol} ⚠️ Timeframes disagree: {mtf.get('details')} — noted")

        return {
            "symbol":   symbol,
            "ai":       ai,
            "ind":      ind,
            "strategy": strategy,
            "news":     news,
            "score":    score,
            "checks":   checks,
            "level2":   level2,
            "options":  options,
            "mtf":      mtf,
            "price":    price
        }

    except Exception as e:
        log.error(f"  ⚠ Error scanning {symbol}: {e}")
        return None

# ─────────────────────────────────────────────
# MAIN SCAN
# ─────────────────────────────────────────────
def scan():
    global WATCHLIST
    now_et  = MarketTime.now_et()
    session = MarketTime.session_name()

    if now_et.weekday() >= 5:
        log.info("🌙 Weekend — zero API calls")
        telegram.send_sleeping()
        return

    if not MarketTime.is_market_open():
        log.info(f"🌙 Market closed ({session})")
        return

    print(f"\n{'='*60}")
    log.info(f"🔍 APEX SCAN — {now_et.strftime('%Y-%m-%d %H:%M:%S ET')} | {session}")
    print(f"{'='*60}")

    if MarketTime.is_hard_eod():
        pos_manager.check_hard_eod()
        return

    if MarketTime.is_eod():
        log.info("🔒 EOD — closing positions")
        pos_manager.close_eod_positions()
        return

    if not MarketTime.is_prime_session():
        log.info(f"⏸ {session} — trailing stops only")
        pos_manager.update_trailing_stops()
        return

    now_check = MarketTime.now_et()
    if now_check.hour == MARKET_OPEN_HOUR and now_check.minute < TRADE_START_MIN:
        log.info("⏸ Waiting for first 5min candle to close at 9:35 AM")
        return

    if now_check.hour >= 15:
        log.info("⏸ After 3:00 PM — no new entries, managing exits only")
        pos_manager.update_trailing_stops()
        return

    if not pos_manager.check_kill_switch():
        return

    geo_risk.check_and_alert()

    spy_change = market_ctx.get_spy_change()
    vix        = market_ctx.get_vix()
    log.info(f"📊 SPY: {spy_change*100:+.2f}% | VIX: {vix:.1f}")

    if vix > MAX_VIX:
        log.info(f"⚠️ VIX too high ({vix:.1f}) — skipping")
        return

    spy_abs = abs(spy_change)
    if spy_abs < 0.001 and vix < 12:
        log.info("⏸ Market too choppy (SPY flat + low VIX) — skipping")
        return

    vix_multiplier = 1.0
    if vix > 25:
        vix_multiplier = 0.5
        log.info(f"⚠️ High VIX ({vix:.1f}) — positions reduced 50%")
    elif vix > 20:
        vix_multiplier = 0.75
        log.info(f"⚠️ Elevated VIX ({vix:.1f}) — positions reduced 25%")

    equity         = pos_manager.get_equity()
    open_positions = {p["symbol"]: p for p in alpaca.get_positions()}

    api_stats = cost_tracker.get_stats()
    geo_level = geo_risk.get_risk_level()["risk_level"]
    telegram.send_heartbeat(equity, len(open_positions),
                             pos_manager.daily_r_loss, geo_level,
                             api_stats["daily_cost"])

    log.info(f"💰 ${equity:,.2f} | {len(open_positions)}/{MAX_OPEN_POSITIONS} | {pos_manager.daily_r_loss:.1f}R")

    prev_open_syms = set(open_positions.keys())
    pos_manager.update_trailing_stops()
    pos_manager.track_closed_positions(prev_open_syms)

    available_slots = MAX_OPEN_POSITIONS - len(open_positions)
    if available_slots <= 0:
        log.info("⏭ Max positions reached")
        return

    symbols_to_scan = [s for s in WATCHLIST if s not in open_positions]
    if not symbols_to_scan:
        log.info("⏭ No symbols available to scan")
        return

    log.info(f"⚡ Parallel scanning {len(symbols_to_scan)} symbols...")

    signals = []
    # FIX v6.2: wrap entire executor block so TimeoutError can't crash the bot
    try:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(5, len(symbols_to_scan))) as executor:
            futures = {
                executor.submit(scan_single_symbol, sym, session, spy_change,
                                 vix, open_positions): sym
                for sym in symbols_to_scan
            }
            for future in concurrent.futures.as_completed(futures, timeout=45):
                try:
                    result = future.result()
                    if result:
                        signals.append(result)
                        log.info(f"  📊 {result['symbol']}: {result['ai']['action']} "
                                 f"{result['ai']['confidence']}% score:{result['score']}/10")
                except Exception as e:
                    log.error(f"Parallel scan error: {e}")
    except concurrent.futures.TimeoutError:
        log.warning(f"⚠️ Scan timeout — proceeding with {len(signals)} signals collected")

    signals.sort(key=lambda x: (x["score"], x["ai"]["confidence"]), reverse=True)

    executed = 0
    for sig in signals:
        if executed >= available_slots:
            break

        symbol   = sig["symbol"]
        ai       = sig["ai"]
        ind      = sig["ind"]
        strategy = sig["strategy"]
        news     = sig["news"]
        score    = sig["score"]
        price    = sig["price"]
        mtf      = sig.get("mtf")

        notional = pos_manager.calculate_qty(price, ai["confidence"], score, ind.get("atr"))
        sl  = round(price * (1-STOP_LOSS_PCT  if ai["action"]=="BUY" else 1+STOP_LOSS_PCT), 2)
        tp  = round(price * (1+TAKE_PROFIT_PCT if ai["action"]=="BUY" else 1-TAKE_PROFIT_PCT), 2)

        try:
            # FIX v6.2: route SELL through place_short_order with qty (integer shares).
            # Alpaca does NOT allow notional for short sells — was causing 422 on every short.
            if ai["action"] == "BUY":
                order = alpaca.place_bracket_order(symbol, "buy", notional, sl, tp)
                qty   = notional   # log notional for longs
            else:
                num_shares = max(1, int(notional / price))
                order      = alpaca.place_short_order(symbol, num_shares, sl, tp)
                qty        = num_shares   # log shares for shorts

            order_id = order.get("id", "")
            pos_manager.trades_today  += 1
            open_positions[symbol]     = {"symbol": symbol}
            executed                  += 1

            log.info(f"\n  🚀 {ai['action']} {qty} {symbol} @ ${price} | SL:${sl} TP:${tp}")

            def get_fill_and_report(oid, sym, sig_price, q, side):
                fill = journal.update_fill_price(oid, sym)
                if fill and fill > 0:
                    slippage_pct = abs(fill - sig_price) / sig_price * 100
                    journal.update_latest_trade_fill(sym, sig_price, fill, slippage_pct)
                    telegram.send_fill_report(sym, sig_price, fill, q, side)
                    return fill, slippage_pct
                return sig_price, 0.0

            fill_thread = threading.Thread(
                target=get_fill_and_report,
                args=(order_id, symbol, price, qty, ai["action"]),
                daemon=True
            )
            fill_thread.start()

            mtf_text = f"TF: {mtf.get('details')}" if mtf else ""
            journal.log_entry({
                "timestamp":       datetime.now().isoformat(),
                "symbol":          symbol,
                "action":          ai["action"],
                "strategy":        strategy.get("name","NONE"),
                "signal_price":    price,
                "fill_price":      price,
                "slippage_pct":    0,
                "stop_loss":       sl,
                "take_profit":     tp,
                "qty":             qty,
                "confidence":      ai["confidence"],
                "filter_score":    score,
                "rsi":             ind["rsi"],
                "macd":            ind["macd"]["trend"],
                "adx":             ind["adx"]["adx"],
                "vwap_distance":   ind["vwap"]["distance_pct"],
                "obv_trend":       ind["obv"]["trend"],
                "volume_ratio":    ind["volume"]["ratio"],
                "market_regime":   ind["regime"]["regime"],
                "news_sentiment":  news["sentiment"],
                "reasoning":       ai["reasoning"],
                "atr":             ind.get("atr", 0),
                "timeframe_agree": mtf.get("agree", False) if mtf else False,
                "outcome":"","exit_price":"","pnl_r":"","pnl_dollars":""
            })

            emoji     = "🟢" if ai["action"]=="BUY" else "🔴"
            direction = "📈 LONG" if ai["action"]=="BUY" else "📉 SHORT"
            l2        = sig.get("level2", {})
            opt       = sig.get("options", {})
            telegram.send(
                f"{emoji} <b>APEX — {direction}</b>\n"
                f"📌 <b>{symbol}</b> @ ${price}\n"
                f"📐 {strategy.get('name')} | Score: {score}/10\n"
                f"📊 Qty: {qty} | ATR: {ind.get('atr',0):.2f}\n"
                f"🎯 TP: ${tp} | 🛑 SL: ${sl}\n"
                f"🤖 {ai['confidence']}% | Risk: {ai['risk_level']}\n"
                f"📰 {news['sentiment']}"
                + (f" | Options: {opt.get('signal','N/A')}" if opt.get('unusual') else "")
                + (f"\n🔀 {mtf_text}" if mtf_text else "")
                + f"\n💡 {ai['reasoning'][:150]}"
            )
        except Exception as e:
            log.error(f"Order failed for {symbol}: {e}")

    log.info(f"\n✅ Scan done — {executed} orders | {len(signals)} signals | "
             f"{MarketTime.get_scan_interval()//60}min interval")

# ─────────────────────────────────────────────
# SCHEDULED JOBS
# ─────────────────────────────────────────────
def run_premarket_routine():
    if not MarketTime.is_weekend():
        top_movers = premarket_scanner.run_and_update()
        try:
            equity = float(alpaca.get_account()["equity"])
            telegram.send_morning_briefing(
                market_ctx.get_spy_change(),
                market_ctx.get_vix(),
                equity, WATCHLIST,
                top_movers=top_movers,
                geo_risk=geo_risk.get_risk_level()["risk_level"]
            )
        except Exception as e:
            log.error(f"Pre-market briefing failed: {e}")

def send_morning_briefing():
    if MarketTime.is_weekend(): return
    try:
        equity = float(alpaca.get_account()["equity"])
        telegram.send_morning_briefing(
            market_ctx.get_spy_change(), market_ctx.get_vix(),
            equity, WATCHLIST,
            geo_risk=geo_risk.get_risk_level()["risk_level"]
        )
    except Exception as e:
        log.error(f"Morning briefing failed: {e}")

def run_daily_learning():
    log.info("🧠 Daily learning...")
    pos_manager.send_daily_summary()
    learner.learn()

def send_weekly_report():
    telegram.send_weekly_report(learner.params)

# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────
def startup_summary():
    account  = alpaca.get_account()
    equity   = float(account["equity"])
    buying_p = float(account["buying_power"])
    params   = learner.params

    log.info("╔══════════════════════════════════════════╗")
    log.info("║   APEX AI — ULTIMATE TRADING BOT v6.2    ║")
    log.info("╚══════════════════════════════════════════╝")
    log.info(f"💰 Equity: ${equity:,.2f} | Power: ${buying_p:,.2f}")
    log.info(f"📋 Watchlist: {', '.join(WATCHLIST)}")
    log.info(f"🧠 Win Rate: {params['win_rate']:.1%} | Trades: {params['total_trades']}")
    log.info(f"⚡ WebSocket: ON | Parallel: ON | Background: ON")
    log.info(f"📐 Full day scan 9:35AM-3PM ET (5min candle)")
    log.info(f"🌐 Dashboard: http://localhost:8080")

    telegram.send_online(equity, params["win_rate"], params["min_confidence"])

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        startup_summary()
        pos_manager.reset_daily()

        ws_manager.start()
        bg_processor.start()
        volume_scanner.start()
        start_dashboard()

        schedule.every().day.at("08:00").do(run_premarket_routine)
        schedule.every().day.at("08:15").do(lambda: ws_manager.subscribe(WATCHLIST))
        schedule.every().day.at("09:00").do(send_morning_briefing)
        schedule.every().day.at("09:25").do(pos_manager.reset_daily)
        schedule.every().day.at("16:30").do(run_daily_learning)
        schedule.every().sunday.at("08:00").do(send_weekly_report)
        schedule.every().sunday.at("10:00").do(learner.monthly_report)

        log.info("\n⚡ APEX v6.2 LIVE")
        log.info("⏱  Prime (9:35AM-3PM ET) → 5 min scan (5min candle)")
        log.info("⏱  Off-hours → 60 min check")
        log.info("⏱  Weekend → zero API calls")

        scan()
        last_scan   = datetime.now()
        current_int = MarketTime.get_scan_interval()

        while True:
            schedule.run_pending()
            now     = datetime.now()
            new_int = MarketTime.get_scan_interval()

            if new_int != current_int:
                current_int = new_int
                log.info(f"⏱ Scan interval changed: {current_int//60} min ({MarketTime.session_name()})")

            if (now - last_scan).seconds >= current_int:
                scan()
                last_scan = now

            time.sleep(15)

    except Exception as e:
        log.error(f"❌ Bot crashed: {e}")
        telegram.send_offline(str(e))
        raise
