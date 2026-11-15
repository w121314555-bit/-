import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
import requests

# ==========================================
# 1. 從 GitHub Secrets 安全讀取通知設定
# ==========================================
DISCORD_WEBHOOK_URL = os.getenv("https://discord.com/api/webhooks/1571173815436705802/bAEet3HBx2uoZE0zm5kgowcwrU31JZ7gJD4GEmZrnTBI0doh-gZO_X4CDZH0_hZsOGgp", "").strip()
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# ==========================================
# 2. 全幣種掃描設定 (OKX 公開 API，免 API Key)
# ==========================================
MARKET_TYPE      = "SWAP"    # "SWAP" = 全部 USDT 永續合約；"SPOT" = 全部 USDT 現貨
MIN_24H_VOL_USDT = 1   # 全週期統一門檻：500 萬 USDT
MAX_WORKERS      = 5
MIN_REQ_INTERVAL = 0.08      # 約每秒 12.5 次請求，避免觸發 OKX 限流

# ==========================================
# 3. 7-in-1 Smart v4 分級參數 (與 TradingView 同步)
# ==========================================
ADX_THRESH     = 15.0        # ADX 門檻調整為 15
VOL_MULT       = 1.2         # 放量倍數
MIN_GRADE_SCORE = 2          # 最低推播分數：3 = 只推 S級；2 = 推 S級 + A級

SWING_LOOKBACK = 10
SL_ATR_BUFFER  = 0.5
TP1_RR         = 1.5
TP2_RR         = 2.0

# 自動階梯大週期對照表
HTF_MAP = {
    "5m":  ["15m", "1h"],
    "15m": ["1h"],
    "1h":  ["4h"],
}

exchange = ccxt.okx({"enableRateLimit": False, "timeout": 15000})
rate_lock = threading.Lock()
notify_lock = threading.Lock()
last_req_time = 0.0


def throttled_fetch_ohlcv(symbol: str, timeframe: str, limit: int = 250):
    global last_req_time
    with rate_lock:
        now = time.monotonic()
        wait = MIN_REQ_INTERVAL - (now - last_req_time)
        if wait > 0:
            time.sleep(wait)
        last_req_time = time.monotonic()
    try:
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except ccxt.RateLimitExceeded:
        time.sleep(1.5)
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception:
            return None
    except Exception:
        return None


def fmt_price(price: float) -> str:
    if price >= 1000:
        return f"{price:,.1f}"
    elif price >= 10:
        return f"{price:,.3f}"
    elif price >= 0.1:
        return f"{price:,.4f}"
    elif price >= 0.001:
        return f"{price:,.6f}"
    else:
        return f"{price:,.8f}"


def send_notification(message: str):
    """執行緒安全推播，避免多筆訊號同時觸發 Discord Webhook 限流"""
    with notify_lock:
        print(f"\n{message}\n" + "-" * 45)
        if DISCORD_WEBHOOK_URL:
            try:
                requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)
                time.sleep(0.35)
            except Exception as e:
                print(f"[Discord 推播失敗] {e}")

        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            try:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
                time.sleep(0.1)
            except Exception as e:
                print(f"[Telegram 推播失敗] {e}")


def get_all_okx_symbols() -> list:
    try:
        markets = exchange.load_markets(reload=True)
        tickers = exchange.fetch_tickers(params={"instType": MARKET_TYPE})
        symbols_with_vol = []
        for sym, m in markets.items():
            if not m.get("active") or m.get("quote") != "USDT":
                continue
            if MARKET_TYPE == "SWAP" and not m.get("swap"):
                continue
            if MARKET_TYPE == "SPOT" and not m.get("spot"):
                continue
            vol_usdt = float(tickers.get(sym, {}).get("quoteVolume") or 0)
            if vol_usdt >= MIN_24H_VOL_USDT:
                symbols_with_vol.append((sym, vol_usdt))
        symbols_with_vol.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in symbols_with_vol]
    except Exception as e:
        print(f"[抓取 OKX 幣種清單失敗] {e}")
        return []


def fetch_df(symbol: str, timeframe: str, limit: int = 250):
    ohlcv = throttled_fetch_ohlcv(symbol, timeframe, limit)
    if not ohlcv or len(ohlcv) < 50:
        return None
    return pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])


def calc_rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def calc_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = calc_rma(gain, length) / calc_rma(loss, length).replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(100)


def calc_atr_and_adx(df: pd.DataFrame, length: int = 14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = calc_rma(tr, length)

    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    plus_di = (100 * calc_rma(plus_dm, length) / atr.replace(0, np.nan)).fillna(0)
    minus_di = (100 * calc_rma(minus_dm, length) / atr.replace(0, np.nan)).fillna(0)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = calc_rma(dx, length)
    return atr, adx


def check_htf_trend(symbol: str, current_price: float, side: str, base_tf: str):
    """依目前週期自動對應階梯大週期 (新幣不足 200 根自動放行)"""
    tfs = HTF_MAP.get(base_tf, ["1h"])
    results = []
    all_pass = True

    for tf in tfs:
        df_htf = fetch_df(symbol, tf, limit=250)
        if df_htf is None or len(df_htf) < 200:
            results.append(f"{tf.upper()}無")
            continue
        ema200 = df_htf["close"].ewm(span=200, adjust=False).mean().iloc[-1]
        is_bull = current_price > ema200
        results.append(f"{tf.upper()}多" if is_bull else f"{tf.upper()}空")
        if (side == "BUY" and not is_bull) or (side == "SELL" and is_bull):
            all_pass = False

    return all_pass, " / ".join(results)


def analyze_symbol(symbol: str, timeframe: str) -> bool:
    df = fetch_df(symbol, timeframe, limit=250)
    if df is None or len(df) < 205:
        return False

    close = df["close"]
    ema9   = close.ewm(span=9, adjust=False).mean()
    ema21  = close.ewm(span=21, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    sig_line  = macd_line.ewm(span=9, adjust=False).mean()
    hist_line = macd_line - sig_line

    rsi = calc_rsi(close, 14)
    atr, adx = calc_atr_and_adx(df, 14)
    vol_sma = df["volume"].rolling(20).mean()

    i = -2  # 取上一根剛收盤的 K 棒
    c_close, c_ema200 = close.iloc[i], ema200.iloc[i]
    c_rsi, c_adx, c_atr, c_vol_ma = rsi.iloc[i], adx.iloc[i], atr.iloc[i], vol_sma.iloc[i]

    if any(pd.isna(v) for v in [c_close, c_ema200, c_rsi, c_adx, c_atr, c_vol_ma]) or c_close <= 0:
        return False

    c_vol_r = df["volume"].iloc[i] / c_vol_ma if c_vol_ma > 0 else 0.0
    ema_cross_up = (ema9.iloc[i] > ema21.iloc[i]) and (ema9.iloc[i - 1] <= ema21.iloc[i - 1])
    ema_cross_dn = (ema9.iloc[i] < ema21.iloc[i]) and (ema9.iloc[i - 1] >= ema21.iloc[i - 1])
    macd_bull = (macd_line.iloc[i] > sig_line.iloc[i]) and (hist_line.iloc[i] > 0)
    macd_bear = (macd_line.iloc[i] < sig_line.iloc[i]) and (hist_line.iloc[i] < 0)

    # 1. 基礎必備條件檢查
    side = None
    if c_close > c_ema200 and ema_cross_up and macd_bull and c_rsi < 70:
        side = "BUY"
    elif c_close < c_ema200 and ema_cross_dn and macd_bear and c_rsi > 30:
        side = "SELL"

    if not side:
        return False

    # 2. 進階濾網計分 (若 ADX 與量能雙雙未過，最高僅 1 分，直接跳過省下 HTF API 請求)
    adx_ok = c_adx >= ADX_THRESH
    vol_ok = c_vol_r >= VOL_MULT
    if int(adx_ok) + int(vol_ok) + 1 < MIN_GRADE_SCORE:
        return False

    htf_ok, htf_str = check_htf_trend(symbol, c_close, side, timeframe)
    score = int(adx_ok) + int(vol_ok) + int(htf_ok)

    if score < MIN_GRADE_SCORE:
        return False

    # 3. 評定等級與缺項說明
    if score == 3:
        grade_badge = "🔥 【S級 最高品質 3/3】"
    else:
        missing = "缺ADX" if not adx_ok else "缺放量" if not vol_ok else "缺大週期"
        grade_badge = f"⭐ 【A級 高品質 2/3｜{missing}】"

    # 4. 結構止損與止盈計算
    swing_low  = df["low"].iloc[i - SWING_LOOKBACK + 1 : i + 1].min()
    swing_high = df["high"].iloc[i - SWING_LOOKBACK + 1 : i + 1].max()

    if side == "BUY":
        raw_sl = swing_low - (c_atr * SL_ATR_BUFFER)
        sl = raw_sl if raw_sl < c_close else c_close - (c_atr * 1.5)
        risk = c_close - sl
        tp1, tp2 = c_close + risk * TP1_RR, c_close + risk * TP2_RR
        dir_icon = "🟢 BUY"
    else:
        raw_sl = swing_high + (c_atr * SL_ATR_BUFFER)
        sl = raw_sl if raw_sl > c_close else c_close + (c_atr * 1.5)
        risk = sl - c_close
        tp1, tp2 = c_close - risk * TP1_RR, c_close - risk * TP2_RR
        dir_icon = "🔴 SELL"

    sl_pct = (risk / c_close) * 100
    clean_sym = symbol.replace(":USDT", "")

    adx_mark = "✅" if adx_ok else "❌"
    vol_mark = "✅" if vol_ok else "❌"
    htf_mark = "✅" if htf_ok else "❌"

    msg = (
        f"{grade_badge}\n"
        f"[{timeframe.upper()}] {dir_icon} **{clean_sym}** (OKX {MARKET_TYPE})\n"
        f"進場現價：{fmt_price(c_close)}\n"
        f"結構止損 SL：{fmt_price(sl)} (-{sl_pct:.2f}%)\n"
        f"止盈目標 TP1 (1:{TP1_RR})：{fmt_price(tp1)}\n"
        f"止盈目標 TP2 (1:{TP2_RR})：{fmt_price(tp2)}\n"
        f"品質明細：ADX {c_adx:.1f}{adx_mark} | 量比 {c_vol_r:.1f}x{vol_mark} | {htf_str}{htf_mark} | RSI {c_rsi:.1f}"
    )
    send_notification(msg)
    return True


def get_target_timeframes() -> list:
    """依目前 UTC 時間自動分配要掃描的週期"""
    now_utc = datetime.now(timezone.utc)
    tfs = ["5m"]
    if now_utc.minute % 15 < 5:
        tfs.append("15m")
    if now_utc.minute < 5:
        tfs.append("1h")
    return tfs


if __name__ == "__main__":
    start_t = time.monotonic()
    target_tfs = get_target_timeframes()
    symbols = get_all_okx_symbols()
    print(f"本輪掃描週期：{target_tfs} | 共 {len(symbols)} 個 OKX {MARKET_TYPE} 幣種 (>= {MIN_24H_VOL_USDT:,} USDT)")

    total_hits = 0
    for tf in target_tfs:
        tf_start = time.monotonic()
        hits = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(analyze_symbol, sym, tf) for sym in symbols]
            for fut in as_completed(futures):
                if fut.result():
                    hits += 1
        total_hits += hits
        print(f"  -> [{tf}] 掃描完成：觸發 {hits} 個 S/A 級訊號，耗時 {time.monotonic() - tf_start:.1f} 秒。")

    print(f"全部完成！總計觸發 {total_hits} 個訊號，總耗時 {time.monotonic() - start_t:.1f} 秒。")
