import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ccxt
import numpy as np
import pandas as pd
import requests

# ==========================================
# 1. 從 GitHub Secrets 安全讀取通知設定
# ==========================================
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# ==========================================
# 2. 全幣種掃描設定 (OKX 公開 API，免 API Key)
# ==========================================
MARKET_TYPE      = "SWAP"    # "SWAP" = 全部 USDT 永續合約；"SPOT" = 全部 USDT 現貨
MIN_24H_VOL_USDT = 5000000   # 最低 24H 成交額過濾 (500萬 USDT，避開一字線冷門合約；設 0 為全掃)
MAX_WORKERS      = 5
MIN_REQ_INTERVAL = 0.08      # 約每秒 12.5 次請求，避免觸發 OKX 限流

# ==========================================
# 3. 7-in-1 Smart v3 參數設定 (與 TradingView 同步)
# ==========================================
TIMEFRAME      = "15m"
USE_ADX_FILTER = True
ADX_THRESH     = 15.0
USE_VOL_FILTER = True
VOL_MULT       = 1.2
USE_HTF_FILTER = True
HTF_MODE       = "1H"        # 可選 "1H", "4H", "1H + 4H"

SWING_LOOKBACK = 10
SL_ATR_BUFFER  = 0.5
TP1_RR         = 1.5
TP2_RR         = 2.0

exchange = ccxt.okx({"enableRateLimit": False, "timeout": 15000})
rate_lock = threading.Lock()
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
    print(f"\n{message}\n" + "-" * 45)
    if DISCORD_WEBHOOK_URL:
        try:
            requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)
        except Exception as e:
            print(f"[Discord 推播失敗] {e}")

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
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

    # 無波動一字線補 0，確保 ADX 正常衰減並對齊 TradingView
    plus_di = (100 * calc_rma(plus_dm, length) / atr.replace(0, np.nan)).fillna(0)
    minus_di = (100 * calc_rma(minus_dm, length) / atr.replace(0, np.nan)).fillna(0)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = calc_rma(dx, length)
    return atr, adx


def check_htf_trend(symbol: str, current_price: float, side: str):
    results = {}
    tfs = ["1h"] if HTF_MODE == "1H" else ["4h"] if HTF_MODE == "4H" else ["1h", "4h"]
    all_pass = True
    for tf in tfs:
        df_htf = fetch_df(symbol, tf, limit=250)
        if df_htf is None or len(df_htf) < 200:
            results[tf] = f"{tf.upper()}無"
            continue
        ema200 = df_htf["close"].ewm(span=200, adjust=False).mean().iloc[-1]
        is_bull = current_price > ema200
        results[tf] = f"{tf.upper()}多" if is_bull else f"{tf.upper()}空"
        if (side == "BUY" and not is_bull) or (side == "SELL" and is_bull):
            all_pass = False
    return (True if not USE_HTF_FILTER else all_pass), " / ".join(results.values())


def analyze_symbol(symbol: str) -> bool:
    df = fetch_df(symbol, TIMEFRAME, limit=250)
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

    i = -2  # 取上一根剛收盤的 15m K 棒
    c_close, c_ema200 = close.iloc[i], ema200.iloc[i]
    c_rsi, c_adx, c_atr, c_vol_ma = rsi.iloc[i], adx.iloc[i], atr.iloc[i], vol_sma.iloc[i]

    if any(pd.isna(v) for v in [c_close, c_ema200, c_rsi, c_adx, c_atr, c_vol_ma]) or c_close <= 0:
        return False

    c_vol_r = df["volume"].iloc[i] / c_vol_ma if c_vol_ma > 0 else 0.0
    ema_cross_up = (ema9.iloc[i] > ema21.iloc[i]) and (ema9.iloc[i - 1] <= ema21.iloc[i - 1])
    ema_cross_dn = (ema9.iloc[i] < ema21.iloc[i]) and (ema9.iloc[i - 1] >= ema21.iloc[i - 1])
    macd_bull = (macd_line.iloc[i] > sig_line.iloc[i]) and (hist_line.iloc[i] > 0)
    macd_bear = (macd_line.iloc[i] < sig_line.iloc[i]) and (hist_line.iloc[i] < 0)

    adx_pass = (not USE_ADX_FILTER) or (c_adx >= ADX_THRESH)
    vol_pass = (not USE_VOL_FILTER) or (c_vol_r >= VOL_MULT)

    side = None
    if c_close > c_ema200 and ema_cross_up and macd_bull and c_rsi < 70 and adx_pass and vol_pass:
        side = "BUY"
    elif c_close < c_ema200 and ema_cross_dn and macd_bear and c_rsi > 30 and adx_pass and vol_pass:
        side = "SELL"

    if not side:
        return False

    htf_pass, htf_str = check_htf_trend(symbol, c_close, side)
    if not htf_pass:
        return False

    swing_low  = df["low"].iloc[i - SWING_LOOKBACK + 1 : i + 1].min()
    swing_high = df["high"].iloc[i - SWING_LOOKBACK + 1 : i + 1].max()

    if side == "BUY":
        raw_sl = swing_low - (c_atr * SL_ATR_BUFFER)
        sl = raw_sl if raw_sl < c_close else c_close - (c_atr * 1.5)
        risk = c_close - sl
        tp1, tp2 = c_close + risk * TP1_RR, c_close + risk * TP2_RR
        emoji = "🟢 【多頭共振 BUY】"
    else:
        raw_sl = swing_high + (c_atr * SL_ATR_BUFFER)
        sl = raw_sl if raw_sl > c_close else c_close + (c_atr * 1.5)
        risk = sl - c_close
        tp1, tp2 = c_close - risk * TP1_RR, c_close - risk * TP2_RR
        emoji = "🔴 【空頭共振 SELL】"

    sl_pct = (risk / c_close) * 100
    clean_sym = symbol.replace(":USDT", "")

    msg = (
        f"{emoji} {clean_sym} (OKX {MARKET_TYPE} 15m)\n"
        f"進場現價：{fmt_price(c_close)}\n"
        f"結構止損 SL：{fmt_price(sl)} (-{sl_pct:.2f}%)\n"
        f"止盈目標 TP1 (1:{TP1_RR})：{fmt_price(tp1)}\n"
        f"止盈目標 TP2 (1:{TP2_RR})：{fmt_price(tp2)}\n"
        f"濾網數據：ADX {c_adx:.1f} | 量比 {c_vol_r:.1f}x | RSI {c_rsi:.1f} | {htf_str}"
    )
    send_notification(msg)
    return True


if __name__ == "__main__":
    start_t = time.monotonic()
    symbols = get_all_okx_symbols()
    print(f"共取得 {len(symbols)} 個 OKX {MARKET_TYPE} 幣種 (24H 成交額 >= {MIN_24H_VOL_USDT:,} USDT)，開始並行掃描...")

    hit_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(analyze_symbol, sym) for sym in symbols]
        for fut in as_completed(futures):
            if fut.result():
                hit_count += 1

    print(f"掃描完成！共掃描 {len(symbols)} 幣、觸發 {hit_count} 個訊號，耗時 {time.monotonic() - start_t:.1f} 秒。")
