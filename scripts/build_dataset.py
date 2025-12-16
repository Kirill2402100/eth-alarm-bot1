# scripts/build_dataset.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
from tqdm import tqdm


# TwelveData intervals
TD_INTERVALS = {
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1day",
}


def normalize_twelvedata_symbol(sym: str) -> str:
    """
    TwelveData FX expects format like 'EUR/USD'. In config we keep 'EURUSD'.
    """
    s = (sym or "").strip().upper()
    if "/" in s:
        return s
    if len(s) == 6 and s.isalpha():
        return f"{s[:3]}/{s[3:]}"
    return s


@dataclass
class TDConfig:
    api_key: str
    symbol: str
    interval: str


def _is_rate_limit_err(msg: str) -> bool:
    m = (msg or "").lower()
    return ("run out of api credits" in m) or ("rate limit" in m) or ("too many requests" in m)


def td_fetch_chunk(cfg: TDConfig, start: datetime, end: datetime) -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": cfg.symbol,
        "interval": TD_INTERVALS[cfg.interval],
        "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "UTC",
        "order": "ASC",
        "format": "JSON",
        "apikey": cfg.api_key,
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    data = r.json()
    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(str(data.get("message", "unknown TwelveData error")))

    rows = (data or {}).get("values", [])
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows)
    if "datetime" not in df.columns:
        raise RuntimeError("Unexpected TwelveData response: no datetime column")

    df["time"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[["open", "high", "low", "close", "volume"]].dropna(how="any")
    df = df[~df.index.duplicated(keep="last")]
    return df


def choose_chunk_days(interval: str) -> int:
    """
    Keep under typical 5000 points/request.
    """
    if interval == "5m":
        return 17   # 17*288=4896 (чуть эффективнее, чем 14д)
    if interval == "15m":
        return 45   # 45*96=4320
    if interval == "1h":
        return 180  # 180*24=4320
    if interval == "4h":
        return 700  # 700*6=4200
    return 365


class MinuteRateLimiter:
    """
    Простая защита от лимита credits/min.
    Мы делаем паузу между запросами так, чтобы не превышать лимит.
    """
    def __init__(self, credits_per_minute: int = 8, safety: float = 0.85):
        self.credits_per_minute = max(1, int(credits_per_minute))
        # safety<1 => чуть реже, чем лимит
        self.min_interval = (60.0 / self.credits_per_minute) / max(safety, 0.1)
        self._last_ts = 0.0

    def wait(self):
        now = time.time()
        if self._last_ts <= 0:
            self._last_ts = now
            return
        elapsed = now - self._last_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_ts = time.time()


def fetch_twelvedata_to_parquet(
    api_key: str,
    symbol_raw: str,
    interval: str,
    years: int,
    out_path: str,
    credits_per_minute: int = 8,
) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=365 * years)

    symbol_td = normalize_twelvedata_symbol(symbol_raw)
    cfg = TDConfig(api_key=api_key, symbol=symbol_td, interval=interval)

    # Resume: if file exists, continue from last index
    if os.path.exists(out_path):
        try:
            existing = pd.read_parquet(out_path).sort_index()
            if not existing.empty and isinstance(existing.index, pd.DatetimeIndex):
                last_ts = existing.index.max()
                # чуть сдвинем вперед, чтобы не дублировать последнюю свечу
                start = max(start, (last_ts + pd.Timedelta(minutes=1)).to_pydatetime())
        except Exception:
            # если parquet битый/не читается — начнём заново
            pass

    chunk_days = choose_chunk_days(interval)
    limiter = MinuteRateLimiter(credits_per_minute=credits_per_minute, safety=0.80)

    cur = start
    frames: list[pd.DataFrame] = []

    # если есть существующие данные — подтянем их в начало
    if os.path.exists(out_path):
        try:
            existing = pd.read_parquet(out_path).sort_index()
            if not existing.empty:
                frames.append(existing)
        except Exception:
            pass

    # оценка шагов только для tqdm (не критично)
    steps = max(1, ((end - cur).days // chunk_days) + 1)
    pbar = tqdm(total=steps, desc=f"TD {symbol_raw} ({symbol_td}) {interval}")

    while cur < end:
        nxt = min(end, cur + timedelta(days=chunk_days))

        # ВАЖНО: соблюдаем темп
        limiter.wait()

        # Retry на лимиты/временные ошибки
        attempt = 0
        while True:
            try:
                df = td_fetch_chunk(cfg, cur, nxt)
                break
            except Exception as e:
                attempt += 1
                msg = str(e)
                if _is_rate_limit_err(msg):
                    # подождём до следующего “окна”
                    time.sleep(65)
                    continue
                # сетевые/прочее — несколько попыток
                if attempt <= 3:
                    time.sleep(2.0 * attempt)
                    continue
                raise

        if not df.empty:
            frames.append(df)

        cur = nxt
        pbar.update(1)

    pbar.close()

    if not frames:
        raise RuntimeError("No data received from TwelveData")

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.to_parquet(out_path, index=True)
    return out_path


def main():
    import argparse
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    api_key_env = cfg["provider"]["api_key_env"]
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key env: {api_key_env}")

    symbol = cfg["symbol"]
    years = int(cfg["data"]["years"])
    data_dir = cfg["data"]["data_dir"]

    p5 = os.path.join(data_dir, cfg["data"]["bars_5m"])
    p1 = os.path.join(data_dir, cfg["data"]["bars_1h"])

    # Можно переопределить лимит через env, если вдруг план другой
    credits_per_minute = int(os.getenv("TWELVEDATA_CREDITS_PER_MIN", "8"))

    # 5m
    fetch_twelvedata_to_parquet(api_key, symbol, "5m", years, p5, credits_per_minute=credits_per_minute)
    print("Saved:", p5)

    # 1h
    fetch_twelvedata_to_parquet(api_key, symbol, "1h", years, p1, credits_per_minute=credits_per_minute)
    print("Saved:", p1)


if __name__ == "__main__":
    main()
