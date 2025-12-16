# scripts/build_dataset.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
from tqdm import tqdm

# TwelveData интервалы
TD_INTERVALS = {
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1day",
}

def normalize_twelvedata_symbol(sym: str) -> str:
    """
    TwelveData для FX обычно ожидает формат "EUR/USD", "GBP/JPY", "XAU/USD" и т.п.
    У тебя в конфиге удобнее держать "EURUSD", поэтому нормализуем на лету.
    """
    s = (sym or "").strip().upper()
    if "/" in s:
        return s
    # EURUSD -> EUR/USD
    if len(s) == 6 and s.isalpha():
        return f"{s[:3]}/{s[3:]}"
    # XAUUSD / XAGUSD -> XAU/USD (3+3)
    if len(s) == 6 and s[:3].isalpha() and s[3:].isalpha():
        return f"{s[:3]}/{s[3:]}"
    return s

@dataclass
class TDConfig:
    api_key: str
    symbol: str
    interval: str
    start: datetime
    end: datetime

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
        raise RuntimeError(f"TwelveData error: {data.get('message')}")

    rows = (data or {}).get("values", [])
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows)

    # TwelveData обычно отдаёт "datetime"
    if "datetime" not in df.columns:
        raise RuntimeError("Unexpected TwelveData response: no datetime column")

    df["time"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()

    # Приводим OHLCV
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[["open", "high", "low", "close", "volume"]].dropna(how="any")
    return df

def choose_chunk_days(interval: str) -> int:
    """
    TwelveData лимитирует количество точек на запрос (часто 5000).
    Выбираем размер чанка так, чтобы примерно не превышать лимит.
    """
    if interval == "5m":
        return 14    # 14*288 = 4032
    if interval == "15m":
        return 45    # 45*96 = 4320 (безопаснее, чем 60)
    if interval == "1h":
        return 180   # 180*24 = 4320
    if interval == "4h":
        return 700   # 700*6 = 4200
    return 365

def fetch_twelvedata_to_parquet(api_key: str, symbol_raw: str, interval: str, years: int, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=365 * years)

    symbol_td = normalize_twelvedata_symbol(symbol_raw)
    cfg = TDConfig(api_key=api_key, symbol=symbol_td, interval=interval, start=start, end=end)

    chunk_days = choose_chunk_days(interval)
    cur = start
    frames: list[pd.DataFrame] = []

    steps = max(1, ((end - start).days // chunk_days) + 1)
    pbar = tqdm(total=steps, desc=f"TD {symbol_raw} ({symbol_td}) {interval}")

    while cur < end:
        nxt = min(end, cur + timedelta(days=chunk_days))
        df = td_fetch_chunk(cfg, cur, nxt)
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

    if not os.path.exists(p5):
        fetch_twelvedata_to_parquet(api_key, symbol, "5m", years, p5)
        print("Saved:", p5)
    else:
        print("Exists:", p5)

    if not os.path.exists(p1):
        fetch_twelvedata_to_parquet(api_key, symbol, "1h", years, p1)
        print("Saved:", p1)
    else:
        print("Exists:", p1)

if __name__ == "__main__":
    main()
