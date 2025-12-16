#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
import yaml
from tqdm import tqdm


# ----------------------------
# Helpers
# ----------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_parquet(df: pd.DataFrame, out_path: Path):
    ensure_dir(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)


def file_ok(path: Path, min_bytes: int = 1024) -> bool:
    return path.exists() and path.stat().st_size >= min_bytes


# ----------------------------
# TwelveData client
# ----------------------------

@dataclass
class TDConfig:
    api_key: str
    symbol: str
    interval: str
    start: str
    end: str
    retries: int = 8


TD_URL = "https://api.twelvedata.com/time_series"


def _td_sleep_for_rate_limit(msg: str) -> int:
    if "current minute" in (msg or "").lower():
        now = datetime.now(timezone.utc)
        nxt = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
        return int((nxt - now).total_seconds()) + 2
    return 0


def td_fetch_chunk(cfg: TDConfig) -> pd.DataFrame:
    params = {
        "symbol": cfg.symbol,
        "interval": cfg.interval,
        "apikey": cfg.api_key,
        "start_date": cfg.start,
        "end_date": cfg.end,
        "outputsize": 5000,
        "timezone": "UTC",
        "format": "JSON",
        "order": "ASC",
    }

    last_err = None
    for attempt in range(1, cfg.retries + 1):
        try:
            r = requests.get(TD_URL, params=params, timeout=40)
            data = r.json()

            if data.get("status") == "error":
                msg = data.get("message", "")
                if "invalid" in msg.lower() and "/" not in cfg.symbol:
                    alt = cfg.symbol[:3] + "/" + cfg.symbol[3:]
                    print(f"[TD] symbol '{cfg.symbol}' invalid → retry with '{alt}'", flush=True)
                    cfg.symbol = alt
                    params["symbol"] = alt
                    continue
                slp = _td_sleep_for_rate_limit(msg)
                if slp > 0:
                    print(f"[TD] rate limit → sleep {slp}s", flush=True)
                    time.sleep(slp)
                    continue
                raise RuntimeError(f"TwelveData error: {msg}")

            values = data.get("values", [])
            if not values:
                return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
            return df

        except Exception as e:
            last_err = e
            backoff = min(60, 2 ** attempt)
            print(f"[TD] attempt {attempt}/{cfg.retries} failed: {e}. sleep {backoff}s", flush=True)
            time.sleep(backoff)
    raise RuntimeError(f"[TD] failed after retries: {last_err}")


def year_windows(years: int) -> List[tuple]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(years * 365.25))
    cur = start
    step = timedelta(days=30)
    out = []
    while cur < end:
        nxt = min(end, cur + step)
        out.append((cur, nxt))
        cur = nxt
    return out


def fetch_twelvedata_to_parquet(api_key, symbol, interval, years, out_path):
    ensure_dir(out_path.parent)
    if file_ok(out_path):
        print(f"[DATA] Cached exists → {out_path}")
        return out_path

    dfs = []
    for (a, b) in tqdm(year_windows(years), desc=f"TD {symbol} {interval}"):
        df = td_fetch_chunk(TDConfig(api_key, symbol, interval, a.date().isoformat(), b.date().isoformat()))
        if not df.empty:
            dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    atomic_write_parquet(df_all, out_path)
    print(f"Saved: {out_path}")
    return out_path


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    symbol = cfg.get("symbol", "EUR/USD")
    data_cfg = cfg.get("data", {})
    years = int(data_cfg.get("years", 5))
    data_dir = Path(data_cfg.get("data_dir", "/data"))
    ensure_dir(data_dir)

    api_key = os.getenv(cfg.get("provider", {}).get("api_key_env", "TWELVEDATA_API_KEY"), "")
    if not api_key:
        raise RuntimeError("Missing TwelveData API key")

    bars_5m = data_dir / data_cfg.get("bars_5m", f"{symbol.replace('/', '')}_5m.parquet")
    bars_1h = data_dir / data_cfg.get("bars_1h", f"{symbol.replace('/', '')}_1h.parquet")

    print("=== DEBUG ===")
    print(f"cwd: {os.getcwd()}")
    print(f"APP_ROOT: /app")
    print(f"data_dir: {data_dir}")
    if not data_dir.exists():
        print("(creating /data...)")
    print("=============")

    fetch_twelvedata_to_parquet(api_key, symbol, "5m", years, bars_5m)
    fetch_twelvedata_to_parquet(api_key, symbol, "1h", years, bars_1h)
    print("[DATA] Built OK.")


if __name__ == "__main__":
    main()
