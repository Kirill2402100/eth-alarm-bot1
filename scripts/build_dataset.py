#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml
from tqdm import tqdm


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_fx_symbol(sym: str) -> str:
    """
    TwelveData по FX часто любит формат 'EUR/USD'.
    Если дали 'EURUSD' (6 букв) — превратим в 'EUR/USD'.
    """
    s = (sym or "").strip()
    if "/" in s:
        return s
    if len(s) == 6 and s.isalpha():
        return f"{s[:3]}/{s[3:]}"
    return s


@dataclass
class TDConfig:
    api_key: str
    base_url: str = "https://api.twelvedata.com/time_series"


def td_request(cfg: TDConfig, params: Dict[str, Any], max_retries: int = 8) -> Dict[str, Any]:
    backoff = 2
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(cfg.base_url, params=params, timeout=30)
            data = r.json()

            # TwelveData ошибки часто кладёт в {"status":"error","message":...}
            if isinstance(data, dict) and data.get("status") == "error":
                raise RuntimeError(f"TwelveData error: {data.get('message')}")

            return data
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                break
            print(f"[TD] attempt {attempt}/{max_retries} failed: {e}. sleep {backoff}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

    raise RuntimeError(f"TwelveData request failed after retries: {last_err}") from last_err


def td_fetch_timeseries(
    cfg: TDConfig,
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    outputsize: int = 5000,
) -> pd.DataFrame:
    """
    Качаем кусками (по времени), чтобы не упираться в лимиты outputsize.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": cfg.api_key,
        "format": "JSON",
        "outputsize": outputsize,
        "start_date": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "UTC",
        "order": "ASC",
    }

    data = td_request(cfg, params=params)

    values = data.get("values") or []
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)
    # TwelveData отдаёт datetime строкой
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.sort_values("datetime").drop_duplicates("datetime")
    return df


def fetch_range_in_windows(
    cfg: TDConfig,
    symbol: str,
    interval: str,
    years: int,
    window_days: int,
) -> pd.DataFrame:
    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=int(years * 365))

    windows: List[Tuple[datetime, datetime]] = []
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + timedelta(days=window_days), end_dt)
        windows.append((cur, nxt))
        cur = nxt

    all_parts: List[pd.DataFrame] = []
    desc = f"TD {symbol} {interval}"
    for (a, b) in tqdm(windows, desc=desc):
        part = td_fetch_timeseries(cfg, symbol, interval, a, b)
        if not part.empty:
            all_parts.append(part)

    if not all_parts:
        return pd.DataFrame()

    df = pd.concat(all_parts, ignore_index=True)
    df = df.sort_values("datetime").drop_duplicates("datetime")
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved: {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(str(cfg_path))

    symbol_raw = str(cfg.get("symbol") or "").strip()
    if not symbol_raw:
        raise RuntimeError("config missing: symbol")

    # provider
    provider = cfg.get("provider") or {}
    api_env = str(provider.get("api_key_env") or "TWELVEDATA_API_KEY")
    api_key = os.getenv(api_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing env var {api_env} (TwelveData API key)")

    td_cfg = TDConfig(api_key=api_key)

    # data
    data_cfg = cfg.get("data") or {}
    years = int(data_cfg.get("years", 5))
    data_dir = Path(str(data_cfg.get("data_dir") or "/data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    symbol_td = normalize_fx_symbol(symbol_raw)

    bars_5m = str(data_cfg.get("bars_5m") or f"{symbol_raw}_5m.parquet")
    bars_1h = str(data_cfg.get("bars_1h") or f"{symbol_raw}_1h.parquet")

    p5 = data_dir / bars_5m
    p1 = data_dir / bars_1h

    # если уже есть — не качаем
    if p5.exists() and p1.exists():
        print("[DATA] Already exists, skip download:", flush=True)
        print(f" {p5}", flush=True)
        print(f" {p1}", flush=True)
        return

    # 5m — окно помельче, 1h — можно крупнее
    df5 = fetch_range_in_windows(td_cfg, symbol_td, "5min", years=years, window_days=30)
    if df5.empty:
        raise RuntimeError("Downloaded 5min dataframe is empty (check symbol / API key / limits)")
    save_parquet(df5, p5)

    df1 = fetch_range_in_windows(td_cfg, symbol_td, "1h", years=years, window_days=180)
    if df1.empty:
        raise RuntimeError("Downloaded 1h dataframe is empty (check symbol / API key / limits)")
    save_parquet(df1, p1)

    print("[DATA] Built:", flush=True)
    print(f"  {p5}", flush=True)
    print(f"  {p1}", flush=True)


if __name__ == "__main__":
    main()
