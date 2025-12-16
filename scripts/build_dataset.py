#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import requests
import yaml
from tqdm import tqdm


# ----------------------------
# Config helpers
# ----------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)


def file_ok(path: Path, min_bytes: int = 1024) -> bool:
    # простая проверка "файл не пустой"
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


# ----------------------------
# TwelveData client
# ----------------------------

@dataclass
class TDConfig:
    api_key: str
    symbol: str
    interval: str
    start: str  # YYYY-MM-DD
    end: str    # YYYY-MM-DD
    outputsize: int = 5000
    timezone: str = "UTC"
    retries: int = 8


TD_URL = "https://api.twelvedata.com/time_series"


def _td_sleep_for_rate_limit(msg: str) -> int:
    """
    TwelveData пишет:
    'You have run out of API credits for the current minute...'
    Спим до следующей минуты + небольшой буфер.
    """
    if "current minute" in (msg or "").lower() or "credits" in (msg or "").lower():
        now = datetime.now(timezone.utc)
        # до следующей минуты + 2 сек
        nxt = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
        return max(5, int((nxt - now).total_seconds()) + 2)
    return 0


def td_fetch_chunk(cfg: TDConfig) -> pd.DataFrame:
    params = {
        "symbol": cfg.symbol,
        "interval": cfg.interval,
        "apikey": cfg.api_key,
        "start_date": cfg.start,
        "end_date": cfg.end,
        "outputsize": cfg.outputsize,
        "timezone": cfg.timezone,
        "format": "JSON",
        "order": "ASC",
    }

    last_err = None
    for attempt in range(1, cfg.retries + 1):
        try:
            r = requests.get(TD_URL, params=params, timeout=40)
            data = r.json()

            if isinstance(data, dict) and data.get("status") == "error":
                msg = str(data.get("message") or data)
                # rate limit
                slp = _td_sleep_for_rate_limit(msg)
                if slp > 0:
                    print(f"[TD] Rate limit: sleep {slp}s. msg={msg}", flush=True)
                    time.sleep(slp)
                    continue
                raise RuntimeError(f"TwelveData error: {msg}")

            values = (data or {}).get("values") or []
            if not values:
                # иногда бывают пустые окна — просто возвращаем пустой DF
                return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

            df = pd.DataFrame(values)
            # TwelveData отдаёт строки
            df.rename(columns={"datetime": "datetime"}, inplace=True)
            for c in ["open", "high", "low", "close", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
            df = df.sort_values("datetime").reset_index(drop=True)
            return df

        except Exception as e:
            last_err = e
            backoff = min(60, 2 ** attempt)
            print(f"[TD] attempt {attempt}/{cfg.retries} failed: {e}. sleep {backoff}s", flush=True)
            time.sleep(backoff)

    raise RuntimeError(f"[TD] failed after retries: {last_err}")


def year_windows(years: int) -> List[Tuple[datetime, datetime]]:
    """
    Делаем окна по ~30 дней (чтобы не упираться в лимиты outputsize/объёмы).
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(years * 365.25))
    cur = start
    out = []
    step = timedelta(days=30)
    while cur < end:
        nxt = min(end, cur + step)
        out.append((cur, nxt))
        cur = nxt
    return out


def fetch_twelvedata_to_parquet(
    api_key: str,
    symbol: str,
    interval: str,
    years: int,
    out_path: Path,
    force: bool = False,
) -> Path:
    if file_ok(out_path) and not force:
        print(f"[DATA] Cached exists, skip: {out_path}", flush=True)
        return out_path

    ensure_dir(out_path.parent)

    windows = year_windows(years)
    dfs = []

    label = f"TD {symbol} {interval}"
    for (a, b) in tqdm(windows, desc=label):
        cfg = TDConfig(
            api_key=api_key,
            symbol=symbol,
            interval=interval,
            start=a.date().isoformat(),
            end=b.date().isoformat(),
            outputsize=5000,
        )
        df = td_fetch_chunk(cfg)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise RuntimeError(f"[DATA] No data downloaded for {symbol} {interval}")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    atomic_write_parquet(df_all, out_path)
    print(f"Saved: {out_path}", flush=True)
    return out_path


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="rebuild even if cached files exist")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(str(cfg_path))

    symbol = str(cfg.get("symbol") or "").strip()
    if not symbol:
        raise RuntimeError("config missing: symbol")

    data_cfg = cfg.get("data") or {}
    years = int(data_cfg.get("years") or 5)
    data_dir = Path(str(data_cfg.get("data_dir") or "/data"))

    bars_5m = str(data_cfg.get("bars_5m") or f"{symbol}_5m.parquet")
    bars_1h = str(data_cfg.get("bars_1h") or f"{symbol}_1h.parquet")

    provider = cfg.get("provider") or {}
    api_key_env = str(provider.get("api_key_env") or "TWELVEDATA_API_KEY")
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing env var {api_key_env} with TwelveData api key")

    p5 = data_dir / bars_5m
    p1 = data_dir / bars_1h

    print("=== DEBUG ===", flush=True)
    print(f"cwd: {os.getcwd()}", flush=True)
    print(f"APP_ROOT: {Path(__file__).resolve().parents[1]}", flush=True)
    print(f"data_dir: {data_dir}", flush=True)
    try:
        if data_dir.exists():
            print("ls -lah /data:", flush=True)
            for x in sorted(data_dir.iterdir()):
                if x.is_dir():
                    print(f"  {x.name}/", flush=True)
                else:
                    mb = x.stat().st_size / (1024 * 1024)
                    print(f"  {x.name:30s} {mb:6.2f} MB", flush=True)
    except Exception as e:
        print(f"[WARN] list data_dir failed: {e}", flush=True)
    print("=============", flush=True)

    fetch_twelvedata_to_parquet(api_key, symbol, "5m", years, p5, force=args.force)
    fetch_twelvedata_to_parquet(api_key, symbol, "1h", years, p1, force=args.force)

    print("=============", flush=True)
    print("[DATA] Built:", flush=True)
    print(f"  {p5}", flush=True)
    print(f"  {p1}", flush=True)


if __name__ == "__main__":
    main()
