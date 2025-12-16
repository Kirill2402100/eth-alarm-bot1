#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import requests
import yaml
from tqdm import tqdm


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_fx_symbol(sym: str) -> str:
    """
    Для TwelveData Forex часто нужен формат 'EUR/USD'.
    'EURUSD' -> 'EUR/USD'
    """
    s = (sym or "").strip()
    if "/" in s:
        return s
    if len(s) == 6 and s.isalpha():
        return f"{s[:3]}/{s[3:]}"
    return s


def normalize_interval(tf: str) -> str:
    """
    Приводим к интервалам TwelveData:
      5m -> 5min
      1h -> 1h
    """
    s = (tf or "").strip().lower()
    mapping = {
        "5m": "5min",
        "5min": "5min",
        "1h": "1h",
        "60min": "1h",
    }
    return mapping.get(s, s)


def _is_daily_limit(msg: str) -> bool:
    m = (msg or "").lower()
    return "run out of api credits for the day" in m or "current limit being" in m and "day" in m


def _is_minute_limit(msg: str) -> bool:
    m = (msg or "").lower()
    return "current minute" in m or "limit for the current minute" in m


def _sleep_to_next_minute() -> None:
    now = datetime.now(timezone.utc)
    nxt = (now + timedelta(minutes=1)).replace(second=2, microsecond=0)
    sec = max(2, int((nxt - now).total_seconds()))
    print(f"[TD] minute limit hit -> sleeping {sec}s to next minute...", flush=True)
    time.sleep(sec)


@dataclass
class TDConfig:
    api_key: str
    base_url: str = "https://api.twelvedata.com/time_series"
    search_url: str = "https://api.twelvedata.com/symbol_search"


def td_request_json(url: str, params: Dict[str, Any], max_retries: int = 8) -> Dict[str, Any]:
    backoff = 2
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            data = r.json()

            if isinstance(data, dict) and data.get("status") == "error":
                msg = str(data.get("message") or "")
                # ВАЖНО: daily-limit нельзя “переждать” ретраями — только жжём кредиты
                if _is_daily_limit(msg):
                    raise RuntimeError(f"TwelveData DAILY limit: {msg}")
                # minute-limit можно переждать до следующей минуты
                if _is_minute_limit(msg):
                    _sleep_to_next_minute()
                    raise RuntimeError(f"TwelveData MINUTE limit: {msg}")
                raise RuntimeError(f"TwelveData error: {msg}")

            return data
        except Exception as e:
            last_err = e

            # Если это daily limit — НЕ ретраим
            if "DAILY limit" in str(e):
                raise

            if attempt == max_retries:
                break

            print(f"[TD] attempt {attempt}/{max_retries} failed: {e}. sleep {backoff}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

    raise RuntimeError(f"[TD] failed after retries: {last_err}") from last_err


def td_symbol_search(cfg: TDConfig, query: str) -> Optional[str]:
    """
    Если прямой symbol не проходит — попробуем symbol_search.
    Возвращаем строку symbol вида 'EUR/USD' если нашли.
    """
    params = {
        "symbol": query,
        "apikey": cfg.api_key,
        "outputsize": 10,
    }
    try:
        data = td_request_json(cfg.search_url, params, max_retries=3)
        items = data.get("data") or []
        for it in items:
            sym = (it.get("symbol") or "").strip()
            exch = (it.get("exchange") or "").lower()
            if sym and ("/" in sym) and ("forex" in exch or "fx" in exch):
                return sym
        for it in items:
            sym = (it.get("symbol") or "").strip()
            if sym and "/" in sym:
                return sym
    except Exception:
        return None
    return None


def td_fetch_timeseries(cfg: TDConfig, symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": cfg.api_key,
        "format": "JSON",
        "outputsize": 5000,
        "timezone": "UTC",
        "order": "ASC",
        "start_date": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
    }

    data = td_request_json(cfg.base_url, params=params)
    values = data.get("values") or []
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.sort_values("datetime").drop_duplicates("datetime")
    return df


def _load_partial(partial_path: Path) -> pd.DataFrame:
    try:
        if partial_path.exists() and partial_path.stat().st_size > 0:
            df = pd.read_parquet(partial_path)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                df = df.sort_values("datetime").drop_duplicates("datetime")
                return df
    except Exception:
        pass
    return pd.DataFrame()


def fetch_range_in_windows(
    cfg: TDConfig,
    symbol: str,
    interval: str,
    years: int,
    window_days: int,
    partial_path: Optional[Path] = None,
) -> pd.DataFrame:
    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=int(years * 365))

    existing = pd.DataFrame()
    if partial_path is not None:
        existing = _load_partial(partial_path)
        if not existing.empty:
            last_dt = existing["datetime"].max()
            # продолжаем чуть дальше последней точки, чтобы не дублировать окно
            start_dt = max(start_dt, (last_dt + timedelta(minutes=1)).replace(second=0, microsecond=0))
            print(f"[DATA] resume from partial: {partial_path} (last_dt={last_dt})", flush=True)

    windows: List[Tuple[datetime, datetime]] = []
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + timedelta(days=window_days), end_dt)
        windows.append((cur, nxt))
        cur = nxt

    parts: List[pd.DataFrame] = []
    if not existing.empty:
        parts.append(existing)

    for a, b in tqdm(windows, desc=f"TD {symbol} {interval}"):
        part = td_fetch_timeseries(cfg, symbol, interval, a, b)
        if not part.empty:
            parts.append(part)

        # сохраняем прогресс после каждого окна
        if partial_path is not None and parts:
            df_tmp = pd.concat(parts, ignore_index=True).sort_values("datetime").drop_duplicates("datetime")
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            df_tmp.to_parquet(partial_path, index=False)
            print(f"Saved: {partial_path}", flush=True)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
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

    provider = cfg.get("provider") or {}
    api_env = str(provider.get("api_key_env") or "TWELVEDATA_API_KEY")
    api_key = os.getenv(api_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing env var {api_env} (TwelveData API key)")

    td_cfg = TDConfig(api_key=api_key)

    data_cfg = cfg.get("data") or {}
    years = int(data_cfg.get("years", 5))
    data_dir = Path(str(data_cfg.get("data_dir") or "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)

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

    symbol_td = normalize_fx_symbol(symbol_raw)
    interval_5m = normalize_interval("5m")
    interval_1h = normalize_interval("1h")

    print(f"[TD] symbol_raw={symbol_raw} -> symbol_td={symbol_td}", flush=True)
    print(f"[TD] intervals: 5m -> {interval_5m}, 1h -> {interval_1h}", flush=True)

    try_symbols = [symbol_td, symbol_raw]
    found = td_symbol_search(td_cfg, symbol_raw)
    if found and found not in try_symbols:
        try_symbols.insert(0, found)

    last_err = None
    for sym in try_symbols:
        try:
            partial5 = data_dir / f".partial_{bars_5m}"
            partial1 = data_dir / f".partial_{bars_1h}"

            df5 = fetch_range_in_windows(td_cfg, sym, interval_5m, years=years, window_days=30, partial_path=partial5)
            if df5.empty:
                raise RuntimeError("downloaded 5min dataframe is empty")
            save_parquet(df5, p5)

            df1 = fetch_range_in_windows(td_cfg, sym, interval_1h, years=years, window_days=180, partial_path=partial1)
            if df1.empty:
                raise RuntimeError("downloaded 1h dataframe is empty")
            save_parquet(df1, p1)

            # можно подчистить partial
            try:
                if partial5.exists():
                    partial5.unlink()
                if partial1.exists():
                    partial1.unlink()
            except Exception:
                pass

            print("[DATA] Built:", flush=True)
            print(f"  {p5}", flush=True)
            print(f"  {p1}", flush=True)
            return
        except Exception as e:
            last_err = e
            print(f"[TD] symbol variant '{sym}' failed: {e}", flush=True)

    raise RuntimeError(f"All symbol variants failed. Last error: {last_err}") from last_err


if __name__ == "__main__":
    main()
