#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

from sim.broker_fx import BrokerFX
from sim.engine import run_strategy


SWEEP_KEYS = {
    "strategic_lookback_days",
    "tactical_lookback_days",
    "q_lower",
    "q_upper",
    "range_min_atr_mult",
    "dca_levels",
    "dca_growth",
    "bank_usd",
    "cum_deposit_frac_at_full",
    "tp_pct",
}


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_repo_on_syspath() -> Path:
    app_root = Path(__file__).resolve().parents[1]
    os.chdir(app_root)
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    return app_root


def list_data_dir(data_dir: Path) -> None:
    print("=== DEBUG ===", flush=True)
    print(f"cwd: {os.getcwd()}", flush=True)
    print(f"APP_ROOT: {Path(__file__).resolve().parents[1]}", flush=True)
    print(f"data_dir: {data_dir}", flush=True)
    print("ls -lah /data:", flush=True)
    try:
        if data_dir.exists():
            items = sorted(list(data_dir.iterdir()))
            if not items:
                print("  (empty)", flush=True)
            for x in items:
                if x.is_dir():
                    print(f"  {x.name}/", flush=True)
                else:
                    mb = x.stat().st_size / (1024 * 1024)
                    print(f"  {x.name:30s} {mb:6.2f} MB", flush=True)
        else:
            print("  (missing dir)", flush=True)
    except Exception as e:
        print(f"  (list failed: {e})", flush=True)
    print("=============", flush=True)


def ensure_data(cfg_path: Path, cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    data_cfg = cfg.get("data") or {}
    symbol = str(cfg.get("symbol") or "").strip()

    data_dir = Path(str(data_cfg.get("data_dir") or "/data"))
    bars_5m = str(data_cfg.get("bars_5m") or f"{symbol}_5m.parquet")
    bars_1h = str(data_cfg.get("bars_1h") or f"{symbol}_1h.parquet")

    p5 = data_dir / bars_5m
    p1 = data_dir / bars_1h

    list_data_dir(data_dir)

    if p5.exists() and p1.exists():
        print("[DATA] Using cached files:", flush=True)
        print(f"  {p5}", flush=True)
        print(f"  {p1}", flush=True)
        return p5, p1

    print("[DATA] Missing parquet(s), building dataset now...", flush=True)
    subprocess.check_call([sys.executable, "-u", "-m", "scripts.build_dataset", "--config", str(cfg_path)])
    list_data_dir(data_dir)

    if not (p5.exists() and p1.exists()):
        raise RuntimeError(f"[DATA] build_dataset finished but files still missing: {p5}, {p1}")

    print("[DATA] Built:", flush=True)
    print(f"  {p5}", flush=True)
    print(f"  {p1}", flush=True)
    return p5, p1


def build_grid(strategy_cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Tuple[str, List[Any]]]]:
    base: Dict[str, Any] = {}
    sweeps: List[Tuple[str, List[Any]]] = []

    for k, v in (strategy_cfg or {}).items():
        if k in SWEEP_KEYS and isinstance(v, list):
            sweeps.append((k, v))
        else:
            base[k] = v

    return base, sweeps


def iter_product(sweeps: List[Tuple[str, List[Any]]]) -> List[Dict[str, Any]]:
    if not sweeps:
        return [{}]

    # вручную без itertools.product, чтобы проще отлаживать
    out = [{}]
    for (k, vals) in sweeps:
        new_out = []
        for cur in out:
            for x in vals:
                d = dict(cur)
                d[k] = x
                new_out.append(d)
        out = new_out
    return out


def call_run_strategy_compat(
    df_entry: pd.DataFrame,
    df_range: pd.DataFrame,
    broker: BrokerFX,
    symbol: str,
    params: Dict[str, Any],
    out_dir: Path,
    save_trades: bool,
) -> Any:
    """
    Универсальный вызов run_strategy, который не ломается от различий в сигнатуре.
    - Если run_strategy принимает df_entry/df_range — передадим так
    - Если принимает entry_df/range_df/bars_* — подберём
    - Если принимает отдельные параметры (tp_pct, q_lower...) — тоже подставим
    """
    sig = inspect.signature(run_strategy)
    kwargs: Dict[str, Any] = {}

    # 1) Сначала — "сервисные" штуки
    name_map_entry = {"df_entry", "entry_df", "bars_entry", "bars_5m", "df_5m", "entry"}
    name_map_range = {"df_range", "range_df", "bars_range", "bars_1h", "df_1h", "range"}
    name_map_broker = {"broker", "brk"}
    name_map_symbol = {"symbol", "sym", "pair"}
    name_map_out = {"out_dir", "report_dir", "output_dir"}
    name_map_save_trades = {"save_trades", "write_trades", "dump_trades"}

    for pname, p in sig.parameters.items():
        if pname in name_map_entry:
            kwargs[pname] = df_entry
        elif pname in name_map_range:
            kwargs[pname] = df_range
        elif pname in name_map_broker:
            kwargs[pname] = broker
        elif pname in name_map_symbol:
            kwargs[pname] = symbol
        elif pname in name_map_out:
            kwargs[pname] = str(out_dir)
        elif pname in name_map_save_trades:
            kwargs[pname] = save_trades

    # 2) Потом — параметры стратегии.
    # Если run_strategy ожидает их поимённо — отлично.
    # Если у него есть **kwargs — тоже подойдёт (передадим лишнее туда).
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    for pname, p in sig.parameters.items():
        if pname in kwargs:
            continue
        if pname in params:
            kwargs[pname] = params[pname]

    if has_varkw:
        # добавим всё остальное, чего нет в явных именах
        for k, v in params.items():
            if k not in kwargs:
                kwargs[k] = v

    try:
        return run_strategy(**kwargs)
    except TypeError as e:
        # fallback: попробуем позиционно df_entry, df_range, broker, затем всё остальное как kwargs
        try:
            return run_strategy(df_entry, df_range, broker, **params)
        except Exception:
            raise TypeError(f"run_strategy call failed. signature={sig}. error={e}") from e


def main() -> None:
    ensure_repo_on_syspath()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(str(cfg_path))

    symbol = str(cfg.get("symbol") or "").strip()
    if not symbol:
        raise RuntimeError("config missing: symbol")

    # ensure data
    p5, p1 = ensure_data(cfg_path, cfg)

    # load data
    df_entry = pd.read_parquet(p5)
    df_range = pd.read_parquet(p1)

    # broker
    bcfg = cfg.get("broker") or {}
    broker = BrokerFX(
        leverage=float(bcfg.get("leverage", 200)),
        spread_points=float(bcfg.get("spread_points", 8)),
        tick=float(bcfg.get("tick", 0.00001)),
        margin_call_level=float(bcfg.get("margin_call_level", 0.20)),
        stop_out_level=float(bcfg.get("stop_out_level", 0.00)),
    )

    # strategy config
    strat = cfg.get("strategy") or {}
    base_params, sweeps = build_grid(strat)
    combos = iter_product(sweeps)

    report_cfg = cfg.get("report") or {}
    out_dir = Path(str(report_cfg.get("out_dir") or "/data/out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_trades = bool(report_cfg.get("save_trades", True))

    total = len(combos)
    print(f"[GRID] total combinations: {total}", flush=True)

    results: List[Dict[str, Any]] = []
    for i, combo in enumerate(combos, 1):
        params = dict(base_params)
        params.update(combo)

        print(f"[GRID] {i}/{total} params={params}", flush=True)

        res = call_run_strategy_compat(
            df_entry=df_entry,
            df_range=df_range,
            broker=broker,
            symbol=symbol,
            params=params,
            out_dir=out_dir,
            save_trades=save_trades,
        )

        # сохраним минимально полезное, не зная точный формат res
        row = dict(params)
        row["_i"] = i
        row["_ok"] = True
        row["_result_type"] = type(res).__name__
        results.append(row)

        # периодически сбрасываем на диск
        if i % 50 == 0:
            pd.DataFrame(results).to_csv(out_dir / "grid_progress.csv", index=False)

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "grid_results.csv", index=False)
    print(f"[GRID] done. saved: {out_dir / 'grid_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
