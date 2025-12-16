# scripts/run_grid.py
from __future__ import annotations

import os
import sys
import itertools
import subprocess
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import yaml
import pandas as pd

# --- FIX: ensure /app is on sys.path so `import sim...` works even if cwd != /app
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from sim.broker_fx import BrokerFX
from sim.engine import run_strategy
from sim.metrics import compute_equity_stats


def _print_data_dir_state(data_dir: str) -> None:
    print("=== DEBUG ===")
    print("cwd:", os.getcwd())
    print("APP_ROOT:", APP_ROOT)
    print("data_dir:", data_dir)
    try:
        if os.path.isdir(data_dir):
            print(f"ls -lah {data_dir}:")
            for name in sorted(os.listdir(data_dir)):
                p = os.path.join(data_dir, name)
                st = os.stat(p)
                print(f"  {name:30s}  {st.st_size/1024/1024:8.2f} MB")
        else:
            print(f"{data_dir} does not exist / not a dir")
    except Exception as e:
        print("list /data failed:", repr(e))
    print("=============")


def ensure_data(cfg_path: str, cfg: dict) -> Tuple[str, str]:
    data_dir = cfg["data"]["data_dir"]
    p5 = os.path.join(data_dir, cfg["data"]["bars_5m"])
    p1 = os.path.join(data_dir, cfg["data"]["bars_1h"])

    os.makedirs(data_dir, exist_ok=True)
    _print_data_dir_state(data_dir)

    # если уже есть оба файла — используем
    if os.path.exists(p5) and os.path.exists(p1):
        print(f"[DATA] Using cached files:\n  {p5}\n  {p1}")
        return p5, p1

    print("[DATA] Missing parquet(s), building dataset now...")
    subprocess.check_call([sys.executable, "-m", "scripts.build_dataset", "--config", cfg_path])

    # после сборки — ещё раз проверим
    _print_data_dir_state(data_dir)
    if not (os.path.exists(p5) and os.path.exists(p1)):
        raise RuntimeError("Dataset build finished but parquet files are still missing")

    print(f"[DATA] Built:\n  {p5}\n  {p1}")
    return p5, p1


def _load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # ожидаем: index=DatetimeIndex(UTC) + open/high/low/close/volume
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError(f"{path}: index is not DatetimeIndex")
    return df.sort_index()


def grid(cfg: dict) -> List[Dict[str, Any]]:
    s = cfg["strategy"]
    keys = [
        "strategic_lookback_days",
        "q_lower",
        "q_upper",
        "range_min_atr_mult",
        "dca_levels",
        "dca_growth",
        "bank_usd",
        "cum_deposit_frac_at_full",
        "tp_pct",
    ]
    values = [s[k] if isinstance(s[k], list) else [s[k]] for k in keys]

    out = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        # фиксированные параметры тоже добавим
        fixed = {
            "tf_entry": s["tf_entry"],
            "tf_range": s["tf_range"],
            "tactical_lookback_days": s.get("tactical_lookback_days", [3])[0],
            "entry_long_thr": s["entry_long_thr"],
            "entry_short_thr": s["entry_short_thr"],
            "break_eps": s["break_eps"],
            "reentry_band": s["reentry_band"],
            "tactical_pcts": s["tactical_pcts"],
            "strategic_pcts": s["strategic_pcts"],
            "trailing_stages": s["trailing_stages"],
            "rsi_len": s["rsi_len"],
            "adx_len": s["adx_len"],
            "vol_win": s["vol_win"],
        }
        params.update(fixed)
        out.append(params)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1) данные
    p5, p1 = ensure_data(args.config, cfg)
    df_5m = _load_parquet(p5)
    df_1h = _load_parquet(p1)

    # 2) брокер (сим)
    bcfg = cfg["broker"]
    broker = BrokerFX(
        leverage=float(bcfg["leverage"]),
        spread_points=float(bcfg["spread_points"]),
        tick=float(bcfg["tick"]),
        margin_call_level=float(bcfg["margin_call_level"]),
        stop_out_level=float(bcfg["stop_out_level"]),
    )

    # 3) прогон сетки
    out_dir = cfg["report"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    save_trades = bool(cfg["report"].get("save_trades", True))

    all_rows = []
    best = None

    combos = grid(cfg)
    print(f"[GRID] total combinations: {len(combos)}")

    for i, params in enumerate(combos, 1):
        print(f"[GRID] {i}/{len(combos)} params={params}")

        result = run_strategy(
            df_entry=df_5m,
            df_range=df_1h,
            broker=broker,
            params=params,
        )

        stats = compute_equity_stats(result.equity_curve)
        row = {**params, **asdict(stats)}
        all_rows.append(row)

        if best is None or row.get("total_return", -1e9) > best.get("total_return", -1e9):
            best = row

        if save_trades and getattr(result, "trades", None) is not None:
            trades_path = os.path.join(out_dir, f"trades_{i:04d}.parquet")
            pd.DataFrame(result.trades).to_parquet(trades_path, index=False)

    # 4) отчёт
    df_res = pd.DataFrame(all_rows)
    res_path = os.path.join(out_dir, "grid_results.parquet")
    df_res.to_parquet(res_path, index=False)

    print("[DONE] saved:", res_path)
    if best:
        print("[BEST]", best)


if __name__ == "__main__":
    main()
