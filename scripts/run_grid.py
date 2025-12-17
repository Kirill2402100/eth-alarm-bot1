cd ~/eth-alarm-bot1
source .venv/bin/activate
mkdir -p data/out

cat > scripts/run_grid.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
import os
import sys
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def ensure_repo_on_syspath() -> Path:
    """
    ВАЖНО: это должно выполняться ДО любых `from sim...`
    """
    app_root = Path(__file__).resolve().parents[1]
    os.chdir(app_root)
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    return app_root


APP_ROOT = ensure_repo_on_syspath()

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from sim.broker_fx import BrokerFX  # noqa: E402
from sim.engine import run_strategy  # noqa: E402


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


def list_data_dir(data_dir: Path) -> None:
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    print("=== DEBUG ===", flush=True)
    print(f"cwd: {os.getcwd()}", flush=True)
    print(f"APP_ROOT: {APP_ROOT}", flush=True)
    print(f"data_dir: {data_dir}", flush=True)
    print(f"ls -lah {data_dir}:", flush=True)

    try:
        if data_dir.exists():
            items = sorted(list(data_dir.iterdir()))
            if not items:
                print(" (empty)", flush=True)
            for x in items:
                if x.is_dir():
                    print(f" {x.name}/", flush=True)
                else:
                    mb = x.stat().st_size / (1024 * 1024)
                    print(f" {x.name:35s} {mb:8.2f} MB", flush=True)
        else:
            print(" (missing dir)", flush=True)
    except Exception as e:
        print(f" (list failed: {e})", flush=True)

    print("=============", flush=True)


def ensure_data(cfg_path: Path, cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    """
    Гарантирует наличие parquet-файлов.
    ВАЖНО: если DISABLE_DATASET_DOWNLOAD=1 или data.disable_download=true — НЕ качаем.
    """
    data_cfg = cfg.get("data") or {}
    symbol = str(cfg.get("symbol") or "").strip()
    if not symbol:
        raise RuntimeError("config missing: symbol")

    data_dir = Path(str(data_cfg.get("data_dir") or "data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    bars_5m = str(data_cfg.get("bars_5m") or f"{symbol}_5m.parquet")
    bars_1h = str(data_cfg.get("bars_1h") or f"{symbol}_1h.parquet")

    p5 = data_dir / bars_5m
    p1 = data_dir / bars_1h

    list_data_dir(data_dir)

    if p5.exists() and p1.exists():
        print("[DATA] Using cached files:", flush=True)
        print(f" {p5}", flush=True)
        print(f" {p1}", flush=True)
        return p5, p1

    disable_env = os.getenv("DISABLE_DATASET_DOWNLOAD", "").strip() == "1"
    disable_cfg = bool(data_cfg.get("disable_download", False))
    if disable_env or disable_cfg:
        raise RuntimeError(
            "[DATA] Dataset download is DISABLED (DISABLE_DATASET_DOWNLOAD=1 or data.disable_download=true). "
            "But parquet files are missing."
        )

    print("[DATA] Missing parquet(s), building dataset now...", flush=True)
    subprocess.check_call([sys.executable, "-u", "-m", "scripts.build_dataset", "--config", str(cfg_path)])

    list_data_dir(data_dir)

    if not (p5.exists() and p1.exists()):
        raise RuntimeError(f"[DATA] build_dataset finished but files still missing: {p5}, {p1}")

    print("[DATA] Built:", flush=True)
    print(f" {p5}", flush=True)
    print(f" {p1}", flush=True)
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
    symbol: str,
    df_entry: pd.DataFrame,
    df_range: pd.DataFrame,
    broker: BrokerFX,
    params: Dict[str, Any],
) -> Any:
    sig = inspect.signature(run_strategy)
    p = sig.parameters
    has_varkw = any(pp.kind == inspect.Parameter.VAR_KEYWORD for pp in p.values())

    df5_names = {"df5", "df_5m", "bars_5m", "df_entry", "entry_df"}
    df1h_names = {"df1h", "df_1h", "bars_1h", "df_range", "range_df"}

    if "params" in p:
        kwargs: Dict[str, Any] = {}
        for name in p.keys():
            if name in {"symbol", "sym", "pair"}:
                kwargs[name] = symbol
            elif name in df5_names:
                kwargs[name] = df_entry
            elif name in df1h_names:
                kwargs[name] = df_range
            elif name in {"broker", "brk"}:
                kwargs[name] = broker
            elif name == "params":
                kwargs[name] = params
        try:
            return run_strategy(**kwargs)
        except TypeError:
            return run_strategy(symbol, df_entry, df_range, broker, params)

    if has_varkw:
        try:
            return run_strategy(symbol, df_entry, df_range, broker, **params)
        except TypeError:
            try:
                return run_strategy(symbol, df_entry, df_range, **params)
            except TypeError:
                return run_strategy(symbol, df_entry, df_range, broker)

    try:
        return run_strategy(symbol, df_entry, df_range, broker, params)
    except TypeError:
        return run_strategy(symbol, df_entry, df_range, broker)


def flatten_summary(summary: Any) -> Dict[str, Any]:
    try:
        return asdict(summary)
    except Exception:
        return dict(summary.__dict__)


def compute_derived_metrics(
    params: Dict[str, Any],
    summary_dict: Dict[str, Any],
    trades_df: Optional[pd.DataFrame],
    eq_series: Optional[pd.Series],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    trades_n = int(summary_dict.get("trades", 0) or 0)
    wins_n = int(summary_dict.get("wins", 0) or 0)
    bank = float(summary_dict.get("bank_usd", params.get("bank_usd", 0.0)) or 0.0)
    net_profit = float(summary_dict.get("net_profit", 0.0) or 0.0)

    out["win_rate"] = (wins_n / trades_n) if trades_n > 0 else 0.0
    out["roi_pct"] = (net_profit / bank * 100.0) if bank > 0 else 0.0

    months = 0.0
    if eq_series is not None and len(eq_series) >= 2:
        dt_days = (eq_series.index[-1] - eq_series.index[0]).total_seconds() / 86400.0
        months = max(1e-9, dt_days / 30.4375)
    out["months_covered"] = months
    out["profit_per_month_usd"] = (net_profit / months) if months > 0 else 0.0
    out["roi_per_month_pct"] = (out["roi_pct"] / months) if months > 0 else 0.0

    if trades_df is not None and len(trades_df) > 0:
        out["avg_steps"] = float(trades_df["steps"].mean())
        out["p95_steps"] = float(trades_df["steps"].quantile(0.95))
        out["max_steps"] = int(trades_df["steps"].max())

        out["max_used_margin"] = float(trades_df["used_margin"].max())
        out["max_used_margin_pct_of_bank"] = (out["max_used_margin"] / bank * 100.0) if bank > 0 else 0.0

        if "min_margin_level_seen" in trades_df.columns:
            out["min_trade_margin_level"] = float(trades_df["min_margin_level_seen"].min())
        else:
            out["min_trade_margin_level"] = float("inf")
    else:
        out["avg_steps"] = 0.0
        out["p95_steps"] = 0.0
        out["max_steps"] = 0
        out["max_used_margin"] = 0.0
        out["max_used_margin_pct_of_bank"] = 0.0
        out["min_trade_margin_level"] = float("inf")

    if eq_series is not None and len(eq_series) >= 2:
        eq_m = eq_series.resample("M").last()
        if len(eq_m) >= 2:
            mret = eq_m.pct_change().dropna()
            out["mret_mean_pct"] = float(mret.mean() * 100.0)
            out["mret_worst_pct"] = float(mret.min() * 100.0)
            out["mret_best_pct"] = float(mret.max() * 100.0)
            out["months_pos"] = int((mret > 0).sum())
            out["months_total"] = int(len(mret))
        else:
            out["mret_mean_pct"] = 0.0
            out["mret_worst_pct"] = 0.0
            out["mret_best_pct"] = 0.0
            out["months_pos"] = 0
            out["months_total"] = 0
    else:
        out["mret_mean_pct"] = 0.0
        out["mret_worst_pct"] = 0.0
        out["mret_best_pct"] = 0.0
        out["months_pos"] = 0
        out["months_total"] = 0

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--save_every", type=int, default=50)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(str(cfg_path))

    symbol = str(cfg.get("symbol") or "").strip()
    if not symbol:
        raise RuntimeError("config missing: symbol")

    p5, p1 = ensure_data(cfg_path, cfg)

    df_entry = pd.read_parquet(p5)
    df_range = pd.read_parquet(p1)

    bcfg = cfg.get("broker") or {}
    broker = BrokerFX(
        leverage=float(bcfg.get("leverage", 200)),
        spread_points=float(bcfg.get("spread_points", 8)),
        tick=float(bcfg.get("tick", 0.00001)),
        margin_call_level=float(bcfg.get("margin_call_level", 0.20)),
        stop_out_level=float(bcfg.get("stop_out_level", 0.00)),
    )

    strat = cfg.get("strategy") or {}
    base_params, sweeps = build_grid(strat)
    combos = iter_product(sweeps)

    report_cfg = cfg.get("report") or {}
    out_dir = Path(str(report_cfg.get("out_dir") or "data/out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(combos)
    print(f"[GRID] total combinations: {total}", flush=True)

    results: List[Dict[str, Any]] = []

    for i, combo in enumerate(combos, 1):
        params = dict(base_params)
        params.update(combo)

        print(f"[GRID] {i}/{total} params={params}", flush=True)

        row: Dict[str, Any] = dict(params)
        row["_i"] = i

        try:
            summary, trades_df, eq_series = call_run_strategy_compat(
                symbol=symbol,
                df_entry=df_entry,
                df_range=df_range,
                broker=broker,
                params=params,
            )

            sdict = flatten_summary(summary)
            row.update(sdict)
            row.update(compute_derived_metrics(params, sdict, trades_df, eq_series))

            row["_ok"] = True
            row["_error"] = ""
        except Exception as e:
            row["_ok"] = False
            row["_error"] = str(e)

        results.append(row)

        if i % int(args.save_every) == 0:
            pd.DataFrame(results).to_csv(out_dir / "grid_progress.csv", index=False)

    pd.DataFrame(results).to_csv(out_dir / "grid_results.csv", index=False)
    print(f"[GRID] done. saved: {out_dir / 'grid_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
PY

# на всякий случай убираем кеш байткода
find . -name "__pycache__" -type d -print -exec rm -rf {} + 2>/dev/null || true
