# scripts/run_grid.py
from __future__ import annotations
import os
import itertools
import yaml
import pandas as pd

from sim.broker_fx import BrokerFX
from sim.engine import run_strategy

def ensure_data(cfg_path: str, cfg: dict) -> tuple[str, str]:
    data_dir = cfg["data"]["data_dir"]
    p5 = os.path.join(data_dir, cfg["data"]["bars_5m"])
    p1 = os.path.join(data_dir, cfg["data"]["bars_1h"])
    if os.path.exists(p5) and os.path.exists(p1):
        return p5, p1

    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "scripts.build_dataset", "--config", cfg_path])
    if not (os.path.exists(p5) and os.path.exists(p1)):
        raise RuntimeError("Dataset build failed: parquet files not found")
    return p5, p1

def grid(cfg: dict) -> list[dict]:
    s = cfg["strategy"]
    keys = [
        "strategic_lookback_days",
        "tactical_lookback_days",
        "q_lower", "q_upper",
        "range_min_atr_mult",
        "dca_levels", "dca_growth",
        "bank_usd", "cum_deposit_frac_at_full",
        "tp_pct",
    ]
    vals = [s[k] for k in keys]
    combos = []
    for prod in itertools.product(*vals):
        d = {k: v for k, v in zip(keys, prod)}
        # фиксированные параметры стратегии
        d.update({
            "tf_entry": s["tf_entry"],
            "tf_range": s["tf_range"],
            "entry_long_thr": s["entry_long_thr"],
            "entry_short_thr": s["entry_short_thr"],
            "break_eps": s["break_eps"],
            "reentry_band": s["reentry_band"],
            "tactical_pcts": s["tactical_pcts"],
            "strategic_pcts": s["strategic_pcts"],
            "trailing_stages": s["trailing_stages"],
            "rsi_len": s.get("rsi_len", 14),
            "adx_len": s.get("adx_len", 14),
            "vol_win": s.get("vol_win", 50),
        })
        combos.append(d)
    return combos

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    symbol = cfg["symbol"]

    p5, p1 = ensure_data(args.config, cfg)
    df5 = pd.read_parquet(p5).sort_index()
    df1 = pd.read_parquet(p1).sort_index()

    b = cfg["broker"]
    broker = BrokerFX(
        leverage=float(b["leverage"]),
        tick=float(b["tick"]),
        spread_points=float(b["spread_points"]),
        margin_call_level=float(b["margin_call_level"]),
        stop_out_level=float(b["stop_out_level"]),
    )

    out_dir = cfg["report"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    save_trades = bool(cfg["report"].get("save_trades", True))

    rows = []
    for idx, p in enumerate(grid(cfg), start=1):
        summary, trades_df, eq = run_strategy(symbol, df5, df1, broker, p)
        row = summary.__dict__.copy()
        rows.append(row)

        if save_trades:
            # отдельные трейды — чтобы потом анализировать “где риск по марже”
            tag = f"{idx:04d}_L{p['dca_levels']}_G{p['dca_growth']}_TP{p['tp_pct']}_Q{p['q_lower']}-{p['q_upper']}_LB{p['strategic_lookback_days']}"
            trades_path = os.path.join(out_dir, f"trades_{tag}.parquet")
            eq_path = os.path.join(out_dir, f"equity_{tag}.parquet")
            trades_df.to_parquet(trades_path, index=False)
            eq.to_frame().to_parquet(eq_path, index=True)

        print(f"[{idx}] net={row['net_profit']:.2f} trades={row['trades']} minML={row['min_margin_level']:.4f} stopouts={row['stop_out_events']}")

    summary_df = pd.DataFrame(rows)
    summary_df["winrate"] = (summary_df["wins"] / summary_df["trades"].replace(0, pd.NA)).astype("float64")
    summary_df = summary_df.sort_values(["stop_out_events","net_profit"], ascending=[True, False])

    out_csv = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

if __name__ == "__main__":
    main()
