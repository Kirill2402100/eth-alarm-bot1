# sim/engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import pandas as pd
import pandas_ta as ta

from sim.broker_fx import BrokerFX


# -------------------- helpers --------------------

def plan_margins_bank_first(bank: float, levels: int, growth: float, frac: float) -> list[float]:
    total = bank * frac
    if levels <= 0 or total <= 0:
        return []
    if abs(growth - 1.0) < 1e-12:
        return [total / levels] * levels
    base = total * (growth - 1.0) / (growth**levels - 1.0)
    return [base * (growth**i) for i in range(levels)]


def break_levels(lower: float, upper: float, break_eps: float) -> tuple[float, float]:
    up = upper * (1.0 + break_eps)
    dn = lower * (1.0 - break_eps)
    return up, dn


def chandelier_stop(side: str, price: float, atr: float, mult: float = 3.0) -> float:
    return price - mult * atr if side == "LONG" else price + mult * atr


def compute_pct_targets(entry: float, side: str, lower: float, upper: float, atr1h: float, tick: float,
                        pcts: list[float], break_eps: float) -> list[float]:
    if side == "SHORT":
        cap = upper
        path = max(0.0, cap - entry)
        raw = [entry + path * p for p in pcts]
        brk_up, _ = break_levels(lower, upper, break_eps)
        buf = max(tick, 0.05 * max(atr1h, 1e-12))
        capped = [min(x, brk_up - buf) for x in raw]
        out = sorted(set(round(x / tick) * tick for x in capped))
    else:
        cap = lower
        path = max(0.0, entry - cap)
        raw = [entry - path * p for p in pcts]
        _, brk_dn = break_levels(lower, upper, break_eps)
        buf = max(tick, 0.05 * max(atr1h, 1e-12))
        capped = [max(x, brk_dn + buf) for x in raw]
        out = sorted(set(round(x / tick) * tick for x in capped), reverse=True)

    dedup = []
    for x in out:
        if not dedup:
            dedup.append(x)
        else:
            if side == "SHORT":
                if x > dedup[-1] + tick:
                    dedup.append(x)
            else:
                if x < dedup[-1] - tick:
                    dedup.append(x)
    return dedup


def compute_mixed_targets(entry: float, side: str,
                          strat: dict[str, float], tac: dict[str, float],
                          tick: float, tactical_pcts: list[float], strategic_pcts: list[float],
                          break_eps: float) -> list[dict[str, Any]]:
    t_prices = compute_pct_targets(entry, side, tac["lower"], tac["upper"], tac["atr1h"], tick, tactical_pcts, break_eps)
    s_prices = compute_pct_targets(entry, side, strat["lower"], strat["upper"], strat["atr1h"], tick, strategic_pcts, break_eps)

    targets = []
    for i, pr in enumerate(t_prices):
        pct = int(round(tactical_pcts[min(i, len(tactical_pcts)-1)] * 100))
        targets.append({"price": pr, "label": f"TAC {pct}%"})
    for i, pr in enumerate(s_prices):
        pct = int(round(strategic_pcts[min(i, len(strategic_pcts)-1)] * 100))
        targets.append({"price": pr, "label": f"STRAT {pct}%"})

    if side == "SHORT":
        targets = sorted(targets, key=lambda x: x["price"])
    else:
        targets = sorted(targets, key=lambda x: x["price"], reverse=True)

    out = []
    for t in targets:
        if not out:
            out.append(t)
        else:
            if side == "SHORT":
                if t["price"] > out[-1]["price"] + tick:
                    out.append(t)
            else:
                if t["price"] < out[-1]["price"] - tick:
                    out.append(t)
    return out


def trend_reversal_confirmed(side: str, supertrend_state: str) -> bool:
    return (side == "SHORT" and supertrend_state in ("up_to_down_near", "down")) or \
           (side == "LONG"  and supertrend_state in ("down_to_up_near", "up"))


def supertrend_state_from_dirs(dir_prev: int, dir_now: int) -> str:
    if dir_prev == -1 and dir_now == 1:
        return "down_to_up_near"
    if dir_prev == 1 and dir_now == -1:
        return "up_to_down_near"
    return "up" if dir_now == 1 else "down"


# -------------------- rolling ranges (без lookahead) --------------------

def build_ranges_1h(df1h: pd.DataFrame,
                    strategic_lookback_days: int,
                    tactical_lookback_days: int,
                    q_lower: float, q_upper: float,
                    range_min_atr_mult: float) -> pd.DataFrame:
    closes = df1h["close"].astype(float)

    ema50 = ta.ema(closes, length=50)
    atr14 = ta.atr(df1h["high"], df1h["low"], df1h["close"], length=14)

    def roll_quantile(win: int, q: float) -> pd.Series:
        return closes.rolling(win, min_periods=max(50, int(win*0.5))).quantile(q)

    strat_win = int(strategic_lookback_days * 24)
    tac_win   = int(tactical_lookback_days * 24)

    strat_lo = roll_quantile(strat_win, q_lower)
    strat_hi = roll_quantile(strat_win, q_upper)
    tac_lo   = roll_quantile(tac_win,   q_lower)
    tac_hi   = roll_quantile(tac_win,   q_upper)

    def adjust(lo: pd.Series, hi: pd.Series) -> tuple[pd.Series, pd.Series]:
        mid = ema50
        a = atr14.fillna(0.0)
        lo2 = pd.concat([lo, mid - range_min_atr_mult * a], axis=1).min(axis=1)
        hi2 = pd.concat([hi, mid + range_min_atr_mult * a], axis=1).max(axis=1)
        return lo2, hi2

    strat_lo2, strat_hi2 = adjust(strat_lo, strat_hi)
    tac_lo2, tac_hi2     = adjust(tac_lo, tac_hi)

    out = pd.DataFrame({
        "strat_lower": strat_lo2,
        "strat_upper": strat_hi2,
        "strat_mid": ema50,
        "strat_atr1h": atr14,
        "tac_lower": tac_lo2,
        "tac_upper": tac_hi2,
        "tac_mid": ema50,
        "tac_atr1h": atr14,
    }, index=df1h.index)

    out["strat_width"] = out["strat_upper"] - out["strat_lower"]
    out["tac_width"]   = out["tac_upper"] - out["tac_lower"]

    return out.shift(1)


# -------------------- позиция --------------------

@dataclass
class Position:
    side: str
    open_time: pd.Timestamp
    leverage: float
    step_margins: list[float]
    max_steps: int

    steps_filled: int = 0
    qty: float = 0.0
    avg: float = 0.0
    used_margin: float = 0.0

    tp_pct: float = 0.0
    tp_price: float = 0.0
    sl_price: Optional[float] = None
    trail_stage: int = -1

    reserved_one: bool = False
    ordinary_targets: list[dict[str, Any]] = None

    def add_step(self, price_exec: float):
        m = float(self.step_margins[self.steps_filled])
        notional = m * self.leverage
        new_qty = notional / max(price_exec, 1e-12)
        self.avg = (self.avg * self.qty + price_exec * new_qty) / max(self.qty + new_qty, 1e-12) if self.qty > 0 else price_exec
        self.qty += new_qty
        self.used_margin += m
        self.steps_filled += 1
        self.tp_price = self.avg * (1.0 + self.tp_pct) if self.side == "LONG" else self.avg * (1.0 - self.tp_pct)
        return m


# -------------------- результат --------------------

@dataclass
class RunSummary:
    symbol: str
    tf_entry: str
    tf_range: str

    strategic_lookback_days: int
    tactical_lookback_days: int
    q_lower: float
    q_upper: float
    range_min_atr_mult: float

    dca_levels: int
    dca_growth: float
    bank_usd: float
    cum_deposit_frac_at_full: float
    tp_pct: float

    trades: int
    wins: int
    net_profit: float
    max_dd_usd: float
    max_dd_pct: float

    min_margin_level: float
    margin_call_events: int
    stop_out_events: int


# -------------------- основной прогон --------------------

def run_strategy(
    symbol: str,
    df5: pd.DataFrame,
    df1h: pd.DataFrame,
    broker: BrokerFX,
    params: dict[str, Any],
) -> tuple[RunSummary, pd.DataFrame, pd.Series]:

    tf_entry = params["tf_entry"]
    tf_range = params["tf_range"]

    strategic_lookback_days = int(params["strategic_lookback_days"])
    tactical_lookback_days  = int(params["tactical_lookback_days"])
    q_lower = float(params["q_lower"])
    q_upper = float(params["q_upper"])
    range_min_atr_mult = float(params["range_min_atr_mult"])

    entry_long_thr  = float(params["entry_long_thr"])
    entry_short_thr = float(params["entry_short_thr"])
    break_eps = float(params["break_eps"])
    reentry_band = float(params["reentry_band"])

    dca_levels = int(params["dca_levels"])
    dca_growth = float(params["dca_growth"])
    bank_usd = float(params["bank_usd"])
    cum_frac = float(params["cum_deposit_frac_at_full"])
    tp_pct = float(params["tp_pct"])

    tactical_pcts  = list(map(float, params["tactical_pcts"]))
    strategic_pcts = list(map(float, params["strategic_pcts"]))
    trailing_stages = [(float(a), float(b)) for a, b in params["trailing_stages"]]

    rng1h = build_ranges_1h(
        df1h,
        strategic_lookback_days=strategic_lookback_days,
        tactical_lookback_days=tactical_lookback_days,
        q_lower=q_lower, q_upper=q_upper,
        range_min_atr_mult=range_min_atr_mult,
    )
    rng5 = rng1h.reindex(df5.index, method="ffill")
    valid = rng5["strat_lower"].notna() & rng5["tac_lower"].notna() & (rng5["tac_width"] > 0)
    df5 = df5.loc[valid]
    rng5 = rng5.loc[valid]

    # --- индикаторы считаем 1 раз ---
    atr5 = ta.atr(df5["high"], df5["low"], df5["close"], length=14)

    # supertrend direction series (1 раз!)
    st_state_default = "flat"
    st_dir = None
    st = ta.supertrend(df5["high"], df5["low"], df5["close"], length=10, multiplier=3.0)
    d_col = next((c for c in st.columns if c.startswith("SUPERTd_")), None)
    if d_col is not None and len(st) > 1:
        st_dir = st[d_col].astype("Int64")

    equity = bank_usd
    equity_curve = []
    times = []

    peak = equity
    max_dd = 0.0
    max_dd_pct = 0.0

    pos: Optional[Position] = None
    trades = []
    wins = 0
    trades_n = 0

    min_ml = float("inf")
    margin_call_events = 0
    stop_out_events = 0

    def mark_to_market(side: str, avg: float, qty: float, bid_c: float, ask_c: float) -> float:
        if qty <= 0:
            return 0.0
        return (bid_c - avg) * qty if side == "LONG" else (avg - ask_c) * qty

    for i, (ts, row) in enumerate(df5.iterrows()):
        mid_h = float(row.high); mid_l = float(row.low); mid_c = float(row.close)

        bid_c, ask_c = broker.bid_ask_from_mid(mid_c)

        low_bid, low_ask = broker.bid_ask_from_mid(mid_l)
        high_bid, high_ask = broker.bid_ask_from_mid(mid_h)

        strat = {
            "lower": float(rng5.loc[ts, "strat_lower"]),
            "upper": float(rng5.loc[ts, "strat_upper"]),
            "mid": float(rng5.loc[ts, "strat_mid"]) if pd.notna(rng5.loc[ts, "strat_mid"]) else mid_c,
            "atr1h": float(rng5.loc[ts, "strat_atr1h"]) if pd.notna(rng5.loc[ts, "strat_atr1h"]) else 0.0,
        }
        strat["width"] = max(1e-12, strat["upper"] - strat["lower"])

        tac = {
            "lower": float(rng5.loc[ts, "tac_lower"]),
            "upper": float(rng5.loc[ts, "tac_upper"]),
            "mid": float(rng5.loc[ts, "tac_mid"]) if pd.notna(rng5.loc[ts, "tac_mid"]) else mid_c,
            "atr1h": float(rng5.loc[ts, "tac_atr1h"]) if pd.notna(rng5.loc[ts, "tac_atr1h"]) else 0.0,
        }
        tac["width"] = max(1e-12, tac["upper"] - tac["lower"])

        # supertrend state from precomputed dirs
        st_state = st_state_default
        if st_dir is not None:
            d_now = st_dir.loc[ts]
            if pd.notna(d_now):
                d_prev = st_dir.shift(1).loc[ts]
                if pd.notna(d_prev):
                    st_state = supertrend_state_from_dirs(int(d_prev), int(d_now))
                else:
                    st_state = "up" if int(d_now) == 1 else "down"

        # equity now
        if pos:
            floating = mark_to_market(pos.side, pos.avg, pos.qty, bid_c, ask_c)
            equity_now = equity + floating
            used_margin = pos.used_margin
        else:
            equity_now = equity
            used_margin = 0.0

        ml = broker.margin_level(equity_now, used_margin)
        min_ml = min(min_ml, ml)

        if pos and ml <= broker.margin_call_level:
            margin_call_events += 1

        # stop out intrabar worst
        if pos:
            worst_float = mark_to_market(
                pos.side, pos.avg, pos.qty,
                bid_c=(low_bid if pos.side == "LONG" else bid_c),
                ask_c=(ask_c if pos.side == "LONG" else high_ask),
            )
            worst_equity = equity + worst_float
            worst_ml = broker.margin_level(worst_equity, pos.used_margin)
            if worst_ml <= broker.stop_out_level:
                exit_px = low_bid if pos.side == "LONG" else high_ask
                pnl = (exit_px - pos.avg) * pos.qty if pos.side == "LONG" else (pos.avg - exit_px) * pos.qty
                equity += pnl
                stop_out_events += 1

                trades.append({
                    "open_time": pos.open_time,
                    "close_time": ts,
                    "side": pos.side,
                    "reason": "STOP_OUT",
                    "steps": pos.steps_filled,
                    "avg": pos.avg,
                    "exit": exit_px,
                    "pnl": pnl,
                    "used_margin": pos.used_margin,
                    "min_margin_level_seen": float(min_ml),
                })
                trades_n += 1
                pos = None

        # entry
        if pos is None:
            pos_in = (mid_c - tac["lower"]) / tac["width"]
            side = "LONG" if pos_in <= entry_long_thr else ("SHORT" if pos_in >= entry_short_thr else None)
            if side:
                margins = plan_margins_bank_first(bank_usd, dca_levels, dca_growth, cum_frac)
                pos = Position(
                    side=side,
                    open_time=ts,
                    leverage=broker.leverage,
                    step_margins=margins,
                    max_steps=dca_levels,
                )
                pos.tp_pct = tp_pct
                pos.ordinary_targets = compute_mixed_targets(
                    entry=mid_c, side=pos.side, strat=strat, tac=tac,
                    tick=broker.tick, tactical_pcts=tactical_pcts, strategic_pcts=strategic_pcts,
                    break_eps=break_eps
                )

                entry_exec = ask_c if pos.side == "LONG" else bid_c
                pos.add_step(entry_exec)

        # manage
        if pos:
            brk_up, brk_dn = break_levels(strat["lower"], strat["upper"], break_eps)

            if not pos.reserved_one and (mid_c >= brk_up or mid_c <= brk_dn):
                pos.max_steps = min(pos.steps_filled + 1, dca_levels)
                pos.reserved_one = True

            if pos.steps_filled < pos.max_steps:
                if pos.reserved_one:
                    need_retest = (pos.side == "SHORT" and mid_c <= strat["upper"] * (1 - reentry_band)) or \
                                  (pos.side == "LONG"  and mid_c >= strat["lower"] * (1 + reentry_band))
                    if need_retest and trend_reversal_confirmed(pos.side, st_state):
                        exec_px = ask_c if pos.side == "LONG" else bid_c
                        pos.add_step(exec_px)
                        pos.max_steps = pos.steps_filled
                else:
                    idx = pos.steps_filled - 1
                    nxt = pos.ordinary_targets[idx] if (pos.ordinary_targets and 0 <= idx < len(pos.ordinary_targets)) else None
                    if nxt:
                        trg = float(nxt["price"])
                        triggered = (pos.side == "LONG" and mid_l <= trg) or (pos.side == "SHORT" and mid_h >= trg)
                        if triggered:
                            exec_px = ask_c if pos.side == "LONG" else bid_c
                            pos.add_step(exec_px)

            atr_now = float(atr5.loc[ts]) if pd.notna(atr5.loc[ts]) else 0.0
            if pos.side == "LONG":
                gain_to_tp = max(0.0, (bid_c / max(pos.avg, 1e-12) - 1.0) / max(tp_pct, 1e-12))
            else:
                gain_to_tp = max(0.0, (pos.avg / max(ask_c, 1e-12) - 1.0) / max(tp_pct, 1e-12))

            for stage_idx, (arm, lock) in enumerate(trailing_stages):
                if pos.trail_stage >= stage_idx:
                    continue
                if gain_to_tp < arm:
                    break
                lock_pct = lock * tp_pct
                locked = pos.avg * (1 + lock_pct) if pos.side == "LONG" else pos.avg * (1 - lock_pct)
                chand = chandelier_stop(pos.side, mid_c, atr_now, mult=3.0)
                new_sl = max(locked, chand) if pos.side == "LONG" else min(locked, chand)
                new_sl = round(new_sl / broker.tick) * broker.tick

                improves = (pos.sl_price is None) or (pos.side == "LONG" and new_sl > pos.sl_price) or (pos.side == "SHORT" and new_sl < pos.sl_price)
                if improves:
                    pos.sl_price = new_sl
                    pos.trail_stage = stage_idx

            tp = pos.tp_price
            sl = pos.sl_price

            tp_hit = (pos.side == "LONG" and high_bid >= tp) or (pos.side == "SHORT" and low_ask <= tp)
            sl_hit = (sl is not None) and ((pos.side == "LONG" and low_bid <= sl) or (pos.side == "SHORT" and high_ask >= sl))

            if tp_hit or sl_hit:
                reason = "TP" if tp_hit else "SL"
                exit_px = tp if tp_hit else float(sl)

                pnl = (exit_px - pos.avg) * pos.qty if pos.side == "LONG" else (pos.avg - exit_px) * pos.qty
                equity += pnl
                if pnl > 0:
                    wins += 1

                trades.append({
                    "open_time": pos.open_time,
                    "close_time": ts,
                    "side": pos.side,
                    "reason": reason,
                    "steps": pos.steps_filled,
                    "avg": pos.avg,
                    "exit": exit_px,
                    "pnl": pnl,
                    "used_margin": pos.used_margin,
                    "min_margin_level_seen": float(min_ml),
                })
                trades_n += 1
                pos = None

        if pos:
            floating = mark_to_market(pos.side, pos.avg, pos.qty, bid_c, ask_c)
            eq_c = equity + floating
        else:
            eq_c = equity

        peak = max(peak, eq_c)
        dd = peak - eq_c
        max_dd = max(max_dd, dd)
        dd_pct = (dd / peak) * 100.0 if peak > 0 else 0.0
        max_dd_pct = max(max_dd_pct, dd_pct)

        times.append(ts)
        equity_curve.append(eq_c)

    eq_series = pd.Series(index=pd.Index(times, name="time"), data=equity_curve, name="equity")
    trades_df = pd.DataFrame(trades)
    net_profit = float(eq_series.iloc[-1] - eq_series.iloc[0]) if len(eq_series) else 0.0

    summary = RunSummary(
        symbol=symbol,
        tf_entry=tf_entry,
        tf_range=tf_range,
        strategic_lookback_days=strategic_lookback_days,
        tactical_lookback_days=tactical_lookback_days,
        q_lower=q_lower,
        q_upper=q_upper,
        range_min_atr_mult=range_min_atr_mult,
        dca_levels=dca_levels,
        dca_growth=dca_growth,
        bank_usd=bank_usd,
        cum_deposit_frac_at_full=cum_frac,
        tp_pct=tp_pct,
        trades=int(trades_n),
        wins=int(wins),
        net_profit=float(net_profit),
        max_dd_usd=float(max_dd),
        max_dd_pct=float(max_dd_pct),
        min_margin_level=float(min_ml if min_ml != float("inf") else float("inf")),
        margin_call_events=int(margin_call_events),
        stop_out_events=int(stop_out_events),
    )

    return summary, trades_df, eq_series
