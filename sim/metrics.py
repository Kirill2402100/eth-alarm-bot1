# sim/metrics.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class EquityStats:
    net_profit: float
    max_drawdown_usd: float
    max_drawdown_pct: float

def compute_equity_stats(equity_curve: pd.Series) -> EquityStats:
    eq = equity_curve.astype(float)
    net_profit = float(eq.iloc[-1] - eq.iloc[0])
    peak = eq.cummax()
    dd = peak - eq
    max_dd = float(dd.max())
    max_dd_pct = float((dd / peak.replace(0, np.nan)).max() * 100.0) if (peak > 0).any() else float("nan")
    return EquityStats(net_profit=net_profit, max_drawdown_usd=max_dd, max_drawdown_pct=max_dd_pct)
