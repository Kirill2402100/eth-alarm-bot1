# sim/broker_fx.py
from dataclasses import dataclass

@dataclass(frozen=True)
class BrokerFX:
    leverage: float
    tick: float
    spread_points: float
    margin_call_level: float  # 0.20 = 20%
    stop_out_level: float     # 0.00 = 0%

    def spread_price(self) -> float:
        return float(self.spread_points) * float(self.tick)

    def bid_ask_from_mid(self, mid: float) -> tuple[float, float]:
        half = 0.5 * self.spread_price()
        return float(mid) - half, float(mid) + half

    def margin_level(self, equity: float, used_margin: float) -> float:
        if used_margin <= 0:
            return float("inf")
        return equity / used_margin  # 1.0 == 100%
