from __future__ import annotations

from datetime import datetime, timezone, timedelta
import gspread
from typing import Optional, List, Dict

HEADERS = [
    "pair", "risk", "bias", "ttl", "updated_at",
    "scan_lock_until", "reserve_off", "dca_scale"
]

class FAStore:
    """Хранит / обновляет политику в листе FA_Signals."""

    def __init__(self, sheet_id: str, creds: dict):
        gc = gspread.service_account_from_dict(creds)
        self.sh = gc.open_by_key(sheet_id)
        self.ws = self._ensure_ws("FA_Signals", HEADERS)

    def _ensure_ws(self, title: str, headers: List[str]):
        try:
            ws = self.sh.worksheet(title)
        except gspread.WorksheetNotFound:
            ws = self.sh.add_worksheet(title=title, rows=2000, cols=max(20, len(headers)))
            ws.append_row(headers)
            return ws
        # гарантируем заголовки
        if [h.lower() for h in ws.row_values(1)] != headers:
            try:
                ws.delete_rows(1)
            except Exception:
                pass
            ws.insert_row(headers, 1)
        return ws

    def _find_row_idx(self, pair_upper: str) -> Optional[int]:
        col = self.ws.col_values(1)  # колонка "pair"
        for i, v in enumerate(col, start=1):
            if i == 1:  # headers
                continue
            if str(v).upper() == pair_upper:
                return i
        return None

    def get(self, pair: str) -> Optional[Dict]:
        pair = pair.upper()
        for row in self.ws.get_all_records():
            if str(row.get("pair", "")).upper() == pair:
                return row
        return None

    def list_all(self) -> List[Dict]:
        return self.ws.get_all_records()

    def upsert(self, pair: str, **fields) -> Dict:
        """Дописывает/обновляет пару. Поддерживает:
        risk(Green/Amber/Red), bias(neutral/long-only/short-only), ttl(int, min),
        dca_scale(float 0..1), reserve_off(0/1/true/false), scan_lock_until (ISO либо минуты int/float).
        """
        pair = pair.upper()
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # привести поля
        norm: Dict[str, str] = {k: "" for k in HEADERS}
        norm.update(self.get(pair) or {})
        for k, v in fields.items():
            k = k.lower()
            if k not in HEADERS:
                continue
            if k == "ttl":
                try: v = int(float(v))
                except: v = 0
            elif k == "dca_scale":
                try:
                    v = float(v)
                    if v < 0: v = 0.0
                    if v > 1: v = 1.0
                except:
                    v = 1.0
            elif k == "reserve_off":
                v = str(v).strip().lower()
                v = "1" if v in ("1", "true", "yes", "on") else "0"
            elif k == "scan_lock_until":
                # если число — это минуты от сейчас
                if isinstance(v, (int, float)) or str(v).replace(".", "", 1).isdigit():
                    minutes = float(v)
                    until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
                    v = until.strftime("%Y-%m-%dT%H:%M:%SZ")
            norm[k] = v

        norm["pair"] = pair
        norm["updated_at"] = now_iso

        values = [norm.get(h, "") for h in HEADERS]
        row_idx = self._find_row_idx(pair)
        if row_idx:
            self.ws.update(f"A{row_idx}", [values])
        else:
            self.ws.append_row(values)
        return norm
