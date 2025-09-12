# news_augment.py — добавляет в лист NEWS недостающие источники для GBP/JPY (не трогая основной календарь)
import os, re, json, base64, logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime, timedelta, timezone

import httpx

# ---- Google Sheets ----
try:
    import gspread
    from google.oauth2 import service_account
    _GSHEETS_AVAILABLE = True
except Exception:
    gspread = None
    service_account = None
    _GSHEETS_AVAILABLE = False

# ---- ENV ----
SHEET_ID = os.getenv("SHEET_ID", "").strip()
NEWS_WS  = "NEWS"
RUN_FOREVER = (os.getenv("RUN_FOREVER", "0").lower() in ("1","true","yes","on"))
AUGMENT_EVERY_MIN = int(os.getenv("AUGMENT_EVERY_MIN", "30") or "30")

# Если основной MoF ENG URL 404 — используем JP
MOF_ENG = "https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/"
MOF_JP  = "https://www.mof.go.jp/policy/international_policy/reference/foreign_exchange_intervention/"

JINA_PROXY = os.getenv("JINA_PROXY","https://r.jina.ai/http/").rstrip("/") + "/"

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("news_augment")

# ---- helpers: creds ----
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def _decode_b64_to_json(s: str) -> Optional[dict]:
    s = (s or "").strip()
    if not s: return None
    s += "=" * ((4 - len(s) % 4) % 4)
    try:
        return json.loads(base64.b64decode(s).decode("utf-8","strict"))
    except Exception:
        return None

def _load_service_info() -> Optional[dict]:
    info = _decode_b64_to_json(os.getenv("GOOGLE_CREDENTIALS_JSON_B64",""))
    if info: return info
    for k in ("GOOGLE_CREDENTIALS_JSON","GOOGLE_CREDENTIALS"):
        raw = os.getenv(k,"").strip()
        if raw:
            try: return json.loads(raw)
            except Exception: pass
    return None

def build_sheets_client(sheet_id: str):
    if not _GSHEETS_AVAILABLE: return None, "gsheets libs not installed"
    if not sheet_id: return None, "SHEET_ID empty"
    info = _load_service_info()
    if not info: return None, "no service account json"
    try:
        creds = service_account.Credentials.from_service_account_info(info, scopes=SHEETS_SCOPES)
        gc = gspread.authorize(creds)
        return gc.open_by_key(sheet_id), "ok"
    except Exception as e:
        return None, f"auth/open error: {e}"

def ensure_news_ws(sh):
    headers = ["ts_utc","source","title","url","countries","ccy","tags","importance_guess","hash"]
    for ws in sh.worksheets():
        if ws.title == NEWS_WS:
            try:
                cur = ws.get_values("A1:I1") or [[]]
                if not cur or cur[0] != headers:
                    ws.update(range_name="A1", values=[headers])
            except Exception:
                pass
            return ws
    ws = sh.add_worksheet(title=NEWS_WS, rows=500, cols=9)
    ws.update(range_name="A1", values=[headers])
    return ws

def _read_existing_hashes(ws) -> set:
    try:
        col = ws.get_values("I2:I10000")
        return {r[0] for r in col if r and r[0]}
    except Exception:
        return set()

def _to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---- HTTP ----
def fetch_text(url: str, timeout=15.0) -> tuple[str,int,str]:
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent":"Mozilla/5.0 (LPBot/1.0)"}) as cli:
            r = cli.get(url)
            if r.status_code in (403,404):
                pr = cli.get(JINA_PROXY + url)
                return pr.text, pr.status_code, str(pr.url)
            return r.text, r.status_code, str(r.url)
    except Exception as e:
        log.warning("fetch failed %s: %s", url, e)
        return "", 0, url

def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s or "").replace("\xa0"," ").strip()

# ---- model ----
@dataclass
class NewsItem:
    ts_utc: datetime
    source: str
    title: str
    url: str
    countries: str
    ccy: str
    tags: str
    importance_guess: str

    def key_hash(self) -> str:
        return f"{self.source}|{self.url}"[:180]

    def to_row(self) -> List[str]:
        return [
            _to_utc_iso(self.ts_utc),
            self.source,
            _strip_html(self.title),
            self.url,
            self.countries,
            self.ccy,
            self.tags,
            self.importance_guess,
            self.key_hash(),
        ]

# ---- collectors (добавочные источники) ----
def collect_extra_news() -> List[NewsItem]:
    items: List[NewsItem] = []
    now = datetime.now(timezone.utc)

    # BoE: Monetary Policy Summary & Minutes (главная MPC-лента) — GBP / high
    txt, code, _ = fetch_text("https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes")
    if code:
        for m in re.finditer(r'href="(/monetary-policy-summary-and-minutes/20\d{2}/[^"]+)"[^>]*>(.*?)</a>', txt, re.I):
            u = "https://www.bankofengland.co.uk" + m.group(1)
            t = _strip_html(m.group(2)) or "Monetary Policy Summary and Minutes"
            items.append(NewsItem(now, "BOE_PR", t, u, "united kingdom", "GBP", "boe mpc", "high"))

    # BoJ: Monetary Policy decisions — JPY / high
    # сводная страница решений/заявлений
    txt, code, _ = fetch_text("https://www.boj.or.jp/en/mopo/mpmdeci/index.htm")
    if code:
        for m in re.finditer(r'href="(/en/mopo/[^"]+)"[^>]*>(.*?)</a>', txt, re.I):
            href = m.group(1)
            if not re.search(r"(mpm|statement|policy|decision)", href, re.I):
                continue
            u = "https://www.boj.or.jp" + href
            t = _strip_html(m.group(2)) or "BoJ Monetary Policy"
            items.append(NewsItem(now, "BOJ_PR", t, u, "japan", "JPY", "boj mpm", "high"))

    # JP MoF FX: если ENG-страница недоступна, добавляем JP (JPY / high)
    txt, code, _ = fetch_text(MOF_ENG)
    if code != 200:
        txt, code, url = fetch_text(MOF_JP)
        if code == 200 and re.search(r"(為替|介入|外国為替|announcement|intervention)", txt, re.I):
            items.append(NewsItem(now, "JP_MOF_FX", "FX intervention reference page", url, "japan", "JPY", "mof", "high"))

    log.info("augment collected: %d new candidates", len(items))
    return items

# ---- write to sheet ----
def append_news_rows(ws, rows: List[List[str]]):
    if not rows: return 0
    ws.append_rows(rows, value_input_option="RAW")
    return len(rows)

# ---- main once ----
def run_once():
    if not SHEET_ID:
        raise RuntimeError("SHEET_ID env empty")
    sh, _ = build_sheets_client(SHEET_ID)
    if not sh:
        raise RuntimeError("Sheets auth/open failed")

    ws = ensure_news_ws(sh)
    existing = _read_existing_hashes(ws)

    items = collect_extra_news()
    rows = [n.to_row() for n in items if n.key_hash() not in existing]

    added = append_news_rows(ws, rows)
    log.info("NEWS augment: +%d rows", added)

# ---- entry ----
def main():
    if not RUN_FOREVER:
        run_once()
        return
    interval = max(1, AUGMENT_EVERY_MIN) * 60
    while True:
        try:
            run_once()
        except Exception:
            log.exception("augment iteration failed")
        import time as _t; _t.sleep(interval)

if __name__ == "__main__":
    main()
