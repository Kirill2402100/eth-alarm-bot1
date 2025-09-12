# calendar_collector.py — собирает CALENDAR и NEWS в Google Sheets

import os, re, json, base64, time, logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Iterable
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ","Europe/Belgrade")) if ZoneInfo else None

import httpx

try:
    import gspread
    from google.oauth2 import service_account
    _GSHEETS_AVAILABLE = True
except Exception:
    gspread = None
    service_account = None
    _GSHEETS_AVAILABLE = False

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("calendar_collector")

SHEET_ID = os.getenv("SHEET_ID","").strip()

CAL_WS_OUT = os.getenv("CAL_WS_OUT","CALENDAR").strip() or "CALENDAR"
CAL_WS_RAW = os.getenv("CAL_WS_RAW","CALENDAR_RAW").strip() or "CALENDAR_RAW"

RUN_FOREVER = (os.getenv("RUN_FOREVER","1").lower() in ("1","true","yes","on"))
COLLECT_EVERY_MIN = int(os.getenv("COLLECT_EVERY_MIN","20") or "20")

FREE_LOOKBACK_DAYS  = int(os.getenv("FREE_LOOKBACK_DAYS","7") or "7")
FREE_LOOKAHEAD_DAYS = int(os.getenv("FREE_LOOKAHEAD_DAYS","30") or "30")

JINA_PROXY = os.getenv("JINA_PROXY","https://r.jina.ai/http/").rstrip("/") + "/"

ALLOWED_SOURCES = {s.strip().upper() for s in os.getenv("FA_NEWS_SOURCES","US_FED_PR,ECB_PR,BOE_PR,BOJ_PR,RBA_MR,US_TREASURY,JP_MOF_FX").split(",") if s.strip()}
KW_RE = re.compile(os.getenv("FA_NEWS_KEYWORDS","rate decision|monetary policy|bank rate|policy decision|unscheduled|emergency|intervention|FX intervention|press conference"), re.I)
FA_NEWS_RECENT_MIN = int(os.getenv("FA_NEWS_RECENT_MIN","30"))
FA_NEWS_MIN_COUNT  = int(os.getenv("FA_NEWS_MIN_COUNT","2"))
FA_NEWS_LOCK_MIN   = int(os.getenv("FA_NEWS_LOCK_MIN","30"))
FA_NEWS_TTL_MIN    = int(os.getenv("FA_NEWS_TTL_MIN","120"))

SYMBOLS = ["USDJPY","AUDUSD","EURUSD","GBPUSD"]
PAIR_COUNTRIES = {
    "USDJPY": {"united states","japan"},
    "AUDUSD": {"australia","united states"},
    "EURUSD": {"euro area","united states"},
    "GBPUSD": {"united kingdom","united states"},
}

SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def _decode_b64_to_json(s: str) -> Optional[dict]:
    s = (s or "").strip()
    if not s: return None
    s += "=" * ((4 - len(s) % 4) % 4)
    try: return json.loads(base64.b64decode(s).decode("utf-8","strict"))
    except Exception: return None

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

def ensure_worksheet(sh, title: str, headers: List[str]):
    for ws in sh.worksheets():
        if ws.title == title:
            try:
                cur = ws.get_values("A1:Z1") or [[]]
                if not cur or cur[0] != headers:
                    ws.update("A1", [headers])
            except Exception:
                pass
            return ws, False
    ws = sh.add_worksheet(title=title, rows=200, cols=max(10,len(headers)))
    ws.update("A1", [headers])
    return ws, True

def _to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _localize(dt: datetime) -> datetime:
    return dt.astimezone(LOCAL_TZ) if LOCAL_TZ else dt

# -------- HTTP --------
def fetch_text(url: str, timeout=15.0) -> tuple[str,int,str]:
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent":"Mozilla/5.0 (LPBot/1.0)"}) as cli:
            r = cli.get(url)
            if r.status_code in (403,404):
                pr = cli.get(JINA_PROXY + url.replace("://", "://"))
                return pr.text, pr.status_code, str(pr.url)
            return r.text, r.status_code, str(r.url)
    except Exception as e:
        log.warning("fetch failed %s: %s", url, e)
        return "", 0, url

# -------- Models --------
@dataclass
class CalEvent:
    utc: datetime
    country: str
    currency: str
    title: str
    impact: str
    source: str
    url: str
    def to_row(self) -> List[str]:
        local = _localize(self.utc)
        return [_to_utc_iso(self.utc), local.strftime("%Y-%m-%d %H:%M"),
                self.country, self.currency, self.title, self.impact, self.source, self.url]

CAL_HEADERS  = ["utc_iso","local_time","country","currency","title","impact","source","url"]
NEWS_HEADERS = ["ts_utc","source","title","url","countries","ccy","tags","importance_guess","hash"]

# -------- Calendar parsers (миним.) --------
def _parse_month_day_year(s: str) -> Optional[datetime]:
    m = re.search(r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:–\d{1,2})?,\s*(20\d{2})", s, re.I)
    if not m: return None
    month = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}[m.group(1).lower()[:3]]
    return datetime(int(m.group(3)), month, int(m.group(2)), 14, 0, 0, tzinfo=timezone.utc)

def parse_fomc_calendar(text: str, url: str) -> List[CalEvent]:
    out: List[CalEvent] = []
    for m in re.finditer(r"(?:FOMC|Meeting)[^<>\n]{0,80}?(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:–\d{1,2})?,\s*20\d{2}", text, re.I):
        dt = _parse_month_day_year(m.group(0))
        if dt:
            out.append(CalEvent(utc=dt, country="united states", currency="USD", title="FOMC Meeting / Rate Decision", impact="high", source="FOMC", url=url))
    uniq = {}; [uniq.setdefault(ev.utc.date(), ev) for ev in out]
    return list(uniq.values())

def parse_boj_calendar(text: str, url: str) -> List[CalEvent]:
    out: List[CalEvent] = []
    for m in re.finditer(r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2}).{0,64}?Monetary Policy Meeting", text, re.I):
        dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), 3, 0, 0, tzinfo=timezone.utc)
        out.append(CalEvent(utc=dt, country="japan", currency="JPY", title="BoJ Monetary Policy Meeting", impact="high", source="BoJ", url=url))
    uniq = {}; [uniq.setdefault(ev.utc.date(), ev) for ev in out]
    return list(uniq.values())[:20]

def collect_calendar() -> List[CalEvent]:
    events: List[CalEvent] = []
    txt, code, u = fetch_text("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    if code: events += parse_fomc_calendar(txt, u)
    txt, code, u = fetch_text("https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm")
    if code: events += parse_boj_calendar(txt, u)
    return events

# -------- NEWS scrapers --------
@dataclass
class NewsItem:
    ts_utc: datetime; source: str; title: str; url: str; countries: str; ccy: str; tags: str; importance_guess: str
    def key_hash(self) -> str: return re.sub(r"\s+"," ", f"{self.source}|{self.title}".strip())[:180]
    def to_row(self) -> List[str]:
        return [_to_utc_iso(self.ts_utc), self.source, self.title, self.url, self.countries, self.ccy, self.tags, self.importance_guess, self.key_hash()]

def _parse_time_attr(html: str) -> Optional[datetime]:
    m = re.search(r'<time[^>]+datetime="([^"]+)"', html, re.I)
    if not m: return None
    try: return datetime.fromisoformat(m.group(1).replace("Z","+00:00")).astimezone(timezone.utc)
    except Exception: return None

def collect_news() -> List[NewsItem]:
    items: List[NewsItem] = []
    now = datetime.now(timezone.utc)

    # FED PR (списки разделов — хватит для сигналов)
    txt, code, _ = fetch_text("https://www.federalreserve.gov/newsevents/pressreleases.htm")
    if code:
        for m in re.finditer(r'<a[^>]+href="(/newsevents/pressreleases/[^"]+)"[^>]*>(.*?)</a>', txt, re.I):
            u = "https://www.federalreserve.gov"+m.group(1)
            t = re.sub(r"<[^>]+>","",m.group(2)).strip()
            items.append(NewsItem(now,"US_FED_PR",t,u,"united states","USD","policy","high" if KW_RE.search(t) else "medium"))

    # US Treasury
    txt, code, _ = fetch_text("https://home.treasury.gov/news/press-releases")
    if code:
        for m in re.finditer(r'<a[^>]+href="(/news/press-releases/[^"]+)"[^>]*>(.*?)</a>', txt, re.I):
            u = "https://home.treasury.gov"+m.group(1)
            t = re.sub(r"<[^>]+>","",m.group(2)).strip()
            items.append(NewsItem(now,"US_TREASURY",t,u,"united states","USD","treasury","medium"))

    # ECB PR
    txt, code, _ = fetch_text("https://www.ecb.europa.eu/press/pubbydate/html/index.en.html?name_of_publication=Press%20release")
    if code:
        for m in re.finditer(r'<a[^>]+href="(/press/[^"]+)"[^>]*>(.*?)</a>', txt, re.I):
            u = "https://www.ecb.europa.eu"+m.group(1)
            t = re.sub(r"<[^>]+>","",m.group(2)).strip()
            items.append(NewsItem(now,"ECB_PR",t,u,"euro area","EUR","ecb","high" if KW_RE.search(t) else "medium"))

    # BoE
    txt, code, _ = fetch_text("https://www.bankofengland.co.uk/news")
    if code:
        for m in re.finditer(r'<a[^>]+href="(/news/[^"]+)"[^>]*>(.*?)</a>', txt, re.I):
            u = "https://www.bankofengland.co.uk"+m.group(1)
            t = re.sub(r"<[^>]+>","",m.group(2)).strip()
            items.append(NewsItem(now,"BOE_PR",t,u,"united kingdom","GBP","boe","high" if KW_RE.search(t) else "medium"))

    # RBA
    txt, code, _ = fetch_text("https://www.rba.gov.au/media-releases/")
    if code:
        for m in re.finditer(r'<a[^>]+href="(/media-releases/\d{4}/[^"]+)"[^>]*>(.*?)</a>', txt, re.I):
            u = "https://www.rba.gov.au"+m.group(1)
            t = re.sub(r"<[^>]+>","",m.group(2)).strip()
            items.append(NewsItem(now,"RBA_MR",t,u,"australia","AUD","rba","high" if KW_RE.search(t) else "medium"))

    # JP MoF FX (флаг)
    txt, code, url = fetch_text("https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/index.html")
    if code and re.search(r"intervention|announcement", txt, re.I):
        items.append(NewsItem(now,"JP_MOF_FX","FX intervention reference page",url,"japan","JPY","mof","high"))

    log.info("NEWS collected: %d", len(items))
    return items

# -------- NEWS → FA (строгая агрегация для FA_Signals) --------
def _news_is_high(row: dict) -> bool:
    if ALLOWED_SOURCES and row["source"].upper() not in ALLOWED_SOURCES: return False
    hay = f"{row['title']} {row['tags']}"
    return bool(KW_RE.search(hay))

def _read_news_rows(sh) -> List[dict]:
    try: ws = sh.worksheet("NEWS")
    except Exception: return []
    rows = ws.get_all_records()
    out = []
    for r in rows:
        try:
            ts = datetime.fromisoformat(str(r.get("ts_utc")).replace("Z","+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        out.append({
            "ts_utc": ts,
            "source": str(r.get("source","")).strip(),
            "title":  str(r.get("title","")).strip(),
            "url":    str(r.get("url","")).strip(),
            "countries": str(r.get("countries","")).strip().lower(),
            "ccy":    str(r.get("ccy","")).strip().upper(),
            "tags":   str(r.get("tags","")).strip(),
        })
    return out

def compute_fa_from_news(all_news: List[dict], now_utc: datetime) -> Dict[str, dict]:
    window_start = now_utc - timedelta(minutes=FA_NEWS_RECENT_MIN)
    recent = [r for r in all_news if r["ts_utc"] >= window_start and _news_is_high(r)]
    cty_hits: dict[str, int] = {}
    for r in recent:
        for c in [x.strip() for x in (r["countries"] or "").split(",") if x.strip()]:
            cty_hits[c] = cty_hits.get(c, 0) + 1
    out: Dict[str, dict] = {}
    for sym, countries in PAIR_COUNTRIES.items():
        cnt = sum(cty_hits.get(c,0) for c in countries)
        is_red = cnt >= FA_NEWS_MIN_COUNT
        out[sym] = {
            "pair": sym, "risk": "Red" if is_red else "Green", "bias": "neutral",
            "ttl": FA_NEWS_TTL_MIN, "updated_at": _to_utc_iso(now_utc),
            "scan_lock_until": _to_utc_iso(now_utc + timedelta(minutes=FA_NEWS_LOCK_MIN)) if is_red else "",
            "reserve_off": 1 if is_red else 0, "dca_scale": 0.5 if is_red else 1.0,
            "reason": "news-high" if is_red else "base", "risk_pct": 0,
        }
    return out

def write_fa_signals(sh, signals: Dict[str, dict]):
    headers = ["pair","risk","bias","ttl","updated_at","scan_lock_until","reserve_off","dca_scale","reason","risk_pct"]
    ws, _ = ensure_worksheet(sh, "FA_Signals", headers)
    order = ["USDJPY","AUDUSD","EURUSD","GBPUSD"]
    values = []
    for sym in order:
        s = signals.get(sym, {})
        values.append([
            s.get("pair", sym), s.get("risk","Green"), s.get("bias","neutral"),
            s.get("ttl",""), s.get("updated_at",""), s.get("scan_lock_until",""),
            int(bool(s.get("reserve_off",0))), float(s.get("dca_scale",1.0)),
            s.get("reason","base"), int(s.get("risk_pct",0)),
        ])
    ws.update("A2", values)

def _read_existing_hashes(ws, hash_col_letter: str) -> set[str]:
    try:
        rng = f"{hash_col_letter}2:{hash_col_letter}10000"
        col = ws.get_values(rng)
        return {r[0] for r in col if r and r[0]}
    except Exception:
        return set()

def _calendar_window_filter(events: Iterable[CalEvent]) -> List[CalEvent]:
    now = datetime.now(timezone.utc)
    d1 = now - timedelta(days=FREE_LOOKBACK_DAYS)
    d2 = now + timedelta(days=FREE_LOOKAHEAD_DAYS)
    return [e for e in events if d1 <= e.utc <= d2]

def collect_once():
    log.info("collector… sheet=%s ws=%s tz=%s window=[-%dd,+%dd]", SHEET_ID, CAL_WS_OUT, (LOCAL_TZ.key if LOCAL_TZ else "UTC"), FREE_LOOKBACK_DAYS, FREE_LOOKAHEAD_DAYS)
    sh, _ = build_sheets_client(SHEET_ID)
    if not sh:
        log.error("Sheets not available — exit")
        return

    # CALENDAR
    cal_events = _calendar_window_filter(collect_calendar())
    ws_cal, _ = ensure_worksheet(sh, CAL_WS_OUT, CAL_HEADERS)
    try:
        existing = ws_cal.get_all_records()
        seen = {(r["utc_iso"], r.get("title","")) for r in existing}
    except Exception:
        seen = set()
    new_rows = []
    for ev in cal_events:
        key = (_to_utc_iso(ev.utc), ev.title)
        if key not in seen:
            new_rows.append(ev.to_row())
    if new_rows:
        ws_cal.append_rows(new_rows, value_input_option="RAW")
    log.info("CALENDAR: +%d rows", len(new_rows))

    # NEWS
    news = collect_news()
    ws_news, _ = ensure_worksheet(sh, "NEWS", NEWS_HEADERS)
    existing_hash = _read_existing_hashes(ws_news, "I")
    news_rows = [n.to_row() for n in news if n.key_hash() not in existing_hash]
    if news_rows:
        ws_news.append_rows(news_rows, value_input_option="RAW")
    log.info("NEWS: +%d rows", len(news_rows))

    # FA signals (строго)
    all_news = _read_news_rows(sh)
    fa = compute_fa_from_news(all_news, datetime.now(timezone.utc))
    write_fa_signals(sh, fa)
    log.info("cycle done.")

def main():
    if not SHEET_ID: raise RuntimeError("SHEET_ID env empty")
    if RUN_FOREVER:
        interval = max(1, COLLECT_EVERY_MIN)*60
        while True:
            try: collect_once()
            except Exception: log.exception("collect_once failed")
            time.sleep(interval)
    else:
        collect_once()

if __name__ == "__main__":
    main()
