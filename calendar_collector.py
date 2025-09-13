# calendar_collector.py — собирает CALENDAR, NEWS, FA_Signals и INVESTOR_DIGEST в Google Sheets
import os, re, json, base64, time, logging, html as _html
from dataclasses import dataclass
from typing import Optional, List, Dict, Iterable
from datetime import datetime, timedelta, timezone, time as _time
from html import unescape as _html_unescape

# ---- TZ ----
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/Belgrade")) if ZoneInfo else None

# ---- HTTP ----
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

# ---- Logging ----
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("calendar_collector")

# ---- ENV & Constants ----
SHEET_ID = os.getenv("SHEET_ID", "").strip()

CAL_WS_OUT = os.getenv("CAL_WS_OUT", "CALENDAR").strip() or "CALENDAR"
CAL_WS_RAW = os.getenv("CAL_WS_RAW", "CALENDAR_RAW").strip() or "CALENDAR_RAW"

RUN_FOREVER = (os.getenv("RUN_FOREVER", "1").lower() in ("1", "true", "yes", "on"))
COLLECT_EVERY_MIN = int(os.getenv("COLLECT_EVERY_MIN", "20") or "20")

FREE_LOOKBACK_DAYS  = int(os.getenv("FREE_LOOKBACK_DAYS", "8") or "8")
FREE_LOOKAHEAD_DAYS = int(os.getenv("FREE_LOOKAHEAD_DAYS", "30") or "30")

JINA_PROXY = os.getenv("JINA_PROXY", "https://r.jina.ai/http/").rstrip("/") + "/"

ALLOWED_SOURCES = {
    s.strip().upper()
    for s in os.getenv("FA_NEWS_SOURCES", "US_FED_PR,ECB_PR,BOE_PR,BOJ_PR,RBA_MR,US_TREASURY,JP_MOF_FX").split(",")
    if s.strip()
}

KW_RE = re.compile(os.getenv(
    "FA_NEWS_KEYWORDS",
    "rate decision|monetary policy|bank rate|policy decision|unscheduled|emergency|"
    "intervention|FX intervention|press conference|policy statement|policy statements|"
    "rate statement|cash rate|fomc|mpc"
), re.I)

FA_NEWS_RECENT_MIN = int(os.getenv("FA_NEWS_RECENT_MIN", "30"))
FA_NEWS_MIN_COUNT  = int(os.getenv("FA_NEWS_MIN_COUNT", "2"))
FA_NEWS_LOCK_MIN   = int(os.getenv("FA_NEWS_LOCK_MIN", "30"))
FA_NEWS_TTL_MIN    = int(os.getenv("FA_NEWS_TTL_MIN", "120"))

SYMBOLS = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]
PAIR_COUNTRIES = {
    "USDJPY": {"united states", "japan"},
    "AUDUSD": {"australia", "united states"},
    "EURUSD": {"euro area", "united states"},
    "GBPUSD": {"united kingdom", "united states"},
}

SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
FA_BADGE = {"Green": "🟢", "Amber": "🟡", "Red": "🔴"}

ECB_CAL_URLS = [
    "https://www.ecb.europa.eu/press/govcdec/html/index.en.html",
    "https://www.ecb.europa.eu/press/calendars/govc/html/index.en.html",
    "https://www.ecb.europa.eu/press/calendar/html/index.en.html",
    "https://www.ecb.europa.eu/press/calendar",
]
BOE_CAL_URLS = [
    "https://www.bankofengland.co.uk/interest-rates/meeting-dates",
    "https://www.bankofengland.co.uk/monetary-policy",
]
RBA_CAL_URLS = [
    "https://www.rba.gov.au/monetary-policy/2025.html",
    "https://www.rba.gov.au/media-releases/2025/mr-25-02.html",
    "https://www.rba.gov.au/monetary-policy/rba-board.html",
]

# ---- Helpers: creds ----
def _decode_b64_to_json(s: str) -> Optional[dict]:
    s = (s or "").strip()
    if not s:
        return None
    s += "=" * ((4 - len(s) % 4) % 4)
    try:
        return json.loads(base64.b64decode(s).decode("utf-8", "strict"))
    except Exception:
        return None

def _load_service_info() -> Optional[dict]:
    info = _decode_b64_to_json(os.getenv("GOOGLE_CREDENTIALS_JSON_B64", ""))
    if info:
        return info
    for k in ("GOOGLE_CREDENTIALS_JSON", "GOOGLE_CREDENTIALS"):
        raw = os.getenv(k, "").strip()
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
    return None

def build_sheets_client(sheet_id: str):
    if not _GSHEETS_AVAILABLE:
        return None, "gsheets libs not installed"
    if not sheet_id:
        return None, "SHEET_ID empty"
    info = _load_service_info()
    if not info:
        return None, "no service account json"
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
                    ws.update(range_name="A1", values=[headers])
            except Exception:
                pass
            return ws, False
    ws = sh.add_worksheet(title=title, rows=200, cols=max(10, len(headers)))
    ws.update(range_name="A1", values=[headers])
    return ws, True

# ---- Time helpers ----
def _to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _localize(dt: datetime) -> datetime:
    return dt.astimezone(LOCAL_TZ) if LOCAL_TZ else dt

def _norm_iso_key(v: str) -> str:
    s = str(v or "").strip()
    s = s.replace(" ", "T").replace("Z", "+00:00")
    s = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", s)
    try:
        dt = datetime.fromisoformat(s).astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return s

# ---- HTTP helpers ----
def fetch_text(url: str, timeout=15.0) -> tuple[str, int, str]:
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (LPBot/1.0)",
                "Accept-Language": "en,en-US;q=0.9"
            }
        ) as cli:
            r = cli.get(url)
            if r.status_code in (403, 404, 429, 503):
                pr = cli.get(JINA_PROXY + url)
                return pr.text or "", (200 if pr.text else pr.status_code), url
            return r.text, r.status_code, str(r.url)
    except Exception as e:
        log.warning("fetch failed %s: %s (retry via proxy)", url, e)
        try:
            pr = httpx.get(JINA_PROXY + url, timeout=timeout)
            return pr.text or "", (200 if pr.text else pr.status_code), url
        except Exception as e2:
            log.warning("proxy fetch failed %s: %s", url, e2)
            return "", 0, url

def _html_to_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = _html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---- Models ----
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
        return [
            _to_utc_iso(self.utc),
            local.strftime("%Y-%m-%d %H:%M"),
            self.country,
            self.currency,
            self.title,
            self.impact,
            self.source,
            self.url,
        ]

CAL_HEADERS  = ["utc_iso", "local_time", "country", "currency", "title", "impact", "source", "url"]
NEWS_HEADERS = ["ts_utc", "source", "title", "url", "countries", "ccy", "tags", "importance_guess", "hash"]

# ---- Universal Calendar Parsing Helpers ----
def _has_policy_context(html_text: str, patterns: List[str]) -> bool:
    t = _html_to_text(html_text)
    return any(re.search(p, t, re.I) for p in patterns)

def _anchor_hm(env_name: str, default: str) -> tuple[int, int]:
    s = (os.getenv(env_name, default) or default).strip()
    m = re.match(r"^\s*(\d{1,2})(?::(\d{1,2}))?\s*$", s)
    if not m:
        return 12, 0
    h = max(0, min(23, int(m.group(1))))
    mi = max(0, min(59, int(m.group(2) or "0")))
    return h, mi

def _mk_dt(y: int, mo: int, d: int, env_name: str, default: str) -> datetime:
    h, mi = _anchor_hm(env_name, default)
    return datetime(y, mo, d, h, mi, 0, tzinfo=timezone.utc)

def _guess_context_year(text: str) -> int:
    cur_year = datetime.now(timezone.utc).year
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", _html_to_text(text))]
    if not years:
        return cur_year
    if cur_year in years:
        return cur_year
    if (cur_year + 1) in years:
        return cur_year + 1
    return max(years)

_MONTH = {m:i for i,m in enumerate(
    ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], start=1
)}

def _scan_dates_any(text: str) -> list[tuple[int,int,int]]:
    t = _html_to_text(text)
    ctx_year = _guess_context_year(t)
    out: list[tuple[int,int,int]] = []
    MON = (
        r"(Jan(?:\.|uary)?|Feb(?:\.|ruary)?|Mar(?:\.|ch)?|Apr(?:\.|il)?|"
        r"May|Jun(?:\.|e)?|Jul(?:\.|y)?|Aug(?:\.|ust)?|"
        r"Sep(?:\.|t|tember)?|Oct(?:\.|ober)?|Nov(?:\.|ember)?|Dec(?:\.|ember)?)"
    )
    def mon2num(s: str) -> int:
        return _MONTH[s.lower().replace(".", "")[:3]]
    p1 = re.compile(rf"\b(\d{{1,2}})(?:\s*[–—\-]\s*\d{{1,2}})?\s+{MON}"
                    r"\s*,?\s*(20\d{2})\b", re.I)
    for m in p1.finditer(t):
        d = int(m.group(1)); mo = mon2num(m.group(2)); y = int(m.group(3))
        out.append((y, mo, d))
    p2 = re.compile(rf"\b{MON}\s+(\d{{1,2}})(?:\s*[–—\-]\s*\d{{1,2}})?,\s*(20\d{{2}})\b", re.I)
    for m in p2.finditer(t):
        mo = mon2num(m.group(1)); d = int(m.group(2)); y = int(m.group(3))
        out.append((y, mo, d))
    p3 = re.compile(r"\b(20\d{2})[./\-](\d{1,2})[./\-](\d{1,2})\b")
    for m in p3.finditer(t):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        out.append((y, mo, d))
    p4 = re.compile(rf"\b(\d{{1,2}})\s+{MON}[a-z]*\s+(20\d{{2}})\b", re.I)
    for m in p4.finditer(t):
        d = int(m.group(1)); mo = mon2num(m.group(2)); y = int(m.group(3))
        out.append((y, mo, d))
    p5 = re.compile(rf"\b{MON}\s+(\d{{1,2}})(?:\s*[–—\-]\s*\d{{1,2}})?\b", re.I)
    for m in p5.finditer(t):
        mo = mon2num(m.group(1)); d = int(m.group(2))
        out.append((ctx_year, mo, d))
    p6 = re.compile(rf"\b(\d{{1,2}})\s+{MON}\b", re.I)
    for m in p6.finditer(t):
        d = int(m.group(1)); mo = mon2num(m.group(2))
        out.append((ctx_year, mo, d))
    cur_year = datetime.now(timezone.utc).year
    uniq = {}
    for y, mo, d in out:
        if not (cur_year - 1 <= y <= cur_year + 1):
            continue
        try:
            datetime(y, mo, d)
            uniq[(y, mo, d)] = (y, mo, d)
        except ValueError:
            pass
    return list(uniq.values())

def _selftest_scan_dates():
    """For local testing of date regexes."""
    log.info("--- Running date regex selftest ---")
    samples = {
        "11 September 2025": (2025, 9, 11), "September 11, 2025": (2025, 9, 11),
        "2025-09-11": (2025, 9, 11), "Sep. 20–21": (datetime.now(timezone.utc).year, 9, 20),
        "11 Sep 2025": (2025, 9, 11), "Sep 11": (datetime.now(timezone.utc).year, 9, 11),
    }
    for s, expected in samples.items():
        res = _scan_dates_any(s)
        if res and res[0] == expected:
            log.info("OK: '%s' -> %s", s, res)
        else:
            log.error("FAIL: '%s' -> %s (expected %s)", s, res, expected)
    log.info("--- Selftest done ---")

# ---- Calendar parsers ----
def _fomc_range_end_dates(text: str) -> set[tuple[int,int,int]]:
    """Концы диапазонов (вторые дни) для FOMC: 'Sep 16–17, 2025', '16–17 Sep 2025',
    а также варианты БЕЗ года ('Sep 16–17' / '16–17 Sep') с подстановкой контекстного года."""
    t = _html_to_text(text)
    ctx_year = _guess_context_year(t)
    cur_year = datetime.now(timezone.utc).year
    MON = (
        r"(Jan(?:\.|uary)?|Feb(?:\.|ruary)?|Mar(?:\.|ch)?|Apr(?:\.|il)?|"
        r"May|Jun(?:\.|e)?|Jul(?:\.|y)?|Aug(?:\.|ust)?|"
        r"Sep(?:\.|t|tember)?|Oct(?:\.|ober)?|Nov(?:\.|ember)?|Dec(?:\.|ember)?)"
    )
    def mon2num(s: str) -> int:
        return _MONTH[s.lower().replace('.', '')[:3]]
    ends: set[tuple[int,int,int]] = set()
    # Вариант: Month d1–d2, YYYY
    pA = re.compile(rf"\b{MON}\s+(\d{{1,2}})\s*[–—\-]\s*(\d{{1,2}})\s*,\s*(20\d{{2}})\b", re.I)
    for m in pA.finditer(t):
        mo = mon2num(m.group(1)); d2 = int(m.group(3)); y = int(m.group(4))
        ends.add((y, mo, d2))
    # Вариант: d1–d2 Month YYYY
    pB = re.compile(rf"\b(\d{{1,2}})\s*[–—\-]\s*(\d{{1,2}})\s+{MON}\s*(20\d{{2}})\b", re.I)
    for m in pB.finditer(t):
        d2 = int(m.group(2)); mo = mon2num(m.group(3)); y = int(m.group(4))
        ends.add((y, mo, d2))
    # Вариант БЕЗ года: Month d1–d2  -> год = контекстный
    pC = re.compile(rf"\b{MON}\s+(\d{{1,2}})\s*[–—\-]\s*(\d{{1,2}})\b", re.I)
    for m in pC.finditer(t):
        mo = mon2num(m.group(1)); d2 = int(m.group(3))
        ends.add((ctx_year, mo, d2))
    # Вариант БЕЗ года: d1–d2 Month  -> год = контекстный
    pD = re.compile(rf"\b(\d{{1,2}})\s*[–—\-]\s*(\d{{1,2}})\s+{MON}\b", re.I)
    for m in pD.finditer(t):
        d2 = int(m.group(2)); mo = mon2num(m.group(3))
        ends.add((ctx_year, mo, d2))
    # Слегка ограничим «дальние» года, чтобы не тащить 2028+ в coverage.
    ends = { (y, mo, d) for (y, mo, d) in ends if (cur_year - 1) <= y <= (cur_year + 1) }
    return ends

def parse_ecb_calendar(html: str, url: str) -> list[CalEvent]:
    if "govc/html" not in url and not _has_policy_context(html, [
        r"governing council", r"monetary policy", r"rate decision"
    ]): return []
    out: list[CalEvent] = []
    for y, mo, d in _scan_dates_any(html):
        try:
            dt = _mk_dt(y, mo, d, "CAL_ECB_ANCHOR_UTC", "12:45")
            out.append(CalEvent(utc=dt, country="euro area", currency="EUR",
                title="ECB Governing Council — Monetary Policy Meeting",
                impact="high", source="ECB", url=url))
        except Exception: continue
    return list({ev.utc.date(): ev for ev in out}.values())

def parse_boe_calendar(html: str, url: str) -> list[CalEvent]:
    if not _has_policy_context(html, [r"monetary policy committee", r"\bMPC\b", r"bank rate", r"policy decision"]): return []
    out: list[CalEvent] = []
    for y, mo, d in _scan_dates_any(html):
        try:
            dt = _mk_dt(y, mo, d, "CAL_BOE_ANCHOR_UTC", "11:00")
            out.append(CalEvent(utc=dt, country="united kingdom", currency="GBP",
                title="BoE MPC Meeting / Rate Decision",
                impact="high", source="BoE", url=url))
        except Exception: continue
    return list({ev.utc.date(): ev for ev in out}.values())

def parse_rba_calendar(html: str, url: str) -> list[CalEvent]:
    if not _has_policy_context(html, [
        r"monetary policy board", r"cash rate", r"board meeting",
        r"reserve bank board", r"meeting dates"
    ]): return []
    out: list[CalEvent] = []
    for y, mo, d in _scan_dates_any(html):
        try:
            dt = _mk_dt(y, mo, d, "CAL_RBA_ANCHOR_UTC", "04:30")
            out.append(CalEvent(utc=dt, country="australia", currency="AUD",
                title="RBA Monetary Policy Board Meeting",
                impact="high", source="RBA", url=url))
        except Exception: continue
    return list({ev.utc.date(): ev for ev in out}.values())

def parse_fomc_calendar(html: str, url: str) -> List[CalEvent]:
    """FOMC: берём вторые дни диапазонов (с учётом строк без года). Если ничего не нашли — fallback на общий сканер."""
    out: List[CalEvent] = []
    range_ends = _fomc_range_end_dates(html)
    triples = range_ends or set(_scan_dates_any(html))
    for (y, mo, d) in triples:
        try:
            dt = _mk_dt(y, mo, d, "CAL_FOMC_ANCHOR_UTC", "18:00")
            out.append(CalEvent(
                utc=dt, country="united states", currency="USD",
                title="FOMC Meeting / Rate Decision", impact="high", source="FOMC", url=url
            ))
        except Exception:
            continue
    return list({ev.utc.date(): ev for ev in out}.values())

def parse_boj_calendar(html: str, url: str) -> List[CalEvent]:
    if not _has_policy_context(html, [r"monetary policy meeting", r"\bMPM\b", r"monetary policy"]): return []
    out: list[CalEvent] = []
    for y, mo, d in _scan_dates_any(html):
        try:
            dt = _mk_dt(y, mo, d, "CAL_BOJ_ANCHOR_UTC", "03:00")
            out.append(CalEvent(utc=dt, country="japan", currency="JPY",
                title="BoJ Monetary Policy Meeting", impact="high", source="BoJ", url=url))
        except Exception: continue
    return list({ev.utc.date(): ev for ev in out}.values())

def _log_calendar_coverage(events: List[CalEvent]):
    cnt = {}
    for e in events: cnt[e.currency] = cnt.get(e.currency, 0) + 1
    for c in ["USD","JPY","EUR","GBP","AUD"]:
        log.info("CALENDAR coverage %s: %d", c, cnt.get(c, 0))
    examples = {c: [] for c in ["USD","JPY","EUR","GBP","AUD"]}
    for ev in sorted(events, key=lambda e: e.utc):
        if ev.currency in examples and len(examples[ev.currency]) < 3:
            examples[ev.currency].append(f"{ev.utc.strftime('%Y-%m-%d')} ({ev.source})")
    for c, ex_list in examples.items():
        if ex_list: log.info("CALENDAR examples %s: %s", c, ", ".join(ex_list))

def _rba_schedule_url() -> Optional[str]:
    txt, code, u = fetch_text("https://www.rba.gov.au/media-releases/")
    if code == 200 and txt:
        m = re.search(
            r'href="(/media-releases/\d{4}/[^"]+\.html)"[^>]*>\s*[^<]*Reserve Bank Board Meeting Dates',
            txt, re.I)
        if m: return "https://www.rba.gov.au" + m.group(1)
    return None

def collect_calendar() -> List[CalEvent]:
    events: List[CalEvent] = []
    
    url_fomc = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    txt, code, f_url = fetch_text(url_fomc)
    if code == 200 and txt:
        f = parse_fomc_calendar(txt, f_url)
        if f: log.info("FOMC parsed: %d from %s", len(f), f_url)
        events.extend(f)

    url_boj = "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm"
    txt, code, b_url = fetch_text(url_boj)
    if code == 200 and txt:
        b = parse_boj_calendar(txt, b_url)
        if b: log.info("BoJ parsed: %d from %s", len(b), b_url)
        events.extend(b)

    for name, urls in [("ECB", ECB_CAL_URLS), ("BoE", BOE_CAL_URLS)]:
        cb_events, sources = [], []
        parser = globals()[f"parse_{name.lower()}_calendar"]
        for url in urls:
            txt, code, u = fetch_text(url)
            if code == 200 and txt:
                parsed = parser(txt, u)
                if parsed:
                    cb_events.extend(parsed); sources.append(f"{u} ({len(parsed)})")
        final_cb = list({ev.utc.date(): ev for ev in cb_events}.values())
        if sources: log.info("%s parsed: %d from %s", name, len(final_cb), ", ".join(sources))
        events.extend(final_cb)
    
    rba_urls = list(RBA_CAL_URLS)
    if u := _rba_schedule_url(): rba_urls.insert(0, u)
    rba_events = []
    for url in rba_urls:
        txt, code, ru = fetch_text(url)
        if code == 200 and txt: rba_events.extend(parse_rba_calendar(txt, ru))
    uniq_rba = {ev.utc.date(): ev for ev in rba_events}
    final_rba = list(uniq_rba.values())
    if final_rba:
        log.info("RBA parsed: %d from %s", len(final_rba), ", ".join(sorted({e.url for e in final_rba})))
    events.extend(final_rba)

    final_unique_events = { (ev.utc.date(), ev.country): ev for ev in events }
    out = list(final_unique_events.values())
    out.sort(key=lambda e: (e.utc, e.country))
    _log_calendar_coverage(out)
    return out

# ---- NEWS scrapers ----
@dataclass
class NewsItem:
    ts_utc: datetime; source: str; title: str; url: str
    countries: str; ccy: str; tags: str; importance_guess: str

    def key_hash(self) -> str:
        return re.sub(r"\s+", " ", f"{self.source}|{self.url}".strip())[:180]

    def to_row(self) -> List[str]:
        clean_title = _html_unescape(re.sub(r"<[^>]+>", "", self.title)).strip()
        return [
            _to_utc_iso(self.ts_utc), self.source, clean_title, self.url,
            self.countries, self.ccy, self.tags, self.importance_guess, self.key_hash(),
        ]

def _mof_fx_best_url() -> Optional[str]:
    cands = [
        "https://www.mof.go.jp/english/policy/international_policy/reference/feio/index.html",
        "https://www.mof.go.jp/en/policy/international_policy/reference/feio/index.html",
        "https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/",
    ]
    for u in cands:
        txt, code, _ = fetch_text(u)
        if code == 200 and re.search(r"(foreign|fx).{0,20}intervention|feio", txt, re.I): return u
    return None

def collect_news() -> List[NewsItem]:
    items: List[NewsItem] = []
    now = datetime.now(timezone.utc)

    txt, code, _ = fetch_text("https://www.federalreserve.gov/newsevents/pressreleases.htm")
    if code == 200 and txt:
        for m in re.finditer(r'href="(/newsevents/pressreleases/(?:20\d{2}-press-(?:fomc|monetary)|20\d{2}-press-\w+)\.htm)"[^>]*>(.*?)</a>', txt, re.I):
            u, t = "https://www.federalreserve.gov" + m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip() or "Fed press release"
            imp = "high" if KW_RE.search(t) else "medium"
            items.append(NewsItem(now, "US_FED_PR", t, u, "united states", "USD", "policy", imp))

    txt, code, _ = fetch_text("https://home.treasury.gov/news/press-releases")
    if code == 200 and txt:
        for m in re.finditer(r'href="(/news/press-releases/sb\d+)"[^>]*>(.*?)</a>', txt, re.I):
            u, t = "https://home.treasury.gov" + m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
            items.append(NewsItem(now, "US_TREASURY", t, u, "united states", "USD", "treasury", "medium"))

    txt, code, _ = fetch_text("https://www.ecb.europa.eu/press/pubbydate/html/index.en.html?name_of_publication=Press%20release")
    if code == 200 and txt:
        for m in re.finditer(r'href="(/press/(?:govcdec|press_conference|pr)/[^"]+)"[^>]*>(.*?)</a>', txt, re.I):
            u, t = "https://www.ecb.europa.eu" + m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
            imp = "high" if KW_RE.search(t) else "medium"
            items.append(NewsItem(now, "ECB_PR", t, u, "euro area", "EUR", "ecb", imp))

    txt, code, _ = fetch_text("https://www.bankofengland.co.uk/news")
    if code == 200 and txt:
        for m in re.finditer(r'href="(/news/20\d{2}/[^"]+)"[^>]*>(.*?)</a>', txt, re.I):
            u, t = "https://www.bankofengland.co.uk" + m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
            imp = "high" if KW_RE.search(t) else "medium"
            items.append(NewsItem(now, "BOE_PR", t, u, "united kingdom", "GBP", "boe", imp))

    txt, code, _ = fetch_text("https://www.rba.gov.au/media-releases/")
    if code == 200 and txt:
        for m in re.finditer(r'href="(/media-releases/\d{4}/[^"]+)"[^>]*>(.*?)</a>', txt, re.I):
            u, t = "https://www.rba.gov.au" + m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
            imp = "high" if KW_RE.search(t) else "medium"
            items.append(NewsItem(now, "RBA_MR", t, u, "australia", "AUD", "rba", imp))

    if u := _mof_fx_best_url():
        items.append(NewsItem(now, "JP_MOF_FX", "FX intervention reference page", u, "japan", "JPY", "mof", "high"))

    max_age_days = int(os.getenv("NEWS_MAX_AGE_DAYS", "90") or "90")
    cutoff = now - timedelta(days=max_age_days)
    items = [i for i in items if i.ts_utc >= cutoff]
    log.info("NEWS collected: %d items", len(items))
    return items

# ---- NEWS → FA ----
def _news_is_high(row: dict) -> bool:
    if ALLOWED_SOURCES and row["source"].upper() not in ALLOWED_SOURCES: return False
    return bool(KW_RE.search(f"{row['title']} {row['tags']}"))

def _read_news_rows(sh) -> List[dict]:
    try: ws = sh.worksheet("NEWS")
    except Exception: return []
    out = []
    for r in ws.get_all_records():
        try:
            ts = datetime.fromisoformat(str(r.get("ts_utc")).replace("Z", "+00:00")).astimezone(timezone.utc)
            out.append({ "ts_utc": ts, "source": str(r.get("source", "")).strip(),
                "title":  str(r.get("title", "")).strip(), "url": str(r.get("url", "")).strip(),
                "countries": str(r.get("countries", "")).strip().lower(),
                "ccy": str(r.get("ccy", "")).strip().upper(), "tags": str(r.get("tags", "")).strip() })
        except Exception: continue
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
        cnt = sum(1 if cty_hits.get(c, 0) > 0 else 0 for c in countries)
        if cnt >= max(1, FA_NEWS_MIN_COUNT):
            risk, dca, reserve, lock, reason = "Red", 0.5, 1, FA_NEWS_LOCK_MIN, "news-high"
        elif cnt == 1:
            risk, dca, reserve, lock, reason = "Amber", 0.75, 0, max(1, FA_NEWS_LOCK_MIN // 2), "news-medium"
        else:
            risk, dca, reserve, lock, reason = "Green", 1.0, 0, 0, "base"
        out[sym] = {
            "pair": sym, "risk": risk, "bias": "neutral", "ttl": FA_NEWS_TTL_MIN,
            "updated_at": _to_utc_iso(now_utc),
            "scan_lock_until": _to_utc_iso(now_utc + timedelta(minutes=lock)) if lock else "",
            "reserve_off": reserve, "dca_scale": dca, "reason": reason, "risk_pct": 0,
        }
    return out

def write_fa_signals(sh, signals: Dict[str, dict]):
    headers = ["pair", "risk", "bias", "ttl", "updated_at", "scan_lock_until", "reserve_off", "dca_scale", "reason", "risk_pct"]
    ws, _ = ensure_worksheet(sh, "FA_Signals", headers)
    values = []
    for sym in ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]:
        s = signals.get(sym, {})
        values.append([
            s.get("pair", sym), s.get("risk", "Green"), s.get("bias", "neutral"),
            s.get("ttl", ""), s.get("updated_at", ""), s.get("scan_lock_until", ""),
            int(bool(s.get("reserve_off", 0))), float(s.get("dca_scale", 1.0)),
            s.get("reason", "base"), int(s.get("risk_pct", 0)),
        ])
    ws.update(range_name="A2", values=values)

def _read_existing_hashes(ws, col: str) -> set[str]:
    try: return {r[0] for r in ws.get_values(f"{col}2:{col}10000") if r and r[0]}
    except Exception: return set()

def _calendar_window_filter(events: Iterable[CalEvent]) -> List[CalEvent]:
    today = datetime.now(timezone.utc).date()
    d1 = datetime.combine(today - timedelta(days=FREE_LOOKBACK_DAYS), _time.min, tzinfo=timezone.utc)
    d2 = datetime.combine(today + timedelta(days=FREE_LOOKAHEAD_DAYS), _time.max, tzinfo=timezone.utc)
    return [e for e in events if d1 <= e.utc <= d2]

def _render_investor_digest_text(signals: Dict[str, dict]) -> str:
    lines = ["FA-сводка (aggregated):"]
    for sym in ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD"]:
        s = signals.get(sym, {})
        risk, badge = s.get("risk", "Green"), FA_BADGE.get(s.get("risk", "Green"), "⚪️")
        bias, reason = s.get("bias", "neutral"), s.get("reason", "base")
        dca = s.get("dca_scale", 1.0)
        extra = f" | dca×{dca:g}" if risk != "Green" or dca != 1.0 or reason != "base" else ""
        lines.append(f"• {sym[:3]}/{sym[3:]}: {badge} {risk}/{bias} | {reason}{extra}")
    return "\n".join(lines)

def write_investor_digest_row(sh, signals: Dict[str, dict], now_utc: datetime):
    ws, _ = ensure_worksheet(sh, "INVESTOR_DIGEST", ["ts_utc", "text"])
    ws.append_row([_to_utc_iso(now_utc), _render_investor_digest_text(signals)], value_input_option="RAW")

# ---- Main cycle ----
def collect_once():
    log.info("collector… sheet=%s ws=%s tz=%s window=[-%dd,+%dd]",
        SHEET_ID, CAL_WS_OUT, (LOCAL_TZ.key if LOCAL_TZ else "UTC"),
        FREE_LOOKBACK_DAYS, FREE_LOOKAHEAD_DAYS)
    sh, _ = build_sheets_client(SHEET_ID)
    if not sh: return log.error("Sheets not available — exit")

    ws_cal, _ = ensure_worksheet(sh, CAL_WS_OUT, CAL_HEADERS)
    try:
        existing = ws_cal.get_all_records()
        seen = {(_norm_iso_key(r.get("utc_iso")), (r.get("title") or "").strip()) for r in existing}
        seen_date_cty = set()
        for r in existing:
            try:
                d = datetime.fromisoformat(str(r.get("utc_iso")).replace("Z","+00:00")).date()
                if c := str(r.get("country","")).strip().lower(): seen_date_cty.add((d, c))
            except Exception: pass
    except Exception: seen, seen_date_cty = set(), set()

    all_events = collect_calendar()
    win_events = _calendar_window_filter(all_events)
    if win_events: log.info("CALENDAR window: %d/%d kept", len(win_events), len(all_events))
    else: log.info("CALENDAR window: 0/%d kept", len(all_events))

    new_rows = []
    for ev in win_events:
        key1 = (_norm_iso_key(_to_utc_iso(ev.utc)), ev.title.strip())
        key2 = (ev.utc.date(), ev.country.strip().lower())
        if key1 in seen or key2 in seen_date_cty: continue
        new_rows.append(ev.to_row())
        seen.add(key1); seen_date_cty.add(key2)

    if new_rows:
        ws_cal.append_rows(new_rows, value_input_option="RAW")
        log.info("CALENDAR: +%d rows", len(new_rows))

    news = collect_news()
    ws_news, _ = ensure_worksheet(sh, "NEWS", NEWS_HEADERS)
    existing_hash = _read_existing_hashes(ws_news, "I")
    news_rows = [n.to_row() for n in news if n.key_hash() not in existing_hash]
    if news_rows:
        ws_news.append_rows(news_rows, value_input_option="RAW")
        log.info("NEWS: +%d rows", len(news_rows))

    all_news = _read_news_rows(sh)
    now = datetime.now(timezone.utc)
    fa = compute_fa_from_news(all_news, now)
    write_fa_signals(sh, fa)
    write_investor_digest_row(sh, fa, now)
    log.info("cycle done.")

def main():
    # _selftest_scan_dates()
    if not SHEET_ID: raise RuntimeError("SHEET_ID env empty")
    if RUN_FOREVER:
        interval = max(1, COLLECT_EVERY_MIN) * 60
        while True:
            try: collect_once()
            except Exception: log.exception("collect_once failed")
            time.sleep(interval)
    else:
        collect_once()

if __name__ == "__main__":
    main()
