# calendar_collector.py
import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
import time
import re

# ---- Optional deps: requests / bs4 ----
try:
    import requests
    from bs4 import BeautifulSoup
    _HTTP_OK = True
except Exception:
    requests = None
    BeautifulSoup = None
    _HTTP_OK = False

# ---- Google Sheets ----
try:
    import gspread
    from google.oauth2 import service_account
    _GS_OK = True
except Exception:
    gspread = None
    service_account = None
    _GS_OK = False

# -------------------- LOGS --------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("calendar_collector")

# -------------------- ENV --------------------
SHEET_ID = os.getenv("SHEET_ID", "").strip()
SHEET_WS_OUT = os.getenv("CAL_WS_OUT", "CALENDAR").strip() or "CALENDAR"
SHEET_WS_RAW = os.getenv("CAL_WS_RAW", "CALENDAR_RAW").strip() or "CALENDAR_RAW"

LOCAL_TZ_NAME = os.getenv("LOCAL_TZ", "Europe/Belgrade").strip()
try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)
except Exception:
    LOCAL_TZ = None

LOOKBACK_DAYS = int(os.getenv("FREE_LOOKBACK_DAYS", "7") or "7")
LOOKAHEAD_DAYS = int(os.getenv("FREE_LOOKAHEAD_DAYS", "180") or "180")
COLLECT_EVERY_MIN = int(os.getenv("COLLECT_EVERY_MIN", "180") or "180")
RUN_FOREVER = os.getenv("RUN_FOREVER", "1").strip() not in ("", "0", "false", "False")

# Bypass/reader proxy for tougher sites (optional). If enabled:
#   JINA_PROXY=1  -> we fetch via https://r.jina.ai/http://example.com
JINA_PROXY = os.getenv("JINA_PROXY", "1") not in ("0", "false", "False")

HEADERS = {
    "User-Agent": os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; fund-bot/1.0; +github.com)"),
    "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
}

COUNTRY_BY_CCY = {
    "USD": "united states",
    "JPY": "japan",
    "EUR": "euro area",
    "GBP": "united kingdom",
    "AUD": "australia",
}
CCY_BY_COUNTRY = {v: k for k, v in COUNTRY_BY_CCY.items()}

# -------------------- SHEETS --------------------
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def _decode_b64(s: str) -> Optional[dict]:
    import base64
    s = (s or "").strip()
    if not s:
        return None
    s += "=" * ((4 - len(s) % 4) % 4)
    return json.loads(base64.b64decode(s).decode("utf-8", "strict"))

def _load_service_info() -> Tuple[Optional[dict], str]:
    # 1) GOOGLE_CREDENTIALS_JSON_B64
    b64 = os.getenv("GOOGLE_CREDENTIALS_JSON_B64", "").strip()
    if b64:
        try:
            return _decode_b64(b64), "env:GOOGLE_CREDENTIALS_JSON_B64"
        except Exception as e:
            return None, f"bad b64: {e}"
    # 2) GOOGLE_CREDENTIALS_JSON / GOOGLE_CREDENTIALS
    for name in ("GOOGLE_CREDENTIALS_JSON", "GOOGLE_CREDENTIALS"):
        raw = os.getenv(name, "").strip()
        if raw:
            try:
                return json.loads(raw), f"env:{name}"
            except Exception as e:
                return None, f"{name} invalid json: {e}"
    return None, "not-found"

def open_sheet():
    if not _GS_OK:
        raise RuntimeError("gsheets libs not installed")
    if not SHEET_ID:
        raise RuntimeError("SHEET_ID empty")

    info, src = _load_service_info()
    if not info:
        raise RuntimeError(f"google creds not found ({src})")
    creds = service_account.Credentials.from_service_account_info(info, scopes=SHEETS_SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(SHEET_ID)
    return sh

def ensure_ws(sh, title: str, headers: List[str]):
    for ws in sh.worksheets():
        if ws.title == title:
            # ensure header
            try:
                cur = ws.get_values("A1:Z1") or [[]]
                if cur and cur[0] != headers:
                    ws.update(range_name="A1", values=[headers])
            except Exception:
                pass
            return ws
    ws = sh.add_worksheet(title=title, rows=2000, cols=max(12, len(headers)))
    ws.update(range_name="A1", values=[headers])
    return ws

# -------------------- HELPERS --------------------
def to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def to_local(dt: datetime) -> str:
    tz = LOCAL_TZ or timezone.utc
    return dt.astimezone(tz).strftime("%Y-%m-%d %H:%M")

def is_high(impact: str) -> bool:
    s = (impact or "").lower()
    return "high" in s or s == "3" or s == "red"

def clamp_country(name: str) -> str:
    n = (name or "").strip().lower()
    # normalize a few aliases
    n = {"united states of america":"united states", "u.s.":"united states", "us":"united states"}.get(n, n)
    return n

def event(country: str, title: str, utc_dt: datetime, impact: str, source: str, url: str = "") -> dict:
    c = clamp_country(country)
    return {
        "utc": utc_dt.replace(tzinfo=timezone.utc),
        "country": c,
        "currency": CCY_BY_COUNTRY.get(c, ""),
        "title": (title or "").strip(),
        "impact": (impact or "").strip(),
        "source": source,
        "url": url or "",
    }

# -------------------- SOURCES --------------------
def fetch_ff_json() -> List[dict]:
    """ForexFactory weekly JSON (may be empty or rate-limited)."""
    if not _HTTP_OK: return []
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        r = requests.get(url, timeout=12, headers=HEADERS)
        if r.status_code == 429:
            log.warning("FF JSON 429, skip this round")
            return []
        r.raise_for_status()
        data = []
        for it in r.json() or []:
            ts = it.get("timestamp")
            if not ts: continue
            dt_utc = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            country = it.get("country") or ""
            impact = it.get("impact") or ""
            title = it.get("title") or it.get("event") or "Event"
            data.append(event(country, title, dt_utc, impact, "ff/json", "https://www.forexfactory.com/calendar"))
        log.info("FF JSON parsed: %d", len(data))
        return data
    except Exception as e:
        log.info("FF JSON failed: %s", e)
        return []

def _via_proxy(url: str) -> str:
    # r.jina.ai proxy: r.jina.ai/http://example.com  OR  /https://
    if not JINA_PROXY:
        return url
    if url.startswith("https://"):
        return "https://r.jina.ai/https://" + url[len("https://"):]
    if url.startswith("http://"):
        return "https://r.jina.ai/http://" + url[len("http://"):]
    return url

def fetch_ff_html() -> List[dict]:
    """ForexFactory HTML (High only)."""
    if not (_HTTP_OK and BeautifulSoup):
        return []
    try:
        url = "https://www.forexfactory.com/calendar?week=this&impact=3"
        r = requests.get(_via_proxy(url), headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("table.calendar__table tr.calendar__row")
        out = []
        for tr in rows:
            imp = (tr.get("data-impact") or "").lower()
            if "high" not in imp:
                continue
            title_el = tr.select_one(".calendar__event-title, .calendar__event-title a")
            country_el = tr.select_one(".calendar__country")
            time_el = tr.select_one("[data-event-datetime]")
            if not (title_el and country_el and time_el):
                continue
            title = title_el.get_text(strip=True)
            country = country_el.get_text(strip=True)
            # FF often stores ISO-like datetime in UTC in data-event-datetime
            dt_s = time_el.get("data-event-datetime") or ""
            try:
                # Examples: 2025-09-10 12:30:00
                dt_utc = datetime.fromisoformat(dt_s.replace(" ", "T")).replace(tzinfo=timezone.utc)
            except Exception:
                continue
            out.append(event(country, title, dt_utc, "High", "ff/html", url))
        log.info("FF HTML parsed: %d high-impact", len(out))
        return out
    except Exception as e:
        log.info("FF HTML failed: %s", e)
        return []

def fetch_fomc() -> List[dict]:
    """Federal Reserve FOMC meeting days (approx parsing)."""
    if not _HTTP_OK: return []
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    try:
        r = requests.get(_via_proxy(url), headers=HEADERS, timeout=20)
        r.raise_for_status()
        # crude parse: look for YYYY (current & next), and <td>Month DD-??</td>
        text = r.text
        years = set(re.findall(r"(20\d{2})", text))
        out = []
        for m in re.finditer(r"(\w+\s+\d{1,2}(?:\–|-|\u2013)?\d{0,2}),\s*(20\d{2})", text):
            mm = m.group(1)  # e.g., "September 17–18"
            yy = m.group(2)
            # take first date in range
            dfirst = re.sub(r"(–|-|\u2013).*", "", mm).strip()
            dt_try = f"{dfirst} {yy}"
            try:
                dt_loc = datetime.strptime(dt_try, "%B %d %Y")
                # assume announcement ~ 18:00 UTC? We'll set 18:00 UTC placeholder.
                dt_utc = datetime(dt_loc.year, dt_loc.month, dt_loc.day, 18, 0, tzinfo=timezone.utc)
                out.append(event("united states", "FOMC meeting", dt_utc, "High", "fed"))
            except Exception:
                continue
        log.info("FOMC parsed: %d", len(out))
        return out
    except Exception as e:
        log.info("FOMC failed: %s", e)
        return []

def fetch_ecb() -> List[dict]:
    """ECB monetary policy meeting dates (coarse parse)."""
    if not _HTTP_OK: return []
    url = "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html"
    try:
        r = requests.get(_via_proxy(url), headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        out = []
        for li in soup.select("li"):
            t = li.get_text(" ", strip=True)
            # e.g., "Monetary policy meeting - 12 September 2025"
            if "Monetary policy meeting" in t:
                m = re.search(r"(\d{1,2}\s+\w+\s+20\d{2})", t)
                if not m: continue
                try:
                    dt = datetime.strptime(m.group(1), "%d %B %Y").replace(tzinfo=timezone.utc)
                    out.append(event("euro area", "ECB monetary policy meeting", dt, "High", "ecb"))
                except Exception:
                    pass
        log.info("ECB parsed: %d", len(out))
        return out
    except Exception as e:
        log.info("ECB failed: %s", e)
        return []

def fetch_boe() -> List[dict]:
    """BoE MPC dates (minutes/decision)."""
    if not (_HTTP_OK and BeautifulSoup): return []
    url = "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes"
    try:
        r = requests.get(_via_proxy(url), headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        out = []
        # The page lists upcoming dates in cards; we look for date patterns
        for el in soup.find_all(text=re.compile(r"\d{1,2}\s+\w+\s+20\d{2}")):
            t = el.strip()
            try:
                dt = datetime.strptime(t, "%d %B %Y").replace(tzinfo=timezone.utc)
                out.append(event("united kingdom", "BoE MPC announcement", dt, "High", "boe"))
            except Exception:
                continue
        log.info("BoE parsed: %d", len(out))
        return out
    except Exception as e:
        log.info("BoE failed: %s", e)
        return []

def fetch_boj() -> List[dict]:
    """BoJ MPM schedule (very rough)."""
    if not (_HTTP_OK and BeautifulSoup): return []
    url = "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm"
    try:
        r = requests.get(_via_proxy(url), headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        out = []
        for el in soup.find_all(text=re.compile(r"\d{1,2}\s+\w+\s+20\d{2}")):
            t = el.strip()
            for fmt in ("%d %B %Y", "%B %d %Y"):
                try:
                    dt = datetime.strptime(t, fmt).replace(tzinfo=timezone.utc)
                    out.append(event("japan", "BoJ policy meeting", dt, "High", "boj"))
                    break
                except Exception:
                    continue
        log.info("BoJ parsed: %d", len(out))
        return out
    except Exception as e:
        log.info("BoJ failed: %s", e)
        return []

def fetch_dailyfx() -> List[dict]:
    """DailyFX calendar (often 403; try reader proxy). Returns High only."""
    if not (_HTTP_OK and BeautifulSoup):
        return []
    url = "https://www.dailyfx.com/economic-calendar?tz=0"
    try:
        r = requests.get(_via_proxy(url), headers=HEADERS, timeout=20)
        r.raise_for_status()
        # DailyFX is a Next.js app; look for __NEXT_DATA__ with events inside.
        m = re.search(r'__NEXT_DATA__"\s*type="application/json">\s*({.*})\s*</script>', r.text)
        if not m:
            log.warning("DailyFX: __NEXT_DATA__ not found")
            return []
        data = json.loads(m.group(1))
        # structure may change; do best-effort
        out = []
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k.lower() in ("events", "calendarEvents") and isinstance(v, list):
                        for ev in v:
                            try:
                                # look for keys: country/currency/title/impact/time
                                country = ev.get("country", "") or ev.get("region", "")
                                title = ev.get("title", "") or ev.get("eventName", "")
                                imp = ev.get("impact", "")
                                ts = ev.get("timestamp") or ev.get("dateTime")
                                if not (country and title and ts): continue
                                if not is_high(str(imp)): continue
                                # ts could be epoch ms or iso
                                if isinstance(ts, (int, float)):
                                    dt_utc = datetime.utcfromtimestamp(int(ts)/1000).replace(tzinfo=timezone.utc)
                                else:
                                    dt_utc = datetime.fromisoformat(str(ts).replace("Z","+00:00"))
                                out.append(event(country, title, dt_utc, "High", "dailyfx", url))
                            except Exception:
                                continue
                    else:
                        walk(v)
            elif isinstance(o, list):
                for it in o:
                    walk(it)
        walk(data)
        log.info("DailyFX parsed: %d High", len(out))
        return out
    except Exception as e:
        log.info("DailyFX failed: %s", e)
        return []

def fetch_investing() -> List[dict]:
    """Investing.com — heavily protected; best-effort via reader proxy; may return 0."""
    if not (_HTTP_OK and BeautifulSoup):
        return []
    url = "https://www.investing.com/economic-calendar/"
    try:
        r = requests.get(_via_proxy(url), headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        out = []
        for tr in soup.select("table tr"):
            tds = tr.find_all("td")
            if len(tds) < 5: continue
            imp = tr.select_one(".sentiment") or tr.select_one("[data-img_key='bull3']")
            if not imp: continue
            # crude parse
            time_el = tds[0].get_text(" ", strip=True)
            title = tds[2].get_text(" ", strip=True)
            country = tds[1].get("title") or tds[1].get_text(" ", strip=True)
            if not (time_el and country and title): continue
            # we don't have the exact date here; skip unless we can infer from table heading
            # leaving as placeholder collector
            # out.append(event(country, title, some_dt, "High", "investing", url))
        log.info("Investing parsed: %d (High)", len(out))
        return out
    except Exception as e:
        log.info("Investing failed: %s", e)
        return []

# -------------------- UNION & FILTER --------------------
def dedup(events: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for ev in events:
        key = (int(ev["utc"].timestamp()), ev["country"], ev["title"].lower())
        if key in seen: continue
        seen.add(key)
        out.append(ev)
    return out

def within_window(ev: dict, start: datetime, end: datetime) -> bool:
    return start <= ev["utc"] <= end

def filter_by_countries(events: List[dict], countries: List[str]) -> List[dict]:
    want = {c.lower() for c in countries}
    return [ev for ev in events if ev["country"] in want]

def collect_all() -> List[dict]:
    # Order matters: reliable first
    all_raw: List[dict] = []
    all_raw += fetch_ff_json()
    all_raw += fetch_ff_html()
    all_raw += fetch_fomc()
    all_raw += fetch_ecb()
    all_raw += fetch_boe()
    all_raw += fetch_boj()
    # Optional:
    all_raw += fetch_dailyfx()
    # all_raw += fetch_investing()  # usually blocked, left disabled
    return dedup(all_raw)

# -------------------- WRITE SHEET --------------------
OUT_HEADERS = [
    "utc_iso", "local_time", "country", "currency", "title", "impact", "source", "url"
]

def write_sheet(sh, ws_title: str, rows: List[List[str]]):
    ws = ensure_ws(sh, ws_title, OUT_HEADERS)
    # Clear old (except header) and write new
    ws.resize(rows=len(rows) + 1)
    ws.update(range_name="A1", values=[OUT_HEADERS] + rows)

def run_once():
    start = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    end = datetime.now(timezone.utc) + timedelta(days=LOOKAHEAD_DAYS)

    events = collect_all()
    log.info("Union total_raw=%d", len(events))

    # Keep only relevant FX countries (USD/EUR/GBP/JPY/AUD) to reduce noise
    fx_countries = {v for v in COUNTRY_BY_CCY.values()}
    events = [ev for ev in events if ev["country"] in fx_countries and is_high(ev["impact"] or "High")]
    events = [ev for ev in events if within_window(ev, start, end)]
    events.sort(key=lambda e: e["utc"])

    rows = []
    for ev in events:
        rows.append([
            to_iso(ev["utc"]),
            to_local(ev["utc"]),
            ev["country"],
            ev["currency"],
            ev["title"],
            ev.get("impact") or "High",
            ev.get("source") or "",
            ev.get("url") or "",
        ])

    if not _GS_OK:
        log.error("Sheets libs not installed, cannot write")
        return

    try:
        sh = open_sheet()
        write_sheet(sh, SHEET_WS_OUT, rows)
        # RAW (optional) — dump last full union for debugging
        raw_rows = []
        for ev in collect_all():
            raw_rows.append([
                to_iso(ev["utc"]), to_local(ev["utc"]), ev["country"],
                CCY_BY_COUNTRY.get(ev["country"], ""), ev["title"],
                ev.get("impact") or "", ev.get("source") or "", ev.get("url") or "",
            ])
        write_sheet(sh, SHEET_WS_RAW, raw_rows)
        log.info("Wrote %d events to sheet '%s'", len(rows), SHEET_WS_OUT)
    except Exception as e:
        log.error("Sheets write failed: %s", e)

def main():
    if not SHEET_ID:
        raise SystemExit("SHEET_ID not set")
    if not _HTTP_OK:
        log.warning("requests/bs4 not installed; only sheet ops will work")

    if RUN_FOREVER:
        while True:
            try:
                run_once()
            except Exception as e:
                log.error("collector run failed: %s", e)
            time.sleep(max(60, COLLECT_EVERY_MIN * 60))
    else:
        run_once()

if __name__ == "__main__":
    main()
