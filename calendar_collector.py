# calendar_collector.py
import os
import re
import json
import base64
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

# -------- optional deps ----------
try:
    import requests
    _REQ = True
except Exception:
    requests = None
    _REQ = False

try:
    from bs4 import BeautifulSoup  # type: ignore
    _BS4 = True
except Exception:
    BeautifulSoup = None
    _BS4 = False

# -------- timezone ----------
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

LOCAL_TZ_NAME = os.getenv("LOCAL_TZ", "Europe/Belgrade")
LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME) if ZoneInfo else None

# -------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("calendar_collector")

# -------- env ----------
SHEET_ID = os.getenv("SHEET_ID", "").strip()
CAL_WS_OUT = os.getenv("CAL_WS_OUT", "CALENDAR").strip() or "CALENDAR"
CAL_WS_RAW = os.getenv("CAL_WS_RAW", "").strip()
RUN_FOREVER = os.getenv("RUN_FOREVER", "true").strip().lower() in ("1", "true", "yes", "y")
COLLECT_EVERY_MIN = int(os.getenv("COLLECT_EVERY_MIN", "30") or "30")
FREE_LOOKBACK_DAYS = int(os.getenv("FREE_LOOKBACK_DAYS", "7") or "7")
FREE_LOOKAHEAD_DAYS = int(os.getenv("FREE_LOOKAHEAD_DAYS", "30") or "30")
USE_JINA = os.getenv("JINA_PROXY", "true").strip().lower() in ("1", "true", "yes", "y")

# -------- Google Sheets ----------
try:
    import gspread
    from google.oauth2 import service_account
    _GSHEETS = True
except Exception:
    gspread = None
    service_account = None
    _GSHEETS = False

SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def _env(name: str) -> str:
    v = os.getenv(name, "")
    return v if isinstance(v, str) else ""


def _decode_b64_maybe_padded(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    padded = s + "=" * ((4 - len(s) % 4) % 4)
    return base64.b64decode(padded).decode("utf-8", "strict")


def load_google_service_info() -> Tuple[Optional[dict], str]:
    b64 = _env("GOOGLE_CREDENTIALS_JSON_B64")
    if b64:
        try:
            decoded = _decode_b64_maybe_padded(b64)
            info = json.loads(decoded)
            return info, "env:GOOGLE_CREDENTIALS_JSON_B64"
        except Exception as e:
            return None, f"bad b64 json: {e}"
    raw = _env("GOOGLE_CREDENTIALS_JSON") or _env("GOOGLE_CREDENTIALS")
    if raw:
        try:
            info = json.loads(raw)
            return info, "env:GOOGLE_CREDENTIALS_JSON"
        except Exception as e:
            return None, f"bad raw json: {e}"
    return None, "not-found"


def build_sheets_client(sheet_id: str):
    if not _GSHEETS:
        raise RuntimeError("gspread/google-auth not installed")
    info, src = load_google_service_info()
    if not info:
        raise RuntimeError(f"google creds missing: {src}")
    creds = service_account.Credentials.from_service_account_info(info, scopes=SHEETS_SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    return sh


def ensure_ws(sh, title: str, headers: List[str]):
    try:
        ws = None
        for w in sh.worksheets():
            if w.title == title:
                ws = w
                break
        if ws is None:
            ws = sh.add_worksheet(title=title, rows=100, cols=max(6, len(headers)))
            ws.update(range_name="A1", values=[headers])
        else:
            # гарантируем заголовок
            ws.update(range_name="A1", values=[headers])
        return ws
    except Exception as e:
        raise RuntimeError(f"ensure_ws error: {e}")


# -------- helpers: normalization / date window ----------
COUNTRY_ALIAS = {
    # United States
    "us": "united states", "usa": "united states", "u.s.": "united states",
    "united states": "united states", "united states of america": "united states",
    "america": "united states", "fomc": "united states", "federal reserve": "united states",
    # Japan
    "jp": "japan", "jpn": "japan", "japan": "japan", "boj": "japan", "bank of japan": "japan",
    # Euro area
    "eu": "euro area", "eurozone": "euro area", "euro area": "euro area", "ecb": "euro area",
    "european central bank": "euro area",
    # United Kingdom
    "uk": "united kingdom", "gb": "united kingdom", "great britain": "united kingdom",
    "united kingdom": "united kingdom", "boe": "united kingdom", "bank of england": "united kingdom",
    # Australia
    "au": "australia", "aus": "australia", "australia": "australia", "rba": "australia",
    "reserve bank of australia": "australia",
}

OK_COUNTRIES = {"united states", "japan", "euro area", "united kingdom", "australia"}

PAIR_COUNTRIES = {
    "USDJPY": {"united states", "japan"},
    "AUDUSD": {"australia", "united states"},
    "EURUSD": {"euro area", "united states"},
    "GBPUSD": {"united kingdom", "united states"},
}

HIGH_MARKERS = {"high", "3", "red", "important", "rate decision", "interest rate", "policy meeting"}


def norm_country(s: str) -> str:
    s = (s or "").strip().lower()
    return COUNTRY_ALIAS.get(s, s)


def is_high_impact(rec: dict) -> bool:
    imp = str(rec.get("importance") or rec.get("impact") or "").strip().lower()
    title = (rec.get("title") or "").strip().lower()
    if imp in {"3", "high", "red"}:
        return True
    return any(k in title for k in HIGH_MARKERS)


def in_window(dt_utc: Optional[datetime], d1: datetime, d2: datetime) -> bool:
    return (dt_utc is not None) and (d1 <= dt_utc <= d2)


def noon_utc(date_obj: datetime) -> datetime:
    return datetime(date_obj.year, date_obj.month, date_obj.day, 12, 0, tzinfo=timezone.utc)


# -------- HTTP helpers ----------
def _ua_headers() -> Dict[str, str]:
    return {
        "User-Agent": "calendar-worker/1.0 (+https://example.org)",
        "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
    }


def fetch_text(url: str, timeout: int = 12) -> Optional[str]:
    if not _REQ:
        return None
    # try direct
    try:
        r = requests.get(url, headers=_ua_headers(), timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        if USE_JINA:
            try:
                proxy = f"https://r.jina.ai/{url}"
                r = requests.get(proxy, headers=_ua_headers(), timeout=timeout)
                r.raise_for_status()
                return r.text
            except Exception as e2:
                log.warning("fetch_text failed via proxy: %s", e2)
        else:
            log.warning("fetch_text failed: %s", e)
    return None


def fetch_json(url: str, timeout: int = 12):
    if not _REQ:
        return None
    try:
        r = requests.get(url, headers=_ua_headers(), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning("fetch_json failed: %s", e)
        return None


# -------- Sources --------
def src_ff_json() -> List[dict]:
    """ForexFactory weekly JSON (может отдавать пусто)."""
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    data = fetch_json(url)
    out = []
    if isinstance(data, list):
        for it in data:
            try:
                ts = int(it.get("timestamp", 0))
            except Exception:
                ts = 0
            if not ts:
                continue
            dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
            country_raw = it.get("country") or ""
            country = norm_country(country_raw)
            out.append({
                "utc": dt_utc,
                "country": country,
                "title": it.get("title") or it.get("event") or "Event",
                "importance": it.get("impact") or it.get("importance") or "",
                "source": "FF-JSON",
            })
    log.info("FF JSON parsed: %s", len(out))
    return out


def src_ff_html() -> List[dict]:
    """Парсинг ForexFactory HTML (через прямой или r.jina.ai)."""
    # Берём «текущую/следующую» неделю — FF на JSON уже дернули; HTML пробуем на всякий.
    urls = [
        "https://www.forexfactory.com/calendar?week=this",
        "https://www.forexfactory.com/calendar?week=next",
    ]
    out: List[dict] = []
    for url in urls:
        html = fetch_text(url)
        if not html:
            continue
        text = html.lower()
        # очень грубая эвристика: просто ищем слова "high impact" и даты ГГГГ-ММ-ДД в data-атрибутах
        # (у FF HTML без аутентикации часто меняется верстка; r.jina.ai убирает теги)
        for m in re.finditer(r'data-impact="(high|medium|low)"[^>]*?data-event-datetime="([0-9\-:tzZ\+]+)"[^>]*?data-country="([^"]*)"', html, re.I):
            impact = m.group(1)
            dt_raw = m.group(2)
            c_raw = m.group(3)
            try:
                # data-event-datetime обычно ISO
                dt_utc = datetime.fromisoformat(dt_raw.replace("Z", "+00:00")).astimezone(timezone.utc)
            except Exception:
                continue
            out.append({
                "utc": dt_utc,
                "country": norm_country(c_raw),
                "title": "High impact event",
                "importance": impact,
                "source": "FF-HTML",
            })
    log.info("FF HTML parsed: %s high-impact events (mode=%s)", len(out), "direct")
    return out


def _extract_dates_text(text: str) -> List[datetime]:
    """Достаём даты вида 'September 17-18, 2025' или 'September 20, 2025' — возвращаем 1-й день, 12:00 UTC."""
    if not text:
        return []
    months = ("january|february|march|april|may|june|july|august|september|october|november|december")
    # September 17-18, 2025  |  September 17, 2025
    rx = re.compile(rf'\b({months})\s+(\d{{1,2}})(?:\s*[-–]\s*\d{{1,2}})?\s*,\s*(\d{{4}})\b', re.I)
    out = []
    for m in rx.finditer(text):
        mon, day, year = m.group(1), int(m.group(2)), int(m.group(3))
        mon_num = {
            "january":1, "february":2, "march":3, "april":4, "may":5, "june":6,
            "july":7, "august":8, "september":9, "october":10, "november":11, "december":12
        }[mon.lower()]
        try:
            dt = datetime(year, mon_num, day, tzinfo=timezone.utc)
        except Exception:
            continue
        out.append(noon_utc(dt))
    return out


def src_fomc() -> List[dict]:
    """ФРС: даты заседаний FOMC."""
    urls = [
        "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        "https://www.federalreserve.gov/monetarypolicy.htm",
    ]
    out: List[dict] = []
    for u in urls:
        txt = fetch_text(u)
        if not txt:
            continue
        dates = _extract_dates_text(txt)
        for dt in dates:
            out.append({
                "utc": dt,
                "country": "united states",
                "title": "FOMC Meeting / Rate Decision",
                "importance": "high",
                "source": "FOMC",
            })
        if out:
            break
    log.info("FOMC parsed: %s", len(out))
    return out


def src_ecb() -> List[dict]:
    """ECB: Governing Council / Monetary policy meetings."""
    urls = [
        "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html",
        "https://www.ecb.europa.eu/press/govcdec/meetings/html/index.en.html",
    ]
    out: List[dict] = []
    for u in urls:
        txt = fetch_text(u)
        if not txt:
            continue
        dates = _extract_dates_text(txt)
        for dt in dates:
            out.append({
                "utc": dt,
                "country": "euro area",
                "title": "ECB Governing Council / Rate Decision",
                "importance": "high",
                "source": "ECB",
            })
        if out:
            break
    log.info("ECB parsed: %s", len(out))
    return out


def src_boe() -> List[dict]:
    """Bank of England: MPC meeting dates."""
    urls = [
        "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes",
        "https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp?Travel=NIxAZx",
    ]
    out: List[dict] = []
    for u in urls:
        txt = fetch_text(u)
        if not txt:
            continue
        dates = _extract_dates_text(txt)
        for dt in dates:
            out.append({
                "utc": dt,
                "country": "united kingdom",
                "title": "BoE MPC Meeting / Rate Decision",
                "importance": "high",
                "source": "BoE",
            })
        if out:
            break
    log.info("BoE parsed: %s", len(out))
    return out


def src_boj() -> List[dict]:
    """Bank of Japan: Monetary Policy Meeting."""
    urls = [
        "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm",
        "https://www.boj.or.jp/en/mopo/index.htm",
    ]
    out: List[dict] = []
    for u in urls:
        txt = fetch_text(u)
        if not txt:
            continue
        dates = _extract_dates_text(txt)
        for dt in dates:
            out.append({
                "utc": dt,
                "country": "japan",
                "title": "BoJ Monetary Policy Meeting / Rate Decision",
                "importance": "high",
                "source": "BoJ",
            })
        if out:
            break
    log.info("BoJ parsed: %s", len(out))
    return out


def src_dailyfx() -> List[dict]:
    """DailyFX календарь: пробуем вытащить high-impact (через r.jina.ai)."""
    urls = [
        "https://www.dailyfx.com/economic-calendar?tz=0",
        "https://www.dailyfx.com/economic-calendar",
    ]
    out: List[dict] = []
    for u in urls:
        txt = fetch_text(u)
        if not txt:
            continue
        # Jina даёт плоский текст — ищем строки с HIGH и датами ISO
        # (Очень грубо; часто DailyFX режет, так что норм если 0)
        for m in re.finditer(r'\b(High)\b.*?(\d{4}-\d{2}-\d{2})[T ](\d{2}):(\d{2})', txt, re.I | re.S):
            ymd = m.group(2)
            hh, mm = int(m.group(3)), int(m.group(4))
            try:
                dt = datetime.fromisoformat(f"{ymd}T{hh:02d}:{mm:02d}:00+00:00").astimezone(timezone.utc)
            except Exception:
                continue
            out.append({
                "utc": dt,
                "country": "",  # ниже попытаемся угадать из заголовков
                "title": "High impact event (DailyFX)",
                "importance": "high",
                "source": "DailyFX",
            })
        if out:
            break
    if not out:
        log.warning("DailyFX parse gave 0")
    else:
        log.info("DailyFX parsed: %s", len(out))
    return out


def src_investing() -> List[dict]:
    """Investing.com (часто 403; пробуем через r.jina.ai текст)."""
    urls = [
        "https://www.investing.com/economic-calendar/",
        "https://www.investing.com/economic-calendar/?importance=3",
    ]
    out: List[dict] = []
    for u in urls:
        txt = fetch_text(u)
        if not txt:
            continue
        # грубая эвристика на ISO дату/время
        for m in re.finditer(r'(\d{4}-\d{2}-\d{2})[T ](\d{2}):(\d{2})', txt):
            ymd = m.group(1); hh = int(m.group(2)); mm = int(m.group(3))
            try:
                dt = datetime.fromisoformat(f"{ymd}T{hh:02d}:{mm:02d}:00+00:00").astimezone(timezone.utc)
            except Exception:
                continue
            out.append({
                "utc": dt,
                "country": "",
                "title": "High impact event (Investing)",
                "importance": "high",
                "source": "Investing",
            })
        if out:
            break
    log.info("Investing parsed: %s (mode=%s)", len(out), "direct")
    return out


# -------- Union / Filter / Write --------
def guess_country_from_title(title: str) -> Optional[str]:
    t = (title or "").lower()
    if any(x in t for x in ["fomc", "federal reserve", "usd"]):
        return "united states"
    if any(x in t for x in ["ecb", "euro", "eur"]):
        return "euro area"
    if any(x in t for x in ["boj", "bank of japan", "jpy", "japan"]):
        return "japan"
    if any(x in t for x in ["boe", "bank of england", "gbp", "uk"]):
        return "united kingdom"
    if any(x in t for x in ["rba", "australia", "aud"]):
        return "australia"
    return None


def collect_union() -> List[dict]:
    if not _REQ:
        log.warning("requests/bs4 not installed; only sheet ops will work")
    # собираем с источников
    union: List[dict] = []
    union += src_ff_json()
    union += src_ff_html()
    union += src_dailyfx()
    union += src_investing()
    union += src_fomc()
    union += src_ecb()
    union += src_boe()
    union += src_boj()

    # нормализация и догадка стран
    dedup = set()
    cleaned: List[dict] = []
    for ev in union:
        utc = ev.get("utc")
        title = ev.get("title") or "Event"
        country = norm_country(ev.get("country") or "")
        if not country:
            guess = guess_country_from_title(title)
            if guess:
                country = guess
        if not country:
            # последний шанс: если source один из ЦБ — расставим
            src = (ev.get("source") or "").lower()
            if src == "fomc":
                country = "united states"
            elif src == "ecb":
                country = "euro area"
            elif src == "boe":
                country = "united kingdom"
            elif src == "boj":
                country = "japan"
        key = None
        if isinstance(utc, datetime):
            key = (utc.date().isoformat(), (country or ""), title.lower().strip())
        else:
            continue
        if key in dedup:
            continue
        dedup.add(key)
        ev["country"] = country or ""
        cleaned.append(ev)

    log.info("Union total_raw=%s", len(cleaned))
    return cleaned


def filter_for_window(evts: List[dict], back_days: int, ahead_days: int) -> List[dict]:
    now_utc = datetime.now(timezone.utc)
    d1 = now_utc - timedelta(days=back_days)
    d2 = now_utc + timedelta(days=ahead_days)

    filtered: List[dict] = []
    for ev in evts:
        utc = ev.get("utc")
        country = norm_country(ev.get("country") or "")
        if country not in OK_COUNTRIES:
            continue
        if not is_high_impact(ev):
            continue
        if not in_window(utc, d1, d2):
            continue
        filtered.append(ev)

    log.info("After country+days=%s; days=[%s..%s]",
             len(filtered), d1.date().isoformat(), d2.date().isoformat())
    return filtered


def write_to_sheet(events: List[dict]):
    if not _GSHEETS:
        raise RuntimeError("gspread/google-auth not installed")
    if not SHEET_ID:
        raise RuntimeError("SHEET_ID is empty")

    sh = build_sheets_client(SHEET_ID)
    headers = ["utc_iso", "local_belgrade", "country", "title", "importance", "source"]

    ws = ensure_ws(sh, CAL_WS_OUT, headers)
    # clear all except header
    try:
        ws.resize(rows=1)  # оставить только строку заголовка
    except Exception:
        # если нет прав/метода — просто обновим поверх
        pass

    rows = []
    for ev in sorted(events, key=lambda x: x["utc"]):
        utc: datetime = ev["utc"]
        loc = utc.astimezone(LOCAL_TZ) if LOCAL_TZ else utc
        rows.append([
            utc.isoformat(timespec="minutes"),
            loc.strftime("%Y-%m-%d %H:%M"),
            ev.get("country", ""),
            ev.get("title", ""),
            ev.get("importance", ""),
            ev.get("source", ""),
        ])

    if rows:
        try:
            ws.append_rows(rows, value_input_option="RAW")
        except Exception:
            # совместимость со старыми gspread
            for r in rows:
                ws.append_row(r, value_input_option="RAW")

    log.info("Wrote %s events to sheet '%s'", len(rows), CAL_WS_OUT)

    # сырое хранилище (необязательное)
    if CAL_WS_RAW:
        try:
            ws_raw = ensure_ws(sh, CAL_WS_RAW, ["json"])
            ws_raw.resize(rows=1)
            blob = {"written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "events": [
                        {
                            "utc": ev["utc"].isoformat(),
                            "country": ev.get("country", ""),
                            "title": ev.get("title", ""),
                            "importance": ev.get("importance", ""),
                            "source": ev.get("source", ""),
                        } for ev in events
                    ]}
            ws_raw.append_row([json.dumps(blob, ensure_ascii=False)], value_input_option="RAW")
        except Exception as e:
            log.warning("raw sheet write failed: %s", e)


# -------- Main loop ----------
def run_once():
    union = collect_union()
    filtered = filter_for_window(union, FREE_LOOKBACK_DAYS, FREE_LOOKAHEAD_DAYS)
    write_to_sheet(filtered)


def main():
    log.info("collector starting… (sheet_id=%s, ws_out=%s, tz=%s, back=%s, ahead=%s, loop=%s/%sm)",
             SHEET_ID, CAL_WS_OUT, LOCAL_TZ_NAME, FREE_LOOKBACK_DAYS, FREE_LOOKAHEAD_DAYS,
             RUN_FOREVER, COLLECT_EVERY_MIN)
    if not _REQ:
        log.warning("requests/bs4 not installed; HTTP sources unavailable.")
    if RUN_FOREVER:
        import time
        while True:
            try:
                run_once()
            except Exception as e:
                log.exception("run_once failed: %s", e)
            time.sleep(max(60, COLLECT_EVERY_MIN * 60))
    else:
        run_once()


if __name__ == "__main__":
    main()
