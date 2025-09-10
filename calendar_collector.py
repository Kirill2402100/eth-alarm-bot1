#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calendar_collector.py

Собирает:
 A) Календарь: ForexFactory, DailyFX, Investing, FOMC (+ ECB/BoE/BoJ, best-effort)
 B) Новости: официальные ленты/страницы (best-effort, ключевые слова)
 C) Рынок: из последних строк BMR_DCA_* (Supertrend/VolZ/ATR1h)

Пишет:
 - CALENDAR (для fa_bot.py) — строго с хедерами:
   ["utc_iso","local_time","country","currency","title","impact","source","url"]
 - NEWS (минимальная лента)
 - FA_Signals (risk/bias/ttl/scan_lock_until/reserve_off/dca_scale/reason/risk_pct)
 - INVESTOR_DIGEST (короткая выжимка)

ENV (минимум):
  SHEET_ID, GOOGLE_CREDENTIALS_JSON_B64 (или GOOGLE_CREDENTIALS_JSON/GOOGLE_CREDENTIALS)
Опции см. в блоке CONFIG.
"""

import os, re, json, time, base64, logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

import httpx
import gspread
from google.oauth2 import service_account

# ---------------- Logging ----------------
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("calendar_collector")

# ---------------- CONFIG / ENV ----------------
SHEET_ID   = os.getenv("SHEET_ID","").strip()
LOCAL_TZ   = os.getenv("LOCAL_TZ","Europe/Belgrade")

CAL_WS     = os.getenv("CAL_WS_OUT","CALENDAR")
NEWS_WS    = os.getenv("NEWS_WS","NEWS")
FA_WS      = os.getenv("FA_WS","FA_Signals")
DIGEST_WS  = os.getenv("DIGEST_WS","INVESTOR_DIGEST")

BACK_DAYS  = int(os.getenv("COLLECT_BACK_DAYS","7") or "7")
AHEAD_DAYS = int(os.getenv("COLLECT_AHEAD_DAYS","30") or "30")

RUN_FOREVER = os.getenv("RUN_FOREVER","1").lower() in ("1","true","yes","on")
EVERY_MIN   = int(os.getenv("COLLECT_EVERY_MIN","180") or "180")

# FA настройки
QUIET_BEFORE_MIN = int(os.getenv("QUIET_BEFORE_MIN","45"))
QUIET_AFTER_MIN  = int(os.getenv("QUIET_AFTER_MIN","45"))
NEWS_FRESH_MIN   = int(os.getenv("SE_NEWS_FRESH_MIN","90"))

W_CAL   = int(os.getenv("W_CAL","14"))
W_NEWS  = int(os.getenv("W_NEWS","8"))
W_MKT   = int(os.getenv("W_MKT","6"))
QUIET_BONUS = int(os.getenv("QUIET_BONUS","10"))

RED_THR   = int(os.getenv("FA_RED_THR","70"))
AMBER_THR = int(os.getenv("FA_AMBER_THR","40"))

USE_BMR_FOR_BIAS = os.getenv("USE_BMR_FOR_BIAS","1").lower() in ("1","true","yes","on")

BMR_SHEETS = {
    "USDJPY": os.getenv("BMR_SHEET_USDJPY","BMR_DCA_USDJPY"),
    "EURUSD": os.getenv("BMR_SHEET_EURUSD","BMR_DCA_EURUSD"),
    "GBPUSD": os.getenv("BMR_SHEET_GBPUSD","BMR_DCA_GBPUSD"),
    "AUDUSD": os.getenv("BMR_SHEET_AUDUSD","BMR_DCA_AUDUSD"),
}

PAIR_COUNTRIES = {
    "USDJPY": {"united states","japan"},
    "EURUSD": {"united states","euro area","germany","france","italy","spain"},
    "GBPUSD": {"united states","united kingdom"},
    "AUDUSD": {"united states","australia"},
}

COUNTRY_TO_CCY = {
    "united states":"USD","japan":"JPY","united kingdom":"GBP","euro area":"EUR",
    "germany":"EUR","france":"EUR","italy":"EUR","spain":"EUR","australia":"AUD",
    "canada":"CAD","new zealand":"NZD","switzerland":"CHF","china":"CNY"
}

CAL_HEADERS   = ["utc_iso","local_time","country","currency","title","impact","source","url"]
NEWS_HEADERS  = ["ts_utc","source","title","url","countries","ccy","tags","importance_guess","hash"]
FA_HEADERS    = ["pair","risk","bias","ttl","updated_at","scan_lock_until","reserve_off","dca_scale","reason","risk_pct"]
DIGEST_HEADERS= ["ts_utc","text"]

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
CLIENT = httpx.Client(timeout=12.0, headers={"User-Agent": UA, "Accept-Language":"en;q=0.9"})

# ---------------- Helpers ----------------
def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")

def _parse_dt_guess(s: str) -> Optional[datetime]:
    if not s: return None
    s = s.strip().replace("Z","+00:00")
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

def _local_iso_from_utc_iso(utc_iso: str, tz: str) -> str:
    try:
        dt = _parse_dt_guess(utc_iso)
        if not dt: return ""
        import zoneinfo
        ltz = zoneinfo.ZoneInfo(tz)
        return dt.astimezone(ltz).isoformat(timespec="seconds")
    except Exception:
        return ""

def _load_sheet():
    info = None
    b64 = os.getenv("GOOGLE_CREDENTIALS_JSON_B64","").strip()
    if b64:
        try:
            b64 += "=" * ((4 - len(b64)%4)%4)
            info = json.loads(base64.b64decode(b64).decode("utf-8"))
        except Exception:
            info = None
    if not info:
        raw = os.getenv("GOOGLE_CREDENTIALS_JSON", os.getenv("GOOGLE_CREDENTIALS","")).strip()
        if raw:
            try: info = json.loads(raw)
            except Exception: info = None
    if not info or not SHEET_ID:
        raise RuntimeError("Google credentials or SHEET_ID not set")
    creds = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    gc = gspread.authorize(creds)
    return gc.open_by_key(SHEET_ID)

def _ensure_ws(sh, title, headers):
    try:
        ws = sh.worksheet(title)
        cur = ws.row_values(1)
        if cur != headers:
            ws.update("A1",[headers])
        return ws
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=1000, cols=max(20, len(headers)))
        ws.update("A1",[headers])
        return ws

def _http_get(url: str, allow_proxy=True) -> Optional[str]:
    # прямой запрос
    try:
        r = CLIENT.get(url)
        if r.status_code == 200 and r.text:
            return r.text
    except Exception as e:
        log.debug("direct fail %s: %s", url, e)
    if not allow_proxy:
        return None
    # reader-прокси (устойчивее к сложной верстке)
    try:
        prox = "https://r.jina.ai/http://" + url.replace("https://","").replace("http://","")
        r = CLIENT.get(prox)
        if r.status_code == 200 and r.text:
            return r.text
    except Exception as e:
        log.debug("proxy fail %s: %s", url, e)
    return None

# ---------------- Fetchers: Calendar ----------------
def fetch_ff_html() -> List[Dict]:
    url = "https://www.forexfactory.com/calendar?week=this"
    html = _http_get(url, allow_proxy=True)
    out = []
    if not html:
        log.info("FF HTML parsed: 0")
        return out
    rows = re.findall(r'<tr[^>]*calendar__row[^>]*?>.*?</tr>', html, re.S|re.I)
    for row in rows:
        imp = "High" if re.search(r'High Impact', row, re.I) else ("Medium" if re.search(r'Medium Impact', row, re.I) else "")
        if not imp:
            continue
        mtime = re.search(r'data-event-datetime="([^"]+)"', row, re.I)
        mctry = re.search(r'calendar__flag[^>]+title="([^"]+)"', row, re.I)
        mtitle= re.search(r'calendar__event-title[^>]*>(.*?)</', row, re.S|re.I)
        if not (mtime and mtitle):
            continue
        dt = _parse_dt_guess(mtime.group(1))
        if not dt: continue
        country = (mctry.group(1) if mctry else "").strip().lower()
        title = re.sub(r'<.*?>',' ', mtitle.group(1)).strip()
        ccy = COUNTRY_TO_CCY.get(country,"")
        out.append({
            "utc_iso": _iso_utc(dt),
            "local_time": "",  # заполним позже
            "country": country,
            "currency": ccy,
            "title": title,
            "impact": imp,
            "source": "FF",
            "url": url
        })
    log.info("FF HTML parsed: %d high-impact rows", len(out))
    return out

def fetch_dailyfx() -> List[Dict]:
    url = "https://www.dailyfx.com/economic-calendar?tz=0"
    html = _http_get(url, allow_proxy=True)
    out = []
    if not html:
        log.info("DailyFX parsed: 0")
        return out
    js = re.search(r'__INITIAL_STATE__\s*=\s*({.*?});\s*</script>', html, re.S)
    if not js:
        log.info("DailyFX initial_state not found")
        return out
    try:
        data = json.loads(js.group(1))
        items = (data.get("economicCalendar") or {}).get("economicCalendar") or []
        for it in items:
            try:
                ts = int(it.get("date"))/1000.0
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                title = (it.get("event") or "").strip()
                impact= (it.get("impact") or "").strip().lower()
                imp = "High" if "high" in impact else ("Medium" if "medium" in impact else "Low")
                ctry = (it.get("country") or "").strip().lower()
                ccy  = COUNTRY_TO_CCY.get(ctry,"")
                if not title: 
                    continue
                out.append({
                    "utc_iso": _iso_utc(dt),
                    "local_time":"",
                    "country": ctry,
                    "currency": ccy,
                    "title": title,
                    "impact": imp,
                    "source":"DailyFX",
                    "url": url
                })
            except Exception:
                continue
    except Exception:
        pass
    log.info("DailyFX parsed: %d", len(out))
    return out

def fetch_investing() -> List[Dict]:
    url = "https://www.investing.com/economic-calendar/"
    html = _http_get(url, allow_proxy=True)
    out = []
    if not html:
        log.info("Investing parsed: 0")
        return out
    rows = re.findall(r'<tr[^>]*?js-event-item[^>]*?>.*?</tr>', html, re.S|re.I)
    for row in rows:
        mtime = re.search(r'data-event-datetime="([^"]+)"', row, re.I)
        if not mtime: continue
        dt = _parse_dt_guess(mtime.group(1)); 
        if not dt: continue
        heads = len(re.findall(r'icon-\w*?bull', row, re.I))
        imp = "High" if heads>=3 else ("Medium" if heads==2 else ("Low" if heads==1 else ""))
        mtitle = re.search(r'event__title[^>]*?>(.*?)</', row, re.S|re.I)
        title = re.sub(r'<.*?>',' ', (mtitle.group(1) if mtitle else "")).strip()
        mctry = re.search(r'flag\w*?[^>]*?title="([^"]+)"', row, re.I)
        ctry = (mctry.group(1) if mctry else "").strip().lower()
        ccy  = COUNTRY_TO_CCY.get(ctry,"")
        if not title: continue
        out.append({
            "utc_iso": _iso_utc(dt),
            "local_time":"",
            "country": ctry,
            "currency": ccy,
            "title": title,
            "impact": imp or "Medium",
            "source": "Investing",
            "url": url
        })
    log.info("Investing parsed: %d", len(out))
    return out

def fetch_fomc() -> List[Dict]:
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    html = _http_get(url, allow_proxy=True)
    out = []
    if not html: 
        log.info("FOMC parsed: 0"); 
        return out
    blocks = re.findall(r'(?s)<div class="panel panel-default".*?</div>\s*</div>', html)
    for b in blocks:
        h = re.search(r'(?i)<h4.*?>(.*?)</h4>', b)
        if not h: 
            continue
        head = re.sub(r'<.*?>',' ', h.group(1)).strip()
        y = re.search(r'(20\d{2})', head); m = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)', head, re.I); d = re.search(r'(\d{1,2})', head)
        if not (y and m and d): 
            continue
        MONTHS = {m.lower():i+1 for i,m in enumerate(["January","February","March","April","May","June","July","August","September","October","November","December"])}
        dt = datetime(int(y.group(1)), MONTHS[m.group(1).lower()], int(d.group(1)), 18, 0, tzinfo=timezone.utc)
        out.append({
            "utc_iso": _iso_utc(dt),
            "local_time":"",
            "country": "united states",
            "currency":"USD",
            "title": "FOMC Meeting / Statement / Presser",
            "impact": "High",
            "source": "FOMC",
            "url": url
        })
    log.info("FOMC parsed: %d", len(out))
    return out

def fetch_ecb() -> List[Dict]:
    url = "https://www.ecb.europa.eu/press/calendars/mgc/html/index.en.html"
    html = _http_get(url, allow_proxy=True)
    out = []
    if not html: 
        log.info("ECB parsed: 0"); 
        return out
    items = re.findall(r'(?is)<li[^>]*?>.*?</li>', html)
    for it in items:
        if not re.search(r'Governing Council|Monetary Policy Meeting', it, re.I):
            continue
        t = re.sub(r'<.*?>',' ', it).strip()
        y = re.search(r'(20\d{2})', t); 
        m = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)', t, re.I)
        d = re.search(r'(\d{1,2})', t)
        if not (y and m and d): continue
        MONTHS = {m.lower():i+1 for i,m in enumerate(["January","February","March","April","May","June","July","August","September","October","November","December"])}
        dt = datetime(int(y.group(1)), MONTHS[m.group(1).lower()], int(d.group(1)), 11, 45, tzinfo=timezone.utc)
        out.append({
            "utc_iso": _iso_utc(dt),
            "local_time":"",
            "country": "euro area",
            "currency":"EUR",
            "title": "ECB Governing Council / Rate Decision",
            "impact":"High",
            "source":"ECB","url":url
        })
    log.info("ECB parsed: %d", len(out))
    return out

def fetch_boe() -> List[Dict]:
    url = "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes"
    html = _http_get(url, allow_proxy=True)
    out = []
    if not html:
        log.info("BoE parsed: 0"); 
        return out
    items = re.findall(r'(?is)<time[^>]*datetime="([^"]+)"[^>]*>.*?</time>.*?Monetary Policy', html)
    for iso in items:
        dt = _parse_dt_guess(iso)
        if not dt: 
            continue
        out.append({
            "utc_iso": _iso_utc(dt),
            "local_time":"",
            "country":"united kingdom","currency":"GBP",
            "title":"BoE Monetary Policy / Rate Decision",
            "impact":"High","source":"BoE","url":url
        })
    log.info("BoE parsed: %d", len(out))
    return out

def fetch_boj() -> List[Dict]:
    url = "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm/"
    html = _http_get(url, allow_proxy=True)
    out = []
    if not html:
        log.info("BoJ parsed: 0"); 
        return out
    items = re.findall(r'(?is)Monetary Policy Meeting.*?<time[^>]*datetime="([^"]+)"', html)
    for iso in items:
        dt = _parse_dt_guess(iso)
        if not dt: continue
        out.append({
            "utc_iso": _iso_utc(dt),
            "local_time":"",
            "country":"japan","currency":"JPY",
            "title":"BoJ Monetary Policy Meeting",
            "impact":"High","source":"BoJ","url":url
        })
    log.info("BoJ parsed: %d", len(out))
    return out

def collect_calendar() -> List[Dict]:
    out = []
    try: out += fetch_ff_html()
    except Exception: log.exception("FF failed")
    try: out += fetch_dailyfx()
    except Exception: log.exception("DailyFX failed")
    try: out += fetch_investing()
    except Exception: log.exception("Investing failed")
    try: out += fetch_fomc()
    except Exception: log.exception("FOMC failed")
    try: out += fetch_ecb()
    except Exception: log.exception("ECB failed")
    try: out += fetch_boe()
    except Exception: log.exception("BoE failed")
    try: out += fetch_boj()
    except Exception: log.exception("BoJ failed")
    return out

# ---------------- Fetchers: News (best-effort) ----------------
# Небольшой набор официальных страниц + ключевые слова («intervention», «unscheduled», «emergency»…)
NEWS_SOURCES = [
    # (name, url, countries_hint, ccy)
    ("US_FED_PR", "https://www.federalreserve.gov/newsevents/pressreleases.htm", "united states", "USD"),
    ("US_TREASURY", "https://home.treasury.gov/news/press-releases", "united states", "USD"),
    ("JAPAN_MOF_FX", "https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/index.html", "japan", "JPY"),
    ("BOE_NEWS", "https://www.bankofengland.co.uk/news", "united kingdom", "GBP"),
    ("ECB_PR", "https://www.ecb.europa.eu/press/pr/html/index.en.html", "euro area", "EUR"),
    ("RBA_NEWS", "https://www.rba.gov.au/media-releases/", "australia", "AUD"),
]

NEWS_KEYWORDS = re.compile(
    r"(interven|unscheduled|emergency|surprise|extraordinary|stabiliz|fx|yen|inflation alert|rate corridor|"
    r"liquidity operation|market notice|statement|verbal intervention)", re.I
)

def fetch_news() -> List[Dict]:
    out = []
    for name, url, ctry, ccy in NEWS_SOURCES:
        html = _http_get(url, allow_proxy=True)
        if not html: 
            continue
        # вытаскиваем первые 20 ссылок-заголовков с датой (очень грубо)
        items = re.findall(r'(?is)<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?(?:datetime="([^"]+)")?', html)[:20]
        for href, title_html, dt_iso in items:
            title = re.sub(r'<.*?>',' ', title_html).strip()
            if not title: 
                continue
            tags = []
            if NEWS_KEYWORDS.search(title):
                tags.append("keyword")
            ts = ""
            dt = None
            if dt_iso:
                dt = _parse_dt_guess(dt_iso)
            # если нет datetime, не беда — новость все равно пройдёт только если «свежая» по таймингу ниже
            if dt:
                ts = _iso_utc(dt)
            out.append({
                "ts_utc": ts or "", "source": name, "title": title, "url": url,
                "countries": ctry, "ccy": ccy, "tags": ",".join(tags),
                "importance_guess": "HIGH" if NEWS_KEYWORDS.search(title) else "",
                "hash": f"{name}|{title[:80]}"
            })
    log.info("NEWS collected: %d items", len(out))
    return out

# ---------------- FA Scoring ----------------
RE_IMP = re.compile(r"(cpi|nfp|payroll|employment|pmi|gdp|inflation|rate|decision|press|conference|minutes)", re.I)

def _nearest_high_event(now: datetime, cal_rows: List[Dict], countries: set) -> Optional[Dict]:
    cands = []
    for r in cal_rows:
        if str(r.get("impact","")).lower().startswith("high"):
            c = (r.get("country") or "").lower()
            if not c or c not in countries: 
                continue
            dt = _parse_dt_guess(r.get("utc_iso",""))
            if dt and dt >= now - timedelta(minutes=5):
                cands.append((dt, r))
    cands.sort(key=lambda x: x[0])
    return cands[0][1] if cands else None

def _calendar_score(now: datetime, ev: Optional[Dict]) -> Tuple[int,bool,str,Optional[datetime]]:
    if not ev:
        return 0, False, "", None
    dt = _parse_dt_guess(ev.get("utc_iso",""))
    if not dt:
        return 0, False, ev.get("title",""), None
    title = ev.get("title","")
    quiet = (dt - timedelta(minutes=QUIET_BEFORE_MIN) <= now <= dt + timedelta(minutes=QUIET_AFTER_MIN))
    diff_min = (dt - now).total_seconds()/60.0
    C = 0
    if diff_min <= 15: C = 4
    elif diff_min <= 60: C = 3
    elif diff_min <= 180: C = 2
    else: C = 1
    if RE_IMP.search(title): C = min(5, C+1)
    return C, quiet, title, dt

def _news_score(now: datetime, news_rows: List[Dict], countries: set) -> Tuple[int, Optional[Dict]]:
    fresh = now - timedelta(minutes=NEWS_FRESH_MIN)
    cand = []
    for r in news_rows:
        ts = _parse_dt_guess(r.get("ts_utc",""))
        if ts and ts >= fresh:
            c = (r.get("countries") or "").lower()
            if c and c in countries:
                s = 0
                if "HIGH" in (r.get("importance_guess") or ""): s += 2
                if "keyword" in (r.get("tags") or ""): s += 1
                if RE_IMP.search(r.get("title","")): s += 1
                cand.append((s, r))
    if not cand: 
        return 0, None
    cand.sort(key=lambda x:x[0], reverse=True)
    s, obj = cand[0]
    return min(4, max(0, s)), obj

def _market_bits_from_bmr(sh, pair: str) -> Tuple[int,str,str]:
    if not USE_BMR_FOR_BIAS:
        return 0, "neutral", ""
    name = BMR_SHEETS.get(pair)
    if not name:
        return 0, "neutral", ""
    try:
        ws = sh.worksheet(name)
        rows = ws.get_all_records()
        if not rows:
            return 0, "neutral", ""
        last = rows[-1]
        st  = str(last.get("Supertrend","")).lower()
        volz= float(str(last.get("Vol_z","0")).replace(",",".") or 0.0)
        atr1= float(str(last.get("ATR_1h","0")).replace(",",".") or 0.0)
        m = 0
        if volz > 2.0: m += 2
        elif volz > 1.3: m += 1
        if atr1 > 1.5: m += 1
        bias = "neutral"
        if "up" in st: bias = "long-only"
        if "down" in st: bias = "short-only"
        reason = f"ST={st or 'n/a'}; volZ={volz:.2f}; ATR1h={atr1:.2f}"
        return min(3,m), bias, reason
    except Exception:
        return 0, "neutral", ""

def _risk_percent(C:int,N:int,M:int,quiet:bool) -> int:
    r = C*W_CAL + N*W_NEWS + M*W_MKT
    if quiet and C>=3: r += QUIET_BONUS
    return max(0, min(100, int(round(r))))

def _risk_label(pct:int) -> str:
    if pct >= RED_THR: return "Red"
    if pct >= AMBER_THR: return "Amber"
    return "Green"

# ---------------- Pipeline ----------------
def write_calendar(ws_cal, rows: List[Dict]):
    # dedup по (utc_iso,title,source)
    have = set()
    try:
        ex = ws_cal.get_all_records()
        for r in ex:
            have.add((str(r.get("utc_iso","")), str(r.get("title","")), str(r.get("source",""))))
    except Exception:
        pass

    batch = []
    for e in rows:
        key = (e["utc_iso"], e["title"], e["source"])
        if key in have: 
            continue
        # local_time
        e = dict(e)
        e["local_time"] = _local_iso_from_utc_iso(e["utc_iso"], LOCAL_TZ)
        batch.append([e.get(h,"") for h in CAL_HEADERS])
    if batch:
        ws_cal.append_rows(batch, value_input_option="RAW")
    log.info("CALENDAR: %d new rows appended", len(batch))

def write_news(ws_news, news: List[Dict]):
    if not news: 
        return
    have = set()
    try:
        ex = ws_news.get_all_records()
        for r in ex:
            have.add((str(r.get("hash","")), str(r.get("source",""))))
    except Exception:
        pass
    batch = []
    for n in news:
        key = (n.get("hash",""), n.get("source",""))
        if key in have: 
            continue
        batch.append([n.get(h,"") for h in NEWS_HEADERS])
    if batch:
        ws_news.append_rows(batch, value_input_option="RAW")
    log.info("NEWS: %d new rows appended", len(batch))

def compute_and_write_fa(sh, ws_fa, ws_digest, cal_rows: List[Dict], news_rows: List[Dict]):
    now = datetime.now(timezone.utc)
    updated = []

    for pair, countries in PAIR_COUNTRIES.items():
        ev = _nearest_high_event(now, cal_rows, countries)
        C, quiet, ev_title, ev_dt = _calendar_score(now, ev)
        N, nobj = _news_score(now, news_rows, countries)
        M, bias_bmr, mkt_reason = _market_bits_from_bmr(sh, pair)

        risk_pct = _risk_percent(C, N, M, quiet)
        risk_lbl = _risk_label(risk_pct)
        bias = bias_bmr or "neutral"

        ttl = 180
        scan_lock_until = ""
        reserve_off = 0

        reason_bits = []
        if C: reason_bits.append(f"C{C}{'Q' if quiet else ''}:{(ev_title or '')[:40]}")
        if N: reason_bits.append(f"N{N}:{(nobj.get('title','')[:40] if nobj else '')}")
        if M: reason_bits.append(f"M{M}({mkt_reason})")
        reason = " | ".join(reason_bits) if reason_bits else "base"

        if ev_dt:
            ahead_min = int((ev_dt - now).total_seconds()/60.0)
            if ahead_min <= 60: ttl = 90
            elif ahead_min <= 180: ttl = 120
            else: ttl = 180
            if quiet and C >= 3:
                lock_till = max(now, ev_dt + timedelta(minutes=QUIET_AFTER_MIN))
                scan_lock_until = _iso_utc(lock_till)
                reserve_off = 1 if risk_lbl == "Red" else 0

        # ensure headers & upsert
        headers = ws_fa.row_values(1) or []
        if headers != FA_HEADERS:
            ws_fa.update("A1",[FA_HEADERS])
        rows = ws_fa.get_all_records()
        idx = None
        for i, r in enumerate(rows, start=2):
            if str(r.get("pair","")).upper() == pair:
                idx = i; break
        rowdict = {
            "pair": pair, "risk": risk_lbl, "bias": bias, "ttl": ttl,
            "updated_at": _iso_utc(now),
            "scan_lock_until": scan_lock_until, "reserve_off": reserve_off,
            "dca_scale": 1.0, "reason": reason, "risk_pct": risk_pct,
        }
        values = [rowdict.get(h,"") for h in FA_HEADERS]
        if idx is None:
            ws_fa.append_row(values, value_input_option="RAW")
        else:
            ws_fa.update(f"A{idx}", [values])
        updated.append((pair, risk_lbl, bias, risk_pct, reason))

    if updated:
        def tag(p): 
            return "🟢" if p<AMBER_THR else ("🟠" if p<RED_THR else "🔴")
        now_iso = _iso_utc(now)
        lines = ["ФА-оценка (aggregated):"]
        for p,r,b,rv,rsn in updated:
            lines.append(f"• {p}: {tag(rv)} {rv}% | {r}/{b} | {rsn}")
        ws_digest.append_row([now_iso, "\n".join(lines)], value_input_option="RAW")

def do_cycle():
    log.info("collector… sheet=%s ws=%s tz=%s window=[-%dd, +%dd]", SHEET_ID, CAL_WS, LOCAL_TZ, BACK_DAYS, AHEAD_DAYS)
    sh = _load_sheet()
    ws_cal    = _ensure_ws(sh, CAL_WS, CAL_HEADERS)
    ws_news   = _ensure_ws(sh, NEWS_WS, NEWS_HEADERS)
    ws_fa     = _ensure_ws(sh, FA_WS, FA_HEADERS)
    ws_digest = _ensure_ws(sh, DIGEST_WS, DIGEST_HEADERS)

    # A) Календарь
    raw = collect_calendar()
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=BACK_DAYS)
    end   = now + timedelta(days=AHEAD_DAYS)
    cal = []
    for e in raw:
        dt = _parse_dt_guess(e.get("utc_iso",""))
        if dt and start <= dt <= end:
            cal.append(e)
    cal.sort(key=lambda x: x["utc_iso"])
    write_calendar(ws_cal, cal)

    # B) Новости
    news = fetch_news()
    write_news(ws_news, news)

    # Read back (возможно уже были старые строки)
    cal_rows  = ws_cal.get_all_records()
    news_rows = ws_news.get_all_records()

    # C) Рынок + сводный риск → FA_Signals
    compute_and_write_fa(sh, ws_fa, ws_digest, cal_rows, news_rows)
    log.info("cycle done.")

def main():
    if not SHEET_ID:
        raise SystemExit("SHEET_ID is not set")
    if RUN_FOREVER:
        while True:
            try:
                do_cycle()
            except Exception:
                log.exception("cycle failed")
            time.sleep(max(60, EVERY_MIN*60))
    else:
        do_cycle()

if __name__ == "__main__":
    main()
