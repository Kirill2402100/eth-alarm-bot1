# calendar_collector.py
# Вытягивает:
#  A) Календарь (запланированное — FOMC/ECB/BoE/BoJ + источники)
#  B) Новости (пресс-релизы ЦБ/Минфинов, без «мусора» навигации)
#  C) Лёгкую FA-оценку и агрегированную выжимку для INVESTOR_DIGEST
#
# Требуемые ENV:
#   SHEET_ID
#   GOOGLE_CREDENTIALS_JSON_B64 (или GOOGLE_CREDENTIALS_JSON / GOOGLE_CREDENTIALS)
#   CAL_WS_OUT=CALENDAR
#   CAL_WS_RAW=NEWS
#   FREE_LOOKBACK_DAYS=7
#   FREE_LOOKAHEAD_DAYS=30
#   COLLECT_EVERY_MIN=120   (если RUN_FOREVER=1)
#   RUN_FOREVER=1|0
#   JINA_PROXY=1|0
#   LOCAL_TZ=Europe/Belgrade
#
# Дополнительно (необязательные): QUIET_BEFORE_MIN, QUIET_AFTER_MIN

from __future__ import annotations

import os, json, time, re, logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
from urllib.parse import urljoin

# --- logging ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("calendar_collector")

# --- tz ---
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None
LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Europe/Belgrade")) if ZoneInfo else None

# --- http ---
try:
    import httpx
except Exception:
    httpx = None

# --- parsing ---
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# --- sheets ---
try:
    import gspread
    from google.oauth2 import service_account
    _GSHEETS = True
except Exception:
    gspread = None
    service_account = None
    _GSHEETS = False

# ----------------- ENV -----------------
SHEET_ID = os.getenv("SHEET_ID", "").strip()
CAL_WS_OUT = os.getenv("CAL_WS_OUT", "CALENDAR").strip() or "CALENDAR"
CAL_WS_RAW = os.getenv("CAL_WS_RAW", "NEWS").strip() or "NEWS"

RUN_FOREVER = os.getenv("RUN_FOREVER", "1").lower() in ("1", "true", "yes", "on")
COLLECT_EVERY_MIN = int(os.getenv("COLLECT_EVERY_MIN", "180") or "180")

FREE_LOOKBACK_DAYS = int(os.getenv("FREE_LOOKBACK_DAYS", "7") or "7")
FREE_LOOKAHEAD_DAYS = int(os.getenv("FREE_LOOKAHEAD_DAYS", "30") or "30")

QUIET_BEFORE_MIN = int(os.getenv("QUIET_BEFORE_MIN", "45") or "45")
QUIET_AFTER_MIN  = int(os.getenv("QUIET_AFTER_MIN",  "45") or "45")

USE_JINA = os.getenv("JINA_PROXY", "1").lower() in ("1","true","yes","on")

# ----------------- CONSTANTS -----------------
CAL_HEADERS = ["utc_iso","local_time","country","currency","title","impact","source","url"]
NEWS_HEADERS = ["ts_utc","source","title","url","countries","ccy","tags","importance_guess","hash"]
FA_HEADERS = ["pair","risk","bias","ttl","updated_at","scan_lock_until","reserve_off","dca_scale","reason","risk_pct"]

PAIR_LIST = ["USDJPY","AUDUSD","EURUSD","GBPUSD"]
PAIR_COUNTRIES = {
    "USDJPY": {"united states", "japan"},
    "AUDUSD": {"australia", "united states"},
    "EURUSD": {"euro area", "united states"},
    "GBPUSD": {"united kingdom", "united states"},
}
PAIR_CCY = {"USDJPY":"JPY","AUDUSD":"AUD","EURUSD":"EUR","GBPUSD":"GBP"}

# ----------------- UTILS -----------------
def _decode_b64_json(s: str) -> Optional[dict]:
    s = (s or "").strip()
    if not s:
        return None
    import base64
    s += "=" * ((4 - len(s) % 4) % 4)
    try:
        return json.loads(base64.b64decode(s).decode("utf-8", "strict"))
    except Exception:
        return None

def _sheets_client():
    if not _GSHEETS:
        return None, "gsheets libs not installed"
    info = _decode_b64_json(os.getenv("GOOGLE_CREDENTIALS_JSON_B64", "")) \
           or _decode_b64_json(os.getenv("GOOGLE_CREDENTIALS_B64", ""))

    if not info:
        # прямой JSON в переменной
        for k in ("GOOGLE_CREDENTIALS_JSON","GOOGLE_CREDENTIALS"):
            raw = os.getenv(k, "").strip()
            if raw:
                try:
                    info = json.loads(raw)
                    break
                except Exception:
                    pass
    if not info:
        return None, "no credentials in env"

    try:
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID)
        return sh, "ok"
    except Exception as e:
        return None, f"auth/open error: {e}"

def _ensure_ws(sh, title: str, headers: List[str]):
    try:
        ws = sh.worksheet(title)
        # сверим шапку
        cur = ws.get_values("A1:Z1") or [[]]
        if (cur and cur[0] != headers):
            ws.update(range_name="A1", values=[headers])
        return ws
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=1000, cols=max(10, len(headers)))
        ws.update(range_name="A1", values=[headers])
        return ws

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _fmt_local(dt_utc: datetime) -> str:
    if not dt_utc: return ""
    if LOCAL_TZ:
        return dt_utc.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
    return dt_utc.strftime("%Y-%m-%d %H:%M")

# ----------------- HTTP -----------------
def fetch(url: str, timeout: float = 20.0) -> str:
    if not httpx:
        return ""
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        }) as cli:
            r = cli.get(url)
            if r.status_code == 200:
                return r.text
            # fallback на r.jina.ai при запрете/редиректе
            if USE_JINA and r.status_code in (301,302,403,404):
                proxy = "https://r.jina.ai/http://" + url.replace("https://","").replace("http://","")
                r2 = cli.get(proxy)
                if r2.status_code == 200:
                    return r2.text
    except Exception:
        pass
    return ""

# ----------------- CALENDAR PARSERS (минимально необходимые) -----------------
def parse_fomc_calendar(html: str) -> List[dict]:
    """Грубый парсер страницы FOMC calendars; возвращает «событие заседание/решение»."""
    out = []
    if not html:
        return out
    soup = BeautifulSoup(html, "html.parser") if BeautifulSoup else None
    if not soup:
        return out

    # Ищем блоки таблиц с датами заседаний
    txt = soup.get_text("\n", strip=True).lower()
    # по ключевым словам формируем одно «опорное» событие в середине дня
    # (точного часа нет на этой странице)
    for m in re.finditer(r"fomc\s+meeting", txt):
        # просто добавим placeholder на ближайшую середину дня сегодняшнего
        dt = _now_utc().replace(hour=12, minute=0, second=0, microsecond=0)
        out.append({
            "utc": dt,
            "country": "united states",
            "currency": "USD",
            "title": "FOMC Meeting / Rate Decision",
            "impact": "high",
            "source": "FOMC",
            "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        })
        break
    return out

def parse_ecb_govc(html: str) -> List[dict]:
    out = []
    if not html or not BeautifulSoup: return out
    soup = BeautifulSoup(html, "html.parser")
    # Говернинг каунсил: ищем govc / monetary policy meetings
    for a in soup.select('a[href*="/press/calendars/"], a[href*="/press/govcdec/"]'):
        title = (a.get_text(" ", strip=True) or "").strip()
        if not title: continue
        url = urljoin("https://www.ecb.europa.eu", a.get("href",""))
        dt = _now_utc().replace(hour=11, minute=0, second=0, microsecond=0)
        out.append({
            "utc": dt, "country":"euro area","currency":"EUR",
            "title": "ECB Governing Council / Decision",
            "impact":"high","source":"ECB","url":url
        })
        break
    return out

def parse_boe_mpc(html: str) -> List[dict]:
    out = []
    if not html or not BeautifulSoup: return out
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.select('a[href*="/monetary-policy-summary"]'):
        url = urljoin("https://www.bankofengland.co.uk", a.get("href",""))
        dt = _now_utc().replace(hour=11, minute=0, second=0, microsecond=0)
        out.append({
            "utc": dt,"country":"united kingdom","currency":"GBP",
            "title":"BoE Monetary Policy Summary / Decision",
            "impact":"high","source":"BoE","url":url
        })
        break
    return out

def parse_boj_mpm(html: str) -> List[dict]:
    out = []
    if not html or not BeautifulSoup: return out
    soup = BeautifulSoup(html, "html.parser")
    if soup.find(string=re.compile(r"Monetary Policy Meeting", re.I)):
        dt = _now_utc().replace(hour=3, minute=0, second=0, microsecond=0)
        out.append({
            "utc": dt, "country":"japan","currency":"JPY",
            "title": "BoJ Monetary Policy Meeting",
            "impact":"high","source":"BoJ",
            "url":"https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm/"
        })
    return out

# ----------------- NEWS PARSERS (чистые) -----------------
def _clean_text(node) -> str:
    if not node:
        return ""
    txt = node.get_text(" ", strip=True)
    txt = re.sub(r"\s+", " ", txt)
    blacklist = {"skip to main content", "back to home"}
    return "" if txt.lower() in blacklist else txt

def _guess_importance(title: str) -> str:
    t = (title or "").lower()
    hi = [
        "emergency", "unscheduled", "intervention", "fx intervention",
        "rate decision", "policy decision", "statement", "press conference",
        "bank rate", "surprise", "guidance", "hike", "cut", "stability"
    ]
    md = ["minutes", "speech", "remarks", "testimony", "q&a", "fireside"]
    if any(k in t for k in hi): return "high"
    if any(k in t for k in md): return "medium"
    return "low"

def parse_fed_news(html: str) -> list[dict]:
    if not html or not BeautifulSoup: return []
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select('a[href*="/newsevents/pressreleases/"]'):
        href = a.get("href", "")
        title = _clean_text(a)
        if not title or not href:
            continue
        url = urljoin("https://www.federalreserve.gov", href)
        out.append({
            "source": "US_FED_PR", "title": title, "url": url,
            "countries": "united states", "ccy": "USD",
            "importance_guess": _guess_importance(title)
        })
    return out

def parse_ust_news(html: str) -> list[dict]:
    if not html or not BeautifulSoup: return []
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select('a[href*="/news/press-releases/"]'):
        href = a.get("href", "")
        title = _clean_text(a)
        if not title or not href:
            continue
        url = urljoin("https://home.treasury.gov", href)
        out.append({
            "source": "US_TREASURY", "title": title, "url": url,
            "countries": "united states", "ccy": "USD",
            "importance_guess": _guess_importance(title)
        })
    return out

def parse_boe_news(html: str) -> list[dict]:
    if not html or not BeautifulSoup: return []
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select('a[href^="/news/"]'):
        href = a.get("href", "")
        if not re.search(r"/news/\d{4}/", href or ""):
            continue
        title = _clean_text(a)
        if not title:
            continue
        url = urljoin("https://www.bankofengland.co.uk", href)
        out.append({
            "source": "BOE_NEWS", "title": title, "url": url,
            "countries": "united kingdom", "ccy": "GBP",
            "importance_guess": _guess_importance(title)
        })
    return out

def parse_ecb_pr(html: str) -> list[dict]:
    if not html or not BeautifulSoup: return []
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select('a[href*="/press/pr/"], a[href*="/press/govcdec/"]'):
        href = a.get("href", "")
        title = _clean_text(a)
        if not title or not href:
            continue
        url = urljoin("https://www.ecb.europa.eu", href)
        out.append({
            "source": "ECB_PR", "title": title, "url": url,
            "countries": "euro area", "ccy": "EUR",
            "importance_guess": _guess_importance(title)
        })
    return out

def parse_rba_media(html: str) -> list[dict]:
    if not html or not BeautifulSoup: return []
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select('a[href*="/media-releases/"]'):
        href = a.get("href", "")
        title = _clean_text(a)
        if not title or not href:
            continue
        url = urljoin("https://www.rba.gov.au", href)
        out.append({
            "source": "RBA_MR", "title": title, "url": url,
            "countries": "australia", "ccy": "AUD",
            "importance_guess": _guess_importance(title)
        })
    return out

def parse_mof_fx(html: str) -> list[dict]:
    if not html or not BeautifulSoup: return []
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select('a[href], h2, h3'):
        title = _clean_text(a)
        if not title:
            continue
        t = title.lower()
        if "intervention" in t or "foreign exchange" in t:
            href = a.get("href", "")
            url = urljoin("https://www.mof.go.jp", href) if href else "https://www.mof.go.jp/english/"
            out.append({
                "source": "JPN_MOF", "title": title, "url": url,
                "countries": "japan", "ccy": "JPY",
                "importance_guess": "high"
            })
    return out

def collect_news_all(html_map: dict[str,str]) -> list[dict]:
    items = []
    items += parse_fed_news(html_map.get("fed",""))
    items += parse_ust_news(html_map.get("ust",""))
    items += parse_boe_news(html_map.get("boe",""))
    items += parse_ecb_pr(html_map.get("ecb",""))
    items += parse_rba_media(html_map.get("rba",""))
    items += parse_mof_fx(html_map.get("mof",""))

    # важность
    items = [x for x in items if x.get("importance_guess") in ("high","medium")]
    # хеш
    for x in items:
        x["hash"] = f'{x.get("source","")}|{x.get("title","")}'
    # дедуп внутри батча
    seen = set()
    out = []
    for x in items:
        h = x["hash"]
        if h in seen: 
            continue
        seen.add(h)
        out.append(x)
    return out

# ----------------- SHEETS IO -----------------
def _read_existing_hashes(ws, hash_col_letter="I") -> set:
    try:
        vals = ws.col_values(ord(hash_col_letter)-ord("A")+1)[1:]  # без хедера
        return set(v.strip() for v in vals if v.strip())
    except Exception:
        return set()

def append_calendar(sh, events: List[dict]) -> int:
    if not events:
        return 0
    ws = _ensure_ws(sh, CAL_WS_OUT, CAL_HEADERS)
    # построим хеш по (utc_iso|title|source)
    try:
        existing = ws.get_all_records()
    except Exception:
        existing = []
    existed = set(f"{r.get('utc_iso','')}|{r.get('title','')}|{r.get('source','')}" for r in existing)

    new_rows = []
    for e in events:
        key = f"{e['utc'].strftime('%Y-%m-%dT%H:%M:%S%z')}|{e['title']}|{e['source']}"
        if key in existed:
            continue
        new_rows.append([
            e["utc"].strftime("%Y-%m-%d %H:%M:%S%z"),
            _fmt_local(e["utc"]),
            e["country"], e["currency"], e["title"], e["impact"], e["source"], e["url"],
        ])
    if not new_rows:
        return 0
    ws.append_rows(new_rows, value_input_option="RAW")
    return len(new_rows)

def append_news(sh, items: List[dict]) -> int:
    if not items:
        return 0
    ws = _ensure_ws(sh, CAL_WS_RAW, NEWS_HEADERS)
    existed = _read_existing_hashes(ws, "I")

    rows = []
    now = _now_utc().strftime("%Y-%m-%d %H:%M:%S%z")
    for x in items:
        if x["hash"] in existed:
            continue
        rows.append([
            now, x.get("source",""), x.get("title",""), x.get("url",""),
            x.get("countries",""), x.get("ccy",""), x.get("tags",""),
            x.get("importance_guess",""), x["hash"]
        ])
    if not rows:
        return 0
    ws.append_rows(rows, value_input_option="RAW")
    return len(rows)

# ----------------- SIMPLE FA -----------------
def fa_policy_from_signals(has_red_event: bool, news_hits: Dict[str,int]) -> dict:
    # очень простая политика:
    # если рядом (±60м) high событие — risk=Amber, lock на QUIET окно
    # если «intervention/decision/statement» в новостях — risk=Red 90 минут
    if news_hits.get("high", 0) >= 1:
        return {"risk":"Red","ttl":180, "scan_lock_min":90, "reserve_off":1, "dca_scale":0.5, "reason":"news-high"}
    if has_red_event:
        return {"risk":"Amber","ttl":120, "scan_lock_min":QUIET_BEFORE_MIN+QUIET_AFTER_MIN, "reserve_off":0, "dca_scale":0.75, "reason":"calendar-window"}
    return {"risk":"Green","ttl":180, "scan_lock_min":0, "reserve_off":0, "dca_scale":1.0, "reason":"base"}

def update_fa_sheet(sh, cal_rows: List[dict], news_rows: List[dict]):
    ws = _ensure_ws(sh, "FA_Signals", FA_HEADERS)
    now = _now_utc()

    # подмножества для каждой пары
    per_pair = {}
    for p in PAIR_LIST:
        cset = PAIR_COUNTRIES[p]
        around = []
        for r in cal_rows:
            if r.get("country") in cset:
                around.append(r)

        # «красное событие рядом» — в течении 60 минут
        red_soon = any(abs((r["utc"] - now).total_seconds())/60.0 <= 60 for r in around)

        # новости по валюте пары (по стране или по ccy)
        hits = {"high":0, "medium":0}
        for n in news_rows:
            if n.get("countries") in cset or n.get("ccy","") == PAIR_CCY[p]:
                lvl = n.get("importance_guess")
                if lvl in hits:
                    hits[lvl] += 1

        pol = fa_policy_from_signals(red_soon, hits)
        bias = "neutral"  # можно усложнить при необходимости
        scan_lock_until = ""
        if pol["scan_lock_min"] > 0:
            scan_lock_until = (now + timedelta(minutes=pol["scan_lock_min"])).strftime("%Y-%m-%dT%H:%M:%SZ")

        per_pair[p] = {
            "risk": pol["risk"], "bias": bias, "ttl": pol["ttl"],
            "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scan_lock_until": scan_lock_until,
            "reserve_off": 1 if pol["reserve_off"] else 0,
            "dca_scale": pol["dca_scale"], "reason": pol["reason"], "risk_pct": 0
        }

    # переписываем лист (маленький и простой)
    values = [FA_HEADERS]
    for p in PAIR_LIST:
        row = per_pair[p]
        values.append([
            p, row["risk"], row["bias"], row["ttl"], row["updated_at"],
            row["scan_lock_until"], row["reserve_off"], row["dca_scale"], row["reason"], row["risk_pct"]
        ])
    ws.clear()
    ws.update(range_name="A1", values=values)

def append_digest(sh, per_pair: Dict[str,dict]):
    ws = _ensure_ws(sh, "INVESTOR_DIGEST", ["ts_utc","text"])
    # соберём мини-выжимку
    lines = ["Об-оценка (aggregated):"]
    for p in PAIR_LIST:
        row = per_pair.get(p, {})
        risk = row.get("risk","Green")
        bias = row.get("bias","neutral")
        badge = "🟢" if risk.lower()=="green" else "🟡" if risk.lower().startswith("amber") else "🔴"
        lines.append(f"• {p}: {badge} 0% | {risk}/{bias} | base")
    text = "\n".join(lines)
    ws.append_row([_now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"), text], value_input_option="RAW")

# ----------------- PIPELINE -----------------
def collect_once():
    if not SHEET_ID:
        log.error("SHEET_ID is empty")
        return
    sh, state = _sheets_client()
    if not sh:
        log.error("Sheets error: %s", state)
        return

    log.info("collector… sheet=%s ws=%s tz=%s window=[-%dd, +%dd]",
             SHEET_ID, CAL_WS_OUT, (LOCAL_TZ.key if LOCAL_TZ else "UTC"),
             FREE_LOOKBACK_DAYS, FREE_LOOKAHEAD_DAYS)

    # -------- CALENDAR --------
    cal_events: List[dict] = []

    # источники (минимум — FOMC/ECB/BoE/BoJ)
    html_fomc = fetch("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    ev_fomc = parse_fomc_calendar(html_fomc)
    log.info("FOMC parsed: %d", len(ev_fomc)); cal_events += ev_fomc

    html_ecb = fetch("https://www.ecb.europa.eu/press/calendars/mgc/html/index.en.html")
    ev_ecb = parse_ecb_govc(html_ecb)
    log.info("ECB parsed: %d", len(ev_ecb)); cal_events += ev_ecb

    html_boe = fetch("https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes")
    ev_boe = parse_boe_mpc(html_boe)
    log.info("BoE parsed: %d", len(ev_boe)); cal_events += ev_boe

    html_boj = fetch("https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm/")
    ev_boj = parse_boj_mpm(html_boj)
    log.info("BoJ parsed: %d", len(ev_boj)); cal_events += ev_boj

    # окно дат
    now = _now_utc()
    d1 = now - timedelta(days=FREE_LOOKBACK_DAYS)
    d2 = now + timedelta(days=FREE_LOOKAHEAD_DAYS)
    cal_events = [e for e in cal_events if d1 <= e["utc"] <= d2]

    # запись в CALENDAR
    appended_cal = append_calendar(sh, cal_events)
    log.info("CALENDAR: %d new rows appended", appended_cal)

    # -------- NEWS --------
    html_map = {
        "fed": fetch("https://www.federalreserve.gov/newsevents/pressreleases.htm"),
        "ust": fetch("https://home.treasury.gov/news/press-releases"),
        "ecb": fetch("https://www.ecb.europa.eu/press/pr/html/index.en.html"),
        "boe": fetch("https://www.bankofengland.co.uk/news"),
        "rba": fetch("https://www.rba.gov.au/media-releases/"),
        "mof": fetch("https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/index.html"),
    }
    news_items = collect_news_all(html_map)
    log.info("NEWS collected: %d items", len(news_items))
    appended_news = append_news(sh, news_items)
    log.info("NEWS: %d new rows appended", appended_news)

    # -------- FA_Signals + DIGEST (лёгкая версия) --------
    # Для FA используем уже собранные cal_events и news_items
    try:
        update_fa_sheet(sh, cal_events, news_items)
    except Exception as e:
        log.warning("FA update failed: %s", e)

    # Для INVESTOR_DIGEST берём только что записанные FA-строки обратно (просто соберём ещё раз локально)
    per_pair: Dict[str,dict] = {}
    try:
        ws_fa = _ensure_ws(sh, "FA_Signals", FA_HEADERS)
        rows = ws_fa.get_all_records()
        for r in rows:
            per_pair[str(r.get("pair","")).upper()] = r
        if per_pair:
            append_digest(sh, per_pair)
    except Exception as e:
        log.warning("INVESTOR_DIGEST append failed: %s", e)

def main():
    try:
        while True:
            start = time.time()
            collect_once()
            if not RUN_FOREVER:
                break
            # сон до следующего цикла
            sleep_sec = max(10.0, COLLECT_EVERY_MIN * 60.0 - (time.time() - start))
            log.info("cycle done.")
            time.sleep(sleep_sec)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    print("Starting Container")
    main()
