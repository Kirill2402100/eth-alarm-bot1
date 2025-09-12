# calendar_collector.py — собирает CALENDAR и NEWS в Google Sheets

import os, re, json, base64, time, logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Iterable
from datetime import datetime, timedelta, timezone
from html import unescape as _html_unescape
import html as _html

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
INVESTOR_WS = (os.getenv("INVESTOR_WS","INVESTOR_DIGEST").strip() or "INVESTOR_DIGEST")

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

# emoji для рисков
RISK_EMOJI = {"Green": "🟢", "Amber": "🟡", "Red": "🔴"}

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
                    ws.update(range_name="A1", values=[headers])
            except Exception:
                pass
            return ws, False
    ws = sh.add_worksheet(title=title, rows=200, cols=max(10, len(headers)))
    ws.update(range_name="A1", values=[headers])
    return ws, True

def _to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _localize(dt: datetime) -> datetime:
    return dt.astimezone(LOCAL_TZ) if LOCAL_TZ else dt

def _norm_iso_key(v: str) -> str:
    import re
    s = str(v or "").strip()
    s = s.replace(" ", "T").replace("Z", "+00:00")
    s = re.sub(r'([+-]\d{2})(\d{2})$', r'\1:\2', s)
    try:
        dt = datetime.fromisoformat(s).astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return s

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

def _html_to_text(s: str) -> str:
    """Грубое, но очень устойчивое «выковыривание» текста из HTML."""
    if not s:
        return ""
    # убираем теги и сжимаем пробелы/переносы
    s = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = _html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

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
def parse_fomc_calendar(html: str, url: str) -> List[CalEvent]:
    """
    Страница FOMC часто меняет верстку. Делаем устойчиво:
    1) превращаем HTML в сплошной текст
    2) ищем все шаблоны дат типа 'September 16–17, 2025' / 'Sep 16-17, 2025'
    3) берем ПЕРВЫЙ день интервала как ориентир (14:00Z)
    """
    text = _html_to_text(html)
    # Sep / September и т.п. + день (иногда диапазон) + год
    pat = re.compile(
        r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
        r"Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+"
        r"(\d{1,2})(?:\s*[–—\-]\s*(\d{1,2}))?,\s*(20\d{2})",
        re.I
    )

    out: List[CalEvent] = []
    for m in pat.finditer(text):
        mon_s, d1_s, d2_s, y_s = m.group(1), m.group(2), m.group(3), m.group(4)
        month = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,
                 "sep":9,"oct":10,"nov":11,"dec":12}[mon_s.lower()[:3]]
        day = int(d1_s)
        year = int(y_s)
        dt = datetime(year, month, day, 14, 0, 0, tzinfo=timezone.utc)
        out.append(CalEvent(
            utc=dt, country="united states", currency="USD",
            title="FOMC Meeting / Rate Decision", impact="high", source="FOMC", url=url
        ))
    # дедуп по дате
    uniq = {}
    for ev in out:
        uniq[ev.utc.date()] = ev
    return list(uniq.values())

def parse_boj_calendar(html: str, url: str) -> List[CalEvent]:
    """
    BoJ публикует «MPM schedule» в разных форматах. Берем все yyyy.mm.dd/ yyyy-mm-dd.
    """
    text = _html_to_text(html)
    out: List[CalEvent] = []

    for m in re.finditer(r"(20\d{2})[./-]\s?(\d{1,2})[./-]\s?(\d{1,2})", text):
        try:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            dt = datetime(y, mo, d, 3, 0, 0, tzinfo=timezone.utc)  # утро Токио ~03:00Z
            out.append(CalEvent(
                utc=dt, country="japan", currency="JPY",
                title="BoJ Monetary Policy Meeting", impact="high", source="BoJ", url=url
            ))
        except Exception:
            continue

    # дедуп по дате и отсечка странных лет
    uniq = {}
    for ev in out:
        if 2000 <= ev.utc.year <= 2100:
            uniq[ev.utc.date()] = ev
    return list(uniq.values())

def collect_calendar() -> List[CalEvent]:
    events: List[CalEvent] = []

    url_fomc = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    txt, code, f_url = fetch_text(url_fomc)
    if code:
        f = parse_fomc_calendar(txt, f_url)
        log.info("FOMC parsed: %d", len(f))
        events += f

    url_boj = "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm"
    txt, code, f_url = fetch_text(url_boj)
    if code:
        b = parse_boj_calendar(txt, f_url)
        log.info("BoJ parsed: %d", len(b))
        events += b

    # (ECB/BoE оставляем заглушкой)
    return events

# -------- NEWS scrapers --------
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
        # Дедуп по источнику+URL (а не по title)
        base = f"{self.source}|{self.url}"
        return re.sub(r"\s+", " ", base.strip())[:180]

    def to_row(self) -> List[str]:
        clean_title = _html_unescape(re.sub(r"<[^>]+>", "", self.title)).strip()
        return [
            _to_utc_iso(self.ts_utc),
            self.source,
            clean_title,
            self.url,
            self.countries,
            self.ccy,
            self.tags,
            self.importance_guess,
            self.key_hash()
        ]

def collect_news() -> List[NewsItem]:
    """
    Собираем только релевант:
    - Fed: ссылки вида /newsevents/pressreleases/20xx-press-*.htm и /20xx-press-fomc.htm
    - ECB: press, govcdec, press_conference, monetary policy statements
    - BoE: /news/20xx/*
    - RBA: /media-releases/YYYY/*
    - US Treasury: /news/press-releases/sb\d+
    """
    items: List[NewsItem] = []
    now = datetime.now(timezone.utc)

    # FED
    txt, code, url = fetch_text("https://www.federalreserve.gov/newsevents/pressreleases.htm")
    if code:
        for m in re.finditer(r'href="(/newsevents/pressreleases/(?:20\d{2}-press-(?:fomc|monetary)|20\d{2}-press-\w+)\.htm)"[^>]*>(.*?)</a>', txt, re.I):
            u = "https://www.federalreserve.gov" + m.group(1)
            t = re.sub(r"<[^>]+>", "", m.group(2)).strip() or "Fed press release"
            items.append(NewsItem(now, "US_FED_PR", t, u, "united states", "USD", "policy", "high"))

    # US Treasury
    txt, code, url = fetch_text("https://home.treasury.gov/news/press-releases")
    if code:
        for m in re.finditer(r'href="(/news/press-releases/sb\d+)"[^>]*>(.*?)</a>', txt, re.I):
            u = "https://home.treasury.gov" + m.group(1)
            t = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            items.append(NewsItem(now, "US_TREASURY", t, u, "united states", "USD", "treasury", "medium"))

    # ECB (правление/заявления/конфы)
    txt, code, url = fetch_text("https://www.ecb.europa.eu/press/pubbydate/html/index.en.html?name_of_publication=Press%20release")
    if code:
        for m in re.finditer(r'href="(/press/(?:govcdec|press_conference|pr)/[^"]+)"[^>]*>(.*?)</a>', txt, re.I):
            u = "https://www.ecb.europa.eu" + m.group(1)
            t = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            imp = "high" if KW_RE.search(t) else "medium"
            items.append(NewsItem(now, "ECB_PR", t, u, "euro area", "EUR", "ecb", imp))

    # BoE
    txt, code, url = fetch_text("https://www.bankofengland.co.uk/news")
    if code:
        for m in re.finditer(r'href="(/news/20\d{2}/[^"]+)"[^>]*>(.*?)</a>', txt, re.I):
            u = "https://www.bankofengland.co.uk" + m.group(1)
            t = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            imp = "high" if KW_RE.search(t) else "medium"
            items.append(NewsItem(now, "BOE_PR", t, u, "united kingdom", "GBP", "boe", imp))

    # RBA
    txt, code, url = fetch_text("https://www.rba.gov.au/media-releases/")
    if code:
        for m in re.finditer(r'href="(/media-releases/\d{4}/[^"]+)"[^>]*>(.*?)</a>', txt, re.I):
            u = "https://www.rba.gov.au" + m.group(1)
            t = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            imp = "high" if KW_RE.search(t) else "medium"
            items.append(NewsItem(now, "RBA_MR", t, u, "australia", "AUD", "rba", imp))

    # JP MoF FX (страницу часто меняют; если код !=200 — просто пропускаем)
    txt, code, url = fetch_text("https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/")
    if code == 200 and re.search(r"intervention|announcement", txt, re.I):
        items.append(NewsItem(now, "JP_MOF_FX", "FX intervention reference page", url, "japan", "JPY", "mof", "high"))

    # Легкая отсечка древности по названию (если вдруг соберем архив)
    max_age_days = int(os.getenv("NEWS_MAX_AGE_DAYS", "90") or "90")
    cutoff = now - timedelta(days=max_age_days)
    # пока все ts = now — фильтр на будущее, если появится нормальный парсинг времени
    items = [i for i in items if i.ts_utc >= cutoff]

    log.info("NEWS collected: %d items", len(items))
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
    """
    Три статуса по плотности релевантных новостей в «скользящем окне»:
      - cnt == 0  -> Green:      dca_scale=1.0, reserve_off=0
      - cnt == 1  -> Amber:      dca_scale=0.75, reserve_off=0, lock ~ FA_NEWS_LOCK_MIN/2
      - cnt >= 2  -> Red:        dca_scale=0.5, reserve_off=1,  lock ~ FA_NEWS_LOCK_MIN
    Порог «много новостей» (=красный) регулируется FA_NEWS_MIN_COUNT (обычно 2).
    """
    window_start = now_utc - timedelta(minutes=FA_NEWS_RECENT_MIN)
    recent = [r for r in all_news if r["ts_utc"] >= window_start and _news_is_high(r)]

    cty_hits: dict[str, int] = {}
    for r in recent:
        for c in [x.strip() for x in (r["countries"] or "").split(",") if x.strip()]:
            cty_hits[c] = cty_hits.get(c, 0) + 1

    out: Dict[str, dict] = {}
    for sym, countries in PAIR_COUNTRIES.items():
        cnt = sum(cty_hits.get(c, 0) for c in countries)

        if cnt >= max(1, FA_NEWS_MIN_COUNT):
            risk        = "Red"
            dca_scale   = 0.5
            reserve_off = 1
            lock_min    = FA_NEWS_LOCK_MIN
            reason      = "news-high"
        elif cnt == 1:
            risk        = "Amber"
            dca_scale   = 0.75
            reserve_off = 0
            lock_min    = max(1, FA_NEWS_LOCK_MIN // 2)
            reason      = "news-medium"
        else:
            risk        = "Green"
            dca_scale   = 1.0
            reserve_off = 0
            lock_min    = 0
            reason      = "base"

        out[sym] = {
            "pair": sym, "risk": risk, "bias": "neutral",
            "ttl": FA_NEWS_TTL_MIN, "updated_at": _to_utc_iso(now_utc),
            "scan_lock_until": _to_utc_iso(now_utc + timedelta(minutes=lock_min)) if lock_min else "",
            "reserve_off": reserve_off, "dca_scale": dca_scale,
            "reason": reason, "risk_pct": 0,
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
            s.get("pair", sym),
            s.get("risk", "Green"),
            s.get("bias", "neutral"),
            s.get("ttl", ""),
            s.get("updated_at", ""),
            s.get("scan_lock_until", ""),
            int(bool(s.get("reserve_off", 0))),
            float(s.get("dca_scale", 1.0)),
            s.get("reason", "base"),
            int(s.get("risk_pct", 0)),
        ])
    ws.update(range_name="A2", values=values)

# ---------- INVESTOR_DIGEST ----------
def _read_fa_signals_for_digest(sh) -> List[dict]:
    """Читает лист FA_Signals как список dict'ов; если нет — возвращает пустой список."""
    try:
        ws = sh.worksheet("FA_Signals")
        return ws.get_all_records()
    except Exception:
        return []

def _build_investor_digest_text(rows: List[dict]) -> str:
    """Строит компактную сводку по парам из FA_Signals."""
    by_pair = {str(r.get("pair","")).upper(): r for r in rows}
    lines = []
    for sym in SYMBOLS:
        r = by_pair.get(sym, {})
        risk = str(r.get("risk","Green")).capitalize()
        bias = str(r.get("bias","neutral")).lower()
        reason = str(r.get("reason","")).strip()
        dca = r.get("dca_scale", 1.0)
        # хвост с деталями
        tail = f" | {risk}/{bias}"
        if reason:
            tail += f" | {reason}"
        try:
            dca_val = float(dca)
        except Exception:
            dca_val = 1.0
        if abs(dca_val - 1.0) > 1e-9:
            # печатаем как 0.5 / 0.75 / 1.2 и т.п. без лишних нулей
            dtxt = f"{dca_val:.3f}".rstrip("0").rstrip(".")
            tail += f" | dca×{dtxt}"
        emoji = RISK_EMOJI.get(risk, "🟢")
        lines.append(f"• {sym}: {emoji}{tail}")
    return "FA-сводка (aggregated):\n" + "\n".join(lines)

def write_investor_digest(sh):
    """Пишет одну запись в INVESTOR_DIGEST, если текст изменился со времени прошлой записи."""
    ws, _ = ensure_worksheet(sh, INVESTOR_WS, ["ts_utc","text"])
    text = _build_investor_digest_text(_read_fa_signals_for_digest(sh))
    # проверка на дубликат: сравниваем с последней строкой (B2)
    try:
        last = ws.get_values("B2:B2")
        if last and last[0] and (last[0][0] == text):
            log.info("INVESTOR_DIGEST: unchanged")
            return
    except Exception:
        pass
    ts = _to_utc_iso(datetime.now(timezone.utc))
    ws.insert_row([ts, text], index=2)
    log.info("INVESTOR_DIGEST: +1 row")

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
    ws_cal, _ = ensure_worksheet(sh, CAL_WS_OUT, CAL_HEADERS)
    try:
        existing = ws_cal.get_all_records()
        seen = {(_norm_iso_key(r.get("utc_iso")), (r.get("title") or "").strip()) for r in existing}
    except Exception:
        seen = set()

    new_rows = []
    for ev in _calendar_window_filter(collect_calendar()):
        key = (_norm_iso_key(_to_utc_iso(ev.utc)), ev.title.strip())
        if key not in seen:
            new_rows.append(ev.to_row())
            seen.add(key)

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
    # INVESTOR_DIGEST: агрегированная строка на основе FA_Signals
    try:
        write_investor_digest(sh)
    except Exception as e:
        log.warning("INVESTOR_DIGEST write failed: %s", e)
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
