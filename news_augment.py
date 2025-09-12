# news_augment.py — дополняет лист NEWS локальными источниками (GBP/JPY), без изменений основного кода
import os, re, json, base64, time, logging, html as _html
from typing import Optional, List, Tuple, Iterable
from datetime import datetime, timezone
from html import unescape as _html_unescape

import httpx

# ====== ENV / CONFIG ======
SHEET_ID = os.getenv("SHEET_ID","").strip()
RUN_FOREVER = (os.getenv("NEWS_AUGMENT_RUN_FOREVER","1").lower() in ("1","true","yes","on"))
EVERY_MIN = int(os.getenv("NEWS_AUGMENT_EVERY_MIN","30") or "30")
LOG_LEVEL = os.getenv("LOG_LEVEL","INFO").upper()
JINA_PROXY = os.getenv("JINA_PROXY","https://r.jina.ai/http/").rstrip("/") + "/"

NEWS_HEADERS = ["ts_utc","source","title","url","countries","ccy","tags","importance_guess","hash"]

KW_RE = re.compile(os.getenv(
    "FA_NEWS_KEYWORDS",
    "rate decision|monetary policy|bank rate|policy decision|unscheduled|emergency|"
    "intervention|FX intervention|press conference|policy statement|policy statements|"
    "rate statement|cash rate|fomc|mpc"
), re.I)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("news_augment")

# ====== Google Sheets ======
try:
    import gspread
    from google.oauth2 import service_account
    _GSHEETS = True
except Exception:
    gspread = None
    service_account = None
    _GSHEETS = False

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

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

def open_sheet(sheet_id: str):
    if not _GSHEETS:
        return None, "gsheets libs not installed"
    if not sheet_id:
        return None, "SHEET_ID empty"
    info = _load_service_info()
    if not info:
        return None, "no service account json"
    try:
        creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_id)
        return sh, "ok"
    except Exception as e:
        return None, f"auth/open error: {e}"

def ensure_news_ws(sh):
    for ws in sh.worksheets():
        if ws.title == "NEWS":
            try:
                cur = ws.get_values("A1:I1") or [[]]
                if not cur or cur[0] != NEWS_HEADERS:
                    ws.update(range_name="A1", values=[NEWS_HEADERS])
            except Exception:
                pass
            return ws
    ws = sh.add_worksheet(title="NEWS", rows=1000, cols=9)
    ws.update(range_name="A1", values=[NEWS_HEADERS])
    return ws

def _read_existing_hashes(ws) -> set:
    try:
        col = ws.get_values("I2:I10000")
        return {r[0] for r in col if r and r[0]}
    except Exception:
        return set()

# ====== HTTP helpers ======
def fetch_text(url: str, timeout=15.0) -> Tuple[str,int,str]:
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

def _clean_html_text(s: str) -> str:
    if not s: return ""
    s = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = _html.unescape(s)
    return re.sub(r"\s+"," ", s).strip()

# ====== News row model ======
class NewsItem:
    __slots__ = ("ts_utc","source","title","url","countries","ccy","tags","importance")
    def __init__(self, ts_utc: datetime, source: str, title: str, url: str, countries: str, ccy: str, tags: str, importance: str):
        self.ts_utc = ts_utc
        self.source = source
        self.title = title
        self.url = url
        self.countries = countries
        self.ccy = ccy
        self.tags = tags
        self.importance = importance

    def key_hash(self) -> str:
        return f"{self.source}|{self.url}"[:180]

    def to_row(self) -> List[str]:
        clean_title = _html_unescape(re.sub(r"<[^>]+>", "", self.title)).strip() or self.title.strip()
        return [
            self.ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            self.source,
            clean_title,
            self.url,
            self.countries,
            self.ccy,
            self.tags,
            self.importance,
            self.key_hash(),
        ]

def _importance(title: str, url: str) -> str:
    hay = f"{title} {url}"
    return "high" if KW_RE.search(hay) else "medium"

# ====== GBP: Bank of England (improved) ======
BOE_BASE = "https://www.bankofengland.co.uk"
BOE_CATEGORIES = ["news", "publications", "speeches", "statistics", "prudential-regulation"]
BOE_MAX_PAGES = int(os.getenv("BOE_MAX_PAGES", "3") or "3")  # сколько страниц листинга пройти на раздел

def _boe_extract_articles(html: str, cat: str) -> List[tuple[str, str]]:
    """
    Достаём статьи из листинга /news/<cat>[?page=N] только с годом в URL: /news/<cat>/20YY/...
    Возвращаем (url, title).
    """
    out: List[tuple[str, str]] = []

    # Вариант 1: <a href="...">Текст</a> — берём текст как title
    pat1 = re.compile(
        rf'<a[^>]+href="(?P<u>(?:https?://(?:www\.)?bankofengland\.co\.uk)?/news/{cat}/20\d{{2}}/[^"#?]+)"[^>]*>(?P<t>.*?)</a>',
        re.I
    )
    for m in pat1.finditer(html):
        u = m.group("u")
        if not u.startswith("http"):
            u = BOE_BASE + u
        t = _clean_html_text(m.group("t")) or u.rsplit("/", 1)[-1].replace("-", " ").title()
        out.append((u, t))

    # Вариант 2: если в листинге ссылки без видимого текста внутри того же тега
    # (подпись рендерится соседним элементом), то подхватим просто href и синтезируем title из слага
    if not out:
        pat2 = re.compile(
            rf'href="(?P<u>(?:https?://(?:www\.)?bankofengland\.co\.uk)?/news/{cat}/20\d{{2}}/[^"#?]+)"',
            re.I
        )
        for m in pat2.finditer(html):
            u = m.group("u")
            if not u.startswith("http"):
                u = BOE_BASE + u
            t = u.rsplit("/", 1)[-1].replace("-", " ").title()
            out.append((u, t))

    # Страховка от дублей на странице
    seen = set()
    uniq = []
    for u, t in out:
        if u not in seen:
            seen.add(u)
            uniq.append((u, t))
    return uniq

def collect_boe() -> List[NewsItem]:
    """
    Ходим по разделам /news/<category> и пагинации ?page=2..N, собираем статьи за все доступные страницы.
    Фильтруем любые «категории» без /20YY/ в пути.
    """
    now = datetime.now(timezone.utc)
    out: List[NewsItem] = []
    seen_urls = set()

    # Быстрый шаг: пробуем /news/news (основная лента)
    first_urls = [f"{BOE_BASE}/news/news"] + [f"{BOE_BASE}/news/{c}" for c in BOE_CATEGORIES if c != "news"]
    for url0 in first_urls:
        txt, code, eff = fetch_text(url0)
        if not code:
            continue
        arts = []
        # первая страница без параметров
        cat = url0.split("/news/")[-1] or "news"
        cat = cat.strip("/").split("?", 1)[0]
        arts += _boe_extract_articles(txt, cat)
        log.info("BOE %s p1: %d", cat, len(arts))

        # пагинация ?page=2..N
        for p in range(2, BOE_MAX_PAGES + 1):
            urlp = f"{BOE_BASE}/news/{cat}?page={p}"
            txtp, codep, _ = fetch_text(urlp)
            if not codep:
                continue
            arts_p = _boe_extract_articles(txtp, cat)
            log.info("BOE %s p%d: %d", cat, p, len(arts_p))
            arts += arts_p

        for u, t in arts:
            if "/20" not in u or u in seen_urls:
                continue
            seen_urls.add(u)
            imp = _importance(t, u)
            out.append(NewsItem(now, "BOE_PR", t, u, "united kingdom", "GBP", "boe", imp))

    log.info("BOE collected total: %d", len(out))
    return out
    
# ====== JPY: Bank of Japan + MoF ======
def collect_boj() -> List[NewsItem]:
    out: List[NewsItem] = []
    now = datetime.now(timezone.utc)

    def _grab(url: str):
        txt, code, _ = fetch_text(url)
        if not code: return
        for m in re.finditer(r'href="(?:(https?://www\.boj\.or\.jp)?(/en/mopo/(?:mpmdeci|mpmsche_minu)/[^"]+\.(?:htm|pdf)))"', txt, re.I):
            host, rel = m.group(1) or "https://www.boj.or.jp", m.group(2)
            u = host + rel
            t = rel.rsplit("/",1)[-1]
            title = _clean_html_text(t.replace("_"," ").replace("-", " ").split(".")[0]).title()
            out.append(NewsItem(now, "BOJ_PR", title or "BoJ document", u, "japan", "JPY", "boj mpm", "high"))
        for m in re.finditer(r'href="(?:(https?://www\.boj\.or\.jp)?(/en/mopo/(?:mpmdeci|mpmsche_minu)/[^"]+/))"', txt, re.I):
            host, rel = m.group(1) or "https://www.boj.or.jp", m.group(2)
            u = host + rel
            seg = rel.strip("/").rsplit("/",1)[-1]
            title = _clean_html_text(seg.replace("_"," ").replace("-", " ")).title()
            out.append(NewsItem(now, "BOJ_PR", title or "BoJ page", u, "japan", "JPY", "boj mpm", "high"))

    _grab("https://www.boj.or.jp/en/mopo/mpmdeci/index.htm")
    _grab("https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm")

    log.info("BoJ collected: %d", len(out))
    return out

def collect_jp_mof_fx() -> List[NewsItem]:
    urls = [
        "https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/",
        "https://www.mof.go.jp/policy/international_policy/reference/foreign_exchange_intervention/",
    ]
    now = datetime.now(timezone.utc)
    out: List[NewsItem] = []
    for u in urls:
        txt, code, eff = fetch_text(u)
        if code == 200 and re.search(r"intervention|announcement", txt, re.I):
            out.append(NewsItem(now, "JP_MOF_FX", "FX intervention reference page", eff, "japan", "JPY", "mof", "high"))
            break
    log.info("JP MoF FX collected: %d", len(out))
    return out

# ====== Write to NEWS ======
def append_news_rows(sh, items: List[NewsItem]) -> int:
    if not items: return 0
    ws = ensure_news_ws(sh)
    existing = _read_existing_hashes(ws)
    rows = [n.to_row() for n in items if n.key_hash() not in existing]
    if rows:
        ws.append_rows(rows, value_input_option="RAW")
    log.info("NEWS augment: +%d rows", len(rows))
    return len(rows)

# ====== Main cycle ======
def run_once():
    if not SHEET_ID:
        raise RuntimeError("SHEET_ID env empty")
    sh, status = open_sheet(SHEET_ID)
    if not sh:
        raise RuntimeError(f"Sheets not ready: {status}")

    items: List[NewsItem] = []
    items += collect_boe()      # GBP
    items += collect_boj()      # JPY
    items += collect_jp_mof_fx()# JPY (при доступности)

    log.info("augment collected: %d new candidates", len(items))
    append_news_rows(sh, items)

def main():
    if not SHEET_ID:
        raise RuntimeError("SHEET_ID env empty")
    if RUN_FOREVER:
        interval = max(1, EVERY_MIN) * 60
        while True:
            try:
                run_once()
            except Exception:
                log.exception("news_augment iteration failed")
            time.sleep(interval)
    else:
        run_once()

if __name__ == "__main__":
    main()
