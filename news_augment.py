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

# -------- Bank of England (GBP) --------
_BOE_SITEMAPS = [
    "https://www.bankofengland.co.uk/sitemap.xml",
    "https://www.bankofengland.co.uk/news/sitemap.xml",
    "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/sitemap.xml",
]

def _boe_is_high(url: str, title: str) -> bool:
    hay = f"{title} {url}".lower()
    # всё, что про MPC/Bank Rate/MPS — high
    return (
        "monetary-policy-summary-and-minutes" in hay
        or "bank rate" in hay
        or "mpc" in hay
        or bool(KW_RE.search(hay))
    )

def _slug_to_title(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    t = re.sub(r"[-_]+", " ", slug).strip()
    return t[:1].upper() + t[1:] if t else "BoE item"

def _boe_from_sitemaps(now: datetime) -> List[NewsItem]:
    items: List[NewsItem] = []
    seen: set[str] = set()
    for sm in _BOE_SITEMAPS:
        xml, code, _ = fetch_text(sm)
        # у них бывает 302→500 — пробуем Jina-прокси вручную
        if code != 200:
            try_xml, try_code, _ = fetch_text(os.getenv("JINA_PROXY", "https://r.jina.ai/http/").rstrip("/") + "/" + sm)
            if try_code == 200 and try_xml:
                xml, code = try_xml, 200
        if code != 200 or not xml:
            continue
        for loc in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml, flags=re.I):
            u = loc.strip()
            if "bankofengland.co.uk" not in u:
                continue
            # берем только реально контентные страницы
            if ("/news/" in u or "/monetary-policy-summary-and-minutes/" in u) and re.search(r"/20\d{2}/", u):
                if u in seen:
                    continue
                seen.add(u)
                title = _slug_to_title(u)
                imp = "high" if _boe_is_high(u, title) else "medium"
                items.append(NewsItem(
                    ts_utc=now, source="BOE_PR", title=title, url=u,
                    countries="united kingdom", ccy="GBP", tags="boe", importance_guess=imp
                ))
    return items

def _ddg_links(query: str) -> List[str]:
    # DuckDuckGo HTML выдаёт статические ссылки вида /l/?uddg=<url>
    import urllib.parse as _up
    url = "https://duckduckgo.com/html/?q=" + _up.quote_plus(query)
    html, code, _ = fetch_text(url)
    out: List[str] = []
    if not code or not html:
        return out
    for m in re.finditer(r'href="(?:/l/)?\?uddg=([^"&]+)"', html):
        target = _up.unquote(m.group(1))
        if "bankofengland.co.uk" in target:
            out.append(target)
    for m in re.finditer(r'href="(https://www\.bankofengland\.co\.uk/[^"]+)"', html):
        out.append(m.group(1))
    # лёгкий дедуп
    uniq: List[str] = []
    seen = set()
    for u in out:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq

def _boe_from_search(now: datetime) -> List[NewsItem]:
    items: List[NewsItem] = []
    queries = [
        'site:bankofengland.co.uk "Monetary Policy Summary" 2025',
        'site:bankofengland.co.uk "Monetary Policy Committee" 2025',
        'site:bankofengland.co.uk "Bank Rate" "Monetary Policy" 2025',
        'site:bankofengland.co.uk/monetary-policy-summary-and-minutes 2025',
    ]
    seen: set[str] = set()
    for q in queries:
        for u in _ddg_links(q):
            if u in seen:
                continue
            seen.add(u)
            title = _slug_to_title(u)
            imp = "high" if _boe_is_high(u, title) else "medium"
            items.append(NewsItem(
                ts_utc=now, source="BOE_PR", title=title, url=u,
                countries="united kingdom", ccy="GBP", tags="boe", importance_guess=imp
            ))
    return items

def collect_boe() -> List[NewsItem]:
    now = datetime.now(timezone.utc)
    # 1) пробуем sitemap
    items = _boe_from_sitemaps(now)
    if items:
        log.info("news_augment: BOE via sitemap collected: %d", len(items))
        return items
    # 2) fallback: DuckDuckGo
    items = _boe_from_search(now)
    log.info("news_augment: BOE via search collected: %d", len(items))
    return items
    
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
