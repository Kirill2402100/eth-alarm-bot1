# news_augment.py — докидывает недостающие источники в лист NEWS
# GBP: Bank of England /news (фикс 302→404)
# JPY: Bank of Japan (mpm decisions / minutes / statements)
# JP MoF FX: страница про интервенции (en/jp) — если доступна
#
# ENV (минимум):
#   SHEET_ID
#   GOOGLE_CREDENTIALS_JSON_B64  (или GOOGLE_CREDENTIALS_JSON / GOOGLE_CREDENTIALS)
#   RUN_FOREVER=1                (по умолчанию крутится циклом)
#   AUGMENT_EVERY_MIN=30         (как часто пробегать)
#   FA_NEWS_KEYWORDS             (регексп слов для high-важности; есть дефолт)
#   JINA_PROXY=https://r.jina.ai/http/  (фолбэк на 403/404)
#   LOCAL_TZ (опционально, не используется тут)

import os, re, time, json, base64, logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
from datetime import datetime, timezone

import httpx

# ----- Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("news_augment")

# ----- ENV
SHEET_ID = (os.getenv("SHEET_ID", "") or "").strip()
RUN_FOREVER = os.getenv("RUN_FOREVER", "1").lower() in ("1", "true", "yes", "on")
AUGMENT_EVERY_MIN = int(os.getenv("AUGMENT_EVERY_MIN", "30") or "30")
JINA_PROXY = os.getenv("JINA_PROXY", "https://r.jina.ai/http/").rstrip("/") + "/"

KW_RE = re.compile(os.getenv(
    "FA_NEWS_KEYWORDS",
    # покрывает MPC/BoE, решения, заявления, пресс-конфы, интервенции
    r"(bank rate|monetary policy|mpc|policy decision|rate decision|policy statement|"
    r"press conference|statement|minutes|fx intervention|intervention|unscheduled|emergency)"
), re.I)

NEWS_HEADERS = ["ts_utc", "source", "title", "url", "countries", "ccy", "tags", "importance_guess", "hash"]

# ----- Google Sheets
try:
    import gspread
    from google.oauth2 import service_account
    _GSHEETS_AVAILABLE = True
except Exception:
    gspread = None
    service_account = None
    _GSHEETS_AVAILABLE = False


def _decode_b64_json(s: str) -> dict | None:
    s = (s or "").strip()
    if not s:
        return None
    s += "=" * ((4 - len(s) % 4) % 4)
    try:
        return json.loads(base64.b64decode(s).decode("utf-8", "strict"))
    except Exception:
        return None


def _load_sa_info() -> dict | None:
    info = _decode_b64_json(os.getenv("GOOGLE_CREDENTIALS_JSON_B64", ""))
    if info:
        return info
    for k in ("GOOGLE_CREDENTIALS_JSON", "GOOGLE_CREDENTIALS"):
        raw = (os.getenv(k, "") or "").strip()
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
        return None, "SHEET_ID env empty"
    info = _load_sa_info()
    if not info:
        return None, "no service account json"
    try:
        creds = service_account.Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
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
    ws = sh.add_worksheet(title=title, rows=300, cols=max(10, len(headers)))
    ws.update(range_name="A1", values=[headers])
    return ws, True


def _existing_hashes(ws, col_letter: str = "I") -> Set[str]:
    try:
        rng = f"{col_letter}2:{col_letter}10000"
        col = ws.get_values(rng)
        return {r[0] for r in col if r and r[0]}
    except Exception:
        return set()


# ----- HTTP helpers
def fetch_text(url: str, timeout=15.0) -> Tuple[str, int, str]:
    """GET с fallback через Jina proxy на 403/404."""
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (LPBot/news-augment)"}) as cli:
            r = cli.get(url)
            if r.status_code in (403, 404):
                pr = cli.get(JINA_PROXY + url)
                return pr.text, pr.status_code, str(pr.url)
            return r.text, r.status_code, str(r.url)
    except Exception as e:
        log.warning("fetch failed %s: %s", url, e)
        return "", 0, url


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s or "").strip()


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

    def row(self) -> List[str]:
        clean_title = re.sub(r"\s+", " ", _strip_html(self.title)).strip()
        h = re.sub(r"\s+", " ", f"{self.source}|{self.url}".strip())[:180]
        return [
            self.ts_utc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            self.source,
            clean_title,
            self.url,
            self.countries,
            self.ccy,
            self.tags,
            self.importance_guess,
            h,
        ]


# ----- Scrapers (добавляем недостающее)
def scrape_boe(now: datetime) -> List[NewsItem]:
    """Bank of England: общая лента /news; high если триггерится по ключевым словам."""
    items: List[NewsItem] = []
    txt, code, _ = fetch_text("https://www.bankofengland.co.uk/news")
    if not code:
        return items
    for m in re.finditer(r'href="(/news/20\d{2}/[^"]+)"[^>]*>(.*?)</a>', txt, re.I):
        u = "https://www.bankofengland.co.uk" + m.group(1)
        t = _strip_html(m.group(2))
        imp = "high" if KW_RE.search(t) else "medium"
        items.append(NewsItem(now, "BOE_PR", t or "BoE News", u, "united kingdom", "GBP", "boe", imp))
    return items


def scrape_boj(now: datetime) -> List[NewsItem]:
    """Bank of Japan: набор страниц с решениями/минутами/заявлениями."""
    items: List[NewsItem] = []
    pages = [
        "https://www.boj.or.jp/en/mopo/mpmdeci/index.htm",                  # Monetary Policy Releases
        "https://www.boj.or.jp/en/mopo/mpmdeci/mpr_2025/index.htm",        # All decisions (by year)
        "https://www.boj.or.jp/en/mopo/mpmdeci/state_2025/index.htm",      # Statements on Monetary Policy
        "https://www.boj.or.jp/en/mopo/mpmdeci/minu_2025/index.htm",       # Minutes
        "https://www.boj.or.jp/en/mopo/mpmdeci/opinion_2025/index.htm",    # Summary of Opinions
        "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm",            # Meetings & minutes hub
    ]
    seen_urls: Set[str] = set()
    for url in pages:
        txt, code, _ = fetch_text(url)
        if not code:
            continue
        for m in re.finditer(r'href="(/en/mopo/[^"]+)"[^>]*>(.*?)</a>', txt, re.I):
            href = m.group(1)
            # ограничим на осмысленные разделы
            if not re.search(r"(mpmdeci|mpmsche_minu)", href, re.I):
                continue
            u = "https://www.boj.or.jp" + href
            if u in seen_urls:
                continue
            seen_urls.add(u)
            t = _strip_html(m.group(2)) or "BoJ monetary policy"
            imp = "high" if KW_RE.search(t + " monetary policy mpm decision statement minutes") else "medium"
            items.append(NewsItem(now, "BOJ_PR", t, u, "japan", "JPY", "boj mpm", imp))
    return items


def scrape_mof_fx(now: datetime) -> List[NewsItem]:
    """JP MoF FX page — если открывается и содержит keywords «intervention/announcement», добавляем одну ссылку."""
    items: List[NewsItem] = []
    candidates = [
        "https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/",
        "https://www.mof.go.jp/policy/international_policy/reference/foreign_exchange_intervention/",
    ]
    for url in candidates:
        txt, code, real = fetch_text(url)
        if code == 200 and re.search(r"(intervention|announcement)", txt, re.I):
            items.append(NewsItem(now, "JP_MOF_FX", "FX intervention reference page", real, "japan", "JPY", "mof", "high"))
            break
    return items


def collect_candidates() -> List[NewsItem]:
    now = datetime.now(timezone.utc)
    all_items: List[NewsItem] = []
    all_items += scrape_boe(now)
    all_items += scrape_boj(now)
    all_items += scrape_mof_fx(now)
    return all_items


# ----- Writer
def write_to_news(sh, items: List[NewsItem]) -> int:
    ws, _ = ensure_worksheet(sh, "NEWS", NEWS_HEADERS)
    existing = _existing_hashes(ws, "I")
    rows = []
    added = 0
    for it in items:
        h = re.sub(r"\s+", " ", f"{it.source}|{it.url}".strip())[:180]
        if h in existing:
            continue
        rows.append(it.row())
        existing.add(h)
        added += 1
    if rows:
        ws.append_rows(rows, value_input_option="RAW")
    return added


# ----- Main loop
def run_once():
    if not SHEET_ID:
        raise RuntimeError("SHEET_ID env empty")
    sh, why = build_sheets_client(SHEET_ID)
    if not sh:
        raise RuntimeError(why)

    items = collect_candidates()
    log.info("augment collected: %d new candidates", len(items))
    added = write_to_news(sh, items)
    log.info("NEWS augment: +%d rows", added)


def main():
    if RUN_FOREVER:
        interval = max(1, AUGMENT_EVERY_MIN) * 60
        while True:
            try:
                run_once()
            except Exception:
                log.exception("news_augment: augment iteration failed")
            time.sleep(interval)
    else:
        run_once()


if __name__ == "__main__":
    main()
