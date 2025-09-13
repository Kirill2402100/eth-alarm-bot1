#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_augment.py — добавляет «недостающие» источники в лист NEWS (GBP/JPY/AUD),
не меняя существующий рабочий collector.

Источники:
- Bank of England (BoE): /news/* списки + site search (строгая фильтрация по ключевым словам)
- Bank of Japan (BoJ): Monetary Policy Releases + Schedules/Minutes/Summary of Opinions
- Japan MoF FX: reference page для интервенций (устойчивый перебор путей + fallback)
- Reserve Bank of Australia (RBA): media releases, publications, speeches + search
"""

import os, re, json, base64, logging, html as _html
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional, Dict
from urllib.parse import urljoin, urlparse
from datetime import datetime, timezone

import httpx

try:
    import gspread
    from google.oauth2 import service_account
    _GSHEETS_AVAILABLE = True
except Exception:
    gspread = None
    service_account = None
    _GSHEETS_AVAILABLE = False

# --------- ENV / CONST ----------
SHEET_ID = os.getenv("SHEET_ID", "").strip()
NEWS_WS  = os.getenv("NEWS_WS", "NEWS").strip() or "NEWS"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("news_augment")

JINA_PROXY = os.getenv("JINA_PROXY", "https://r.jina.ai/http/").rstrip("/") + "/"
STRICT_BOE = os.getenv("STRICT_BOE", "1").lower() in ("1", "true", "yes", "on")

# Те же ключевые слова, что у основного collector, но добавим пару боевских терминов.
KW_RE = re.compile(
    r"(rate decision|monetary policy|bank rate|policy decision|unscheduled|emergency|"
    r"press conference|policy statement|policy statements|minutes|mpc|fomc|"
    r"monetary policy summary|interest rate|statement|decision)",
    re.I
)

NEWS_HEADERS = ["ts_utc", "source", "title", "url", "countries", "ccy", "tags", "importance_guess", "hash"]

# --- RBA (AUD) ---
RBA_BASE = "https://www.rba.gov.au"
# берем свежие годы; при желании расширь до 2023
RBA_YEAR_OK_RE = re.compile(r"/20(24|25)\b", re.I)

# Поисковые доборы на случай, если что-то не попало из хабов
RBA_SEARCH_QUERIES = [
    "Monetary Policy Decision",
    "Cash rate decision",
    "Statement on Monetary Policy",
    "Minutes of the Monetary Policy Meeting",
    "RBA Board minutes",
    "SOMP",
]


# --------- Google Sheets helpers ----------
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
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
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


def _read_existing_hashes(ws, hash_col_letter: str) -> set:
    try:
        rng = f"{hash_col_letter}2:{hash_col_letter}100000"
        col = ws.get_values(rng)
        return {r[0] for r in col if r and r[0]}
    except Exception:
        return set()


# --------- HTTP / HTML helpers ----------
def fetch_text(url: str, timeout=15.0) -> Tuple[str, int, str]:
    """GET c fallback через Jina proxy на 403/404."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (LPBot/news_augment)"}
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as cli:
            r = cli.get(url)
            if r.status_code in (403, 404):
                pr = cli.get(JINA_PROXY + url)
                return pr.text, pr.status_code, str(pr.url)
            return r.text, r.status_code, str(r.url)
    except Exception as e:
        log.warning("fetch failed %s: %s", url, e)
        return "", 0, url


def _html_to_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = _html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _iter_links(html: str, base_url: str, domain_must: Optional[str] = None) -> List[Tuple[str, str]]:
    """Возвращает [(url, title_text)], с абсолютными URL. Простая regex-выборка."""
    out: List[Tuple[str, str]] = []
    for m in re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.I | re.S):
        href = m.group(1)
        text = re.sub(r"<[^>]+>", " ", m.group(2) or "")
        text = _html.unescape(re.sub(r"\s+", " ", text)).strip()
        url = urljoin(base_url, href)
        if domain_must:
            try:
                if urlparse(url).netloc and domain_must not in urlparse(url).netloc:
                    continue
            except Exception:
                continue
        out.append((url, text))
    return out


# --------- Models ----------
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
        base = f"{self.source}|{self.url}"
        return re.sub(r"\s+", " ", base.strip())[:180]

    def to_row(self) -> List[str]:
        clean_title = re.sub(r"<[^>]+>", "", self.title or "").strip()
        return [
            self.ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            self.source,
            clean_title,
            self.url,
            self.countries,
            self.ccy,
            self.tags,
            self.importance_guess,
            self.key_hash(),
        ]


# --------- Collectors ----------
def _boe_is_relevant(title: str, url: str) -> bool:
    hay = f"{title} {url}"
    if STRICT_BOE:
        # Строго: нужны явные намёки на MPC / Bank Rate / Monetary Policy / Minutes / Summary / Decision
        strict = re.compile(
            r"(monetary policy summary|monetary policy committee|bank rate|mpc|minutes|policy decision|rate decision)",
            re.I
        )
        return bool(strict.search(hay))
    else:
        return bool(KW_RE.search(hay))


def collect_boe(now: datetime) -> List[NewsItem]:
    base = "https://www.bankofengland.co.uk"
    sections = [
        "/news/news",
        "/news/publications",
        "/news/speeches",
        "/news/statistics",
        "/news/prudential-regulation",
        "/news/upcoming",
    ]
    pages = ["", "?page=2", "?page=3"]

    kept: Dict[str, NewsItem] = {}

    # 1) Списки разделов
    kept_total = 0
    for path in sections:
        for pg in pages:
            url = f"{base}{path}{pg}"
            html, code, final = fetch_text(url)
            if code != 200 or not html:
                continue
            links = _iter_links(html, final, domain_must="bankofengland.co.uk")
            anchors = len(links)
            added = 0
            for u, t in links:
                if not _boe_is_relevant(t, u):
                    continue
                key = f"BOE|{u}"
                if key in kept:
                    continue
                kept[key] = NewsItem(now, "BOE_PR", t, u, "united kingdom", "GBP", "boe", "high")
                added += 1
            log.info("news_augment: BOE list %s: anchors=%d kept=%d", final, anchors, added)
            kept_total += added

    if kept_total > 0:
        log.info("news_augment: BOE HTML lists kept total: %d", kept_total)

    # 2) Site search — добираем, если что-то пропустили
    queries = [
        "Monetary Policy Summary 2025",
        "Monetary Policy Committee 2025",
        "Bank Rate Monetary Policy 2025",
        "Monetary Policy Summary 2024",
        "interest rate decision MPC",
    ]
    for q in queries:
        url = f"{base}/search?query={q.replace(' ', '+')}"
        html, code, final = fetch_text(url)
        if code != 200 or not html:
            continue
        links = _iter_links(html, final, domain_must="bankofengland.co.uk")
        anchors = len(links)
        added = 0
        for u, t in links:
            if not _boe_is_relevant(t, u):
                continue
            key = f"BOE|{u}"
            if key in kept:
                continue
            kept[key] = NewsItem(now, "BOE_PR", t, u, "united kingdom", "GBP", "boe", "high")
            added += 1
        log.info("news_augment: BOE site search '%s': links=%d kept=%d", q, anchors, added)

    log.info("news_augment: BOE collected total: %d", len(kept))
    return list(kept.values())


def collect_boj(now: datetime) -> List[NewsItem]:
    items: List[NewsItem] = []

    # Monetary Policy Releases (решения, стейтменты и т.п.)
    u1 = "https://www.boj.or.jp/en/mopo/mpmdeci/index.htm"
    html, code, final = fetch_text(u1)
    if code == 200 and html:
        links = _iter_links(html, final, domain_must="boj.or.jp")
        cnt = 0
        for u, t in links:
            if not re.search(r"/mopo/(mpmdeci|mpmsche_minu)/", u, re.I) and not re.search(r"/mopo/mpmdeci/", u, re.I):
                continue
            # Явные подпути года/решений/стейтментов
            if any(x in u for x in (
                "/mpr_", "/state_", "/other", "/ope_col_", "/transparency", "/index.htm", "/index.html"
            )):
                items.append(NewsItem(now, "BOJ_PR", t, u, "japan", "JPY", "boj mpm", "high"))
                cnt += 1
        log.info("news_augment: BoJ mpmdeci: %d", cnt)

    # Schedules / Minutes / Summary of Opinions
    u2 = "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm"
    html, code, final = fetch_text(u2)
    if code == 200 and html:
        links = _iter_links(html, final, domain_must="boj.or.jp")
        cnt = 0
        for u, t in links:
            if re.search(r"/mpmsche_minu/", u, re.I) and any(s in u for s in (
                "/opinion_", "/minu_", "/past", "/opinion_all", "/minu_all", "/index.htm", "/index.html"
            )):
                items.append(NewsItem(now, "BOJ_PR", t, u, "japan", "JPY", "boj mpm", "high"))
                cnt += 1
        log.info("news_augment: BoJ mpmsche_minu: %d", cnt)

    # Убираем дубляжи по URL
    uniq: Dict[str, NewsItem] = {}
    for it in items:
        uniq[it.url] = it
    out = list(uniq.values())
    log.info("news_augment: BoJ collected: %d", len(out))
    return out


def collect_mof_fx(now: datetime) -> List[NewsItem]:
    """Устойчивый поиск reference-страницы по интервенциям MoF."""
    candidates = [
        # EN новые/старые пути с index.htm(l)
        "https://www.mof.go.jp/en/policy/international_policy/reference/foreign_exchange_intervention/index.htm",
        "https://www.mof.go.jp/en/policy/international_policy/reference/foreign_exchange_intervention/index.html",
        "https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/index.htm",
        "https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/index.html",
        # JP-ветка (иногда только она живая)
        "https://www.mof.go.jp/policy/international_policy/reference/foreign_exchange_intervention/index.htm",
        "https://www.mof.go.jp/policy/international_policy/reference/foreign_exchange_intervention/index.html",
    ]
    items: List[NewsItem] = []
    found_url = None

    # 1) прямые хиты
    for url in candidates:
        html_src, code, final = fetch_text(url)
        if code == 200 and html_src and re.search(r"(intervention|為替介入|foreign exchange)", html_src, re.I):
            found_url = final
            break

    # 2) fallback: на уровень выше и ищем ссылку с нужным хвостом
    if not found_url:
        parents = [
            "https://www.mof.go.jp/en/policy/international_policy/reference/",
            "https://www.mof.go.jp/english/policy/international_policy/reference/",
            "https://www.mof.go.jp/policy/international_policy/reference/",
        ]
        for p in parents:
            html_src, code, final = fetch_text(p)
            if code != 200 or not html_src:
                continue
            links = _iter_links(html_src, final, domain_must="mof.go.jp")
            fx_links = [u for (u, t) in links if "foreign_exchange_intervention" in u]
            # приоритет вариантам с index.htm(l)
            fx_links = sorted(
                fx_links,
                key=lambda u: (("index.htm" in u or "index.html" in u) == False, len(u))
            )
            if fx_links:
                found_url = fx_links[0]
                break

    if found_url:
        items.append(NewsItem(
            now, "JP_MOF_FX", "FX intervention reference page", found_url,
            "japan", "JPY", "mof", "high"
        ))
        log.info("news_augment: JP MoF FX collected: 1 (url=%s)", found_url)
    else:
        log.info("news_augment: JP MoF FX collected: 0")

    return items


def _rba_importance_and_tags(title: str, url: str) -> Tuple[Optional[str], Optional[str]]:
    """Возвращает (importance, tags) для RBA или (None, None) если нерелевантно."""
    t = (title or "").lower()
    u = (url or "").lower()

    # Решения/ставка/политика
    if ("/media-releases/" in u) and any(k in t for k in ("decision", "cash rate", "monetary policy", "board")):
        return "high", "policy"

    # Statement on Monetary Policy (SOMP)
    if "/publications/smp/" in u or "statement on monetary policy" in t or "somp" in t:
        return "high", "policy somp"

    # Протоколы заседаний
    if "/monetary-policy/rba-board-minutes" in u or ("minutes" in t and "monetary policy" in t):
        return "medium", "minutes"

    # Речи по монетарной политике
    if "/speeches/" in u and "monetary policy" in t and any(k in t for k in ("speech", "address", "governor")):
        return "medium", "speech"

    return None, None


def collect_rba(now: datetime) -> List[NewsItem]:
    """AUD: хабы + поисковые доборы. Строгая фильтрация + годовой фильтр."""
    items: List[NewsItem] = []

    hubs = [
        f"{RBA_BASE}/media-releases/",
        f"{RBA_BASE}/publications/smp/",
        f"{RBA_BASE}/monetary-policy/rba-board-minutes/",
        f"{RBA_BASE}/monetary-policy/",
        f"{RBA_BASE}/speeches/",
    ]

    total_kept = 0

    # 1) обходим хабы
    for url in hubs:
        html_src, code, final_url = fetch_text(url)
        if code != 200 or not html_src:
            log.info("news_augment: RBA hub %s: failed status %s", url, code)
            continue

        kept_here = 0
        for link_url, text in _iter_links(html_src, final_url, domain_must="rba.gov.au"):
            if not RBA_YEAR_OK_RE.search(link_url):
                continue
            imp, tags = _rba_importance_and_tags(text.strip(), link_url)
            if not imp:
                continue
            items.append(NewsItem(now, "RBA_PR", text.strip(), link_url, "australia", "AUD", tags, imp))
            kept_here += 1

        total_kept += kept_here
        log.info("news_augment: RBA hub kept from %s: %d", url, kept_here)

    # 2) поисковые доборы
    for q in RBA_SEARCH_QUERIES:
        search_url = f"{RBA_BASE}/search/?{os.environ.get('RBA_QUERY_PARAM','q')}={q.replace(' ', '+')}"
        # по умолчанию параметр 'q', но оставил хук через ENV на случай изменений
        html_src, code, final_url = fetch_text(search_url)
        if code != 200 or not html_src:
            log.info("news_augment: RBA search fail %s: status %s", q, code)
            continue

        kept_here = 0
        for link_url, text in _iter_links(html_src, RBA_BASE, domain_must="rba.gov.au"):
            if not RBA_YEAR_OK_RE.search(link_url):
                continue
            imp, tags = _rba_importance_and_tags(text.strip(), link_url)
            if not imp:
                continue
            items.append(NewsItem(now, "RBA_PR", text.strip(), link_url, "australia", "AUD", tags, imp))
            kept_here += 1

        total_kept += kept_here
        log.info("news_augment: RBA search '%s': kept=%d", q, kept_here)

    # 3) если ничего не собрали (403/блок со стороны сайта) — оставим “маяк”
    if not items:
        items.append(NewsItem(
            now, "RBA_PR", "RBA hub (beacon)", f"{RBA_BASE}/media-releases/",
            "australia", "AUD", "rba hub", "medium"
        ))
        log.warning("news_augment: RBA fallback beacon inserted")

    # дедуп по URL внутри батча
    uniq: Dict[str, NewsItem] = {}
    for it in items:
        uniq[it.url] = it
    out = list(uniq.values())

    log.info("news_augment: RBA collected: %d", len(out))
    return out


# --------- Write to NEWS ----------
def write_news_rows(sh, items: List[NewsItem]):
    ws, _ = ensure_worksheet(sh, NEWS_WS, NEWS_HEADERS)
    existing_hash = _read_existing_hashes(ws, "I")
    new_rows = [n.to_row() for n in items if n.key_hash() not in existing_hash]
    if new_rows:
        ws.append_rows(new_rows, value_input_option="RAW")
    log.info("news_augment: NEWS augment: +%d rows", len(new_rows))


# --------- Main ----------
def main():
    log.info("news_augment: Starting Container")
    if not SHEET_ID:
        raise RuntimeError("SHEET_ID env empty")
    sh, status = build_sheets_client(SHEET_ID)
    if not sh:
        raise RuntimeError(f"Sheets not available: {status}")

    now = datetime.now(timezone.utc)

    all_items: List[NewsItem] = []
    # GBP / BoE
    try:
        boe_items = collect_boe(now)
        all_items.extend(boe_items)
    except Exception:
        log.exception("collect_boe failed")

    # JPY / BoJ
    try:
        boj_items = collect_boj(now)
        all_items.extend(boj_items)
    except Exception:
        log.exception("collect_boj failed")

    # JPY / MoF FX
    try:
        mof_items = collect_mof_fx(now)
        all_items.extend(mof_items)
    except Exception:
        log.exception("collect_mof_fx failed")
        
    # AUD / RBA
    try:
        rba_items = collect_rba(now)
        all_items.extend(rba_items)
    except Exception:
        log.exception("collect_rba failed")

    # Дедуп по hash внутри батча
    uniq: Dict[str, NewsItem] = {}
    for it in all_items:
        uniq[it.key_hash()] = it
    deduped = list(uniq.values())
    log.info("news_augment: augment collected: %d new candidates", len(deduped))

    write_news_rows(sh, deduped)


if __name__ == "__main__":
    main()
