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
from typing import List, Tuple, Optional, Dict
from urllib.parse import urljoin, urlparse, urldefrag, urlsplit, urlunsplit
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

NOISE_TEXT = re.compile(r"^(skip to main content|back to main menu|日本語|english)$", re.I)
NEWS_HEADERS = ["ts_utc", "source", "title", "url", "countries", "ccy", "tags", "importance_guess", "hash"]

# --- RBA (AUD) ---
RBA_BASE = "https://www.rba.gov.au"
_CUR_Y = datetime.utcnow().year
RBA_YEAR_OK_RE = re.compile(rf"/({_CUR_Y}|{_CUR_Y-1})\b", re.I)
RBA_SEARCH_QUERIES = [
    "Monetary Policy Decision", "Cash rate decision", "Statement on Monetary Policy",
    "Minutes of the Monetary Policy Meeting", "RBA Board minutes", "SOMP",
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
def canon_url(u: str) -> str:
    try:
        u, _ = urldefrag(u)
        p = urlsplit(u)
        host = p.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        path = re.sub(r"/+$", "", p.path) or "/"
        return urlunsplit((p.scheme.lower(), host, path, p.query, ""))
    except Exception:
        return u


def is_noise_link(text: str, url: str) -> bool:
    if NOISE_TEXT.search((text or "").strip()):
        return True
    if "/search?query=" in url:
        return True
    if "#" in url:
        return True
    return False


def fetch_text(url: str, timeout=15.0) -> Tuple[str, int, str]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (LPBot/news_augment)"}
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as cli:
            r = cli.get(url)
            if r.status_code in (403, 404):
                pr = cli.get(JINA_PROXY + url)
                # не притворяемся 200 — используем фактический код
                return (pr.text or ""), pr.status_code, url
            return r.text, r.status_code, str(r.url)
    except Exception as e:
        log.warning("fetch failed %s: %s", url, e)
        return "", 0, url


def _iter_links(html: str, base_url: str, domain_must: Optional[str] = None) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for m in re.finditer(r'<a\b([^>]+)>(.*?)</a>', html, re.I | re.S):
        attrs, inner = m.group(1) or "", m.group(2) or ""
        m_href = re.search(r'href\s*=\s*(?P<q>[\'"])(?P<u>.*?)(?P=q)|href\s*=\s*(?P<u2>[^\s>]+)', attrs, re.I | re.S)
        if not m_href:
            continue
        href = (m_href.group('u') or m_href.group('u2') or '').strip()
        if not href:
            continue
        text = _html.unescape(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', inner))).strip()
        url = urljoin(base_url, href)
        if domain_must:
            try:
                host = urlparse(url).netloc.lower()
                if not (host == domain_must or host.endswith('.' + domain_must)):
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
        return f"{self.source}|{canon_url(self.url)}"

    def to_row(self) -> List[str]:
        clean_title = re.sub(r"<[^>]+>", "", self.title or "").strip()
        return [
            self.ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            self.source,
            clean_title,
            canon_url(self.url),
            self.countries,
            self.ccy,
            self.tags,
            self.importance_guess,
            self.key_hash(),
        ]


# --------- Collectors ----------
def _boe_is_relevant(title: str, url: str) -> bool:
    if "/search?query=" in url:
        return False
    try:
        path = urlsplit(url).path.lower()
    except Exception:
        path = ""

    # игнорируем чистые хабы без конкретики
    hub_only = path.rstrip("/") in ("/news/news", "/monetary-policy")
    if hub_only:
        return False

    if not any(p in path for p in (
        "/news/", "/monetary-policy/", "/about/people/monetary-policy-committee",
        "/markets/sonia-benchmark"
    )):
        return False

    strict = re.compile(
        r"(monetary policy summary|monetary policy committee|bank rate|mpc|minutes|"
        r"policy decision|interest rate|statement|press conference)", re.I
    )
    return bool(strict.search(f"{title} {path}"))


def collect_boe(now: datetime) -> List[NewsItem]:
    base = "https://www.bankofengland.co.uk"
    sections = ["/news/news", "/news/publications", "/news/speeches"]
    pages = ["", "?page=2"]
    kept: Dict[str, NewsItem] = {}

    for path in sections:
        for pg in pages:
            url = f"{base}{path}{pg}"
            html, code, final = fetch_text(url)
            if code != 200 or not html: continue
            for u, t in _iter_links(html, final, domain_must="bankofengland.co.uk"):
                if is_noise_link(t, u) or not _boe_is_relevant(t, u): continue
                key = f"BOE|{canon_url(u)}"
                if key not in kept:
                    kept[key] = NewsItem(now, "BOE_PR", t, u, "united kingdom", "GBP", "boe", "high")

    queries = ["Monetary Policy Summary", "Bank Rate Monetary Policy", "interest rate decision MPC"]
    for q in queries:
        url = f"{base}/search?query={q.replace(' ', '+')}"
        html, code, final = fetch_text(url)
        if code != 200 or not html: continue
        for u, t in _iter_links(html, final, domain_must="bankofengland.co.uk"):
            if is_noise_link(t, u) or not _boe_is_relevant(t, u): continue
            key = f"BOE|{canon_url(u)}"
            if key not in kept:
                kept[key] = NewsItem(now, "BOE_PR", t, u, "united kingdom", "GBP", "boe", "high")

    log.info("news_augment: BOE collected total: %d", len(kept))
    return list(kept.values())


def collect_boj(now: datetime) -> List[NewsItem]:
    items: List[NewsItem] = []
    urls_to_check = [
        "https://www.boj.or.jp/en/mopo/mpmdeci/index.htm",
        "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm"
    ]

    for url in urls_to_check:
        html, code, final = fetch_text(url)
        if code == 200 and html:
            for u, t in _iter_links(html, final, domain_must="boj.or.jp"):
                if is_noise_link(t, u): continue
                try:
                    path = urlsplit(u).path.lower()
                    if path.startswith(("/en/mopo/mpmdeci/", "/en/mopo/mpmsche_minu/", "/mopo/mpmdeci/", "/mopo/mpmsche_minu/")):
                        items.append(NewsItem(now, "BOJ_PR", t, u, "japan", "JPY", "boj mpm", "high"))
                except Exception:
                    continue

    uniq: Dict[str, NewsItem] = {canon_url(it.url): it for it in items}
    out = list(uniq.values())
    log.info("news_augment: BoJ collected: %d", len(out))
    return out


def collect_mof_fx(now: datetime) -> List[NewsItem]:
    candidates = [
        "https://www.mof.go.jp/en/policy/international_policy/reference/foreign_exchange_intervention/index.htm",
        "https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/index.htm",
        "https://www.mof.go.jp/policy/international_policy/reference/foreign_exchange_intervention/index.htm",
        "https://www.mof.go.jp/english/policy/international_policy/reference/feio/index.html",
        "https://www.mof.go.jp/en/policy/international_policy/reference/feio/index.html",
        "https://www.mof.go.jp/policy/international_policy/reference/feio/index.html",
    ]
    fx_re = re.compile(r"(foreign[\s_\-]*exchange[\s_\-]*intervention|feio|為替介入|外国為替(?:平衡)?操作)", re.I)
    items: List[NewsItem] = []
    found_url = None

    for url in candidates:
        html_src, code, final = fetch_text(url)
        if code == 200 and html_src and fx_re.search(html_src):
            found_url = final
            break

    if not found_url:
        parents = [
            "https://www.mof.go.jp/en/policy/international_policy/reference/",
            "https://www.mof.go.jp/english/policy/international_policy/reference/",
            "https://www.mof.go.jp/policy/international_policy/reference/",
        ]
        for p in parents:
            html_src, code, final = fetch_text(p)
            if code != 200 or not html_src: continue
            
            links = _iter_links(html_src, final, domain_must="mof.go.jp")
            fx_links = [u for (u, t) in links if fx_re.search(f"{u} {t}")]
            if fx_links:
                found_url = sorted(fx_links, key=lambda u: ((("index.htm" not in u) and ("index.html" not in u)), len(u)))[0]
                break

    if found_url:
        items.append(NewsItem(
            now, "JP_MOF_FX", "FX intervention reference page", found_url,
            "japan", "JPY", "mof", "high"
        ))
        log.info("news_augment: JP MoF FX collected: 1 (url=%s)", found_url)
    else:
        log.warning("news_augment: JP MoF FX page not found")
    return items


def _rba_importance_and_tags(title: str, url: str) -> Tuple[Optional[str], Optional[str]]:
    t, u = (title or "").lower(), (url or "").lower()
    if ("/media-releases/" in u) and any(k in t for k in ("decision", "cash rate", "monetary policy", "board")):
        return "high", "policy"
    if "/publications/smp/" in u or "statement on monetary policy" in t or "somp" in t:
        return "high", "policy somp"
    if "/monetary-policy/rba-board-minutes" in u or ("minutes" in t and "monetary policy" in t):
        return "medium", "minutes"
    if "/speeches/" in u and "monetary policy" in t and any(k in t for k in ("speech", "address", "governor")):
        return "medium", "speech"
    return None, None


def collect_rba(now: datetime) -> List[NewsItem]:
    items: List[NewsItem] = []
    hubs = [
        f"{RBA_BASE}/media-releases/", f"{RBA_BASE}/publications/smp/",
        f"{RBA_BASE}/monetary-policy/rba-board-minutes/", f"{RBA_BASE}/speeches/",
    ]

    for url in hubs:
        html_src, code, final_url = fetch_text(url)
        if code != 200 or not html_src: continue
        for link_url, text in _iter_links(html_src, final_url, domain_must="rba.gov.au"):
            if not RBA_YEAR_OK_RE.search(link_url): continue
            imp, tags = _rba_importance_and_tags(text.strip(), link_url)
            if not imp: continue
            items.append(NewsItem(now, "RBA_PR", text.strip(), link_url, "australia", "AUD", tags, imp))

    for q in RBA_SEARCH_QUERIES:
        search_url = f"{RBA_BASE}/search/?{os.environ.get('RBA_QUERY_PARAM','q')}={q.replace(' ', '+')}"
        html_src, code, final_url = fetch_text(search_url)
        if code != 200 or not html_src: continue
        for link_url, text in _iter_links(html_src, final_url, domain_must="rba.gov.au"):
            if not RBA_YEAR_OK_RE.search(link_url): continue
            imp, tags = _rba_importance_and_tags(text.strip(), link_url)
            if not imp: continue
            items.append(NewsItem(now, "RBA_PR", text.strip(), link_url, "australia", "AUD", tags, imp))

    if not items:
        items.append(NewsItem(
            now, "RBA_PR", "RBA hub (beacon)", f"{RBA_BASE}/media-releases/",
            "australia", "AUD", "rba hub", "medium"
        ))
        log.warning("news_augment: RBA fallback beacon inserted")

    uniq: Dict[str, NewsItem] = {canon_url(it.url): it for it in items}
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
    
    collectors = [collect_boe, collect_boj, collect_mof_fx, collect_rba]
    for collector_func in collectors:
        try:
            all_items.extend(collector_func(now))
        except Exception:
            log.exception("%s failed", collector_func.__name__)

    uniq: Dict[str, NewsItem] = {it.key_hash(): it for it in all_items}
    deduped = list(uniq.values())
    log.info("news_augment: augment collected: %d new candidates", len(deduped))

    write_news_rows(sh, deduped)


if __name__ == "__main__":
    main()
