# -*- coding: utf-8 -*-
"""
news_augment.py — сбор ссылок с регуляторов (Fed/ECB/BoJ/BoE/RBA/US Treasury) + JP MoF FX.
"""

from __future__ import annotations

import os
import re
import html
import time
import httpx
import logging
import urllib.parse
from itertools import islice
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Iterator, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Логгер
# -----------------------------------------------------------------------------
LOG = logging.getLogger("news_augment")
if not LOG.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s INFO %(name)s: %(message)s")
    h.setFormatter(fmt)
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Конфиг
# -----------------------------------------------------------------------------
HTTP_TIMEOUT = float(os.getenv("NEWS_AUGMENT_TIMEOUT", "12"))
UA = os.getenv(
    "NEWS_AUGMENT_UA",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
)
MAX_BODIES = int(os.getenv("NEWS_AUGMENT_MAX_BODIES", "500000"))
BOE_PAGES = int(os.getenv("NEWS_AUGMENT_BOE_PAGES", "3"))
BOE_KEEP_KEYWORDS = [
    r"\bmonetary\s+policy\s+summary\b", r"\bmpc\b", r"\binterest\s+rate\b",
    r"\bbank\s+rate\b", r"\bmonetary\s+policy\s+committee\b",
]
BOE_KEEP_RE = re.compile("|".join(BOE_KEEP_KEYWORDS), re.I)
# Путь для вывода TSV, если запускать как скрипт
DEFAULT_TSV_PATH = os.getenv("NEWS_TSV_PATH", "./news_augment_output.tsv")
HDR = "ts_utc\tsource\ttitle\turl\tcountries\tccy\ttags\timportance_guess\thash\n"

# -----------------------------------------------------------------------------
# Модель данных и утилиты нормализации
# -----------------------------------------------------------------------------

def _canon_url(u: str) -> str:
    try:
        u, _frag = urllib.parse.urldefrag(u)
        p = urllib.parse.urlsplit(u)
        netloc = p.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = re.sub(r"/+$", "", p.path) or "/"
        return urllib.parse.urlunsplit((p.scheme.lower(), netloc, path, p.query, ""))
    except Exception:
        return u

def _hash(source: str, url: str) -> str:
    return f"{source}|{url}" # url уже должен быть каноническим

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
    hash: str

    def to_tsv_row(self) -> str:
        def esc(x: str) -> str:
            return str(x).replace("\t", " ").replace("\n", " ").strip()
        ts_str = self.ts_utc.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        return "\t".join([
            ts_str, esc(self.source), esc(self.title), esc(self.url), esc(self.countries),
            esc(self.ccy), esc(self.tags), esc(self.importance_guess), esc(self.hash),
        ])

# ... (Остальные хелперы: fetch_text, _abs_url, _unwrap_mirror_base, и т.д. остаются без изменений)
def fetch_text(url: str, *, max_bytes: int = MAX_BODIES) -> Tuple[Optional[str], int, str]:
    headers = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    try:
        with httpx.Client(follow_redirects=True, timeout=HTTP_TIMEOUT, headers=headers) as cli:
            r = cli.get(url)
            code = r.status_code
            final = str(r.url)
            txt = r.text[:max_bytes] if r.text else ""
            if code == 200 and txt: return txt, code, final
            if not final.startswith("https://r.jina.ai/http/"):
                alt = "https://r.jina.ai/http/" + final
                ra = cli.get(alt)
                return (ra.text[:max_bytes] if ra.text else ""), ra.status_code, str(ra.url)
            return txt, code, final
    except Exception:
        try:
            alt = "https://r.jina.ai/http/" + url
            with httpx.Client(follow_redirects=True, timeout=HTTP_TIMEOUT, headers=headers) as cli:
                ra = cli.get(alt)
                return (ra.text[:max_bytes] if ra.text else ""), ra.status_code, str(ra.url)
        except Exception:
            return None, 0, url

def _abs_url(base: str, href: str) -> str: return urllib.parse.urljoin(base, href)
def _unwrap_mirror_base(base_url: str) -> str:
    pref = "https://r.jina.ai/http/"
    if base_url.startswith(pref):
        raw = base_url[len(pref):]
        if raw.startswith(("http://", "https://")): return raw
    return base_url

A_TAG_RE = re.compile(r"<a\b([^>]+)>(.*?)</a>", re.I | re.DOTALL)
HREF_RE = re.compile(r'href\s*=\s*(?P<q>"|\')(?P<u>.*?)(?P=q)|href\s*=\s*(?P<u2>[^>\s]+)', re.I | re.DOTALL)
def _strip_tags(s: str) -> str:
    s = re.sub(r"<\s*br\s*/?\s*>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()

def _is_noise_anchor(link: str, text: str) -> bool:
    t = (text or "").strip().lower()
    return (not t) or t.startswith("skip to main")
def _host_matches(host: str, must: str) -> bool:
    host = (host or "").lower()
    must = (must or "").lower()
    return host == must or host.endswith("." + must)

def _iter_links(html_src: str, base_url: str, domain_must: Optional[str] = None) -> Iterator[Tuple[str, str]]:
    join_base = _unwrap_mirror_base(base_url)
    for m in A_TAG_RE.finditer(html_src):
        a_attrs, a_inner = m.group(1) or "", m.group(2) or ""
        href_match = HREF_RE.search(a_attrs)
        if not href_match: continue
        href = (href_match.group("u") or href_match.group("u2") or "").strip()
        if not href: continue
        url = _canon_url(_abs_url(join_base, href))
        if domain_must:
            try: netloc = urllib.parse.urlparse(url).netloc
            except Exception: continue
            if not _host_matches(netloc, domain_must): continue
        text = _strip_tags(a_inner)
        if _is_noise_anchor(url, text): continue
        yield (url, text)

def _now_utc() -> datetime: return datetime.now(timezone.utc)

def _mk_item(source: str, title: str, url: str, countries: str, ccy: str,
             tags: str, importance: str) -> NewsItem:
    cu = _canon_url(url)
    return NewsItem(
        ts_utc=_now_utc(), source=source, title=title, url=cu,
        countries=countries, ccy=ccy, tags=tags,
        importance_guess=importance, hash=_hash(source, cu),
    )

# -----------------------------------------------------------------------------
# Collectors (остаются без изменений, т.к. улучшения централизованы)
# -----------------------------------------------------------------------------
def collect_fed_fomc() -> List[NewsItem]:
    # ... код ...
    years = ["2025", "2024", "2023", "2022", "2021", "2020"]
    base = "https://www.federalreserve.gov/newsevents/pressreleases/"
    items: List[NewsItem] = []
    for y in years:
        url = f"{base}{y}-press-fomc.htm"
        _, code, final = fetch_text(url)
        if code == 200:
            items.append(_mk_item("US_FED_PR", f"{y} FOMC", final, "united states", "USD", "policy", "high"))
    return items

def collect_us_treasury() -> List[NewsItem]:
    # ... код ...
    url = "https://home.treasury.gov/news/press-releases"
    html_src, code, final = fetch_text(url)
    items: List[NewsItem] = []
    if code != 200 or not html_src: return items
    links = list(_iter_links(html_src, final, domain_must="home.treasury.gov"))
    kept = []
    for u, t in links:
        if re.search(r"/news/press-releases/s[bp]\d{4,}", u):
            kept.append((u, t or "Treasury press release"))
    def _score(u: str) -> int:
        m = re.search(r"/news/press-releases/s[bp](\d+)", u)
        return int(m.group(1)) if m else -1
    kept = sorted({_canon_url(u) for u, _ in kept}, key=_score, reverse=True)[:40]
    for u in kept:
        page, c2, f2 = fetch_text(u)
        ttl = None
        if c2 == 200 and page:
            m = re.search(r"<title>(.*?)</title>", page, re.I | re.DOTALL)
            if m: ttl = _strip_tags(m.group(1))
            else:
                m = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.I | re.DOTALL)
                if m: ttl = _strip_tags(m.group(1))
        items.append(_mk_item("US_TREASURY", ttl or "Treasury press release", f2, "united states", "USD", "treasury", "medium"))
    return items

# ... Аналогично для collect_ecb, collect_boj, collect_boe ...
def collect_ecb() -> List[NewsItem]:
    pages = [("Press releases", "https://www.ecb.europa.eu/press/pr/html/index.en.html"), ("Governing Council decisions", "https://www.ecb.europa.eu/press/govcdec/html/index.en.html"), ("Monetary policy decisions", "https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html"), ("Other decisions", "https://www.ecb.europa.eu/press/govcdec/otherdec/html/index.en.html"), ("Monetary policy press conference", "https://www.ecb.europa.eu/press/press_conference/html/index.en.html"), ("Monetary policy statements", "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/html/index.en.html"), ("Monetary policy statements at a glance", "https://www.ecb.europa.eu/press/press_conference/visual-mps/html/index.en.html")]
    items: List[NewsItem] = []
    for title, url in pages:
        _, code, final = fetch_text(url)
        if code == 200:
            imp = "high" if "monetary-policy" in final or "mopo" in final or "press_conference" in final else "medium"
            items.append(_mk_item("ECB_PR", title, final, "euro area", "EUR", "ecb", imp))
    return items

def collect_boj() -> List[NewsItem]:
    roots = [("Monetary Policy Meetings", "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm", "high"), ("Summary of Opinions", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2025/index.htm", "high"), ("Minutes", "https://www.boj.or.jp/en/mopo/mpmsche_minu/minu_2025/index.htm", "high"), ("Others", "https://www.boj.or.jp/en/mopo/mpmsche_minu/m_ref/index.htm", "high"), ("Monetary Policy Releases", "https://www.boj.or.jp/en/mopo/mpmdeci/index.htm", "high"), ("All Decisions (by year)", "https://www.boj.or.jp/en/mopo/mpmdeci/mpr_2025/index.htm", "high"), ("Statements on Monetary Policy", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2025/index.htm", "high"), ("Other Statements", "https://www.boj.or.jp/en/mopo/mpmdeci/other/index.htm", "high"), ("Introduction or Modification of Schemes of Operations", "https://www.boj.or.jp/en/mopo/mpmdeci/ope_col_2025/index.htm", "high"), ("Transparency of Monetary Policy", "https://www.boj.or.jp/en/mopo/mpmdeci/transparency/index.htm", "high"), ("List by Year", "https://www.boj.or.jp/en/mopo/mpmdeci/mpr_all/index.htm", "high"), ("List by Year", "https://www.boj.or.jp/en/mopo/mpmdeci/state_all/index.htm", "high"), ("Summary of Opinions at the Monetary Policy Meetings List by Year", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_all/index.htm", "high"), ("Minutes of the Monetary Policy Meetings List by Year", "https://www.boj.or.jp/en/mopo/mpmsche_minu/minu_all/index.htm", "high"), ("Past Monetary Policy Meetings", "https://www.boj.or.jp/en/mopo/mpmsche_minu/past.htm", "high")]
    items: List[NewsItem] = []
    for title, url, imp in roots:
        _, code, final = fetch_text(url)
        if code == 200: items.append(_mk_item("BOJ_PR", title, final, "japan", "JPY", "boj mpm", imp))
    extra_roots = ["https://www.boj.or.jp/en/mopo/mpmdeci/index.htm", "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm"]
    seen = {it.url for it in items}
    added = 0
    for root in extra_roots:
        html_src, code, final = fetch_text(root)
        if code != 200 or not html_src: continue
        for u, t in _iter_links(html_src, final, domain_must="boj.or.jp"):
            if u in seen: continue
            if re.search(r"/mpr_2025/|/state_2025/|/minu_2025/|/opinion_2025/|/k25|/g25|/opi25|/mpr25", u):
                items.append(_mk_item("BOJ_PR", t or "BoJ document", u, "japan", "JPY", "boj mpm", "high"))
                seen.add(u)
                added += 1
                if added >= 40: break
    LOG.info("news_augment: BoJ collected: %d", len(items))
    return items

def _boe_list_pages() -> List[str]:
    base = "https://www.bankofengland.co.uk/news"
    cats = ["news", "publications", "speeches", "statistics", "prudential-regulation", "upcoming"]
    urls = []
    for c in cats:
        for p in range(1, BOE_PAGES + 1): urls.append(f"{base}/{c}" + ("" if p == 1 else f"?page={p}"))
    return urls
def _boe_site_search_queries() -> List[str]: return ["Monetary Policy Summary 2025", "Monetary Policy Committee 2025", "Bank Rate Monetary Policy 2025", "Monetary Policy Summary 2024", "interest rate decision MPC"]
def collect_boe() -> List[NewsItem]:
    items: List[NewsItem] = []
    kept_total = 0
    for u in _boe_list_pages():
        html_src, code, final = fetch_text(u)
        if code != 200 or not html_src: continue
        anchors = list(_iter_links(html_src, final, domain_must="bankofengland.co.uk"))
        kept = []
        for link, text in anchors:
            if BOE_KEEP_RE.search(f"{link} {text}"): kept.append((link, text))
        LOG.info(f"news_augment: BOE list {u}: anchors={len(anchors)} kept={len(kept)}")
        kept_total += len(kept)
        for link, text in kept[:50]: items.append(_mk_item("BOE_PR", text or "BoE item", link, "united kingdom", "GBP", "boe", "medium"))
    LOG.info(f"news_augment: BOE HTML lists kept total: {kept_total}")
    for q in _boe_site_search_queries():
        su = f"https://www.bankofengland.co.uk/search?query={urllib.parse.quote(q)}"
        html_src, code, final = fetch_text(su)
        if code != 200 or not html_src: continue
        anchors = list(_iter_links(html_src, final, domain_must="bankofengland.co.uk"))
        kept = []
        for link, text in anchors:
            if BOE_KEEP_RE.search(f"{link} {text}"): kept.append((link, text))
        LOG.info(f"news_augment: BOE site search '{q}': links={len(anchors)} kept={len(kept)}")
        for link, text in kept[:50]: items.append(_mk_item("BOE_PR", text or "BoE item", link, "united kingdom", "GBP", "boe", "medium"))
    return items

RBA_BASE = "https://www.rba.gov.au"
RBA_YEAR_OK_RE = re.compile(r"/20(24|25)\b")
RBA_SEARCH_QUERIES = ["Monetary Policy Decision", "Cash rate decision", "Statement on Monetary Policy", "Minutes of the Monetary Policy Meeting", "RBA Board minutes", "SOMP"]
def _rba_importance_and_tags(title: str, url: str) -> Tuple[Optional[str], Optional[str]]:
    t, u = (title or "").lower(), (url or "").lower()
    if ("/media-releases/" in u and ("decision" in t or "cash rate" in t or "monetary policy" in t or "board" in t)): return "high", "policy"
    if "/publications/smp/" in u or "statement on monetary policy" in t or "somp" in t: return "high", "policy somp"
    if "/monetary-policy/rba-board-minutes" in u or ("minutes" in t and "monetary policy" in t): return "medium", "minutes"
    if "/speeches/" in u and "monetary policy" in t and ("speech" in t or "address" in t or "governor" in t): return "medium", "speech"
    return None, None
def collect_rba() -> List[NewsItem]:
    items: List[NewsItem] = []
    hubs = [f"{RBA_BASE}/media-releases/", f"{RBA_BASE}/publications/smp/", f"{RBA_BASE}/monetary-policy/rba-board-minutes/", f"{RBA_BASE}/monetary-policy/", f"{RBA_BASE}/speeches/"]
    total_kept = 0
    for url in hubs:
        html_src, code, final_url = fetch_text(url)
        if code != 200 or not html_src: LOG.info(f"news_augment: RBA hub {url}: failed status {code}"); continue
        kept = 0
        for link_url, text in _iter_links(html_src, final_url, domain_must="rba.gov.au"):
            if not RBA_YEAR_OK_RE.search(link_url): continue
            imp, tags = _rba_importance_and_tags(text.strip(), link_url)
            if not imp: continue
            items.append(_mk_item("RBA_PR", text.strip(), link_url, "australia", "AUD", tags, imp)); kept += 1
        total_kept += kept
        LOG.info(f"news_augment: RBA hub kept from {url}: {kept}")
    for q in RBA_SEARCH_QUERIES:
        search_url = f"{RBA_BASE}/search/?{urllib.parse.urlencode({'q': q})}"
        html_src, code, final_url = fetch_text(search_url)
        if code != 200 or not html_src: LOG.info(f"news_augment: RBA search fail {q}: status {code}"); continue
        kept = 0
        for link_url, text in _iter_links(html_src, RBA_BASE, domain_must="rba.gov.au"):
            if not RBA_YEAR_OK_RE.search(link_url): continue
            imp, tags = _rba_importance_and_tags(text.strip(), link_url)
            if not imp: continue
            items.append(_mk_item("RBA_PR", text.strip(), link_url, "australia", "AUD", tags, imp)); kept += 1
        total_kept += kept
        LOG.info(f"news_augment: RBA search '{q}': kept={kept}")
    if not items:
        for url in hubs: items.append(_mk_item("RBA_PR", "RBA hub", url, "australia", "AUD", "rba hub", "medium"))
        LOG.info(f"news_augment: RBA fallback hubs added: {len(hubs)}")
    return items

def collect_mof_fx(now: datetime) -> List[NewsItem]:
    # ... код ...
    items: List[NewsItem] = []
    bases = ["https://www.mof.go.jp/en/policy/international_policy/reference/foreign-exchange-intervention/", "https://www.mof.go.jp/en/policy/international_policy/reference/foreign_exchange_intervention/", "https://www.mof.go.jp/english/policy/international_policy/reference/foreign-exchange-intervention/", "https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/", "https://www.mof.go.jp/policy/international_policy/reference/foreign-exchange-intervention/", "https://www.mof.go.jp/policy/international_policy/reference/foreign_exchange_intervention/"]
    candidates: List[str] = []
    for b in bases:
        candidates.append(b)
        candidates.append(urllib.parse.urljoin(b, "index.htm"))
        candidates.append(urllib.parse.urljoin(b, "index.html"))
    body_pat = re.compile(r"(foreign[\s_\-]*exchange[\s_\-]*intervention|intervention[\s_\-]*in[\s_\-]*the[\s_\-]*foreign[\s_\-]*exchange|為替介入|外国為替(?:平衡)?操作)", re.I)
    found_url: Optional[str] = None
    for url in candidates:
        html_src, code, final = fetch_text(url)
        if code == 200 and html_src and body_pat.search(html_src):
            found_url = final
            LOG.info(f"news_augment: JP MoF FX direct hit: {found_url}"); break
    if not found_url:
        parents = ["https://www.mof.go.jp/en/policy/international_policy/reference/", "https://www.mof.go.jp/english/policy/international_policy/reference/", "https://www.mof.go.jp/policy/international_policy/reference/", "https://www.mof.go.jp/en/policy/international_policy/", "https://www.mof.go.jp/english/policy/international_policy/", "https://www.mof.go.jp/policy/international_policy/"]
        fx_link_pat = re.compile(r"(foreign[\s_\-]*exchange[\s_\-]*intervention|為替介入|外国為替(?:平衡)?操作)", re.I)
        for p in parents:
            html_src, code, final = fetch_text(p)
            if code != 200 or not html_src: continue
            links = list(_iter_links(html_src, final, domain_must="mof.go.jp"))
            matches = []
            for u, t in links:
                if fx_link_pat.search(f"{u} {t}"): matches.append((u, t))
            LOG.info(f"news_augment: JP MoF FX scan parent {p}: links={len(links)} matches={len(matches)}")
            if matches:
                matches.sort(key=lambda x: len(x[0]), reverse=True)
                found_url = matches[0][0]; break
    if found_url:
        items.append(NewsItem(ts_utc=now, source="JP_MOF_FX", title="FX intervention reference page", url=found_url, countries="japan", ccy="JPY", tags="mof fx", importance_guess="high", hash=_hash("JP_MOF_FX", found_url)))
        LOG.info(f"news_augment: JP MoF FX collected: 1 (url={found_url})")
    else: LOG.info("news_augment: JP MoF FX collected: 0")
    return items

def merge_dedup(*groups: Iterable[NewsItem]) -> List[NewsItem]:
    out: List[NewsItem] = []
    seen: set[str] = set()
    for g in groups:
        for it in g:
            if it.hash in seen: continue
            seen.add(it.hash)
            out.append(it)
    return out

# -----------------------------------------------------------------------------
# Основной раннер, sanity checks и экспорт
# -----------------------------------------------------------------------------

def _post_sanity(items: List[NewsItem]):
    """Проверяет инварианты перед записью, чтобы предотвратить запись 'грязных' данных."""
    assert all("://www." not in it.url for it in items if any(h in it.url for h in (
        "federalreserve.gov", "bankofengland.co.uk", "boj.or.jp", "ecb.europa.eu"
    ))), "www.* в url после канонизации"
    assert all("|" in it.hash and it.hash.endswith(it.url) for it in items), "hash != source|url"
    assert all("#" not in it.url for it in items), "anchor в url"
    LOG.info("Post-sanity checks passed for %d items", len(items))

def write_and_verify(out_path: str, items: List[NewsItem]):
    """Записывает TSV и сразу же верифицирует запись для диагностики."""
    # Убедимся, что директория существует
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Запись
    sorted_items = sorted(items, key=lambda x: x.ts_utc, reverse=True)
    with open(out_path, "w", encoding="utf-8") as w:
        w.write(HDR)
        for it in sorted_items:
            w.write(it.to_tsv_row() + "\n")

    size = os.path.getsize(out_path)
    LOG.info("[export] Wrote %d rows, %d bytes -> %s", len(items), size, os.path.abspath(out_path))

    # Контрольное чтение того же файла
    with open(out_path, "r", encoding="utf-8") as r:
        head = "".join(islice(r, 8))
    LOG.info("[export] TSV head:\n---\n%s---", head.strip())

def run_augment() -> List[NewsItem]:
    LOG.info("Starting Container")
    t0 = time.time()

    fed = collect_fed_fomc(); LOG.info("FED collected: %d", len(fed))
    ecb = collect_ecb(); LOG.info("ECB collected: %d", len(ecb))
    boj = collect_boj()
    boe = collect_boe(); LOG.info("BOE collected total: %d", len(boe))
    rba = collect_rba(); LOG.info("RBA collected: %d", len(rba))
    if not any(it.ccy == "AUD" for it in rba):
        rba.append(_mk_item("RBA_PR", "RBA hub (beacon)", f"{RBA_BASE}/media-releases/", "australia", "AUD", "rba hub", "medium"))
        LOG.warning("Injected RBA beacon because no AUD items were found naturally.")
    mof = collect_mof_fx(_now_utc())
    ust = collect_us_treasury(); LOG.info("US Treasury collected: %d", len(ust))

    all_items = merge_dedup(fed, ecb, boj, boe, rba, mof, ust)
    _post_sanity(all_items) # Проверка перед возвратом/записью
    LOG.info("NEWS augment: +%d rows", len(all_items))
    LOG.info("Took %.2fs", time.time() - t0)
    return all_items

if __name__ == "__main__":
    collected_items = run_augment()
    write_and_verify(DEFAULT_TSV_PATH, collected_items)
