# -*- coding: utf-8 -*-
"""
news_augment.py — сбор ссылок с регуляторов (Fed/ECB/BoJ/BoE/US Treasury) + JP MoF FX.
Логи в стиле:
    INFO news_augment: Starting Container
    INFO news_augment: ...
Вывод (если запускать как скрипт): TSV с колонками
ts_utc	source	title	url	countries	ccy	tags	importance_guess	hash
"""

from __future__ import annotations

import os
import re
import html
import time
import httpx
import logging
import urllib.parse
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
# Сколько страниц листать на разделах (BoE):
BOE_PAGES = int(os.getenv("NEWS_AUGMENT_BOE_PAGES", "3"))

# Для BoE иногда HTML отдаёт очень много якорей; отфильтруем по ключам:
BOE_KEEP_KEYWORDS = [
    r"\bmonetary\s+policy\s+summary\b",
    r"\bmpc\b",  # Monetary Policy Committee
    r"\binterest\s+rate\b",
    r"\bbank\s+rate\b",
    r"\bmonetary\s+policy\s+committee\b",
]
BOE_KEEP_RE = re.compile("|".join(BOE_KEEP_KEYWORDS), re.I)

# -----------------------------------------------------------------------------
# Модель данных
# -----------------------------------------------------------------------------
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
            return x.replace("\t", " ").replace("\n", " ").strip()
        ts = self.ts_utc.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        return "\t".join([
            ts,
            esc(self.source),
            esc(self.title),
            esc(self.url),
            esc(self.countries),
            esc(self.ccy),
            esc(self.tags),
            esc(self.importance_guess),
            esc(self.hash),
        ])


def _hash(source: str, url: str) -> str:
    return f"{source}|{url}"


# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------
def fetch_text(url: str, *, max_bytes: int = MAX_BODIES) -> Tuple[Optional[str], int, str]:
    """
    Возвращает (text, status_code, final_url). При 4xx/5xx пытается r.jina.ai/http/…
    """
    headers = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    try:
        with httpx.Client(follow_redirects=True, timeout=HTTP_TIMEOUT, headers=headers) as cli:
            r = cli.get(url)
            code = r.status_code
            final = str(r.url)
            txt = r.text[:max_bytes] if r.text else ""
            if code == 200 and txt:
                return txt, code, final
            # Фоллбэк на r.jina.ai
            if not final.startswith("https://r.jina.ai/http/"):
                alt = "https://r.jina.ai/http/" + final
                ra = cli.get(alt)
                return (ra.text[:max_bytes] if ra.text else ""), ra.status_code, str(ra.url)
            return txt, code, final
    except Exception:
        # Последний шанс — прямой r.jina.ai от исходного URL
        try:
            alt = "https://r.jina.ai/http/" + url
            with httpx.Client(follow_redirects=True, timeout=HTTP_TIMEOUT, headers=headers) as cli:
                ra = cli.get(alt)
                return (ra.text[:max_bytes] if ra.text else ""), ra.status_code, str(ra.url)
        except Exception:
            return None, 0, url


def _abs_url(base: str, href: str) -> str:
    return urllib.parse.urljoin(base, href)


A_TAG_RE = re.compile(
    r"<a\b([^>]+)>(.*?)</a>",
    re.I | re.DOTALL,
)

HREF_RE = re.compile(
    r'href\s*=\s*(?P<q>"|\')(?P<u>.*?)(?P=q)|href\s*=\s*(?P<u2>[^>\s]+)',
    re.I | re.DOTALL,
)


def _strip_tags(s: str) -> str:
    s = re.sub(r"<\s*br\s*/?\s*>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def _iter_links(html_src: str, base_url: str, domain_must: Optional[str] = None) -> Iterator[Tuple[str, str]]:
    """
    Итератор (url, text). Если задан domain_must, пропускаем чужие домены.
    """
    for m in A_TAG_RE.finditer(html_src):
        a_attrs = m.group(1) or ""
        a_inner = m.group(2) or ""
        href_match = HREF_RE.search(a_attrs)
        if not href_match:
            continue
        href = href_match.group("u") or href_match.group("u2") or ""
        if not href:
            continue
        url = _abs_url(base_url, href.strip())
        if domain_must:
            try:
                netloc = urllib.parse.urlparse(url).netloc
            except Exception:
                continue
            if domain_must not in netloc:
                continue
        text = _strip_tags(a_inner)
        yield (url, text)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _mk_item(source: str, title: str, url: str, countries: str, ccy: str,
             tags: str, importance: str) -> NewsItem:
    return NewsItem(
        ts_utc=_now_utc(),
        source=source,
        title=title,
        url=url,
        countries=countries,
        ccy=ccy,
        tags=tags,
        importance_guess=importance,
        hash=_hash(source, url),
    )


# -----------------------------------------------------------------------------
# Collectors
# -----------------------------------------------------------------------------
def collect_fed_fomc() -> List[NewsItem]:
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
    """
    Сканируем последнее из /news/press-releases (простой якорный парсер).
    """
    url = "https://home.treasury.gov/news/press-releases"
    html_src, code, final = fetch_text(url)
    items: List[NewsItem] = []
    if code != 200 or not html_src:
        return items

    links = list(_iter_links(html_src, final, domain_must="home.treasury.gov"))
    kept = []
    for u, t in links:
        if re.search(r"/news/press-releases/s[bp]\d{4,}", u):
            kept.append((u, t or "Treasury press release"))
    # ограничим кол-во, но оставим последние сверху:
    # сортировка по числу в sbNNNN, по убыванию
    def _score(u: str) -> int:
        m = re.search(r"/news/press-releases/s[bp](\d+)", u)
        return int(m.group(1)) if m else -1

    kept = sorted({u for u, _ in kept}, key=_score, reverse=True)[:40]
    for u in kept:
        # вытащим заголовок с самой страницы
        page, c2, f2 = fetch_text(u)
        ttl = None
        if c2 == 200 and page:
            # <title> или h1
            m = re.search(r"<title>(.*?)</title>", page, re.I | re.DOTALL)
            if m:
                ttl = _strip_tags(m.group(1))
            else:
                m = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.I | re.DOTALL)
                if m:
                    ttl = _strip_tags(m.group(1))
        items.append(_mk_item("US_TREASURY", ttl or "Treasury press release", f2, "united states", "USD", "treasury", "medium"))
    return items


def collect_ecb() -> List[NewsItem]:
    """
    Добавляем основные пресc-страницы ЕЦБ (стабильные рубрики).
    """
    pages = [
        ("Press releases", "https://www.ecb.europa.eu/press/pr/html/index.en.html"),
        ("Governing Council decisions", "https://www.ecb.europa.eu/press/govcdec/html/index.en.html"),
        ("Monetary policy decisions", "https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html"),
        ("Other decisions", "https://www.ecb.europa.eu/press/govcdec/otherdec/html/index.en.html"),
        ("Monetary policy press conference", "https://www.ecb.europa.eu/press/press_conference/html/index.en.html"),
        ("Monetary policy statements", "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/html/index.en.html"),
        ("Monetary policy statements at a glance", "https://www.ecb.europa.eu/press/press_conference/visual-mps/html/index.en.html"),
    ]
    items: List[NewsItem] = []
    for title, url in pages:
        _, code, final = fetch_text(url)
        if code == 200:
            imp = "high" if "monetary-policy" in final or "mopo" in final or "press_conference" in final else "medium"
            items.append(_mk_item("ECB_PR", title, final, "euro area", "EUR", "ecb", imp))
    return items


def collect_boj() -> List[NewsItem]:
    """
    Банк Японии: основные индекс-страницы + элементы (pdf/htm) с последних лет.
    """
    roots = [
        ("Monetary Policy Meetings", "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm", "high"),
        ("Summary of Opinions", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_2025/index.htm", "high"),
        ("Minutes", "https://www.boj.or.jp/en/mopo/mpmsche_minu/minu_2025/index.htm", "high"),
        ("Others", "https://www.boj.or.jp/en/mopo/mpmsche_minu/m_ref/index.htm", "high"),
        ("Monetary Policy Releases", "https://www.boj.or.jp/en/mopo/mpmdeci/index.htm", "high"),
        ("All Decisions (by year)", "https://www.boj.or.jp/en/mopo/mpmdeci/mpr_2025/index.htm", "high"),
        ("Statements on Monetary Policy", "https://www.boj.or.jp/en/mopo/mpmdeci/state_2025/index.htm", "high"),
        ("Other Statements", "https://www.boj.or.jp/en/mopo/mpmdeci/other/index.htm", "high"),
        ("Introduction or Modification of Schemes of Operations", "https://www.boj.or.jp/en/mopo/mpmdeci/ope_col_2025/index.htm", "high"),
        ("Transparency of Monetary Policy", "https://www.boj.or.jp/en/mopo/mpmdeci/transparency/index.htm", "high"),
        # агрегаторы:
        ("List by Year", "https://www.boj.or.jp/en/mopo/mpmdeci/mpr_all/index.htm", "high"),
        ("List by Year", "https://www.boj.or.jp/en/mopo/mpmdeci/state_all/index.htm", "high"),
        ("Summary of Opinions at the Monetary Policy Meetings List by Year", "https://www.boj.or.jp/en/mopo/mpmsche_minu/opinion_all/index.htm", "high"),
        ("Minutes of the Monetary Policy Meetings List by Year", "https://www.boj.or.jp/en/mopo/mpmsche_minu/minu_all/index.htm", "high"),
        ("Past Monetary Policy Meetings", "https://www.boj.or.jp/en/mopo/mpmsche_minu/past.htm", "high"),
    ]
    items: List[NewsItem] = []
    for title, url, imp in roots:
        _, code, final = fetch_text(url)
        if code == 200:
            items.append(_mk_item("BOJ_PR", title, final, "japan", "JPY", "boj mpm", imp))

    # Попробуем дополнительно собрать важные PDF/HTM за 2025 (видны на индексах)
    # (Это лёгкий best-effort; если сайт меняет пути — просто пропустим)
    extra_roots = [
        "https://www.boj.or.jp/en/mopo/mpmdeci/index.htm",
        "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm",
    ]
    seen = {it.url for it in items}
    added = 0
    for root in extra_roots:
        html_src, code, final = fetch_text(root)
        if code != 200 or not html_src:
            continue
        for u, t in _iter_links(html_src, final, domain_must="boj.or.jp"):
            if u in seen:
                continue
            if re.search(r"/mpr_2025/|/state_2025/|/minu_2025/|/opinion_2025/|/k25|/g25|/opi25|/mpr25", u):
                items.append(_mk_item("BOJ_PR", t or "BoJ document", u, "japan", "JPY", "boj mpm", "high"))
                seen.add(u)
                added += 1
                if added >= 40:
                    break
    LOG.info("news_augment: BoJ collected: %d", len(items))
    return items


def _boe_list_pages() -> List[str]:
    base = "https://www.bankofengland.co.uk/news"
    cats = ["news", "publications", "speeches", "statistics", "prudential-regulation", "upcoming"]
    urls = []
    for c in cats:
        for p in range(1, BOE_PAGES + 1):
            u = f"{base}/{c}" + ("" if p == 1 else f"?page={p}")
            urls.append(u)
    return urls


def _boe_site_search_queries() -> List[str]:
    return [
        "Monetary Policy Summary 2025",
        "Monetary Policy Committee 2025",
        "Bank Rate Monetary Policy 2025",
        "Monetary Policy Summary 2024",
        "interest rate decision MPC",
    ]


def collect_boe() -> List[NewsItem]:
    """
    BoE: два режима — 1) листаем разделы /news/*, 2) дёргаем встроенный поиск сайта.
    Фильтруем якоря по ключам.
    """
    items: List[NewsItem] = []
    kept_total = 0

    # 1) разделы
    list_urls = _boe_list_pages()
    for u in list_urls:
        html_src, code, final = fetch_text(u)
        if code != 200 or not html_src:
            continue
        anchors = list(_iter_links(html_src, final, domain_must="bankofengland.co.uk"))
        kept = []
        for link, text in anchors:
            s = f"{link} {text}"
            if BOE_KEEP_RE.search(s):
                kept.append((link, text))
        LOG.info("news_augment: BOE list %s: anchors=%d kept=%d", u, len(anchors), len(kept))
        kept_total += len(kept)
        for link, text in kept[:50]:
            items.append(_mk_item("BOE_PR", text or "BoE item", link, "united kingdom", "GBP", "boe", "medium"))
    LOG.info("news_augment: BOE HTML lists kept total: %d", kept_total)

    # 2) поиск сайта
    for q in _boe_site_search_queries():
        su = f"https://www.bankofengland.co.uk/search?query={urllib.parse.quote(q)}"
        html_src, code, final = fetch_text(su)
        if code != 200 or not html_src:
            continue
        anchors = list(_iter_links(html_src, final, domain_must="bankofengland.co.uk"))
        kept = []
        for link, text in anchors:
            s = f"{link} {text}"
            if BOE_KEEP_RE.search(s):
                kept.append((link, text))
        LOG.info("news_augment: BOE site search '%s': links=%d kept=%d", q, len(anchors), len(kept))
        for link, text in kept[:50]:
            items.append(_mk_item("BOE_PR", text or "BoE item", link, "united kingdom", "GBP", "boe", "medium"))

    return items


# ---------------------------- JP MoF FX (с фиксами) ---------------------------
def collect_mof_fx(now: datetime) -> List[NewsItem]:
    """
    JP Ministry of Finance FX interventions reference:
    - поддержка дефисов и подчёркиваний в пути (foreign-exchange-intervention / foreign_exchange_intervention)
    - пробуем базу без 'index.*' (заканчивается на '/')
    - сканируем родительские разделы (EN/english/JP) на наличие ссылок с нужными ключами
    """
    items: List[NewsItem] = []

    # Базовые директории (варианты en/english и -/_ в слаге)
    bases = [
        "https://www.mof.go.jp/en/policy/international_policy/reference/foreign-exchange-intervention/",
        "https://www.mof.go.jp/en/policy/international_policy/reference/foreign_exchange_intervention/",
        "https://www.mof.go.jp/english/policy/international_policy/reference/foreign-exchange-intervention/",
        "https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/",
        "https://www.mof.go.jp/policy/international_policy/reference/foreign-exchange-intervention/",
        "https://www.mof.go.jp/policy/international_policy/reference/foreign_exchange_intervention/",
    ]

    candidates: List[str] = []
    for b in bases:
        # без index.* (просто '/')
        candidates.append(b)
        # с index.* (обе версии)
        candidates.append(urllib.parse.urljoin(b, "index.htm"))
        candidates.append(urllib.parse.urljoin(b, "index.html"))

    # Сигнатуры в тексте/якорях (англ/яп), допускаем пробелы/дефисы/подчёркивания
    body_pat = re.compile(
        r"(foreign[\s_\-]*exchange[\s_\-]*intervention|"
        r"intervention[\s_\-]*in[\s_\-]*the[\s_\-]*foreign[\s_\-]*exchange|"
        r"為替介入|外国為替(?:平衡)?操作)",
        re.I,
    )

    found_url: Optional[str] = None

    # 1) прямые попадания
    for url in candidates:
        html_src, code, final = fetch_text(url)
        if code == 200 and html_src and body_pat.search(html_src):
            found_url = final
            LOG.info("news_augment: JP MoF FX direct hit: %s", found_url)
            break

    # 2) если прямого попадания нет — сканируем родителей
    if not found_url:
        parents = [
            "https://www.mof.go.jp/en/policy/international_policy/reference/",
            "https://www.mof.go.jp/english/policy/international_policy/reference/",
            "https://www.mof.go.jp/policy/international_policy/reference/",
            # иногда ссылки лежат уровнем выше:
            "https://www.mof.go.jp/en/policy/international_policy/",
            "https://www.mof.go.jp/english/policy/international_policy/",
            "https://www.mof.go.jp/policy/international_policy/",
        ]
        fx_link_pat = re.compile(
            r"(foreign[\s_\-]*exchange[\s_\-]*intervention|"
            r"為替介入|外国為替(?:平衡)?操作)",
            re.I,
        )

        for p in parents:
            html_src, code, final = fetch_text(p)
            if code != 200 or not html_src:
                continue
            links = list(_iter_links(html_src, final, domain_must="mof.go.jp"))
            matches = []
            for u, t in links:
                s = f"{u} {t}"
                if fx_link_pat.search(s):
                    matches.append((u, t))
            LOG.info("news_augment: JP MoF FX scan parent %s: links=%d matches=%d",
                     p, len(links), len(matches))
            if matches:
                # предпочтём наиболее «глубокую» ссылку (обычно это целевая)
                matches.sort(key=lambda x: len(x[0]), reverse=True)
                found_url = matches[0][0]
                break

    if found_url:
        items.append(NewsItem(
            ts_utc=now,
            source="JP_MOF_FX",
            title="FX intervention reference page",
            url=found_url,
            countries="japan",
            ccy="JPY",
            tags="mof",
            importance_guess="high",
            hash=_hash("JP_MOF_FX", found_url),
        ))
        LOG.info("news_augment: JP MoF FX collected: 1 (url=%s)", found_url)
    else:
        LOG.info("news_augment: JP MoF FX collected: 0")

    return items


# -----------------------------------------------------------------------------
# Объединение и дедуп
# -----------------------------------------------------------------------------
def merge_dedup(*groups: Iterable[NewsItem]) -> List[NewsItem]:
    out: List[NewsItem] = []
    seen: set[str] = set()
    for g in groups:
        for it in g:
            if it.hash in seen:
                continue
            seen.add(it.hash)
            out.append(it)
    return out


# -----------------------------------------------------------------------------
# Основной раннер
# -----------------------------------------------------------------------------
def run_augment() -> List[NewsItem]:
    LOG.info("Starting Container")
    t0 = time.time()

    fed = collect_fed_fomc()
    LOG.info("news_augment: FED collected: %d", len(fed))

    ecb = collect_ecb()
    LOG.info("news_augment: ECB collected: %d", len(ecb))

    boj = collect_boj()  # внутри пишет лог
    boe = collect_boe()
    LOG.info("news_augment: BOE collected total: %d", len(boe))

    # JP MoF FX (сейчас скан только референсной страницы)
    mof = collect_mof_fx(_now_utc())

    # US Treasury (press releases)
    ust = collect_us_treasury()
    LOG.info("news_augment: US Treasury collected: %d", len(ust))

    all_items = merge_dedup(fed, ecb, boj, boe, mof, ust)
    LOG.info("news_augment: NEWS augment: +%d rows", len(all_items))
    LOG.info("news_augment: took %.2fs", time.time() - t0)
    return all_items


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _print_tsv(items: List[NewsItem]) -> None:
    print("ts_utc\tsource\ttitle\turl\tcountries\tccy\ttags\timportance_guess\thash")
    for it in items:
        print(it.to_tsv_row())


if __name__ == "__main__":
    items = run_augment()
    _print_tsv(items)
