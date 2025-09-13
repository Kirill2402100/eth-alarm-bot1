# -*- coding: utf-8 -*-
"""
NEWS augment collector: FED, US Treasury, ECB, BoJ, BoE, JP MoF FX, RBA
Выходные поля каждой строки:
ts_utc, source, title, url, countries, ccy, tags, importance_guess, hash
"""

import re
import datetime as dt
from urllib.parse import urljoin, urlencode

# ---------- общие утилиты ----------

def _now_utc_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _clean_html(txt: str) -> str:
    # убрать теги и лишние пробелы
    t = re.sub(r"<\s*br\s*/?>", " ", txt, flags=re.I)
    t = re.sub(r"<.*?>", "", t, flags=re.S)
    return re.sub(r"\s+", " ", t).strip()

def _extract_anchors(html: str):
    # (href, inner_html)
    return re.findall(r'<a\s[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.I | re.S)

def _mk_row(source, title, url, countries, ccy, tags, importance):
    row = {
        "ts_utc": _now_utc_iso(),
        "source": source,
        "title": title,
        "url": url,
        "countries": countries,
        "ccy": ccy,
        "tags": tags,
        "importance_guess": importance,
    }
    row["hash"] = f'{row["source"]}|{row["url"]}'
    return row

def _dedup_rows(rows):
    by_url = {}
    for r in rows:
        by_url[r["url"]] = r
    return list(by_url.values())

def _try_get(client, url, logger=None, timeout=20.0):
    """Стандартный GET + опциональный fallback через r.jina.ai при 4xx/5xx."""
    try:
        r = client.get(url, timeout=timeout)
    except Exception as e:
        if logger:
            logger.info(f"news_augment: GET error {url}: {e}")
        r = None

    if r is not None and r.status_code == 200:
        return r

    # fallback для сложных сайтов, если честно  не всегда поможет — но попробуем
    if r is None or r.status_code >= 400:
        mirror = f"https://r.jina.ai/http/{url}"
        try:
            rm = client.get(mirror, timeout=timeout)
            if rm.status_code == 200 and rm.text:
                if logger:
                    logger.info(f"news_augment: fallback via r.jina.ai OK for {url}")
                # возвращаем "как бы r", но с текстом от зеркала
                class DummyResp:
                    def __init__(self, text):
                        self.status_code = 200
                        self.text = text
                return DummyResp(rm.text)
            else:
                if logger:
                    logger.info(f"news_augment: fallback fail {url}: {rm.status_code}")
        except Exception as e:
            if logger:
                logger.info(f"news_augment: fallback error {url}: {e}")

    return r  # пусть вызывающий сам проверит статус

def _log(logger, msg):
    if logger:
        logger.info(f"news_augment: {msg}")

# ---------- FED (FOMC press releases by year, как в логах) ----------

def collect_fed(client, logger=None):
    rows = []
    base = "https://www.federalreserve.gov/newsevents/pressreleases"
    years = [2025, 2024, 2023, 2022, 2021, 2020]
    for y in years:
        url = f"{base}/{y}-press-fomc.htm"
        rows.append(_mk_row(
            "US_FED_PR", f"{y} FOMC", url,
            "united states", "USD", "policy", "high"
        ))
    _log(logger, f"FED collected: {len(rows)}")
    return rows

# ---------- US Treasury (press releases list) ----------

def collect_us_treasury(client, logger=None, max_pages=1):
    rows = []
    # главная лента PR
    base = "https://home.treasury.gov/news/press-releases"
    for page in range(1, max_pages + 1):
        url = base if page == 1 else f"{base}?page={page-1}"
        r = _try_get(client, url, logger)
        if not r or r.status_code != 200:
            _log(logger, f"Treasury PR page fail {url}: {getattr(r,'status_code',None)}")
            continue
        kept = 0
        for href, text in _extract_anchors(r.text):
            if "/news/press-releases/" not in href:
                continue
            title = _clean_html(text)
            if not title:
                continue
            full = href if href.startswith("http") else urljoin(base, href)
            rows.append(_mk_row("US_TREASURY", title, full, "united states", "USD", "treasury", "medium"))
            kept += 1
        _log(logger, f"US Treasury page kept: {kept}")
    return _dedup_rows(rows)

# ---------- ECB (стабильные узлы) ----------

def collect_ecb(client, logger=None):
    base = "https://www.ecb.europa.eu/press"
    items = [
        ("ECB_PR", "Press releases", f"{base}/pr/html/index.en.html", "medium"),
        ("ECB_PR", "Governing Council decisions", f"{base}/govcdec/html/index.en.html", "medium"),
        ("ECB_PR", "Monetary policy decisions", f"{base}/govcdec/mopo/html/index.en.html", "high"),
        ("ECB_PR", "Other decisions", f"{base}/govcdec/otherdec/html/index.en.html", "medium"),
        ("ECB_PR", "Monetary policy press conference", f"{base}/press_conference/html/index.en.html", "high"),
        ("ECB_PR", "Monetary policy statements", f"{base}/press_conference/monetary-policy-statement/html/index.en.html", "high"),
        ("ECB_PR", "Monetary policy statements at a glance", f"{base}/press_conference/visual-mps/html/index.en.html", "high"),
    ]
    rows = [
        _mk_row(src, title, url, "euro area", "EUR", "ecb", imp)
        for (src, title, url, imp) in items
    ]
    _log(logger, f"ECB collected: {len(rows)}")
    return rows

# ---------- BoJ (решения/statement/минуты/summary-of-opinions) ----------

_BOJ_KEEP_PATTERNS = re.compile(
    r"(Statement on Monetary Policy|Monetary Policy Releases|All Decisions|"
    r"Summary of Opinions|Minutes|Transparency of Monetary Policy|"
    r"Introduction or Modification of Schemes|Quarterly Schedule|"
    r"Loan Disbursement|Change in the Guideline|Plan for the Reduction|Timetable and Schedule)",
    re.I
)

def collect_boj(client, logger=None):
    rows = []

    pages = [
        "https://www.boj.or.jp/en/mopo/mpmdeci/index.htm",
        "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm",
    ]
    kept_total = 0
    for url in pages:
        r = _try_get(client, url, logger)
        if not r or r.status_code != 200:
            _log(logger, f"BoJ page fail {url}: {getattr(r,'status_code',None)}")
            continue
        local_kept = 0
        for href, text in _extract_anchors(r.text):
            title = _clean_html(text)
            if not title:
                continue
            if not _BOJ_KEEP_PATTERNS.search(title):
                continue
            full = href if href.startswith("http") else urljoin(url, href)
            imp = "high" if re.search(r"(Statement on Monetary Policy|All Decisions|Monetary Policy Releases)", title, re.I) else "high"
            rows.append(_mk_row("BOJ_PR", title, full, "japan", "JPY", "boj mpm", imp))
            local_kept += 1
        kept_total += local_kept
        _log(logger, f"BoJ kept from {url}: {local_kept}")

    out = _dedup_rows(rows)
    _log(logger, f"BoJ collected: {len(out)}")
    return out

# ---------- BoE (новости с фильтром по монетарной теме) ----------

_BOE_NEWS_SECTIONS = [
    "https://www.bankofengland.co.uk/news/news",
    "https://www.bankofengland.co.uk/news/publications",
    "https://www.bankofengland.co.uk/news/speeches",
    "https://www.bankofengland.co.uk/news/statistics",
    "https://www.bankofengland.co.uk/news/prudential-regulation",
    "https://www.bankofengland.co.uk/news/upcoming",
]

_BOE_KEEP = re.compile(
    r"(Monetary Policy|Monetary Policy Committee|MPC|Bank Rate|interest rate decision)",
    re.I
)

def collect_boe(client, logger=None, pages_per_section=3):
    rows = []
    kept_total = 0
    for base in _BOE_NEWS_SECTIONS:
        for p in range(1, pages_per_section + 1):
            url = base if p == 1 else f"{base}?page={p}"
            r = _try_get(client, url, logger)
            if not r or r.status_code != 200:
                _log(logger, f"BoE list fail {url}: {getattr(r,'status_code',None)}")
                continue
            anchors = _extract_anchors(r.text)
            kept = 0
            for href, text in anchors:
                title = _clean_html(text)
                if not title or not _BOE_KEEP.search(title):
                    continue
                full = href if href.startswith("http") else urljoin(url, href)
                rows.append(_mk_row("BOE_PR", title, full, "united kingdom", "GBP", "boe", "medium"))
                kept += 1
            kept_total += kept
            _log(logger, f"BOE list {url}: anchors={len(anchors)} kept={kept}")
    _log(logger, f"BOE HTML lists kept total: {kept_total}")

    # Подстраховка: site search по нескольким запросам
    queries = [
        "Monetary Policy Summary 2025",
        "Monetary Policy Committee 2025",
        "Bank Rate Monetary Policy 2025",
        "Monetary Policy Summary 2024",
        "interest rate decision MPC",
    ]
    kept_q = 0
    for q in queries:
        s_url = f"https://www.bankofengland.co.uk/search?{urlencode({'query': q})}"
        r = _try_get(client, s_url, logger)
        if not r or r.status_code != 200:
            _log(logger, f"BoE search fail {q}: {getattr(r,'status_code',None)}")
            continue
        anchors = _extract_anchors(r.text)
        kept = 0
        for href, text in anchors:
            title = _clean_html(text)
            if not title or not _BOE_KEEP.search(title):
                continue
            full = href if href.startswith("http") else urljoin("https://www.bankofengland.co.uk", href)
            rows.append(_mk_row("BOE_PR", title, full, "united kingdom", "GBP", "boe", "medium"))
            kept += 1
        kept_q += kept
        _log(logger, f"BoE site search '{q}': links={len(anchors)} kept={kept}")

    _log(logger, f"BOE collected total: {len(rows)}")
    return _dedup_rows(rows)

# ---------- JP MoF FX (фикс дефисов + варианты путей) ----------

_JP_MOF_URLS = [
    # корректный (с дефисами), чаще встречается у англ. раздела
    "https://www.mof.go.jp/en/policy/international_policy/reference/foreign-exchange-intervention/",
    "https://www.mof.go.jp/english/policy/international_policy/reference/foreign-exchange-intervention/",
    # исторические варианты (с подчеркиваниями)
    "https://www.mof.go.jp/en/policy/international_policy/reference/foreign_exchange_intervention/",
    "https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/",
    # fallback: корневые reference разделы
    "https://www.mof.go.jp/en/policy/international_policy/reference/",
    "https://www.mof.go.jp/english/policy/international_policy/reference/",
    "https://www.mof.go.jp/policy/international_policy/reference/",
]

_JP_MOF_KEEP = re.compile(
    r"(Foreign\s*Exchange\s*Intervention|intervention|Results of Foreign Exchange Intervention)",
    re.I
)

def collect_jp_mof_fx(client, logger=None):
    rows = []
    for url in _JP_MOF_URLS:
        r = _try_get(client, url, logger)
        if not r:
            _log(logger, f"JP MoF FX fail {url}: no response")
            continue
        if r.status_code != 200:
            _log(logger, f"JP MoF FX {url}: {r.status_code}")
            continue

        anchors = _extract_anchors(r.text)
        kept = 0
        for href, text in anchors:
            title = _clean_html(text)
            if not title or not _JP_MOF_KEEP.search(title):
                continue
            full = href if href.startswith("http") else urljoin(url, href)
            rows.append(_mk_row("JP_MOF_FX", title, full, "japan", "JPY", "mof fx", "high"))
            kept += 1
        _log(logger, f"JP MoF FX parsed {url}: kept={kept}")

    out = _dedup_rows(rows)
    _log(logger, f"JP MoF FX collected: {len(out)}")
    return out

# ---------- RBA (AUD) ----------

_RBA_BASE = "https://www.rba.gov.au"
_RBA_SEARCH_QUERIES = [
    "Monetary Policy Decision 2025",
    "Statement on Monetary Policy 2025",
    "Minutes of the Monetary Policy Meeting 2025",
    "Governor speech monetary policy 2025",
]

def _rba_imp_tags(title: str):
    t = title.lower()
    if "monetary policy decision" in t:
        return "high", "policy"
    if "statement on monetary policy" in t or "somp" in t:
        return "high", "policy somp"
    if "minutes of the monetary policy meeting" in t:
        return "medium", "minutes"
    if "governor" in t and ("speech" in t or "address" in t) and "monetary policy" in t:
        return "medium", "speech"
    return None, None

def collect_rba(client, logger=None, max_pages=1):
    rows = []
    hubs = [
        f"{_RBA_BASE}/monetary-policy/",
        f"{_RBA_BASE}/media-releases/",
    ]
    for url in hubs:
        r = _try_get(client, url, logger)
        if not r or r.status_code != 200:
            _log(logger, f"RBA hub fail {url}: {getattr(r,'status_code',None)}")
            continue
        anchors = _extract_anchors(r.text)
        kept = 0
        for href, text in anchors:
            title = _clean_html(text)
            if not title:
                continue
            imp, tags = _rba_imp_tags(title)
            if not imp:
                continue
            full = href if href.startswith("http") else urljoin(url, href)
            rows.append(_mk_row("RBA_PR", title, full, "australia", "AUD", tags, imp))
            kept += 1
        _log(logger, f"RBA hub kept from {url}: {kept}")

    for q in _RBA_SEARCH_QUERIES:
        s_url = f"{_RBA_BASE}/search/?{urlencode({'q': q})}"
        r = _try_get(client, s_url, logger)
        if not r or r.status_code != 200:
            _log(logger, f"RBA search fail {q}: {getattr(r,'status_code',None)}")
            continue
        anchors = _extract_anchors(r.text)
        kept = 0
        for href, text in anchors:
            title = _clean_html(text)
            imp, tags = _rba_imp_tags(title)
            if not imp:
                continue
            full = href if href.startswith("http") else urljoin(_RBA_BASE, href)
            rows.append(_mk_row("RBA_PR", title, full, "australia", "AUD", tags, imp))
            kept += 1
        _log(logger, f"RBA search '{q}': kept={kept}")

    out = _dedup_rows(rows)
    _log(logger, f"RBA collected: {len(out)}")
    return out

# ---------- основной агрегатор ----------

def collect_all(client, logger=None):
    """
    Возвращает единый список NEWS-строк.
    Порядок источников подобран по приоритетам/стабильности.
    """
    rows = []
    rows.extend(collect_fed(client, logger=logger))
    rows.extend(collect_us_treasury(client, logger=logger, max_pages=1))
    rows.extend(collect_ecb(client, logger=logger))
    rows.extend(collect_boj(client, logger=logger))
    rows.extend(collect_boe(client, logger=logger, pages_per_section=3))
    rows.extend(collect_jp_mof_fx(client, logger=logger))
    rows.extend(collect_rba(client, logger=logger))

    out = _dedup_rows(rows)
    _log(logger, f"NEWS augment: +{len(out)} rows")
    return out

# ---------- тест локальный ----------
if __name__ == "__main__":
    try:
        import httpx, logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("news_augment")
        with httpx.Client(headers={"User-Agent": "news-augment-bot/1.0"}) as client:
            data = collect_all(client, logger=logger)
        # быстрый вывод
        print("ts_utc\tsource\ttitle\turl\tcountries\tccy\ttags\timportance_guess\thash")
        for r in data:
            print("{ts_utc}\t{source}\t{title}\t{url}\t{countries}\t{ccy}\t{tags}\t{importance_guess}\t{hash}".format(**r))
    except Exception as e:
        print("Run error:", e)
