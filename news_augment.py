#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
news_augment.py — собирает ссылки с первичных источников
(BoE, BoJ, JP MoF FX) и докладывает недостающее в CSV лист NEWS.

Главные фишки:
- STRICT_BOE (ENV): 1 — строгий фильтр (только MPC/Bank Rate/Minutes/Decision/Summary),
                    0 — широкий охват (в т.ч. Monetary Policy Report и др.)
- JP MoF FX: фикс распознавания ссылок с дефисами И подчёркиваниями.
- Дедуп по hash = "{source}|{url}".
- Пишет/дополняет CSV с колонками:
  ts_utc, source, title, url, countries, ccy, tags, importance_guess, hash

ENV:
  STRICT_BOE=1|0
  NEWS_CSV=./news.csv
"""

from __future__ import annotations

import csv
import os
import re
import sys
import time
import logging
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Iterator, List, Optional, Tuple

import httpx

# ---------- logging ----------
LOG = logging.getLogger("news_augment")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Включим и httpx логи, чтобы были строки как в примерах
logging.getLogger("httpx").setLevel(logging.INFO)

# ---------- config ----------
STRICT_BOE = os.getenv("STRICT_BOE", "1").strip() == "1"
NEWS_CSV = os.getenv("NEWS_CSV", "news.csv")

HTTP_TIMEOUT = 20.0
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# ---------- utils ----------

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

    @property
    def hash(self) -> str:
        return f"{self.source}|{self.url}"

    def to_row(self) -> List[str]:
        return [
            self.ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            self.source,
            self.title,
            self.url,
            self.countries,
            self.ccy,
            self.tags,
            self.importance_guess,
            self.hash,
        ]


def _http_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": UA, "Accept-Language": "en, *;q=0.1"},
        follow_redirects=True,
        timeout=HTTP_TIMEOUT,
        verify=True,
    )


def _log_http(url: str, status: int, phrase: str = "") -> None:
    logging.getLogger("httpx").info(
        'HTTP Request: GET %s "HTTP/1.1 %s %s"',
        url, status, phrase or "OK"
    )


def fetch_text(url: str) -> Tuple[str, int, str]:
    """GET текст + простой fallback через r.jina.ai/http/…"""
    try:
        with _http_client() as cl:
            r = cl.get(url)
            _log_http(url, r.status_code, r.reason_phrase or "")
            if r.status_code == 200 and r.text:
                return r.text, r.status_code, str(r.url)
    except Exception as e:
        LOG.info("fetch_text: primary GET failed: %s", e)

    # fallback через r.jina.ai
    try:
        wrap = f"https://r.jina.ai/http/{url}"
        with _http_client() as cl:
            r = cl.get(wrap)
            _log_http(wrap, r.status_code, r.reason_phrase or "")
            if r.status_code == 200 and r.text:
                return r.text, r.status_code, url
            return (r.text or ""), r.status_code, url
    except Exception as e:
        LOG.info("fetch_text: fallback GET failed: %s", e)
        return "", 0, url


def _abs_url(base: str, href: str) -> str:
    try:
        return urllib.parse.urljoin(base, href)
    except Exception:
        return href


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s or "")


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _iter_links(html: str, base_url: str, domain_must: Optional[str] = None) -> Iterator[Tuple[str, str]]:
    """Грубый парс ссылок из HTML: (abs_url, anchor_text)."""
    if not html:
        return
    for m in re.finditer(r"<a\s+[^>]*href\s*=\s*['\"]([^'\"]+)['\"][^>]*>(.*?)</a>", html, re.I | re.S):
        href = m.group(1)
        txt = _norm_space(_strip_tags(m.group(2)))
        if not href:
            continue
        full = _abs_url(base_url, href)
        if domain_must and urllib.parse.urlparse(full).netloc.find(domain_must) == -1:
            continue
        yield full, txt


# ---------- collectors ----------

def _boe_keep(url: str, title: str) -> bool:
    u = url.lower()
    t = title.lower()
    # Чистим явный мусор (pdf, если не policy-материал и т.п. не выкидываем агрессивно)
    if not (u.startswith("https://www.bankofengland.co.uk/") or u.startswith("http://www.bankofengland.co.uk/")):
        return False

    # Строгий набор
    strict_pat = re.compile(
        r"(monetary\s+policy\s+(summary|statement|committee|minutes|decision)|"
        r"\bbank\s+rate\b|"
        r"\bmpc\b|"
        r"interest\s+rate\s+decision|"
        r"\bminutes\b)", re.I
    )
    # Широкий набор
    broad_pat = re.compile(
        r"(monetary\s+policy|bank\s+rate|mpc|"
        r"monetary\s+policy\s+report\b|mpr\b|"
        r"press\s+conference|governing\s+council|"
        r"policy\s+statement|policy\s+summary)", re.I
    )

    if STRICT_BOE:
        return bool(strict_pat.search(t) or strict_pat.search(u))
    else:
        return bool(broad_pat.search(t) or broad_pat.search(u))


def collect_boe(now: datetime) -> List[NewsItem]:
    """Bank of England — листинги + site search, с фильтрацией по MPC/Bank Rate/etc."""
    sections = [
        "https://www.bankofengland.co.uk/news/news",
        "https://www.bankofengland.co.uk/news/publications",
        "https://www.bankofengland.co.uk/news/speeches",
        "https://www.bankofengland.co.uk/news/statistics",
        "https://www.bankofengland.co.uk/news/prudential-regulation",
        "https://www.bankofengland.co.uk/news/upcoming",
    ]

    def scan_list(url: str) -> List[Tuple[str, str]]:
        html, code, final = fetch_text(url)
        kept: List[Tuple[str, str]] = []
        anchors = 0
        seen = set()
        for u, t in _iter_links(html, final, domain_must="bankofengland.co.uk"):
            anchors += 1
            key = (u, t)
            if key in seen:
                continue
            seen.add(key)
            if _boe_keep(u, t):
                kept.append((u, t))
        LOG.info("news_augment: BOE list %s: anchors=%d kept=%d", url, anchors, len(kept))
        return kept

    kept_total: List[Tuple[str, str]] = []
    for sec in sections:
        kept_total.extend(scan_list(sec))
        # пагинация до 3 страниц (как в логах)
        for page in (2, 3):
            kept_total.extend(scan_list(f"{sec}?page={page}"))

    # site search — набор запросов
    queries = [
        "Monetary Policy Summary 2025",
        "Monetary Policy Committee 2025",
        "Bank Rate Monetary Policy 2025",
        "Monetary Policy Summary 2024",
        "interest rate decision MPC",
    ]
    for q in queries:
        url = f"https://www.bankofengland.co.uk/search?query={urllib.parse.quote(q)}"
        html, code, final = fetch_text(url)
        links = list(_iter_links(html, final, domain_must="bankofengland.co.uk"))
        kept = [(u, t) for (u, t) in links if _boe_keep(u, t)]
        LOG.info("news_augment: BOE site search '%s': links=%d kept=%d", q, len(links), len(kept))
        kept_total.extend(kept)

    # в NewsItem
    items: List[NewsItem] = []
    seen_u = set()
    for u, t in kept_total:
        if u in seen_u:
            continue
        seen_u.add(u)
        items.append(NewsItem(
            ts_utc=now,
            source="BOE_PR",
            title=t if t else "Bank of England",
            url=u,
            countries="united kingdom",
            ccy="GBP",
            tags="boe",
            importance_guess="medium" if STRICT_BOE else "high"
        ))

    LOG.info("news_augment: BOE collected total: %d", len(items))
    return items


def collect_boj(now: datetime) -> List[NewsItem]:
    """Bank of Japan — решения/минуты/summary of opinions."""
    items: List[NewsItem] = []
    pages = [
        ("https://www.boj.or.jp/en/mopo/mpmdeci/index.htm", "mpmdeci"),
        ("https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm", "mpmsche_minu"),
    ]
    total_kept = 0
    for url, tag in pages:
        html, code, final = fetch_text(url)
        links = list(_iter_links(html, final, domain_must="boj.or.jp"))
        # берём всё из этих разделов
        kept = []
        seen = set()
        for u, t in links:
            if (u, t) in seen:
                continue
            seen.add((u, t))
            kept.append((u, t))
        total_kept += len(kept)
        LOG.info("news_augment: BoJ %s: %d", tag, len(kept))
        for u, t in kept:
            items.append(NewsItem(
                ts_utc=now,
                source="BOJ_PR",
                title=t if t else "Bank of Japan",
                url=u,
                countries="japan",
                ccy="JPY",
                tags="boj mpm",
                importance_guess="high"
            ))
    LOG.info("news_augment: BoJ collected: %d", total_kept)
    return items


def collect_mof_fx(now: datetime) -> List[NewsItem]:
    """
    JP Ministry of Finance FX interventions reference — фикс под дефисы и подчёркивания.
    Ищем разные варианты путей + fallback с родительских страниц.
    """
    candidates = [
        # EN новые/старые пути с index.htm(l) — и с дефисами, и с подчёркиваниями
        "https://www.mof.go.jp/en/policy/international_policy/reference/foreign-exchange-intervention/index.htm",
        "https://www.mof.go.jp/en/policy/international_policy/reference/foreign-exchange-intervention/index.html",
        "https://www.mof.go.jp/english/policy/international_policy/reference/foreign-exchange-intervention/index.htm",
        "https://www.mof.go.jp/english/policy/international_policy/reference/foreign-exchange-intervention/index.html",

        "https://www.mof.go.jp/en/policy/international_policy/reference/foreign_exchange_intervention/index.htm",
        "https://www.mof.go.jp/en/policy/international_policy/reference/foreign_exchange_intervention/index.html",
        "https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/index.htm",
        "https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/index.html",

        # JP-ветка
        "https://www.mof.go.jp/policy/international_policy/reference/foreign-exchange-intervention/index.htm",
        "https://www.mof.go.jp/policy/international_policy/reference/foreign-exchange-intervention/index.html",
        "https://www.mof.go.jp/policy/international_policy/reference/foreign_exchange_intervention/index.htm",
        "https://www.mof.go.jp/policy/international_policy/reference/foreign_exchange_intervention/index.html",
    ]

    items: List[NewsItem] = []
    found_url: Optional[str] = None

    # 1) прямые варианты
    fx_body_pat = re.compile(r"(intervention|為替介入|foreign[-_ ]exchange)", re.I)
    for url in candidates:
        html_src, code, final = fetch_text(url)
        if code == 200 and html_src and fx_body_pat.search(html_src):
            found_url = final
            break

    # 2) fallback: поднимаемся на уровень выше и ищем ссылку
    if not found_url:
        parents = [
            "https://www.mof.go.jp/en/policy/international_policy/reference/",
            "https://www.mof.go.jp/english/policy/international_policy/reference/",
            "https://www.mof.go.jp/policy/international_policy/reference/",
        ]
        fx_pat = re.compile(r"foreign[-_]?exchange[-_]?intervention", re.I)
        for p in parents:
            html_src, code, final = fetch_text(p)
            if code != 200 or not html_src:
                continue
            for u, t in _iter_links(html_src, final, domain_must="mof.go.jp"):
                if fx_pat.search(u) or fx_pat.search(t):
                    found_url = u
                    break
            if found_url:
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
        ))
        LOG.info("news_augment: JP MoF FX collected: 1 (url=%s)", found_url)
    else:
        LOG.info("news_augment: JP MoF FX collected: 0")

    return items


# ---------- append to NEWS CSV ----------

HEADER = [
    "ts_utc", "source", "title", "url",
    "countries", "ccy", "tags", "importance_guess", "hash"
]


def _read_existing_hashes(csv_path: str) -> set[str]:
    hashes: set[str] = set()
    if not os.path.exists(csv_path):
        return hashes
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for row in rd:
                h = (row.get("hash") or "").strip()
                if h:
                    hashes.add(h)
    except Exception as e:
        LOG.info("read_existing: failed: %s", e)
    return hashes


def _append_rows(csv_path: str, items: List[NewsItem]) -> int:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    exists = os.path.exists(csv_path)
    added = 0
    mode = "a" if exists else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if not exists:
            wr.writerow(HEADER)
        for it in items:
            wr.writerow(it.to_row())
            added += 1
    return added


def augment_news(items: List[NewsItem]) -> int:
    existing = _read_existing_hashes(NEWS_CSV)
    new_items = [i for i in items if i.hash not in existing]
    if not new_items:
        LOG.info("news_augment: NEWS augment: +0 rows")
        return 0
    added = _append_rows(NEWS_CSV, new_items)
    LOG.info("news_augment: NEWS augment: +%d rows", added)
    return added


# ---------- main ----------

def main() -> None:
    LOG.info("news_augment: Starting Container")
    now = datetime.now(timezone.utc)

    all_items: List[NewsItem] = []
    # BoE
    all_items.extend(collect_boe(now))
    # BoJ
    all_items.extend(collect_boj(now))
    # JP MoF FX (фикс дефисы/подчёркивания)
    all_items.extend(collect_mof_fx(now))

    LOG.info("news_augment: augment collected: %d new candidates", len(all_items))
    augment_news(all_items)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
