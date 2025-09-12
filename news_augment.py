#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
news_augment.py — дополняет лист NEWS локальными источниками под GBP/JPY:
  • Bank of England (BoE) — GBP
  • Bank of Japan (BoJ) — JPY
  • Japan MoF FX (страница ссылок по интервенциям) — JPY

Запускается отдельно от calendar_collector. Пишет ТОЛЬКО в лист NEWS.
Дедупликация — по колонке hash ("source|url"), чтобы не плодить дублей.

Требуемые переменные окружения:
  SHEET_ID                       — Google Sheet ID
  GOOGLE_CREDENTIALS_JSON_B64    — service account creds в base64 (или GOOGLE_CREDENTIALS_JSON/GOOGLE_CREDENTIALS)
Необязательные:
  LOG_LEVEL=INFO|DEBUG
  RUN_FOREVER=0/1                — если 1, будет крутиться в цикле
  COLLECT_EVERY_MIN=20           — период опроса в минутах при RUN_FOREVER=1
  NEWS_MAX_AGE_DAYS=365          — обрезка очень старых ссылок
  JINA_PROXY=https://r.jina.ai/http/ — фолбэк-прокси на 403/404

Формат NEWS:
  ["ts_utc","source","title","url","countries","ccy","tags","importance_guess","hash"]
"""

import os, re, json, base64, time, logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime, timedelta, timezone
import html

# ---- Logging ----
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("news_augment")

# ---- ENV ----
SHEET_ID = os.getenv("SHEET_ID", "").strip()
RUN_FOREVER = os.getenv("RUN_FOREVER", "0").lower() in ("1", "true", "yes", "on")
COLLECT_EVERY_MIN = int(os.getenv("COLLECT_EVERY_MIN", "20") or "20")
NEWS_MAX_AGE_DAYS = int(os.getenv("NEWS_MAX_AGE_DAYS", "365") or "365")
JINA_PROXY = os.getenv("JINA_PROXY", "https://r.jina.ai/http/").rstrip("/") + "/"

# ---- HTTP ----
import httpx

UA = {"User-Agent": "Mozilla/5.0 (LPBot/1.0) news_augment"}

def fetch_text(url: str, timeout: float = 20.0) -> Tuple[str, int, str]:
    """GET с авто-редиректами. На 403/404 — пробуем Jina proxy."""
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=UA) as cli:
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
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---- Google Sheets ----
try:
    import gspread
    from google.oauth2 import service_account
    _GSHEETS_AVAILABLE = True
except Exception:
    gspread = None
    service_account = None
    _GSHEETS_AVAILABLE = False

SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

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
        creds = service_account.Credentials.from_service_account_info(info, scopes=SHEETS_SCOPES)
        gc = gspread.authorize(creds)
        return gc.open_by_key(sheet_id), "ok"
    except Exception as e:
        return None, f"auth/open error: {e}"

def ensure_worksheet(sh, title: str, headers: List[str]):
    for ws in sh.worksheets():
        if ws.title == title:
            try:
                cur = ws.get_values("A1:I1") or [[]]
                if not cur or cur[0] != headers:
                    ws.update(range_name="A1", values=[headers])
            except Exception:
                pass
            return ws, False
    ws = sh.add_worksheet(title=title, rows=200, cols=max(10, len(headers)))
    ws.update(range_name="A1", values=[headers])
    return ws, True

def _read_existing_hashes(ws, hash_col_letter: str = "I") -> set[str]:
    try:
        col = ws.get_values(f"{hash_col_letter}2:{hash_col_letter}10000")
        return {r[0] for r in col if r and r[0]}
    except Exception:
        return set()

# ---- Models ----
NEWS_HEADERS = ["ts_utc", "source", "title", "url", "countries", "ccy", "tags", "importance_guess", "hash"]

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
        return [
            self.ts_utc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            self.source,
            self.title.strip(),
            self.url.strip(),
            self.countries.strip().lower(),
            self.ccy.strip().upper(),
            self.tags.strip(),
            self.importance_guess.strip().lower(),
            self.key_hash(),
        ]

# ---- Link helpers ----
from urllib.parse import urljoin as _urljoin

_HREF_RE = re.compile(r'href\s*=\s*(["\'])([^"\']+)\1', re.I)

def _iter_links(html_src: str, base: str, domain_must: Optional[str] = None) -> List[str]:
    out, seen = [], set()
    for m in _HREF_RE.finditer(html_src or ""):
        raw = (m.group(2) or "").strip()
        if not raw or raw.startswith(("javascript:", "mailto:")):
            continue
        u = _urljoin(base, raw.split("#", 1)[0])
        if domain_must and domain_must not in u:
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out

def _slug_to_title(u: str) -> str:
    s = u.rstrip("/").split("/")[-1]
    s = re.sub(r"[-_]+", " ", s)
    s = re.sub(r"\.html?$|\.pdf$|\.htm$", "", s, flags=re.I)
    s = s.strip() or u
    if len(s) < 4:
        return u
    # Заглавные первые буквы
    return " ".join(w.capitalize() for w in s.split())

# ---- Heuristics ----
KW_HIGH = re.compile(
    r"(monetary policy|bank rate|mpc|rate decision|policy decision|policy statement|"
    r"press conference|minutes|summary of opinions|statement on monetary policy|"
    r"intervention|fx intervention)",
    re.I
)

def _importance_high(text: str) -> bool:
    return bool(KW_HIGH.search(text or ""))

def _boe_is_high(url: str, title: str) -> bool:
    return _importance_high(url + " " + title)

# ---- Collectors ----

def collect_boe(now: datetime) -> List[NewsItem]:
    """
    Bank of England:
      1) внутренний поиск сайта (серверный HTML) — даёт стабильные ссылки
      2) парсинг видимых разделов /news/... — вдруг что-то получится «в лоб»
    """
    base = "https://www.bankofengland.co.uk"
    items: List[NewsItem] = []
    seen: set[str] = set()

    # 1) Внутренний поиск сайта по ключам/годам
    search_queries = [
        "Monetary Policy Summary 2025",
        "Monetary Policy Committee 2025",
        "Bank Rate Monetary Policy 2025",
        "Monetary Policy Summary 2024",
    ]

    def _keep_boe(u: str) -> bool:
        # Сохраняем всё релевантное политике/ставке/годам
        if not any(p in u for p in (
            "/monetary-policy-summary-and-minutes/",
            "/monetary-policy/",
            "/news/",
            "/press/",
            "/publication/",
            "/speech/",
            "/speeches/",
        )):
            return False
        return bool(re.search(r"/20\d{2}\b|-20\d{2}-|\b20\d{2}\b", u))

    for q in search_queries:
        url = f"{base}/search?query=" + re.sub(r"\s+", "+", q)
        html_src, code, final_url = fetch_text(url)
        if code != 200 or not html_src:
            continue
        links = _iter_links(html_src, final_url, domain_must="bankofengland.co.uk")
        # нормализация
        links = [u.rstrip("/").split("?")[0] for u in links]
        picked = [u for u in links if _keep_boe(u)]
        log.info("news_augment: BOE site search '%s': links=%d kept=%d", q, len(links), len(picked))
        for u in picked:
            if u in seen:
                continue
            seen.add(u)
            title = _slug_to_title(u)
            imp = "high" if _boe_is_high(u, title) else "medium"
            items.append(NewsItem(now, "BOE_PR", title, u, "united kingdom", "GBP", "boe", imp))

    # 2) Разделы /news/... — иногда в HTML есть прямые <a>, иногда нет (JS). Пробуем всё равно.
    sections = [
        "/news/news",
        "/news/publications",
        "/news/speeches",
        "/news/statistics",
        "/news/prudential-regulation",
        "/news/upcoming",
    ]
    for path in sections:
        for p in ("", "?page=2", "?page=3"):
            url = base + path + p
            html_src, code, final_url = fetch_text(url)
            if code != 200 or not html_src:
                continue
            links = _iter_links(html_src, final_url, domain_must="bankofengland.co.uk")
            # оставляем только материалы 20xx
            picked = [u for u in links if re.search(r"/20\d{2}\b|-20\d{2}-|\b20\d{2}\b", u)]
            log.info("news_augment: BOE list %s: anchors=%d kept=%d", url, len(links), len(picked))
            for u in picked:
                if u in seen:
                    continue
                seen.add(u)
                title = _slug_to_title(u)
                imp = "high" if _boe_is_high(u, title) else "medium"
                items.append(NewsItem(now, "BOE_PR", title, u, "united kingdom", "GBP", "boe", imp))

    log.info("news_augment: BOE collected total: %d", len(items))
    return items

def collect_boj(now: datetime) -> List[NewsItem]:
    """
    Bank of Japan: две витрины — решения/заявления и расписания/мнения/протоколы.
    """
    items: List[NewsItem] = []

    # Monetary Policy Releases (решения, заявления, списки по годам)
    url1 = "https://www.boj.or.jp/en/mopo/mpmdeci/index.htm"
    html1, code1, u1 = fetch_text(url1)
    if code1 == 200 and html1:
        links1 = _iter_links(html1, u1, domain_must="boj.or.jp")
        # отфильтруем только /mopo/mpmdeci/…
        links1 = [u for u in links1 if "/mopo/mpmdeci/" in u]
        for u in links1:
            title = _slug_to_title(u)
            imp = "high" if _importance_high(title + " " + u) else "medium"
            items.append(NewsItem(now, "BOJ_PR", title, u, "japan", "JPY", "boj mpm", imp))

    # Schedule/Minutes/Opinions
    url2 = "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm"
    html2, code2, u2 = fetch_text(url2)
    if code2 == 200 and html2:
        links2 = _iter_links(html2, u2, domain_must="boj.or.jp")
        links2 = [u for u in links2 if "/mopo/mpmsche_minu/" in u]
        for u in links2:
            title = _slug_to_title(u)
            imp = "high" if _importance_high(title + " " + u) else "medium"
            items.append(NewsItem(now, "BOJ_PR", title, u, "japan", "JPY", "boj mpm", imp))

    # нормализуем/дедупим, оставим только наиболее релевантные (mps/mpm/statement/minutes/opinion/…)
    def _boost(u: str, t: str) -> bool:
        return any(k in (u + " " + t).lower() for k in [
            "mpr_", "state_", "k20", "statement", "minutes", "opinion",
            "mpr_202", "state_202", "mpmdeci", "mpmsche_minu",
        ])

    dedup = {}
    for it in items:
        key = it.url.rstrip("/").lower()
        if key not in dedup and _boost(it.url, it.title):
            dedup[key] = it

    items = list(dedup.values())
    log.info("news_augment: BoJ collected: %d", len(items))
    return items

def collect_mof_fx(now: datetime) -> List[NewsItem]:
    """
    Japan MoF FX — справочная страница с историей интервенций.
    У MOF часто гуляют пути и падают в 404. Пробуем оба базовых URL.
    Если доступно — фиксируем ссылку как high.
    """
    candidates = [
        "https://www.mof.go.jp/english/policy/international_policy/reference/foreign_exchange_intervention/",
        "https://www.mof.go.jp/policy/international_policy/reference/foreign_exchange_intervention/",
    ]
    items: List[NewsItem] = []
    for url in candidates:
        html_src, code, final = fetch_text(url)
        if code == 200 and html_src and re.search(r"(intervention|為替介入|foreign exchange)", html_src, re.I):
            items.append(NewsItem(now, "JP_MOF_FX", "FX intervention reference page", final,
                                  "japan", "JPY", "mof", "high"))
            break
    log.info("news_augment: JP MoF FX collected: %d", len(items))
    return items

# ---- Append to Sheets ----

def append_news_rows(sh, items: List[NewsItem]) -> int:
    if not items:
        return 0
    ws, _ = ensure_worksheet(sh, "NEWS", NEWS_HEADERS)
    existing_hash = _read_existing_hashes(ws, "I")
    rows = [n.to_row() for n in items if n.key_hash() not in existing_hash]

    # обрезаем по давности, чтобы мусорные старые ссылки не тащить
    cutoff = datetime.now(timezone.utc) - timedelta(days=NEWS_MAX_AGE_DAYS)
    rows = [r for r in rows if _parse_iso_ts(r[0]) >= cutoff]

    if rows:
        ws.append_rows(rows, value_input_option="RAW")
    return len(rows)

def _parse_iso_ts(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)

# ---- Main ----

def collect_once():
    log.info("Starting Container")
    if not SHEET_ID:
        raise RuntimeError("SHEET_ID env empty")

    sh, msg = build_sheets_client(SHEET_ID)
    if not sh:
        raise RuntimeError(f"Sheets not available: {msg}")

    now = datetime.now(timezone.utc)

    # Сбор по источникам
    boe = collect_boe(now)
    boj = collect_boj(now)
    mof = collect_mof_fx(now)

    all_items = boe + boj + mof
    log.info("news_augment: augment collected: %d new candidates", len(all_items))

    added = append_news_rows(sh, all_items)
    log.info("news_augment: NEWS augment: +%d rows", added)

def main():
    if RUN_FOREVER:
        interval = max(1, COLLECT_EVERY_MIN) * 60
        while True:
            try:
                collect_once()
            except Exception:
                log.exception("collect_once failed")
            time.sleep(interval)
    else:
        collect_once()

# ---- Utils (if run as script) ----
if __name__ == "__main__":
    main()
