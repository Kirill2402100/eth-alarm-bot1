# migrate_tsv.py
import re
import urllib.parse
from datetime import datetime

# --- Утилиты, скопированные из news_augment.py для самодостаточности ---
HDR = "ts_utc\tsource\ttitle\turl\tcountries\tccy\ttags\timportance_guess\thash\n"

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

def _parse_ts(ts_str: str) -> datetime:
    return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")

# --- Основная логика миграции ---

def migrate_tsv(in_path: str, out_path: str):
    """
    Читает TSV, реканонизирует URL и hash, удаляет дубликаты по (source, canon_url),
    оставляя только самую свежую запись для каждой группы.
    """
    try:
        with open(in_path, "r", encoding="utf-8") as f:
            header = f.readline()
            if not header or not header.strip().startswith("ts_utc"):
                f.seek(0)
            rows = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found at {in_path}")
        return

    best = {}  # key: (source, canon_url) -> full_row_tuple
    processed_count = 0
    for line in rows:
        cols = line.strip().split("\t")
        if len(cols) < 9:
            continue
        
        ts, source, title, url, countries, ccy, tags, importance, _ = cols[:9]
        
        # Реканонизация
        canon_url = _canon_url(url)
        new_hash = f"{source}|{canon_url}"
        
        key = (source, canon_url)
        new_row = (ts, source, title, canon_url, countries, ccy, tags, importance, new_hash)
        
        # Оставляем только самую свежую запись
        if key not in best or _parse_ts(ts) > _parse_ts(best[key][0]):
            best[key] = new_row
        
        processed_count += 1

    dedup_rows = sorted(best.values(), key=lambda r: r[0], reverse=True)
    
    with open(out_path, "w", encoding="utf-8") as w:
        w.write(HDR)
        for r in dedup_rows:
            w.write("\t".join(r) + "\n")
            
    print(f"Migration complete. Processed {processed_count} rows, wrote {len(dedup_rows)} unique rows to {out_path}")

if __name__ == '__main__':
    # Пример использования:
    # Создадим тестовый старый файл
    old_data = [
        "2025-09-12T10:00:00Z\tUS_TREASURY\tOld Release\thttps://www.home.treasury.gov/news/press-releases/sb0246#anchor\tusa\tUSD\tt\tmed\told_hash1",
        "2025-09-12T11:00:00Z\tUS_TREASURY\tNewer Release\thttps://home.treasury.gov/news/press-releases/sb0246/\tusa\tUSD\tt\tmed\told_hash2",
        "2025-09-10T09:00:00Z\tBOE_PR\tSome speech\thttps://www.bankofengland.co.uk/speech/2025/governor\tgb\tGBP\ts\tmed\told_hash3"
    ]
    with open("old_data.tsv", "w", encoding="utf-8") as f:
        f.write(HDR)
        f.write("\n".join(old_data))
        
    print("Running migration on test file 'old_data.tsv'...")
    migrate_tsv("old_data.tsv", "clean_data.tsv")
    
    print("\n--- Contents of clean_data.tsv ---")
    with open("clean_data.tsv", "r", encoding="utf-8") as f:
        print(f.read())
