from dataclasses import dataclass
import os
import json

@dataclass
class Settings:
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    SHEET_ID: str = os.getenv("SHEET_ID", "")
    GOOGLE_CREDENTIALS: str = os.getenv("GOOGLE_CREDENTIALS", "")
    # пусто -> разрешить всем; иначе список id через запятую
    ALLOWED_CHAT_IDS: set[int] = None
    DEFAULT_TTL_MIN: int = int(os.getenv("DEFAULT_TTL_MIN", "480"))

    def __post_init__(self):
        raw = os.getenv("ALLOWED_CHAT_IDS", "").replace(" ", "")
        self.ALLOWED_CHAT_IDS = set(int(x) for x in raw.split(",") if x)

settings = Settings()

def creds_dict() -> dict:
    if not settings.GOOGLE_CREDENTIALS:
        return {}
    return json.loads(settings.GOOGLE_CREDENTIALS)
