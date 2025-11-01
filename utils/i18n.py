# utils/i18n.py
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
import streamlit as st


SUPPORTED = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "pt": "Português",
    "zh": "中文",
    "ar": "العربية",
    "hi": "हिन्दी",
    "te": "తెలుగు"
}

DEFAULT_LANG = "en"
LOCALES_DIR = Path("i18n")

@lru_cache(maxsize=16)
def _load_locale(lang: str) -> dict:
    lang = lang if lang in SUPPORTED else DEFAULT_LANG
    fp = LOCALES_DIR / f"{lang}.json"
    if not fp.exists():
        fp = LOCALES_DIR / f"{DEFAULT_LANG}.json"
    return json.loads(fp.read_text(encoding="utf-8"))

def get_lang() -> str:
    return st.session_state.get("lang", DEFAULT_LANG)

def set_lang(lang: str) -> None:
    st.session_state["lang"] = lang if lang in SUPPORTED else DEFAULT_LANG

def t(key: str, default: str | None = None) -> str:
    lang = get_lang()
    data = _load_locale(lang)
    if key in data:
        return data[key]
    # fall back to English for missing keys
    eng = _load_locale("en")
    return eng.get(key, default if default is not None else key)

def language_name(lang: str | None = None) -> str:
    lang = lang or get_lang()
    return SUPPORTED.get(lang, "English")
