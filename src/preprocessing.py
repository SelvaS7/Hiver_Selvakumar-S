import re
import pandas as pd


def normalize_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"https?://\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_probably_english(text: str) -> bool:
    text = normalize_text(text)
    letters = re.findall(r"[A-Za-z]", text)
    if len(letters) < 10:
        return False
    ascii_ratio = sum(ord(c) < 128 for c in text) / max(1, len(text))
    return ascii_ratio >= 0.90
