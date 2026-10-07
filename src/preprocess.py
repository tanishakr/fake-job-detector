import re
import pandas as pd
from src.config import TEXT_COLS


def build_combined_text(df):
    return df[TEXT_COLS].fillna('').agg(' '.join, axis=1)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
