"""Shared utilities for NLU Assignment 1."""

from __future__ import annotations

import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "Datasets"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def ensure_output_dir(part: str) -> Path:
    out = OUTPUT_DIR / part
    out.mkdir(parents=True, exist_ok=True)
    return out


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer using word boundaries."""
    return re.findall(r"\b\w+\b", text.lower())


def load_english_sentences(*filenames: str) -> list[str]:
    """Load English sentences from CSV files in Datasets/."""
    sentences: list[str] = []
    for name in filenames:
        path = DATASETS_DIR / name
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                sentences.append(row["Sentence_en"].strip())
    return sentences


def sentence_tokenize(text: str) -> list[str]:
    """Split text into sentences (simple rule-based)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
