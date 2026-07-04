#!/usr/bin/env python3
"""Part 1: Zipf's Law and Corpus Analysis."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.utils import ensure_output_dir, tokenize

CORPUS_NAME = "Brown Corpus (NLTK)"
CORPUS_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]
MATTR_WINDOW = 500

OPEN_CLASS = {
    "noun", "verb", "adjective", "adverb", "num", "propn",
    "nn", "nns", "nnp", "nnps", "vb", "vbd", "vbg", "vbn", "vbp", "vbz",
    "jj", "jjr", "jjs", "rb", "rbr", "rbs", "fw", "sym",
}
CLOSED_CLASS = {
    "determiner", "preposition", "conjunction", "pronoun", "auxiliary",
    "particle", "punctuation", "to", "existential",
    "dt", "in", "cc", "prp", "prp$", "md", "rp", "wdt", "wp", "wp$",
    "wr", "pdt", "ex", "uh", "pos", "cd",
}


def download_nltk_data() -> None:
    for resource in ("brown", "punkt", "punkt_tab", "universal_tagset"):
        try:
            if resource == "brown":
                nltk.data.find("corpora/brown")
            elif resource == "universal_tagset":
                nltk.data.find("taggers/universal_tagset")
            else:
                nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def load_corpus_tokens(max_tokens: int = 1_000_000) -> list[str]:
    download_nltk_data()
    tokens: list[str] = []
    for file_id in nltk.corpus.brown.fileids():
        for sent in nltk.corpus.brown.sents(file_id):
            for word in sent:
                tokens.append(word.lower())
                if len(tokens) >= max_tokens:
                    return tokens
    return tokens


def compute_frequencies(tokens: list[str]) -> Counter:
    return Counter(tokens)


def fit_zipf(freqs: Counter) -> tuple[float, float, np.ndarray, np.ndarray]:
    ranked = freqs.most_common()
    ranks = np.arange(1, len(ranked) + 1, dtype=float)
    frequencies = np.array([count for _, count in ranked], dtype=float)
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(frequencies)
    slope, intercept, _, _, _ = stats.linregress(log_ranks, log_freqs)
    return slope, intercept, log_ranks, log_freqs


def compute_ttr(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_mattr(tokens: list[str], window: int = MATTR_WINDOW) -> float:
    if len(tokens) < window:
        return compute_ttr(tokens)
    ttrs = []
    for start in range(len(tokens) - window + 1):
        window_tokens = tokens[start : start + window]
        ttrs.append(len(set(window_tokens)) / window)
    return float(np.mean(ttrs))


CLOSED_WORDS = {
    "the", "a", "an", "of", "and", "to", "in", "is", "that", "for", "it",
    "as", "was", "on", "be", "at", "by", "this", "with", "from", "or", "not",
    "but", "are", "his", "he", "has", "had", "were", "which", "they", "their",
    "one", "all", "we", "can", "her", "she", "been", "have", "would", "will",
    "who", "you", "your", "its", "if", "so", "no", "do", "my", "me", "him",
    "than", "them", "then", "there", "when", "what", "up", "out", "about",
    ",", ".", "''", "``", "''",
}


def categorize_word_list(words: list[str]) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {"open-class": [], "closed-class": [], "other": []}
    for word in words:
        if word in CLOSED_WORDS or not word.isalpha():
            categories["closed-class"].append(word)
        else:
            categories["open-class"].append(word)
    return categories


def main() -> None:
    out_dir = ensure_output_dir("part1")
    print(f"Loading {CORPUS_NAME} (up to 1M tokens)...")
    tokens = load_corpus_tokens(1_000_000)
    print(f"Loaded {len(tokens):,} tokens, {len(set(tokens)):,} types.")

    freqs = compute_frequencies(tokens)
    results: dict = {"corpus": CORPUS_NAME, "total_tokens": len(tokens), "total_types": len(freqs)}

    # Q1: Top-10 words
    top10 = freqs.most_common(10)
    results["q1_top10"] = [{"word": w, "frequency": f} for w, f in top10]
    print("\nQ1: Top-10 words")
    for word, freq in top10:
        print(f"  {word:15s} {freq}")

    # Q2: Zipf plot
    slope, intercept, log_ranks, log_freqs = fit_zipf(freqs)
    results["q2"] = {"slope": slope, "intercept": intercept, "zipf_prediction_slope": -1.0}
    print(f"\nQ2: Regression slope={slope:.4f}, intercept={intercept:.4f}")
    confirms = abs(slope - (-1.0)) < 0.15
    results["q2"]["confirms_zipf"] = bool(confirms)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(log_ranks, log_freqs, s=4, alpha=0.5, label="Word types")
    fit_line = slope * log_ranks + intercept
    ax.plot(log_ranks, fit_line, "r-", linewidth=2, label=f"Fit: slope={slope:.3f}")
    ax.set_xlabel("log10(rank)")
    ax.set_ylabel("log10(frequency)")
    ax.set_title(f"Zipf's Law — {CORPUS_NAME}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "q2_zipf_plot.png", dpi=150)
    plt.close(fig)

    # Q3: Top/bottom 20
    top20 = [w for w, _ in freqs.most_common(20)]
    bottom20 = [w for w, _ in freqs.most_common()[-20:]]
    top20_cats = categorize_word_list(top20)
    bottom20_cats = categorize_word_list(bottom20)
    results["q3"] = {
        "top20": top20,
        "bottom20": bottom20,
        "top20_categories": {k: len(v) for k, v in top20_cats.items()},
        "bottom20_categories": {k: len(v) for k, v in bottom20_cats.items()},
    }
    print("\nQ3: Top-20 (mostly closed-class function words expected)")
    print(f"  Categories: {results['q3']['top20_categories']}")
    print(f"  Bottom-20 categories: {results['q3']['bottom20_categories']}")

    # Q4: TTR vs corpus size
    ttr_values = []
    for size in CORPUS_SIZES:
        subset = tokens[:size]
        ttr = compute_ttr(subset)
        ttr_values.append(ttr)
        print(f"  TTR at {size:,} tokens: {ttr:.4f}")

    results["q4"] = {"sizes": CORPUS_SIZES, "ttr": ttr_values}

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(CORPUS_SIZES, ttr_values, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Corpus size (tokens)")
    ax.set_ylabel("Type-Token Ratio (TTR)")
    ax.set_title("TTR vs Corpus Size")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "q4_ttr_plot.png", dpi=150)
    plt.close(fig)

    # Q5: MATTR
    mattr_values = []
    for size in CORPUS_SIZES:
        subset = tokens[:size]
        mattr = compute_mattr(subset, MATTR_WINDOW)
        mattr_values.append(mattr)
        print(f"  MATTR at {size:,} tokens: {mattr:.4f}")

    results["q5"] = {
        "method": f"MATTR (window={MATTR_WINDOW})",
        "sizes": CORPUS_SIZES,
        "mattr": mattr_values,
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(CORPUS_SIZES, ttr_values, "o-", label="Raw TTR", linewidth=2)
    ax.plot(CORPUS_SIZES, mattr_values, "s-", label=f"MATTR (w={MATTR_WINDOW})", linewidth=2)
    ax.set_xlabel("Corpus size (tokens)")
    ax.set_ylabel("Ratio")
    ax.set_title("TTR vs MATTR vs Corpus Size")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "q5_mattr_plot.png", dpi=150)
    plt.close(fig)

    with (out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPart 1 complete. Outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
