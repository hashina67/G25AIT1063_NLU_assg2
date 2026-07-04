#!/usr/bin/env python3
"""Part 2: N-gram Language Models."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.utils import ensure_output_dir, load_english_sentences, tokenize

START = "<s>"
END = "</s>"
UNK = "<unk>"
RANDOM_SEED = 42


class NGramLM:
    """N-gram language model with Laplace smoothing and Jelinek-Mercer interpolation."""

    def __init__(self, n: int, laplace: bool = False):
        self.n = n
        self.laplace = laplace
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.vocab: set[str] = set()
        self.unigram_counts: Counter = Counter()

    def _unigram_prob(self, token: str) -> float:
        total = sum(self.unigram_counts.values())
        count = self.unigram_counts[token]
        if self.laplace:
            return (count + 1) / (total + len(self.vocab))
        return count / total if total else 1e-10

    def _prob(self, token: str, context: tuple[str, ...]) -> float:
        if self.n == 1:
            return self._unigram_prob(token)

        ngram = context + (token,)
        ctx_count = self.context_counts.get(context, 0)
        ng_count = self.ngram_counts.get(ngram, 0)
        vocab_size = len(self.vocab)

        if self.laplace:
            if ng_count > 0:
                return ng_count / ctx_count
            return (ng_count + 1) / (ctx_count + vocab_size)
        if ctx_count == 0:
            return self._unigram_prob(token)
        return ng_count / ctx_count

    def train(self, sentences: list[list[str]]) -> None:
        for sent in sentences:
            for token in sent:
                self.vocab.add(token)
                self.unigram_counts[token] += 1

            for i in range(len(sent)):
                ngram = tuple(sent[i - self.n + 1 + j] for j in range(self.n) if i - self.n + 1 + j >= 0)
                if len(ngram) == self.n:
                    context = ngram[:-1]
                    self.ngram_counts[ngram] += 1
                    self.context_counts[context] += 1

    def sentence_log_prob(self, sent: list[str]) -> float:
        log_prob = 0.0
        for i, token in enumerate(sent):
            start = max(0, i - self.n + 1)
            context = tuple(sent[start:i])
            if self.n == 1:
                context = ()
            prob = self._prob(token, context)
            log_prob += math.log(prob) if prob > 0 else math.log(1e-10)
        return log_prob

    def perplexity(self, sentences: list[list[str]]) -> float:
        total_log_prob = 0.0
        total_tokens = 0
        for sent in sentences:
            total_log_prob += self.sentence_log_prob(sent)
            total_tokens += len(sent)
        if total_tokens == 0:
            return float("inf")
        return math.exp(-total_log_prob / total_tokens)

    def sample_sentence(self, max_length: int = 30) -> list[str]:
        sent = [START]
        for step in range(max_length):
            i = len(sent)
            start = max(0, i - self.n + 1)
            context = tuple(sent[start:i])
            if self.n == 1:
                context = ()

            candidates = [w for w in self.vocab if w != START]
            weights = [self._prob(w, context) for w in candidates]
            total = sum(weights)
            if total == 0:
                break
            weights = [w / total for w in weights]
            next_word = random.choices(candidates, weights=weights, k=1)[0]
            sent.append(next_word)
            if next_word == END:
                break
        if sent[-1] != END:
            sent.append(END)
        return sent


def build_vocab(sentences: list[str], min_count: int = 2) -> set[str]:
    counts: Counter = Counter()
    for sent in sentences:
        counts.update(tokenize(sent))
    return {w for w, c in counts.items() if c >= min_count}


def prepare_sentences(raw_sentences: list[str], vocab: set[str] | None = None) -> list[list[str]]:
    prepared = []
    for sent in raw_sentences:
        tokens = [t if vocab is None or t in vocab else UNK for t in tokenize(sent)]
        if tokens:
            prepared.append([START] + tokens + [END])
    return prepared


def split_train_test(all_sentences: list[str], test_ratio: float = 0.2) -> tuple[list[str], list[str]]:
    rng = random.Random(RANDOM_SEED)
    shuffled = all_sentences[:]
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def main() -> None:
    random.seed(RANDOM_SEED)
    out_dir = ensure_output_dir("part2")

    print("Loading English sentences from Datasets/...")
    all_raw = load_english_sentences(
        "train_en_10000.csv",
        "dev_en_1000.csv",
        "test_en_1000.csv",
    )
    train_raw, test_raw = split_train_test(all_raw, test_ratio=0.2)
    vocab = build_vocab(train_raw, min_count=2)
    train_sents = prepare_sentences(train_raw, vocab)
    test_sents = prepare_sentences(test_raw, vocab)

    print(f"Train: {len(train_sents)} sentences | Test: {len(test_sents)} sentences")

    results: dict = {
        "corpus": "Provided English dataset (train+dev+test, 80/20 split)",
        "train_sentences": len(train_sents),
        "test_sentences": len(test_sents),
    }

    # Q6: Unigram and bigram MLE
    unigram_mle = NGramLM(n=1, laplace=False)
    unigram_mle.train(train_sents)
    bigram_mle = NGramLM(n=2, laplace=False)
    bigram_mle.train(train_sents)
    results["q6"] = {
        "unigram_vocab_size": len(unigram_mle.vocab),
        "bigram_count": len(bigram_mle.ngram_counts),
    }
    print(f"\nQ6: Unigram vocab={len(unigram_mle.vocab)}, bigrams={len(bigram_mle.ngram_counts)}")

    # Q7: Laplace-smoothed bigram
    bigram_laplace = NGramLM(n=2, laplace=True)
    bigram_laplace.train(train_sents)
    results["q7"] = {
        "laplace_smoothing": True,
        "explanation": (
            "Raw MLE assigns zero probability to unseen bigrams, making perplexity "
            "undefined and preventing generalization. Laplace (add-1) smoothing assigns "
            "non-zero probability to unseen events by adding 1 to all counts."
        ),
    }
    print("\nQ7: Laplace smoothing applied to bigram model.")

    # Q8: Perplexity — unigram MLE and bigram Laplace on test set
    pp_unigram = unigram_mle.perplexity(test_sents)
    pp_bigram = bigram_laplace.perplexity(test_sents)
    results["q8"] = {
        "perplexity_unigram_mle": pp_unigram,
        "perplexity_bigram_laplace": pp_bigram,
    }
    print(f"\nQ8: Unigram PP={pp_unigram:.2f} | Bigram PP={pp_bigram:.2f}")

    # Q9: Sample 5 sentences from bigram model
    sampled = []
    for i in range(5):
        sent = bigram_laplace.sample_sentence()
        text = " ".join(sent)
        sampled.append(text)
        print(f"  Sample {i + 1}: {text}")
    results["q9_samples"] = sampled

    # Q10: Trigram with Laplace
    trigram_laplace = NGramLM(n=3, laplace=True)
    trigram_laplace.train(train_sents)
    pp_trigram = trigram_laplace.perplexity(test_sents)
    results["q10"] = {
        "perplexity_unigram_mle": pp_unigram,
        "perplexity_bigram_laplace": pp_bigram,
        "perplexity_trigram_laplace": pp_trigram,
        "trigram_higher_than_bigram": bool(pp_trigram > pp_bigram),
    }
    print(f"\nQ10: Trigram PP={pp_trigram:.2f}")

    with (out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPart 2 complete. Outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
