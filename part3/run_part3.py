#!/usr/bin/env python3
"""Part 3: HMM POS Tagger with Viterbi (from scratch)."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nltk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.utils import ensure_output_dir

RANDOM_SEED = 42
START_TAG = "<START>"
END_TAG = "<END>"


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


class HMMPOSTagger:
    """Hidden Markov Model POS tagger with add-1 smoothing."""

    def __init__(self):
        self.tags: list[str] = []
        self.tag_to_idx: dict[str, int] = {}
        self.transition: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.emission: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.tag_counts: Counter = Counter()
        self.vocab: set[str] = set()

    def train(self, tagged_sents: list[list[tuple[str, str]]]) -> None:
        tag_set: set[str] = set()
        trans_counts: dict[str, Counter] = defaultdict(Counter)
        emit_counts: dict[str, Counter] = defaultdict(Counter)

        for sent in tagged_sents:
            prev_tag = START_TAG
            for word, tag in sent:
                word_lower = word.lower()
                self.vocab.add(word_lower)
                tag_set.add(tag)
                trans_counts[prev_tag][tag] += 1
                emit_counts[tag][word_lower] += 1
                self.tag_counts[tag] += 1
                prev_tag = tag
            trans_counts[prev_tag][END_TAG] += 1

        self.tags = sorted(tag_set)
        self.tag_to_idx = {t: i for i, t in enumerate(self.tags)}
        num_tags = len(self.tags)

        # Transition with add-1 smoothing
        all_dest = self.tags + [END_TAG]
        for prev, dest_counts in trans_counts.items():
            total = sum(dest_counts.values())
            denom = total + num_tags + (1 if prev != START_TAG else 0)
            for dest in all_dest if prev == START_TAG else self.tags + [END_TAG]:
                count = dest_counts.get(dest, 0)
                self.transition[prev][dest] = math.log((count + 1) / denom)

        # Emission with add-1 smoothing
        vocab_size = len(self.vocab)
        for tag, word_counts in emit_counts.items():
            total = sum(word_counts.values())
            denom = total + vocab_size
            for word in self.vocab:
                count = word_counts.get(word, 0)
                self.emission[tag][word] = math.log((count + 1) / denom)

    def viterbi(self, words: list[str]) -> list[str]:
        """Viterbi decode with score and backpointer matrices."""
        words_lower = [w.lower() for w in words]
        n = len(words_lower)
        if n == 0:
            return []

        num_tags = len(self.tags)
        # score[t][j] = best log prob ending in tag j at position t
        score = [[float("-inf")] * num_tags for _ in range(n)]
        backpointer = [[-1] * num_tags for _ in range(n)]

        # Initialization
        for j, tag in enumerate(self.tags):
            trans = self.transition[START_TAG].get(tag, math.log(1e-10))
            emit = self.emission[tag].get(words_lower[0], math.log(1e-10))
            score[0][j] = trans + emit

        # Recursion
        for t in range(1, n):
            for j, tag in enumerate(self.tags):
                emit = self.emission[tag].get(words_lower[t], math.log(1e-10))
                best_score = float("-inf")
                best_prev = -1
                for i, prev_tag in enumerate(self.tags):
                    trans = self.transition[prev_tag].get(tag, math.log(1e-10))
                    candidate = score[t - 1][i] + trans + emit
                    if candidate > best_score:
                        best_score = candidate
                        best_prev = i
                score[t][j] = best_score
                backpointer[t][j] = best_prev

        # Termination
        best_final = float("-inf")
        best_last_tag = 0
        for j, tag in enumerate(self.tags):
            trans = self.transition[tag].get(END_TAG, math.log(1e-10))
            candidate = score[n - 1][j] + trans
            if candidate > best_final:
                best_final = candidate
                best_last_tag = j

        # Backtrack
        path = [0] * n
        path[n - 1] = best_last_tag
        for t in range(n - 2, -1, -1):
            path[t] = backpointer[t + 1][path[t + 1]]

        return [self.tags[path[t]] for t in range(n)]


def split_tagged_sents(sents: list, test_ratio: float = 0.2) -> tuple[list, list]:
    rng = random.Random(RANDOM_SEED)
    indices = list(range(len(sents)))
    rng.shuffle(indices)
    split = int(len(indices) * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    train = [sents[i] for i in train_idx]
    test = [sents[i] for i in test_idx]
    return train, test


def top_n_items(prob_dict: dict[str, dict[str, float]], source: str, n: int = 5) -> list[dict]:
    items = prob_dict.get(source, {})
    sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)[:n]
    return [{"target": k, "log_prob": v, "prob": math.exp(v)} for k, v in sorted_items]


def classify_error(word: str, gold: str, pred: str, in_vocab: bool) -> str:
    if not in_vocab:
        return "OOV word"
    if gold != pred:
        # check if word has multiple possible tags in training
        return "lexical ambiguity / sparse transition"
    return "other"


def main() -> None:
    download_nltk_data()
    out_dir = ensure_output_dir("part3")

    print("Loading Brown corpus (universal tagset)...")
    all_sents = nltk.corpus.brown.tagged_sents(tagset="universal")
    train_sents, test_sents = split_tagged_sents(all_sents, test_ratio=0.2)
    print(f"Train: {len(train_sents)} sents | Test: {len(test_sents)} sents")

    tagger = HMMPOSTagger()
    tagger.train(train_sents)

    results: dict = {"corpus": "Brown (universal tagset)", "train_sents": len(train_sents), "test_sents": len(test_sents)}

    # Q11: Top transitions from NOUN, top emissions from VERB
    noun_trans = top_n_items(tagger.transition, "NOUN", 5)
    verb_emit = top_n_items(tagger.emission, "VERB", 5)
    results["q11"] = {"top_transitions_from_NOUN": noun_trans, "top_emissions_from_VERB": verb_emit}
    print("\nQ11: Top-5 transitions from NOUN:")
    for item in noun_trans:
        print(f"  NOUN -> {item['target']}: P={item['prob']:.4f}")
    print("Top-5 emissions from VERB:")
    for item in verb_emit:
        print(f"  VERB -> '{item['target']}': P={item['prob']:.4f}")

    # Q12 & Q13: Evaluate
    total = 0
    correct = 0
    seen_total = 0
    seen_correct = 0
    oov_total = 0
    oov_correct = 0
    errors: list[dict] = []

    for sent in test_sents:
        words = [w for w, _ in sent]
        gold_tags = [t for _, t in sent]
        pred_tags = tagger.viterbi(words)

        for word, gold, pred in zip(words, gold_tags, pred_tags):
            total += 1
            in_vocab = word.lower() in tagger.vocab
            if gold == pred:
                correct += 1
                if in_vocab:
                    seen_correct += 1
                else:
                    oov_correct += 1
            else:
                errors.append({
                    "token": word,
                    "gold": gold,
                    "predicted": pred,
                    "in_vocab": in_vocab,
                    "reason": classify_error(word, gold, pred, in_vocab),
                })
            if in_vocab:
                seen_total += 1
            else:
                oov_total += 1

    accuracy = correct / total if total else 0
    seen_acc = seen_correct / seen_total if seen_total else 0
    oov_acc = oov_correct / oov_total if oov_total else 0

    results["q13"] = {
        "overall_accuracy": accuracy,
        "seen_accuracy": seen_acc,
        "oov_accuracy": oov_acc,
        "total_tokens": total,
        "seen_tokens": seen_total,
        "oov_tokens": oov_total,
    }
    print(f"\nQ13: Accuracy={accuracy:.4f} | Seen={seen_acc:.4f} | OOV={oov_acc:.4f}")

    # Q14: 10 mis-tagged tokens
    mis_tagged = errors[:10]
    results["q14"] = {"mis_tagged_examples": mis_tagged}

    error_type_counts = Counter(e["reason"] for e in errors)
    results["q14"]["error_type_summary"] = dict(error_type_counts)
    most_frequent = error_type_counts.most_common(1)[0] if error_type_counts else ("none", 0)
    results["q14"]["most_frequent_error"] = most_frequent[0]

    print("\nQ14: Sample mis-tagged tokens:")
    for e in mis_tagged:
        print(f"  '{e['token']}': gold={e['gold']}, pred={e['predicted']} ({e['reason']})")

    with (out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPart 3 complete. Outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
