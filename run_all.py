#!/usr/bin/env python3
"""Run all three parts of NLU Assignment 1."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PARTS = [
    ("Part 1: Zipf's Law", Path("part1/run_part1.py")),
    ("Part 2: Language Models", Path("part2/run_part2.py")),
    ("Part 3: HMM POS Tagger", Path("part3/run_part3.py")),
]


def main() -> None:
    root = Path(__file__).resolve().parent
    for label, script in PARTS:
        print("=" * 60)
        print(label)
        print("=" * 60)
        result = subprocess.run([sys.executable, str(root / script)], cwd=root)
        if result.returncode != 0:
            sys.exit(result.returncode)

    print("\nAll parts completed successfully.")
    print(f"Results saved under {root / 'outputs'}/")


if __name__ == "__main__":
    main()
