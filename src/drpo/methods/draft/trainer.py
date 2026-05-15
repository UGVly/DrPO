from __future__ import annotations

from drpo.methods._baseline_loader import load_baseline_module
from drpo.pickscore import DifferentiablePickScoreScorer


def main() -> None:
    load_baseline_module("draft").main()


if __name__ == "__main__":
    main()
