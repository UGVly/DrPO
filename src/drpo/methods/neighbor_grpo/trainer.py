from __future__ import annotations

from drpo.methods._baseline_loader import load_baseline_module


def main() -> None:
    load_baseline_module("grpo").main()


if __name__ == "__main__":
    main()
