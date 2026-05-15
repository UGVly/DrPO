from __future__ import annotations

from drpo.config import parse_config
from drpo.training.trainer import train


def main() -> None:
    train(parse_config(force_use_lora=True))


if __name__ == "__main__":
    main()
