from __future__ import annotations

import sys
import argparse

from drpo.assets import check_assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check local DrPO model assets.")
    parser.add_argument("--include-eval", action="store_true", help="also check HPSv3 and ImageReward evaluation assets")
    args = parser.parse_args(argv)
    missing = []
    for asset, exists in check_assets(include_eval=args.include_eval):
        state = "OK" if exists else "MISSING"
        print(f"{state:7s} {asset.name:18s} {asset.path}")
        if not exists:
            missing.append(asset)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
