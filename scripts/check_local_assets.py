from __future__ import annotations

import argparse

from drpo.assets import check_assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check local DrPO model assets.")
    parser.parse_args(argv)
    missing = []
    for asset, exists in check_assets():
        state = "OK" if exists else "MISSING"
        print(f"{state:7s} {asset.name:18s} {asset.path}")
        if not exists:
            missing.append(asset)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
