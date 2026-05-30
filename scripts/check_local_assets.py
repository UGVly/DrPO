import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drpo.assets import check_assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check local DrPO model assets.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="also check optional reward and baseline assets",
    )
    args = parser.parse_args(argv)
    missing = []
    for asset, exists in check_assets(include_optional=args.all):
        state = "OK" if exists else "MISSING"
        group = "optional" if asset.optional else "default"
        print(f"{state:7s} {group:8s} {asset.name:18s} {asset.path}")
        if not exists:
            missing.append(asset)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
