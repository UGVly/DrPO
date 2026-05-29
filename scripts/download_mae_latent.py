#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO_ID = "jiangzhou130v1/drpo-mae-latent-256"
FILE_NAME = "mae_latent_256_torch.pth"
SHA256 = "4810249905d2882a41d5a0fe97ebac995af8918dbe121daa9871e5bc605445b1"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the DrPO latent MAE checkpoint from ModelScope.")
    parser.add_argument(
        "--output",
        default=str(project_root() / "drifting" / FILE_NAME),
        help="Destination checkpoint path.",
    )
    parser.add_argument("--repo-id", default=REPO_ID, help="ModelScope model repository id.")
    parser.add_argument("--token", default=None, help="Optional ModelScope access token for private repos.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing destination file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()

    if output.exists() and not args.force:
        actual = sha256sum(output)
        if actual == SHA256:
            print(f"Already present: {output}")
            return 0
        raise SystemExit(f"Refusing to overwrite {output}; pass --force to replace it.")

    try:
        from modelscope.hub.file_download import model_file_download
    except ImportError as exc:
        raise SystemExit("Install ModelScope first: python -m pip install modelscope") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        model_file_download(
            model_id=args.repo_id,
            file_path=FILE_NAME,
            local_dir=str(output.parent),
            token=args.token,
        )
    )
    if downloaded.resolve() != output:
        shutil.move(str(downloaded), str(output))

    actual = sha256sum(output)
    if actual != SHA256:
        output.unlink(missing_ok=True)
        raise SystemExit(f"SHA256 mismatch for {output}: expected {SHA256}, got {actual}")

    print(f"Downloaded: {output}")
    print(f"SHA256: {actual}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
