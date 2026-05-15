from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root for an editable source checkout."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve ``path`` relative to ``base`` or the project root."""
    value = Path(path)
    if value.is_absolute():
        return value
    return (Path(base) if base is not None else project_root()).joinpath(value).resolve()


def require_local_path(path: str | Path, *, description: str, must_be_file: bool | None = None) -> Path:
    """Resolve and validate a local asset path.

    DrPO intentionally does not download model assets during training. Missing
    paths should fail early with a message that tells the user which local asset
    needs to be prepared.
    """
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{description} not found at {resolved}. Prepare the local asset before running DrPO.")
    if must_be_file is True and not resolved.is_file():
        raise FileNotFoundError(f"{description} must be a file, got {resolved}.")
    if must_be_file is False and not resolved.is_dir():
        raise FileNotFoundError(f"{description} must be a directory, got {resolved}.")
    return resolved
