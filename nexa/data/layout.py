"""
Dataset folder layout utilities.

These helpers create the directory structure used by the Nexa dataset
pipeline. They never delete data and never modify downloaded raw files.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path


PIPELINE_DIRECTORIES: tuple[Path, ...] = (
    Path("raw"),
    Path("cleaned"),
    Path("deduplicated"),
    Path("splits") / "train",
    Path("splits") / "validation",
    Path("processed") / "train",
    Path("processed") / "validation",
    Path("manifests"),
    Path("reports"),
)

DATASET_METADATA_FIELDS: tuple[str, ...] = (
    "dataset_name",
    "source",
    "official_url",
    "version",
    "license",
    "download_date",
    "original_filename",
    "checksum_sha256",
    "approximate_size",
)

DATASET_STAGES: tuple[str, ...] = ("raw", "cleaned", "deduplicated")


def ensure_dataset_layout(data_root: str | Path = "data") -> list[Path]:
    """
    Create the canonical dataset pipeline directories.

    Returns the directories that were expected. Existing directories are left
    untouched, and no data files are overwritten or deleted.
    """
    root = Path(data_root)
    created_or_existing: list[Path] = []

    for relative_dir in PIPELINE_DIRECTORIES:
        path = root / relative_dir
        path.mkdir(parents=True, exist_ok=True)
        _touch_gitkeep(path)
        created_or_existing.append(path)

    return created_or_existing


def ensure_dataset_directory(
    dataset_name: str,
    stage: str = "raw",
    data_root: str | Path = "data",
) -> Path:
    """
    Create a per-dataset stage directory and metadata template when needed.

    The metadata file is created only if absent. This preserves any verified
    license, checksum, and source details that have already been recorded.
    """
    if stage not in DATASET_STAGES:
        allowed = ", ".join(DATASET_STAGES)
        raise ValueError(f"stage must be one of: {allowed}")

    safe_name = _safe_dataset_name(dataset_name)
    dataset_dir = Path(data_root) / stage / safe_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _touch_gitkeep(dataset_dir)

    metadata_path = dataset_dir / "metadata.yaml"
    if not metadata_path.exists():
        metadata_path.write_text(_metadata_template(safe_name), encoding="utf-8")

    return dataset_dir


def _touch_gitkeep(path: Path) -> None:
    gitkeep = path / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("Keeps this dataset pipeline directory in Git.\n", encoding="utf-8")


def _safe_dataset_name(dataset_name: str) -> str:
    safe = dataset_name.strip().lower().replace(" ", "_")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in ("_", "-"))
    if not safe:
        raise ValueError("dataset_name must contain at least one alphanumeric character")
    return safe


def _metadata_template(dataset_name: str) -> str:
    today = date.today().isoformat()
    return "\n".join(
        [
            f"dataset_name: {dataset_name}",
            "source: UNKNOWN",
            "official_url: UNKNOWN",
            "version: UNKNOWN",
            "license: UNKNOWN",
            f"download_date: {today}",
            "original_filename: UNKNOWN",
            "checksum_sha256: UNKNOWN",
            "approximate_size: UNKNOWN",
            "",
        ]
    )
