from pathlib import Path

import pytest

from nexa.data import (
    DATASET_METADATA_FIELDS,
    PIPELINE_DIRECTORIES,
    ensure_dataset_directory,
    ensure_dataset_layout,
)


def test_ensure_dataset_layout_creates_required_directories(tmp_path: Path) -> None:
    data_root = tmp_path / "data"

    ensured = ensure_dataset_layout(data_root)

    assert len(ensured) == len(PIPELINE_DIRECTORIES)
    for relative_dir in PIPELINE_DIRECTORIES:
        path = data_root / relative_dir
        assert path.is_dir()
        assert (path / ".gitkeep").is_file()


def test_ensure_dataset_directory_creates_metadata_template(tmp_path: Path) -> None:
    dataset_dir = ensure_dataset_directory("Example Dataset", data_root=tmp_path / "data")

    metadata = dataset_dir / "metadata.yaml"
    assert dataset_dir == tmp_path / "data" / "raw" / "example_dataset"
    assert metadata.is_file()

    text = metadata.read_text(encoding="utf-8")
    for field in DATASET_METADATA_FIELDS:
        assert f"{field}:" in text


def test_ensure_dataset_directory_preserves_existing_metadata(tmp_path: Path) -> None:
    dataset_dir = ensure_dataset_directory("example", data_root=tmp_path / "data")
    metadata = dataset_dir / "metadata.yaml"
    metadata.write_text("dataset_name: curated\nlicense: MIT\n", encoding="utf-8")

    ensure_dataset_directory("example", data_root=tmp_path / "data")

    assert metadata.read_text(encoding="utf-8") == "dataset_name: curated\nlicense: MIT\n"


def test_ensure_dataset_directory_rejects_unknown_stage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stage"):
        ensure_dataset_directory("example", stage="processed", data_root=tmp_path / "data")
