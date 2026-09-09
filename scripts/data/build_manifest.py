"""
Nexa Phase 8.2D - Dataset Manifest Builder
============================================
After collection, generates:
  data/manifests/dataset_manifest.json
  data/manifests/checksums.sha256

Reads _metadata.yaml from each dataset directory and writes a unified
cross-dataset manifest with checksums for every collected file.

Usage:
  python scripts/data/build_manifest.py [--verify]
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASETS = [
    Path("data/raw/wikimedia_english"),
    Path("data/raw/pg19"),
]
MANIFEST_JSON  = Path("data/manifests/dataset_manifest.json")
CHECKSUMS_FILE = Path("data/manifests/checksums.sha256")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metadata(dataset_dir: Path) -> dict | None:
    meta_path = dataset_dir / "_metadata.yaml"
    if not meta_path.exists():
        print(f"WARNING: No _metadata.yaml in {dataset_dir}")
        return None
    with open(meta_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    verify = "--verify" in sys.argv

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "phase": "8.2D",
        "note": (
            "Dataset payloads are NOT tracked by git. "
            "This manifest records metadata and checksums only."
        ),
        "datasets": [],
    }
    checksum_lines: list[str] = []
    all_ok = True

    for dataset_dir in DATASETS:
        if not dataset_dir.exists():
            print(f"WARNING: {dataset_dir} does not exist -- skipping")
            continue

        meta = load_metadata(dataset_dir)
        if not meta:
            continue

        files = meta.get("files", [])
        source_dir = dataset_dir / "source"
        dataset_entry = {
            "dataset_name":      meta.get("dataset_name", dataset_dir.name),
            "source":            meta.get("source", ""),
            "official_url":      meta.get("official_url", ""),
            "version":           meta.get("version", ""),
            "snapshot":          meta.get("snapshot", ""),
            "license":           meta.get("license", ""),
            "license_url":       meta.get("license_url", ""),
            "language":          meta.get("language", ""),
            "selection_method":  meta.get("selection_method", ""),
            "download_date":     meta.get("download_date", ""),
            "phase":             meta.get("phase", ""),
            "files": [],
        }

        for file_rec in files:
            filename   = file_rec.get("filename", "")
            stored_sha = file_rec.get("sha256", "")
            size_bytes = file_rec.get("size_bytes", 0)
            file_path  = source_dir / filename

            file_entry = {
                "filename":    filename,
                "dataset":     dataset_entry["dataset_name"],
                "source_url":  file_rec.get("source_url", ""),
                "size_bytes":  size_bytes,
                "sha256":      stored_sha,
                "download_date": file_rec.get("download_date", ""),
            }

            if verify:
                if not file_path.exists():
                    print(f"  MISSING: {file_path}")
                    file_entry["verify_status"] = "MISSING"
                    all_ok = False
                else:
                    actual_sha = sha256_file(file_path)
                    if actual_sha == stored_sha:
                        print(f"  OK: {filename}")
                        file_entry["verify_status"] = "OK"
                    else:
                        print(f"  CHECKSUM MISMATCH: {filename}")
                        print(f"    expected: {stored_sha}")
                        print(f"    actual:   {actual_sha}")
                        file_entry["verify_status"] = "MISMATCH"
                        all_ok = False
            else:
                if file_path.exists():
                    file_entry["verify_status"] = "not_verified"
                else:
                    file_entry["verify_status"] = "file_not_present"

            dataset_entry["files"].append(file_entry)

            # Add to checksums file
            rel_path = file_path.as_posix().replace("\\", "/")
            checksum_lines.append(f"{stored_sha}  {rel_path}")

        manifest["datasets"].append(dataset_entry)

    # Write manifest JSON
    MANIFEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest written: {MANIFEST_JSON}")

    # Write checksums file
    with open(CHECKSUMS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(checksum_lines) + "\n")
    print(f"Checksums written: {CHECKSUMS_FILE}")

    if verify:
        if all_ok:
            print("\nAll checksums verified OK.")
        else:
            print("\nSome files FAILED verification -- see above.")
            sys.exit(1)


if __name__ == "__main__":
    main()
