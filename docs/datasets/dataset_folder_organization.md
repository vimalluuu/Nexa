# Nexa Dataset Folder Organization

Nexa keeps each dataset pipeline stage separate:

```text
data/
├── raw/
├── cleaned/
├── deduplicated/
├── splits/
│   ├── train/
│   └── validation/
├── processed/
│   ├── train/
│   └── validation/
├── manifests/
└── reports/
```

Each independently sourced dataset must remain identifiable. Raw source files
must never be mixed directly in `data/raw/`; they belong in source-specific
folders such as `data/raw/gutenberg/` or `data/raw/wikipedia_simple/`.

## Required Metadata

Every independently sourced dataset directory must contain `metadata.yaml`
with these fields:

```yaml
dataset_name: UNKNOWN
source: UNKNOWN
official_url: UNKNOWN
version: UNKNOWN
license: UNKNOWN
download_date: UNKNOWN
original_filename: UNKNOWN
checksum_sha256: UNKNOWN
approximate_size: UNKNOWN
```

Use `UNKNOWN` only when the information cannot be verified yet. Raw downloads
should record the checksum and original filename before any cleaning occurs.

## Pipeline Rules

- Keep `raw`, `cleaned`, `deduplicated`, `splits`, and `processed` as separate stages.
- Never overwrite original raw data.
- Never modify downloaded raw files in place.
- Never delete raw data automatically.
- Preserve dataset identity in `cleaned/` and, where practical, `deduplicated/`.
- Combine approved datasets only after cleaning, validation, and license review.
- Write reports under `data/reports/`.
- Write manifests under `data/manifests/`.
- Do not commit large datasets to GitHub.
- Track only metadata, manifests, scripts, documentation, and small permitted sample files.

## Maintenance

Run this command whenever a new checkout or dataset workspace needs the standard
folders:

```bash
python scripts/ensure_dataset_structure.py
```

To create a metadata template for a new independently sourced dataset:

```bash
python scripts/ensure_dataset_structure.py --dataset-name dataset_name_1 --stage raw
```

The script is repeatable. It creates missing folders and missing metadata
templates, but it does not delete data or overwrite existing metadata.
