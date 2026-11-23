
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Remove gold-standard publications from ENGAGE auto-annotated training data.

The pipeline loads gold-standard publication DOIs from a Hugging Face dataset,
reads JSON or JSONL training records, removes records whose DOIs occur in the
gold-standard dataset, and verifies that no shared publications remain.

The original training files are preserved. The script refuses to overwrite
existing output directories and raises errors for invalid records or missing
DOIs.

Example usage:
    python remove_gold_dois.py

    python remove_gold_dois.py \
        --gold gold-standard_subset \
        --train engage-ft-auto-preprocessed \
        --output engage-ft-auto-preprocessed-doi-clean
"""

from argparse import ArgumentParser
from collections import Counter
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote


DEFAULT_ORIGINAL = (
    "engage-ft-auto-preprocessed--engage_251111_ex--"
    "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8-tag-prompt_v2-ex0-fz10--sentence"
)

METADATA_NAMES = {
    "dataset_info.json",
    "state.json",
    "dataset_dict.json",
}


# Configure input datasets and output directory.
def set_args():
    """Parse command-line arguments for dataset paths and output settings.

    Returns:
        Parsed command-line arguments.
    """
    ap = ArgumentParser()

    ap.add_argument(
        "--gold",
        type=Path,
        default=Path("gold-standard_subset"),
    )

    ap.add_argument(
        "--train",
        type=Path,
        default=Path(DEFAULT_ORIGINAL),
    )

    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("engage-ft-auto-preprocessed-doi-clean"),
    )

    return ap.parse_args()


# Normalize DOI strings for consistent publication matching.
def normalize_doi(value):
    """Normalize a DOI by removing prefixes and standardizing its casing.

    Args:
        value: DOI string, optionally containing a URL or DOI prefix.

    Returns:
        Normalized DOI string, or None if the value is invalid or empty.
    """
    if not isinstance(value, str):
        return None

    doi = unquote(value.strip()).lower()

    doi = re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)",
        "",
        doi,
    )

    doi = doi.strip().rstrip(".,; ")

    return doi or None


# Read training examples from JSON arrays, objects, or JSONL files.
def load_json_or_jsonl(path):
    """Load records from a JSON or JSONL file.

    Args:
        path: Path to the input JSON or JSONL file.

    Returns:
        A pair containing the loaded records and their original format.

    Raises:
        ValueError: If the file is empty or contains invalid JSON data.
    """
    content = path.read_text(encoding="utf-8-sig")

    if not content.strip():
        raise ValueError(
            f"Empty JSON/JSONL file: {path}"
        )

    # Attempt to parse the entire file as JSON before trying JSONL.
    try:
        obj = json.loads(content)

    except json.JSONDecodeError as whole_error:
        records = []

        for number, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as line_error:
                raise ValueError(
                    f"Cannot parse {path} as JSON ({whole_error}) "
                    f"or JSONL (line {number}: {line_error})"
                ) from line_error

        if not records:
            raise ValueError(
                f"No JSONL records in {path}"
            )

        return records, "jsonl"

    if isinstance(obj, list):
        return obj, "array"

    if isinstance(obj, dict):
        return [obj], "object"

    raise ValueError(
        f"Unsupported JSON root in {path}: {type(obj).__name__}"
    )


# Extract and validate the DOI associated with a dataset record.
def get_doi(record, path, index):
    """Retrieve a normalized DOI from a dataset record.

    Args:
        record: Dataset record containing a DOI field.
        path: Source file or dataset path.
        index: Position of the record in its source.

    Returns:
        Normalized DOI string.

    Raises:
        ValueError: If the record is invalid or its DOI is missing.
    """
    if not isinstance(record, dict):
        raise ValueError(
            f"Non-object record in {path}, record {index}"
        )

    doi = normalize_doi(record.get("doi"))

    if not doi:
        raise ValueError(
            f"Missing/empty DOI in {path}, record {index}"
        )

    return doi


# Load publication DOIs from the gold-standard Hugging Face dataset.
def load_gold_dois(path):
    """Load and count DOIs from a gold-standard dataset.

    Args:
        path: Path to a saved Hugging Face dataset or Arrow file.

    Returns:
        Counter mapping normalized DOIs to their record frequencies.

    Raises:
        FileNotFoundError: If the gold-standard dataset does not exist.
        ValueError: If the dataset format is unsupported or contains no DOIs.
    """
    from datasets import Dataset, DatasetDict, load_from_disk

    if not path.exists():
        raise FileNotFoundError(
            f"Gold dataset not found: {path}"
        )

    if path.is_dir():
        gold = load_from_disk(str(path))

    elif path.suffix.lower() == ".arrow":
        gold = Dataset.from_file(str(path))

    else:
        raise ValueError(
            "Gold dataset must be a saved Hugging Face "
            "folder or .arrow file"
        )

    # Process every split when the gold dataset contains multiple splits.
    parts = (
        gold.values()
        if isinstance(gold, DatasetDict)
        else [gold]
    )

    counts = Counter()

    for part in parts:
        for index, record in enumerate(part, 1):
            counts[get_doi(record, path, index)] += 1

    if not counts:
        raise ValueError(
            "Gold dataset contains no DOIs"
        )

    return counts


# Discover training JSON and JSONL files while excluding metadata.
def training_files(folder):
    """Find all training data files in a directory.

    Args:
        folder: Root directory containing training files.

    Returns:
        Sorted list of JSON and JSONL file paths.

    Raises:
        FileNotFoundError: If the training directory does not exist.
        ValueError: If no training files are found.
    """
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Training folder not found: {folder}"
        )

    files = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".jsonl"}
        and path.name not in METADATA_NAMES
    )

    if not files:
        raise ValueError(
            f"No training JSON/JSONL files found in {folder}"
        )

    return files


# Write filtered records while preserving their original JSON format.
def write_records(path, records, fmt):
    """Save dataset records as JSON or JSONL.

    Args:
        path: Destination file path.
        records: Dataset records to write.
        fmt: Original format ('jsonl', 'array', or 'object').

    Raises:
        ValueError: If an object-format file contains multiple records.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as file:
        if fmt == "jsonl":
            for record in records:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        elif fmt == "object":
            if len(records) != 1:
                raise ValueError(
                    f"Unexpected object record count for {path}"
                )

            json.dump(
                records[0],
                file,
                ensure_ascii=False,
                indent=2,
            )

            file.write("\n")

        else:
            json.dump(
                records,
                file,
                ensure_ascii=False,
                indent=2,
            )

            file.write("\n")


# Count publication occurrences and validate all training records.
def count_training_dois(files):
    """Validate training records and count their normalized DOIs.

    Args:
        files: Paths to training JSON and JSONL files.

    Returns:
        A pair containing DOI frequencies and original file formats.
    """
    counts = Counter()
    formats = {}

    for path in files:
        records, fmt = load_json_or_jsonl(path)
        formats[path] = fmt

        for index, record in enumerate(records, 1):
            counts[get_doi(record, path, index)] += 1

    return counts, formats


# Remove records whose publication DOIs occur in the gold-standard dataset.
def filter_training_files(
    files,
    train_path,
    output_path,
    gold_counts,
    formats,
):
    """Filter training records and write the remaining examples.

    Args:
        files: Input training file paths.
        train_path: Root directory of the original training data.
        output_path: Destination directory for filtered training data.
        gold_counts: Counter containing gold-standard publication DOIs.
        formats: Mapping of input files to their original JSON formats.

    Returns:
        Counter containing the DOIs of retained training records.
    """
    final_counts = Counter()

    for path in files:
        records, _ = load_json_or_jsonl(path)
        kept = []

        for index, record in enumerate(records, 1):
            doi = get_doi(record, path, index)

            if doi not in gold_counts:
                kept.append(record)
                final_counts[doi] += 1

        # Preserve the relative file structure and original serialization.
        if kept:
            destination = output_path / path.relative_to(train_path)

            write_records(
                destination,
                kept,
                formats[path],
            )

    return final_counts


# Preserve metadata and other non-training files in the output directory.
def copy_metadata_files(train_path, output_path, formats):
    """Copy files that are not part of the training JSON/JSONL datasets.

    Args:
        train_path: Root directory of the original training data.
        output_path: Destination directory for filtered training data.
        formats: Mapping identifying processed training files.
    """
    for path in sorted(train_path.rglob("*")):
        if not path.is_file() or path in formats:
            continue

        if (
            path.suffix.lower() in {".json", ".jsonl"}
            and path.name not in METADATA_NAMES
        ):
            continue

        destination = output_path / path.relative_to(train_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(path, destination)


# Verify that the saved training data contains no gold-standard publications.
def verify_output(
    output_path,
    gold_counts,
    original_counts,
    final_counts,
    removed_records,
):
    """Validate the filtered training files against expected DOI counts.

    Args:
        output_path: Directory containing the filtered training files.
        gold_counts: Counter containing gold-standard publication DOIs.
        original_counts: DOI frequencies before filtering.
        final_counts: Expected DOI frequencies after filtering.
        removed_records: Expected number of removed training records.

    Returns:
        Counter containing DOI frequencies read from saved output files.

    Raises:
        RuntimeError: If saved records differ from expected results or
            contain gold-standard publications.
    """
    verified_counts = Counter()

    # Independently reload the saved files to verify the actual output.
    for path in training_files(output_path):
        records, _ = load_json_or_jsonl(path)

        for index, record in enumerate(records, 1):
            verified_counts[get_doi(record, path, index)] += 1

    if verified_counts != final_counts:
        raise RuntimeError(
            "Output verification failed: "
            "saved counts differ from expected counts"
        )

    if set(verified_counts) & set(gold_counts):
        raise RuntimeError(
            "Output verification failed: shared DOIs remain"
        )

    if (
        sum(original_counts.values())
        - sum(verified_counts.values())
        != removed_records
    ):
        raise RuntimeError(
            "Output verification failed: "
            "record counts do not reconcile"
        )

    return verified_counts


# Save a CSV audit report of publications removed from training.
def write_removed_dois_report(
    output_path,
    overlap,
    gold_counts,
    original_counts,
):
    """Write a CSV report listing removed publication DOIs.

    Args:
        output_path: Directory containing the filtered training data.
        overlap: Set of DOIs shared by gold and training datasets.
        gold_counts: DOI frequencies in the gold-standard dataset.
        original_counts: DOI frequencies in the original training data.

    Returns:
        Path to the generated CSV report.
    """
    report = (
        output_path.parent
        / f"{output_path.name}_removed_dois.csv"
    )

    with report.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "doi",
            "gold_records",
            "removed_training_records",
        ])

        for doi in sorted(overlap):
            writer.writerow([
                doi,
                gold_counts[doi],
                original_counts[doi],
            ])

    return report


def main():
    """Run DOI-based filtering, output verification, and audit reporting."""
    args = set_args()

    gold_path = args.gold.resolve()
    train_path = args.train.resolve()
    output_path = args.output.resolve()

    # Refuse existing output directories and nested input/output paths.
    if output_path.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {output_path}"
        )

    if (
        output_path == train_path
        or train_path in output_path.parents
        or output_path in train_path.parents
    ):
        raise ValueError(
            "Output and input folders must be separate, not nested"
        )

    # Load gold-standard DOIs and discover training data files.
    print(f"Reading gold dataset: {gold_path}")

    gold_counts = load_gold_dois(gold_path)
    files = training_files(train_path)

    print(
        f"Gold records: {sum(gold_counts.values()):,}; "
        f"gold publications: {len(gold_counts):,}"
    )

    print(f"Reading {len(files):,} training JSON/JSONL files...")

    # Validate all training records before creating the output directory.
    original_counts, formats = count_training_dois(files)

    overlap = set(original_counts) & set(gold_counts)

    removed_records = sum(
        original_counts[doi]
        for doi in overlap
    )

    print("\nBEFORE FILTERING")
    print(f"  Training records: {sum(original_counts.values()):,}")
    print(f"  Training publications: {len(original_counts):,}")
    print(f"  Shared publications: {len(overlap):,}")
    print(f"  Training records to remove: {removed_records:,}")

    # Create the output directory only after successful input validation.
    output_path.mkdir(parents=True, exist_ok=False)

    try:
        # Filter overlapping publications and preserve auxiliary files.
        final_counts = filter_training_files(
            files,
            train_path,
            output_path,
            gold_counts,
            formats,
        )

        copy_metadata_files(
            train_path,
            output_path,
            formats,
        )

        # Independently verify saved records and publication counts.
        verified_counts = verify_output(
            output_path,
            gold_counts,
            original_counts,
            final_counts,
            removed_records,
        )

        # Generate the CSV report of removed publication DOIs.
        report = write_removed_dois_report(
            output_path,
            overlap,
            gold_counts,
            original_counts,
        )

    except Exception:
        print(
            f"ERROR: Output at {output_path} may be incomplete; "
            "inspect before reuse.",
            file=sys.stderr,
        )
        raise

    print("\nAFTER FILTERING — VERIFIED FROM OUTPUT FILES")
    print(f"  Training records: {sum(verified_counts.values()):,}")
    print(f"  Training publications: {len(verified_counts):,}")
    print(f"  Removed publications: {len(overlap):,}")
    print(f"  Removed training records: {removed_records:,}")
    print(f"  Output folder: {output_path}")
    print(f"  Removed DOI report: {report}")
    print(
        "PASS: No shared DOIs remain. "
        "Original training files were not changed."
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
