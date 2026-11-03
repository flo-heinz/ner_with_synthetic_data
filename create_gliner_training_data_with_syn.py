
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from engage_nlp.utils.custom_logging import get_logger, setup_logging


logger = get_logger()
setup_logging(verbose=True)


def find_project_root(start: Path) -> Path:
    """
    Find the repository root by searching upward for pyproject.toml.

    This allows file paths to be resolved relative to the project root,
    regardless of where the script is executed.
    """
    start = start.resolve()

    for directory in (start, *start.parents):
        if (directory / "pyproject.toml").exists():
            return directory

    raise RuntimeError(
        "Could not locate the ENGAGE project root. "
        f"Started searching from: {start}"
    )


# Resolve the project root and define default dataset locations.
PROJECT_DIR = find_project_root(Path(__file__).resolve().parent)

DEFAULT_INPUT_FILE = (
    PROJECT_DIR / "data/local/ner/synthesized_data_8shot_250_guidelines.json"
)
DEFAULT_OUTPUT_FILE = (
    PROJECT_DIR / "data/local/ner/synthesized_data_filtered_8shot_250_guidelines.json"
)


# Whether to retain examples without any valid entity annotations.
KEEP_NEGATIVE_EXAMPLES = False

# Only entities belonging to these labels are included in the output dataset.
ALLOWED_LABELS = {
    "biogas_energy",
    "biomass_and_biofuel",
    "ecosystem_service_process_function",
    "geographical_location",
    "geothermal_energy",
    "habitat_ecosystem",
    "hydropower_energy",
    "multiple_renewable_energy_sources",
    "organism_common_name",
    "organism_scientific_name",
    "scientific_method",
    "solar_photovoltaic_energy",
    "temporal_information",
    "tidal_energy",
    "wave_energy",
    "wind_energy",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for input and output dataset paths."""

    parser = argparse.ArgumentParser(
        description="Convert synthetic NER data to GLiNER format."
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Input JSON or JSONL file (default: {DEFAULT_INPUT_FILE})",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT_FILE})",
    )

    return parser.parse_args()


def load_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    """
    Load synthetic NER records from a JSON or JSONL file.

    Supports a JSON list, a single JSON object, or newline-delimited
    JSON objects. Malformed records are skipped and counted.

    Returns:
        A tuple containing valid dictionary records and the number
        of malformed or unsupported records.
    """
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return [], 0

    # First attempt to parse the entire file as standard JSON.
    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            records = [record for record in parsed if isinstance(record, dict)]
            return records, len(parsed) - len(records)

        if isinstance(parsed, dict):
            return [parsed], 0

        return [], 1

    except json.JSONDecodeError:
        # If standard JSON parsing fails, try reading the file as JSONL.
        pass

    records: list[dict[str, Any]] = []
    skipped = 0

    # Parse JSONL records independently so malformed lines do not
    # prevent the remaining records from being processed.
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            skipped += 1
            logger.warning(
                f"Skipping malformed JSONL line {line_number}: {error}"
            )
            continue

        # Each record must be a JSON object containing the example data.
        if not isinstance(record, dict):
            skipped += 1
            logger.warning(
                f"Skipping non-object JSONL value at line {line_number}"
            )
            continue

        records.append(record)

    return records, skipped


def validate_tokens(value: Any) -> list[str] | None:
    """
    Validate the token sequence of an input example.

    A valid sequence must be a non-empty list of non-empty strings.
    """
    if not isinstance(value, list) or not value:
        return None

    if not all(isinstance(token, str) and token for token in value):
        return None

    return value


def convert_entity(entity: Any, tokens: list[str]) -> list[Any] | None:
    """
    Validate and convert an entity annotation to GLiNER format.

    Input spans use exclusive end indices: [start, end).
    GLiNER expects inclusive end indices: [start, end - 1].

    Entities with invalid indices or unsupported labels are discarded.
    """
    if not isinstance(entity, dict):
        return None

    start = entity.get("start_token")
    end = entity.get("end_token")
    label = entity.get("label")

    # Check that the annotation contains the expected field types.
    if not isinstance(start, int):
        return None
    if not isinstance(end, int):
        return None
    if not isinstance(label, str):
        return None

    # Exclude labels that are not part of the target NER schema.
    if label not in ALLOWED_LABELS:
        return None

    # Ensure that the entity span is non-empty and within token bounds.
    if start < 0 or end <= start or end > len(tokens):
        return None

    # Convert the exclusive end index to GLiNER's inclusive format.
    return [start, end - 1, label]


def spans_overlap(first: list[Any], second: list[Any]) -> bool:
    """
    Check whether two inclusive entity spans overlap.

    Spans sharing at least one token are considered overlapping.
    """
    first_start, first_end, _ = first
    second_start, second_end, _ = second

    return first_start <= second_end and second_start <= first_end


def convert_record(
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, int]:
    """
    Convert one synthetic NER example into GLiNER training format.

    Invalid, duplicate, and overlapping entities are removed.
    Examples without valid entities are optionally discarded.

    Returns:
        The converted record (or None) and the number of skipped entities.
    """
    tokens = validate_tokens(record.get("tokens"))

    if tokens is None:
        return None, 0

    raw_entities = record.get("entities", [])

    if not isinstance(raw_entities, list):
        return None, 0

    converted_entities: list[list[Any]] = []
    skipped_entities = 0

    for raw_entity in raw_entities:
        converted = convert_entity(raw_entity, tokens)

        # Discard malformed entities or annotations with unsupported labels.
        if converted is None:
            skipped_entities += 1
            continue

        # Avoid including the same entity annotation more than once.
        if converted in converted_entities:
            skipped_entities += 1
            continue

        # GLiNER training examples should not contain overlapping spans.
        # Keep the first accepted span and discard subsequent overlaps.
        if any(
            spans_overlap(converted, existing)
            for existing in converted_entities
        ):
            skipped_entities += 1
            continue

        converted_entities.append(converted)

    # Sort entities by their token positions for consistent output.
    converted_entities.sort(key=lambda span: (span[0], span[1], span[2]))

    # Optionally remove negative examples with no valid entity annotations.
    if not converted_entities and not KEEP_NEGATIVE_EXAMPLES:
        return None, skipped_entities

    # Construct the token-based representation expected by GLiNER.
    return {
        "tokenized_text": tokens,
        "ner": converted_entities,
    }, skipped_entities


def record_key(record: dict[str, Any]) -> str:
    """
    Generate a deterministic key for exact duplicate detection.

    Sorting dictionary keys ensures that equivalent records produce
    the same serialized representation regardless of key order.
    """
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> None:
    """Run the synthetic NER dataset conversion and filtering pipeline."""

    # Read the configured input and output paths.
    args = parse_args()
    input_file = args.input_file
    output_file = args.output_file

    logger.info(f"Project root: {PROJECT_DIR}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")

    # Verify that the synthetic dataset exists before processing.
    if not input_file.exists():
        raise FileNotFoundError(
            "Synthetic dataset was not found at the expected location:\n"
            f"{input_file}\n\n"
            "Run the synthetic-data generator first or update --input-file."
        )

    # Load input records, keeping track of malformed JSON entries.
    records, malformed_input_records = load_records(input_file)
    logger.info(f"Loaded {len(records)} synthetic records")

    # Store converted examples and track entity label frequencies.
    dataset: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    seen_records: set[str] = set()

    # Counters for records and entities removed during filtering.
    invalid_records = 0
    duplicate_records = 0
    empty_records = 0
    malformed_entities = 0

    for record in records:
        # Validate the example and convert its entity spans.
        converted, skipped_entities = convert_record(record)
        malformed_entities += skipped_entities

        if converted is None:
            raw_entities = record.get("entities", [])

            # Distinguish records with no original annotations from
            # other records rejected during conversion.
            if isinstance(raw_entities, list) and not raw_entities:
                empty_records += 1
            else:
                invalid_records += 1

            continue

        # Remove exact duplicates after conversion and filtering.
        key = record_key(converted)

        if key in seen_records:
            duplicate_records += 1
            continue

        seen_records.add(key)
        dataset.append(converted)

        # Count retained entity mentions for each NER label.
        for _, _, label in converted["ner"]:
            label_counts[label] += 1

    # Create the output directory if it does not already exist.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save the filtered dataset as a formatted JSON array.
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=2)

    # Report dataset size and filtering statistics.
    logger.info(f"Saved {len(dataset)} GLiNER examples")
    logger.info(f"Malformed input records: {malformed_input_records}")
    logger.info(f"Invalid records: {invalid_records}")
    logger.info(f"Empty records skipped: {empty_records}")
    logger.info(f"Duplicate records skipped: {duplicate_records}")
    logger.info(f"Malformed entities skipped: {malformed_entities}")
    logger.info(f"Label counts: {dict(label_counts)}")

    # Print a short summary for command-line users.
    print(f"\nProject root: {PROJECT_DIR}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Examples: {len(dataset)}")
    print(f"Entity mentions: {sum(label_counts.values())}")


if __name__ == "__main__":
    main()
