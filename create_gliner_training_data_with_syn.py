from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any
import argparse

from engage_nlp.utils.custom_logging import get_logger, setup_logging


logger = get_logger()
setup_logging(verbose=True)


def find_project_root(start: Path) -> Path:
    """Find the repository root by searching upward for pyproject.toml."""

    start = start.resolve()

    for directory in (start, *start.parents):
        if (directory / "pyproject.toml").exists():
            return directory

    raise RuntimeError(
        "Could not locate the ENGAGE project root. "
        f"Started searching from: {start}"
    )


PROJECT_DIR = find_project_root(Path(__file__).resolve().parent)
DEFAULT_INPUT_FILE = (
    PROJECT_DIR / "data/local/ner/synthesized_data_8shot_250_guidelines.json"
)
DEFAULT_OUTPUT_FILE = (
    PROJECT_DIR / "data/local/ner/synthesized_data_filtered_8shot_250_guidelines.json"
)


KEEP_NEGATIVE_EXAMPLES = False

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
    """Parse command-line arguments."""

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
    """Load dictionary records from JSON or JSONL."""

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return [], 0

    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            records = [record for record in parsed if isinstance(record, dict)]
            return records, len(parsed) - len(records)

        if isinstance(parsed, dict):
            return [parsed], 0

        return [], 1

    except json.JSONDecodeError:
        pass

    records: list[dict[str, Any]] = []
    skipped = 0

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

        if not isinstance(record, dict):
            skipped += 1
            logger.warning(
                f"Skipping non-object JSONL value at line {line_number}"
            )
            continue

        records.append(record)

    return records, skipped


def validate_tokens(value: Any) -> list[str] | None:
    """Validate and return a non-empty token list."""

    if not isinstance(value, list) or not value:
        return None

    if not all(isinstance(token, str) and token for token in value):
        return None

    return value


def convert_entity(entity: Any, tokens: list[str]) -> list[Any] | None:
    """Convert one exclusive-end entity span to GLiNER inclusive-end format."""

    if not isinstance(entity, dict):
        return None

    start = entity.get("start_token")
    end = entity.get("end_token")
    label = entity.get("label")

    if not isinstance(start, int):
        return None
    if not isinstance(end, int):
        return None
    if not isinstance(label, str):
        return None
    if label not in ALLOWED_LABELS:
        return None
    if start < 0 or end <= start or end > len(tokens):
        return None

    return [start, end - 1, label]


def spans_overlap(first: list[Any], second: list[Any]) -> bool:
    """Return whether two inclusive GLiNER spans overlap."""

    first_start, first_end, _ = first
    second_start, second_end, _ = second

    return first_start <= second_end and second_start <= first_end


def convert_record(
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, int]:
    """Convert one synthetic corpus record to GLiNER format."""

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

        if converted is None:
            skipped_entities += 1
            continue

        if converted in converted_entities:
            skipped_entities += 1
            continue

        if any(
            spans_overlap(converted, existing)
            for existing in converted_entities
        ):
            skipped_entities += 1
            continue

        converted_entities.append(converted)

    converted_entities.sort(key=lambda span: (span[0], span[1], span[2]))

    if not converted_entities and not KEEP_NEGATIVE_EXAMPLES:
        return None, skipped_entities

    return {
        "tokenized_text": tokens,
        "ner": converted_entities,
    }, skipped_entities


def record_key(record: dict[str, Any]) -> str:
    """Create a stable exact-duplicate key."""

    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> None:
    logger.info(f"Project root: {PROJECT_DIR}")
    logger.info(f"Input file: {INPUT_FILE}")
    logger.info(f"Output file: {OUTPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Synthetic dataset was not found at the expected location:\n"
            f"{INPUT_FILE}\n\n"
            "Run the synthetic-data generator first or update INPUT_FILE."
        )

    records, malformed_input_records = load_records(INPUT_FILE)
    logger.info(f"Loaded {len(records)} synthetic records")

    dataset: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    seen_records: set[str] = set()

    invalid_records = 0
    duplicate_records = 0
    empty_records = 0
    malformed_entities = 0

    for record in records:
        converted, skipped_entities = convert_record(record)
        malformed_entities += skipped_entities

        if converted is None:
            raw_entities = record.get("entities", [])

            if isinstance(raw_entities, list) and not raw_entities:
                empty_records += 1
            else:
                invalid_records += 1

            continue

        key = record_key(converted)

        if key in seen_records:
            duplicate_records += 1
            continue

        seen_records.add(key)
        dataset.append(converted)

        for _, _, label in converted["ner"]:
            label_counts[label] += 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(dataset)} GLiNER examples")
    logger.info(f"Malformed input records: {malformed_input_records}")
    logger.info(f"Invalid records: {invalid_records}")
    logger.info(f"Empty records skipped: {empty_records}")
    logger.info(f"Duplicate records skipped: {duplicate_records}")
    logger.info(f"Malformed entities skipped: {malformed_entities}")
    logger.info(f"Label counts: {dict(label_counts)}")

    print(f"\nProject root: {PROJECT_DIR}")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Examples: {len(dataset)}")
    print(f"Entity mentions: {sum(label_counts.values())}")


if __name__ == "__main__":
    main()