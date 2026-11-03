from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from engage_nlp.utils.custom_logging import get_logger, setup_logging


logger = get_logger()
setup_logging(verbose=True)


project_dir = Path(__file__).resolve().parents[5]
logger.info(f"Project directory: {project_dir}")

DEFAULT_INPUT_DIR = (
    project_dir / "data/local/ner/outputs/engage-ft-auto-filtered"
)
DEFAULT_OUTPUT_FILE = (
    project_dir / "data/local/ner/gliner_training_data_filtered/data.json"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Convert annotated NER files to GLiNER training format."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing input JSON files.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to the output GLiNER JSON file.",
    )

    return parser.parse_args()


def load_sentences(path: Path) -> list[dict[str, Any]]:
    """
    Load sentence records from JSON or JSONL.

    Supported input formats:
    - a JSON list of dictionaries
    - a single JSON dictionary
    - JSONL with one dictionary per line
    """

    text = path.read_text(encoding="utf8").strip()

    if not text:
        logger.warning(f"Skipping empty file: {path}")
        return []

    try:
        data = json.loads(text)

        if isinstance(data, list):
            return [
                item
                for item in data
                if isinstance(item, dict)
            ]

        if isinstance(data, dict):
            return [data]

        logger.warning(
            f"Skipping {path}: unsupported JSON root type {type(data)}"
        )
        return []

    except json.JSONDecodeError:
        # The file may be JSONL.
        pass

    sentences: list[dict[str, Any]] = []

    for line_no, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning(
                f"Skipping malformed JSONL line in {path}, "
                f"line {line_no}: {exc}"
            )
            continue

        if not isinstance(item, dict):
            logger.warning(
                f"Skipping non-dict JSONL item in {path}, "
                f"line {line_no}: {type(item)}"
            )
            continue

        sentences.append(item)

    return sentences


def format_annotation_for_gliner(
    annotated_sentence: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Convert one annotated sentence into GLiNER training format.

    The input end_token value is assumed to be exclusive.
    GLiNER expects the end token index to be inclusive.
    """

    tokens = annotated_sentence.get("tokens")
    entities = annotated_sentence.get("entities", [])

    if not tokens:
        return None

    if not isinstance(tokens, list):
        logger.warning(
            f"Skipping sentence with non-list tokens: {tokens}"
        )
        return None

    if not isinstance(entities, list):
        logger.warning(
            f"Treating non-list entities as empty: {entities}"
        )
        entities = []

    result: dict[str, Any] = {
        "tokenized_text": tokens,
        "ner": [],
    }

    for annotation in entities:
        if not isinstance(annotation, dict):
            logger.warning(
                f"Skipping non-dict entity: {annotation}"
            )
            continue

        start_token = annotation.get("start_token")
        end_token_exclusive = annotation.get("end_token")
        label = annotation.get("label")

        if (
            start_token is None
            or end_token_exclusive is None
            or label is None
        ):
            logger.warning(
                f"Skipping malformed entity: {annotation}"
            )
            continue

        if (
            not isinstance(start_token, int)
            or not isinstance(end_token_exclusive, int)
        ):
            logger.warning(
                f"Skipping entity with non-integer span: {annotation}"
            )
            continue

        if start_token < 0:
            logger.warning(
                f"Skipping entity with negative start: {annotation}"
            )
            continue

        if end_token_exclusive <= start_token:
            logger.warning(
                f"Skipping invalid span: {annotation}"
            )
            continue

        end_token_inclusive = end_token_exclusive - 1

        if end_token_inclusive >= len(tokens):
            logger.warning(
                f"Skipping out-of-range span {annotation} "
                f"for token length {len(tokens)}"
            )
            continue

        result["ner"].append(
            [
                start_token,
                end_token_inclusive,
                str(label),
            ]
        )

    return result


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    output_file = args.output_file

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output file: {output_file}")

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    files = sorted(input_dir.glob("*.json"))

    if not files:
        logger.warning(
            f"No JSON files found in {input_dir}"
        )
        return

    logger.info(
        f"Found {len(files)} JSON files in {input_dir}"
    )

    all_gliner_data: list[dict[str, Any]] = []
    total_label_counts: Counter[str] = Counter()

    total_sentences = 0
    total_skipped = 0
    total_positive = 0
    total_negative = 0

    for input_path in files:
        sentences = load_sentences(input_path)

        file_sentences = 0
        file_skipped = 0
        file_positive = 0
        file_negative = 0

        for sentence in sentences:
            total_sentences += 1
            file_sentences += 1

            formatted = format_annotation_for_gliner(sentence)

            if formatted is None:
                total_skipped += 1
                file_skipped += 1
                continue

            if not formatted["ner"]:
                total_negative += 1
                file_negative += 1

                # Preserve the original behavior:
                # negative examples are not added to the output dataset.
                continue

            all_gliner_data.append(formatted)

            total_positive += 1
            file_positive += 1

            for _, _, label in formatted["ner"]:
                total_label_counts[label] += 1

        logger.info(
            f"Processed {input_path.name}: "
            f"sentences={file_sentences}, "
            f"kept={file_positive}, "
            f"negatives={file_negative}, "
            f"skipped={file_skipped}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open(
        "w",
        encoding="utf8",
    ) as file:
        json.dump(
            all_gliner_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"Processed {total_sentences} sentences")
    logger.info(f"Skipped {total_skipped} sentences with missing or invalid tokens")
    logger.info(f"Found {total_negative} negative sentences with no valid entities")
    logger.info(f"Kept {total_positive} sentences with at least one entity")
    logger.info(f"Found {len(total_label_counts)} labels")
    logger.info(f"Label counts: {dict(total_label_counts)}")
    logger.info(f"Saved combined GLiNER dataset to {output_file}")

    if all_gliner_data:
        print("\nExample:")
        print(
            json.dumps(
                all_gliner_data[0],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("No GLiNER examples were created.")

    print(f"\nOutput file: {output_file}")
    print(f"Number of examples: {len(all_gliner_data)}")


if __name__ == "__main__":
    main()
