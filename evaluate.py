#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation script for NER predictions using config-based approach.
"""

from pathlib import Path
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from datasets import load_from_disk
import evaluate
import spacy
from dotenv import load_dotenv

from engage_nlp.components.ner.annotations import (
    AnnotationConfig,
    convert_annotated_document_to_iob_tags,
    convert_label_studio_annotations_to_annotated_spacy_doc,
    convert_spacy_doc_to_inline_xml,
    convert_spacy_doc_to_iob,
    get_annotation_config,
)
from engage_nlp.components.ner.config import NERConfig
from engage_nlp.components.ner.constants import annotator_map
from engage_nlp.components.ner.model_utils.gliner import GlinerAnnotator
from engage_nlp.components.ner.model_utils.llm_ner import LLMAnnotator
from engage_nlp.components.ner.utils import infer_eval_results_file
from engage_nlp.cli import build_base_parser
from engage_nlp.config import load_config
from engage_nlp.utils.custom_logging import get_logger, setup_logging


load_dotenv()

nlp = spacy.load("en_core_web_sm", exclude=["tokenizer"])
logger = get_logger()

seed = 42
np.random.seed(seed)


def resolve_config_paths(config: NERConfig, config_path: str | Path) -> NERConfig:
    config_path = Path(config_path).resolve()

    try:
        repo_root = config_path.parents[5]
    except IndexError:
        repo_root = Path.cwd()

    for field in ["schema", "input_path", "output_dir"]:
        value = getattr(config, field, None)
        if not value:
            continue

        path_value = Path(value)
        if not path_value.is_absolute():
            setattr(config, field, str((repo_root / path_value).resolve()))

    return config


def compute_metrics(
    predictions: list[list[tuple[str, str]]],
    references: list[list[tuple[str, str]]],
) -> pd.DataFrame:
    pred_labels = [[label for _, label in seq] for seq in predictions]
    ref_labels = [[label for _, label in seq] for seq in references]

    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(
        predictions=pred_labels,
        references=ref_labels,
        zero_division=0,
    )

    return pd.DataFrame.from_dict(results).T


def prepare_dataset_for_evaluation(
    df: pd.DataFrame,
    level: str = "sentence",
    config: AnnotationConfig | None = None,
) -> pd.DataFrame:
    if level == "sentence":
        df["doc"] = df["doc"].apply(lambda doc: list(doc.sents))
        df = df.explode("doc").reset_index(drop=True)
    elif level == "section":
        pass
    elif level == "paragraph":
        raise NotImplementedError("Paragraph level processing is not implemented yet.")
    else:
        raise ValueError(f"Unsupported level: {level}")

    df["source"] = df["doc"].apply(lambda doc: doc.text)
    df["target"] = df["doc"].apply(
        lambda doc: convert_spacy_doc_to_inline_xml(doc, config=config)
    )
    df["entity_set"] = df["doc"].apply(
        lambda doc: set(sorted(ent.label_ for ent in doc.ents))
    )
    df["iob"] = df["doc"].apply(convert_spacy_doc_to_iob)

    return df


def replace_empty_prediction_with_placeholder(
    target_iob: list[tuple[str, str]],
    placeholder: str = "O",
) -> list[tuple[str, str]]:
    return [(token, placeholder) for token, _ in target_iob]


def normalize_batch_output(batch_output: Any) -> list[Any]:
    if batch_output is None:
        return []

    if isinstance(batch_output, list):
        return batch_output

    if isinstance(batch_output, tuple):
        return list(batch_output)

    if isinstance(batch_output, Iterable) and not isinstance(
        batch_output, (str, bytes, dict)
    ):
        return list(batch_output)

    return [batch_output]


def count_non_o_tags(seq: list[tuple[str, str]]) -> int:
    return sum(1 for _, tag in seq if tag != "O")


def validate_token_alignment(
    predictions: list[list[tuple[str, str]]],
    references: list[list[tuple[str, str]]],
) -> None:
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions/reference length mismatch: "
            f"{len(predictions)} != {len(references)}"
        )

    for i, (pred_seq, ref_seq) in enumerate(zip(predictions, references)):
        pred_tokens = [tok for tok, _ in pred_seq]
        ref_tokens = [tok for tok, _ in ref_seq]

        if len(pred_tokens) != len(ref_tokens):
            raise ValueError(
                f"Token length mismatch at example {i}: "
                f"{len(pred_tokens)} != {len(ref_tokens)}"
            )

        if pred_tokens != ref_tokens:
            for j, (p, r) in enumerate(zip(pred_tokens, ref_tokens)):
                if p != r:
                    raise ValueError(
                        f"Token mismatch at example {i}, position {j}: "
                        f"pred={p!r}, ref={r!r}"
                    )

            raise ValueError(f"Token mismatch at example {i}")


def get_annotator_from_config(config: NERConfig) -> Any:
    model_path = Path(config.model).expanduser().resolve()
    annotator_class = annotator_map.get(config.model)

    if annotator_class is None:
        if model_path.exists():
            logger.info(f"Detected local GLiNER model path: {model_path}")
            config.model = str(model_path)
            annotator_class = GlinerAnnotator
        else:
            raise ValueError(f"Model '{config.model}' is not supported.")

    logger.info(f"Setting up NER model: {config.model} on device {config.device}")
    return annotator_class.from_config(config)


def run_predictions(
    annotator: Any,
    test_df: pd.DataFrame,
    config: NERConfig,
) -> list[list[tuple[str, str]]]:
    predictions: list[list[tuple[str, str]]] = []
    texts = test_df["source"].tolist()

    if isinstance(annotator, LLMAnnotator) and config.batch_size > 1:
        logger.info(
            f"Using batch processing with batch_size={config.batch_size}, "
            f"max_concurrent={config.max_concurrent}"
        )

        for batch_start in tqdm(
            range(0, len(texts), config.batch_size),
            desc="Processing batches",
        ):
            batch_end = min(batch_start + config.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]

            batch_preds = normalize_batch_output(annotator.process_batch(batch_texts))

            if len(batch_preds) != len(batch_texts):
                raise ValueError(
                    f"Batch prediction count mismatch: got {len(batch_preds)} "
                    f"predictions for {len(batch_texts)} texts."
                )

            for i, pred in enumerate(batch_preds):
                if pred is None:
                    logger.warning(f"[!] Prediction failed at index {batch_start + i}")
                    predictions.append([])
                    continue

                predictions.append(convert_annotated_document_to_iob_tags(pred))

    else:
        logger.info("Using sequential processing")

        for idx, row in tqdm(
            test_df.iterrows(),
            total=len(test_df),
            desc="Running NER predictions",
        ):
            text = row["source"]

            batch_preds = normalize_batch_output(annotator.process_batch([text]))

            if len(batch_preds) != 1:
                raise ValueError(
                    f"Expected exactly 1 prediction for a single input text, "
                    f"got {len(batch_preds)}."
                )

            pred = batch_preds[0]

            if pred is None:
                logger.warning(f"[!] Prediction failed at index {idx}")
                pred_iob = []
            else:
                pred_iob = convert_annotated_document_to_iob_tags(pred)

            if idx < 3:
                logger.info(
                    f"[debug] idx={idx} "
                    f"gold_non_o={count_non_o_tags(row['iob'])} "
                    f"pred_non_o={count_non_o_tags(pred_iob)}"
                )
                logger.info(
                    f"[debug] pred_iob non-O preview="
                    f"{[(tok, tag) for tok, tag in pred_iob if tag != 'O'][:25]}"
                )

            predictions.append(pred_iob)

    return predictions


def main() -> None:
    global logger

    parser = build_base_parser(
        description="NER Evaluation pipeline using config-based approach"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Use 'baseline' or provide the path to a fine-tuned GLiNER model.",
    )
    args = parser.parse_args()

    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config, NERConfig, overrides=args.overrides)

    if args.model:
        if args.model == "baseline":
            config.model = "numind/NuNerZero_span"
        else:
            model_path = Path(args.model).expanduser().resolve()

            if not model_path.is_dir():
                parser.error(
                    f"Model must be 'baseline' or a valid local model directory. "
                    f"Received: {args.model}"
                )

            config.model = str(model_path)

    config = resolve_config_paths(config, args.config)
    config.validate()

    setup_logging(
        verbose=args.verbose,
        log_file=Path(args.log_file) if args.log_file else None,
        component_name="ner_evaluation",
        overwrite=False,
    )
    logger = get_logger()

    logger.info("=" * 60)
    logger.info("NER Evaluation Pipeline")
    logger.info("=" * 60)
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"{config}")

    dataset = load_from_disk(config.input_path)
    df = dataset.to_pandas()

    df["doc"] = df.apply(
        lambda row: convert_label_studio_annotations_to_annotated_spacy_doc(
            row["text"],
            row["annotations"],
            nlp,
        ),
        axis=1,
    )

    annotation_config = get_annotation_config(config.tag_format)
    logger.info(f"Using annotation format: {annotation_config.format.value}")

    test_df = prepare_dataset_for_evaluation(
        df,
        level=config.level,
        config=annotation_config,
    )

    if args.limit > 0:
        logger.info(
            f"Limiting test dataset to first {args.limit} texts for quick evaluation"
        )
        test_df = test_df.head(args.limit).copy()

    annotator = get_annotator_from_config(config)

    output_file = infer_eval_results_file(
        dataset_dir=Path(config.input_path).stem,
        model=getattr(annotator, "id", config.model),
        schema=Path(config.schema),
        n_docs=len(test_df),
        level=config.level,
        base_output_dir=Path(config.output_dir),
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output file: {output_file}")

    if output_file.exists() and not args.force:
        logger.warning(
            f"Output file already exists: {output_file}\n"
            f"Continuing evaluation without overwriting metrics. "
            f"Use --force to overwrite."
        )
    elif output_file.exists() and args.force:
        logger.warning(f"Overwriting existing output file: {output_file}")

    logger.info(f"Running predictions on {len(test_df)} examples...")
    predictions = run_predictions(
        annotator=annotator,
        test_df=test_df,
        config=config,
    )

    test_df = test_df.copy()
    test_df["predictions"] = predictions
    logger.info(f"Completed {len(predictions)} predictions")

    intermediate_output_file = output_file.with_suffix(".predictions_600.jsonl")

    if intermediate_output_file.exists() and not args.force:
        intermediate_output_file = intermediate_output_file.with_name(
            intermediate_output_file.stem + "_new" + intermediate_output_file.suffix
        )
    elif intermediate_output_file.exists() and args.force:
        logger.warning(f"Overwriting existing predictions file: {intermediate_output_file}")

    test_df[
        [
            "doi",
            "section_header",
            "source",
            "target",
            "iob",
            "entity_set",
            "predictions",
        ]
    ].to_json(
        intermediate_output_file,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    logger.info(f"Predictions saved to {intermediate_output_file}")

    logger.info("Fixing empty predictions for evaluation...")
    fix_count = 0

    for idx, row in test_df.iterrows():
        if len(row["predictions"]) == 0:
            logger.warning(
                f"[!] Empty prediction found. Replacing with 'O' tags. "
                f"Index: {idx}\tText: '{row['source'][:50]}...'"
            )
            test_df.at[idx, "predictions"] = replace_empty_prediction_with_placeholder(
                row["iob"],
                placeholder="O",
            )
            fix_count += 1

    if fix_count > 0:
        logger.info(f"Replaced {fix_count} empty predictions with 'O' tags.")

    total_gold_non_o = sum(count_non_o_tags(seq) for seq in test_df["iob"].tolist())
    total_pred_non_o = sum(
        count_non_o_tags(seq) for seq in test_df["predictions"].tolist()
    )

    logger.info(f"Total gold non-O tags: {total_gold_non_o}")
    logger.info(f"Total predicted non-O tags: {total_pred_non_o}")

    if total_pred_non_o == 0 and total_gold_non_o > 0:
        logger.warning(
            "[!] All predictions are O-tags while gold contains entities. "
            "This strongly suggests a model output or conversion bug."
        )

    try:
        validate_token_alignment(
            predictions=test_df["predictions"].tolist(),
            references=test_df["iob"].tolist(),
        )

        results_df = compute_metrics(
            predictions=test_df["predictions"].tolist(),
            references=test_df["iob"].tolist(),
        )

        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(results_df.to_string(float_format="{:.4f}".format))

        if output_file.exists() and not args.force:
            logger.info(
                "Skipping metrics file overwrite because output already exists "
                "and --force was not set."
            )
        else:
            results_df.to_csv(output_file, sep="\t", header=True, float_format="%.4f")
            logger.info(f"Results saved to {output_file}")

    except Exception as e:
        logger.warning(
            f"Error during evaluation:\n"
            f"\t{e}"
            f"\nUnable to save evaluation results. "
            f"See intermediate predictions at {intermediate_output_file}."
        )

    if isinstance(annotator, LLMAnnotator):
        print("\n" + "=" * 60)
        print(annotator.completion_tracker.get_status())
        print("=" * 60)


if __name__ == "__main__":
    main()