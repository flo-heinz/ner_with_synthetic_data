#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Fine-tune the NuNER Zero-span GLiNER model on annotated NER datasets.

The pipeline loads GLiNER-formatted training examples from JSON or JSONL
files, reproducibly shuffles and optionally subsamples the data, and prepares
training and evaluation datasets. Tokens covered by annotated entity spans
are converted to lowercase, while non-entity tokens retain their original
casing.

The model is fine-tuned using configurable optimization, focal loss,
gradient accumulation, learning rate scheduling, and checkpoint settings.
Model artifacts are stored in an output directory derived from the
pretrained model identifier and training dataset directories.

Example usage:
    python finetune_nunerzero_span.py \
        --train_files data/local/ner/training/data.json

    python finetune_nunerzero_span.py \
        --train_files data/local/ner/training/data.json \
        --train_fraction 0.2 --eval_fraction 0.1 \
        --batch_size 4 --force

    python finetune_nunerzero_span.py \
        --train_files train.json --eval_files dev.json

    python finetune_nunerzero_span.py \
        --train_files train.json \
        --num_train_epochs 5 \
        --batch_size 4 --accumulation_steps 2 \
        --learning_rate 5e-6 --others_lr 1e-5
"""


from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import json
import os
import random
import shutil

import torch
from dotenv import load_dotenv
from gliner import GLiNER


# Enable parallel tokenization during data preparation and training.
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Load environment variables required by the training pipeline.
load_dotenv()


# Configure dataset paths, model settings, and training hyperparameters.
def set_args() -> Namespace:
    """Parse command-line arguments for the GLiNER fine-tuning pipeline.

    The arguments control dataset selection, model initialization,
    optimization, training duration, checkpoint management, and
    reproducibility.

    Returns:
        Parsed command-line arguments containing the training configuration.
    """
    ap = ArgumentParser(
        description="Fine-tune the NuNER Zero-span GLiNER model."
    )

    # Configure training and evaluation datasets.
    data_args = ap.add_argument_group("Dataset settings")

    data_args.add_argument("--train_files", nargs="+", required=True)
    data_args.add_argument("--eval_files", nargs="+", default=None)
    data_args.add_argument("--train_fraction", type=float, default=1.0)
    data_args.add_argument("--eval_fraction", type=float, default=0.1)

    # Configure the pretrained model and output directory.
    model_args = ap.add_argument_group("Model and output settings")

    model_args.add_argument(
        "--model_name",
        default="numind/NuNerZero_span",
    )

    model_args.add_argument(
        "-o",
        "--base_output_dir",
        type=Path,
        default=Path("data/local/ner/models"),
    )

    model_args.add_argument("-f", "--force", action="store_true")

    # Configure training duration and effective batch size.
    training_args = ap.add_argument_group("Training settings")

    training_args.add_argument("--batch_size", type=int, default=2)
    training_args.add_argument("--accumulation_steps", type=int, default=1)
    training_args.add_argument("--num_train_epochs", type=float, default=None)

    # Configure learning rates, weight decay, and learning rate warmup.
    optimization_args = ap.add_argument_group("Optimization settings")

    optimization_args.add_argument("--learning_rate", type=float, default=5e-6)
    optimization_args.add_argument("--others_lr", type=float, default=1e-5)
    optimization_args.add_argument("--weight_decay", type=float, default=0.01)
    optimization_args.add_argument(
        "--others_weight_decay",
        type=float,
        default=0.01,
    )
    optimization_args.add_argument("--warmup_ratio", type=float, default=0.1)

    # Configure focal loss for entity classification.
    loss_args = ap.add_argument_group("Loss settings")

    loss_args.add_argument("--focal_loss_alpha", type=float, default=0.75)
    loss_args.add_argument("--focal_loss_gamma", type=float, default=2.0)

    # Configure checkpoint saving and experiment reporting.
    checkpoint_args = ap.add_argument_group("Checkpoint and reporting settings")

    checkpoint_args.add_argument("--save_steps", type=int, default=100)
    checkpoint_args.add_argument("--save_total_limit", type=int, default=1)
    checkpoint_args.add_argument("--report_to", default="none")

    # Configure the random seed for reproducible training.
    ap.add_argument("--seed", type=int, default=42)

    return ap.parse_args()


# Construct an output directory from the model and training dataset names.
def infer_output_dir(
    base_output_dir: Path,
    model_name: str,
    train_files: list[str],
) -> Path:
    """Infer the output directory for the fine-tuned GLiNER model.

    The directory name combines the pretrained model identifier,
    the parent directory names of the training files, and a suffix
    indicating that annotated entity tokens were lowercased.

    Args:
        base_output_dir: Parent directory for saved model artifacts.
        model_name: Identifier of the pretrained GLiNER model.
        train_files: Paths to the training dataset files.

    Returns:
        Path to the output directory for the fine-tuned model.
    """
    # Replace path separators in the pretrained model identifier.
    model_name_clean = model_name.replace("/", "_")

    # Extract and sort dataset directory names for consistent naming.
    train_dirs = sorted(
        Path(file_path).parent.name.lower()
        for file_path in train_files
    )

    train_files_clean = "_".join(train_dirs)

    # Include the preprocessing strategy in the output directory name.
    return base_output_dir / (
        f"{model_name_clean}--{train_files_clean}--lowercase-entities"
    )


# Load and combine GLiNER-formatted examples from multiple input files.
def load_data_from_files(
    file_paths: list[str],
) -> list[dict[str, Any]]:
    """Load annotated NER examples from JSON or JSONL files.

    Each input file may contain a JSON array of GLiNER records or
    individual JSON records stored on separate lines. Examples from
    all supplied files are concatenated into a single dataset.

    Args:
        file_paths: Paths to the JSON or JSONL dataset files.

    Returns:
        Combined list of GLiNER-formatted dataset records.
    """
    data = []

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:

            # Inspect the first character to determine the file format.
            first_char = file.read(1)
            file.seek(0)

            if first_char == "[":
                # Load examples stored as a JSON array.
                file_data = json.load(file)

            else:
                # Load individual JSONL records, ignoring empty lines.
                file_data = [
                    json.loads(line)
                    for line in file
                    if line.strip()
                ]

        # Append the loaded examples to the combined dataset.
        data.extend(file_data)

    return data


# Lowercase annotated entity tokens while preserving non-entity tokens.
def lowercase_entities(
    data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert tokens covered by annotated entity spans to lowercase.

    GLiNER represents entity spans using inclusive start and end token
    indices. Only tokens belonging to annotated entities are modified,
    while tokens outside entity spans retain their original casing.

    The transformation modifies the supplied dataset in place without
    changing token positions, entity boundaries, or entity labels.

    Args:
        data: GLiNER records containing tokenized_text and ner fields.

    Returns:
        Dataset containing the lowercased entity tokens.

    Raises:
        ValueError: If an entity span contains invalid token indices
            or falls outside the corresponding token sequence.
    """
    for record_idx, record in enumerate(data):

        # Copy the token sequence before applying casing transformations.
        tokens = list(record["tokenized_text"])

        for start, end, label in record["ner"]:

            # Validate the inclusive entity span before modifying tokens.
            if not (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start <= end < len(tokens)
            ):
                raise ValueError(
                    f"Invalid entity span in record {record_idx}: "
                    f"{start}, {end}"
                )

            # Include the final token because GLiNER uses inclusive spans.
            tokens[start:end + 1] = [
                token.lower()
                for token in tokens[start:end + 1]
            ]

        # Replace the original token sequence with the transformed tokens.
        record["tokenized_text"] = tokens

    return data


# Shuffle the dataset reproducibly and optionally retain a subset.
def shuffle_and_fraction(
    data: list[dict[str, Any]],
    fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Shuffle dataset records and optionally select a fraction of them.

    A dedicated random number generator ensures that the shuffling
    procedure is reproducible for a given seed without modifying the
    global Python random state.

    Args:
        data: Input dataset records.
        fraction: Proportion of records to retain, in the range (0, 1].
        seed: Random seed used for dataset shuffling.

    Returns:
        Shuffled dataset containing the requested fraction of records.

    Raises:
        ValueError: If the supplied fraction is outside the valid range.
    """
    # Validate the requested dataset fraction.
    if not 0 < fraction <= 1:
        raise ValueError(
            "Fraction must be in the range (0, 1]."
        )

    # Copy the dataset to avoid modifying the original record order.
    data = list(data)

    # Shuffle examples using a dedicated, seeded random generator.
    random.Random(seed).shuffle(data)

    # Retain the requested fraction of the shuffled dataset.
    if fraction < 1.0:
        size = max(1, int(len(data) * fraction))
        data = data[:size]

    return data


# Split the dataset when no separate evaluation files are provided.
def split_train_eval(
    data: list[dict[str, Any]],
    eval_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Partition a shuffled dataset into training and evaluation subsets.

    The evaluation subset is selected from the end of the shuffled
    dataset, while the remaining examples are used for training.
    A non-positive evaluation fraction disables the evaluation split.

    Args:
        data: Previously shuffled dataset records.
        eval_fraction: Proportion of records assigned to evaluation.

    Returns:
        Tuple containing the training dataset and the evaluation dataset.
        The evaluation dataset is None when the split is disabled.

    Raises:
        ValueError: If the evaluation fraction is greater than or equal
            to one.
    """
    # Disable evaluation splitting when no positive fraction is specified.
    if eval_fraction <= 0:
        return data, None

    # Validate the evaluation fraction before splitting the dataset.
    if not 0 < eval_fraction < 1:
        raise ValueError(
            "--eval_fraction must be in the range [0, 1)."
        )

    # Determine the boundary between training and evaluation examples.
    split_idx = int(len(data) * (1 - eval_fraction))

    return data[:split_idx], data[split_idx:]


# Fine-tune the pretrained GLiNER model using the configured hyperparameters.
def run_training(
    model: GLiNER,
    train_dataset: list[dict[str, Any]],
    eval_dataset: list[dict[str, Any]] | None,
    output_dir: Path,
    args: Namespace,
) -> Any:
    """Fine-tune GLiNER on the prepared training and evaluation datasets.

    The training procedure uses configurable learning rates, weight
    decay, linear learning rate scheduling, focal loss, gradient
    accumulation, and checkpoint management.

    CUDA is used when available; otherwise, training runs on the CPU.

    Args:
        model: Initialized pretrained GLiNER model.
        train_dataset: Prepared training examples.
        eval_dataset: Optional evaluation examples.
        output_dir: Directory for model checkpoints and training artifacts.
        args: Parsed command-line arguments containing training settings.

    Returns:
        Training result returned by GLiNER.train_model.
    """
    return model.train_model(

        # Configure the training and evaluation datasets.
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=str(output_dir),

        # Configure optimizer learning rates and regularization.
        learning_rate=args.learning_rate,
        others_lr=args.others_lr,
        weight_decay=args.weight_decay,
        others_weight_decay=args.others_weight_decay,

        # Configure learning rate scheduling and warmup.
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,

        # Configure batch sizes and gradient accumulation.
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,

        # Configure focal loss for entity classification.
        focal_loss_alpha=args.focal_loss_alpha,
        focal_loss_gamma=args.focal_loss_gamma,

        # Configure training duration and checkpoint management.
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        # Configure data loading and device selection.
        dataloader_num_workers=0,
        use_cpu=not torch.cuda.is_available(),

        # Configure experiment reporting.
        report_to=args.report_to,
    )


# Execute the complete GLiNER fine-tuning pipeline.
def main() -> None:
    """Configure and execute the NuNER Zero-span fine-tuning pipeline.

    The pipeline initializes the pretrained model, loads and prepares
    annotated datasets, applies entity-token lowercasing, configures
    the output directory, and starts model fine-tuning.

    Raises:
        RuntimeError: If the output directory already exists and
            overwriting has not been explicitly enabled.
    """
    # Parse command-line arguments.
    args = set_args()

    # Initialize random seeds for reproducible data preparation and training.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load the pretrained GLiNER model.
    model = GLiNER.from_pretrained(args.model_name)

    # Load and combine examples from the supplied training files.
    data = load_data_from_files(args.train_files)

    # Shuffle the dataset and optionally retain a subset of the examples.
    data = shuffle_and_fraction(
        data,
        args.train_fraction,
        args.seed,
    )

    # Prefer explicitly supplied evaluation files over an automatic split.
    if args.eval_files:
        train_dataset = data
        eval_dataset = load_data_from_files(args.eval_files)

    else:
        # Partition the shuffled dataset into training and evaluation sets.
        train_dataset, eval_dataset = split_train_eval(
            data,
            args.eval_fraction,
        )

    # Lowercase annotated entity tokens in the training dataset.
    train_dataset = lowercase_entities(train_dataset)

    # Apply the same preprocessing strategy to the evaluation dataset.
    if eval_dataset is not None:
        eval_dataset = lowercase_entities(eval_dataset)

    # Construct the output directory from the model and dataset names.
    output_dir = infer_output_dir(
        args.base_output_dir,
        args.model_name,
        args.train_files,
    )

    # Prevent accidental overwriting of existing model artifacts.
    if output_dir.exists():
        if not args.force:
            raise RuntimeError(
                f"Output directory {output_dir} already exists."
            )

        # Remove existing model artifacts when overwriting is enabled.
        shutil.rmtree(output_dir)

    # Create the output directory for checkpoints and training artifacts.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fine-tune the model using the prepared datasets and training settings.
    run_training(
        model,
        train_dataset,
        eval_dataset,
        output_dir,
        args,
    )


if __name__ == "__main__":
    main()
