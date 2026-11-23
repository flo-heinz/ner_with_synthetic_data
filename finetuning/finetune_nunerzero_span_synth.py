#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"""
Fine-tune the NuNER Zero-span GLiNER model using real and synthetic NER examples.

The pipeline loads GLiNER-formatted JSON or JSONL training examples, optionally
subsamples real examples, combines them with synthetic examples, splits the data,
lowercases entity tokens, and fine-tunes the NuNER Zero-span model using
configurable optimization and checkpoint settings.

Example usage:
    python finetune_nunerzero_span_synth.py \
        --train_files data/local/ner/real/data.json

    python finetune_nunerzero_span_synth.py \
        --train_files data/local/ner/real/data.json \
        --synthetic_files data/local/ner/synthetic/data.json \
        --train_fraction 0.2 --eval_fraction 0.1

    python finetune_nunerzero_span_synth.py \
        --train_files data/local/ner/real/data.json \
        --synthetic_files data/local/ner/synthetic/data.json \
        --eval_files data/local/ner/eval/data.json
"""



from argparse import ArgumentParser
import json
import os
import random
import shutil
from pathlib import Path
import torch
from dotenv import load_dotenv
from gliner import GLiNER

# Enable tokenizer parallelism before importing libraries that use tokenizers.
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# Load environment variables needed by the training environment.
load_dotenv()


def set_args():
    """
    Parse command-line arguments for the GLiNER fine-tuning pipeline.

    The arguments control input datasets, the pretrained model, output
    handling, data splitting, optimization, checkpointing, and reproducibility.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    ap = ArgumentParser()

    # Real training examples are required; synthetic and evaluation files are optional.
    ap.add_argument("--train_files", nargs="+", required=True)
    ap.add_argument("--synthetic_files", nargs="+", default=None)
    ap.add_argument("--eval_files", nargs="+", default=None)
    ap.add_argument("--model_name", default="numind/NuNerZero_span")

    ap.add_argument(
        "-o",
        "--base_output_dir",
        type=Path,
        default=Path("data/local/ner/models"),
    )

    # Configure training hyperparameters and output/checkpoint behavior.
    ap.add_argument("-f", "--force", action="store_true")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--accumulation_steps", type=int, default=1)
    ap.add_argument("--num_train_epochs", type=float, default=None)
    ap.add_argument("--train_fraction", type=float, default=1.0)
    ap.add_argument("--eval_fraction", type=float, default=0.1)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--others_lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--others_weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--focal_loss_alpha", type=float, default=0.75)
    ap.add_argument("--focal_loss_gamma", type=float, default=2.0)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--save_total_limit", type=int, default=1)
    ap.add_argument("--report_to", default="none")
    ap.add_argument("--seed", type=int, default=42)

    return ap.parse_args()


def infer_output_dir(
    base_output_dir,
    model_name,
    train_files,
    synthetic_files=None,
):
    """
    Construct a descriptive output directory from the model and input datasets.

    Dataset identifiers are taken from the parent directories of both real and
    synthetic input files, deduplicated, and sorted for consistent naming.

    Args:
        base_output_dir (Path): Parent directory for trained model outputs.
        model_name (str): Name or identifier of the pretrained GLiNER model.
        train_files (list[str]): Paths to real training datasets.
        synthetic_files (list[str] | None): Optional synthetic dataset paths.

    Returns:
        Path: Directory where the fine-tuned model will be saved.
    """
    model_name_clean = model_name.replace("/", "_")

    # Include synthetic dataset names when synthetic training is enabled.
    all_files = list(train_files)

    if synthetic_files:
        all_files.extend(synthetic_files)

    dataset_names = sorted({
        Path(file_path).parent.name.lower()
        for file_path in all_files
    })

    dataset_part = "--".join(dataset_names)

    return base_output_dir / (
        f"{model_name_clean}--{dataset_part}--lowercase-entities"
    )


def load_data_from_files(file_paths):
    """
    Load and combine GLiNER training records from JSON or JSONL files.

    Each file may contain a JSON array, a single JSON object, or one JSON
    object per line. Empty files are ignored; malformed JSONL raises an error.

    Args:
        file_paths (list[str]): Paths to the dataset files to load.

    Returns:
        list: Combined records from all provided files.

    Raises:
        ValueError: If a JSON root type is unsupported or JSONL is malformed.
    """
    data = []

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()

        # Skip files that do not contain any data.
        if not content:
            continue

        try:
            # First try parsing the entire file as standard JSON.
            parsed = json.loads(content)

            if isinstance(parsed, list):
                file_data = parsed
            elif isinstance(parsed, dict):
                file_data = [parsed]
            else:
                raise ValueError(
                    f"Unsupported JSON root type in {file_path}: "
                    f"{type(parsed).__name__}"
                )

        except json.JSONDecodeError:
            # Fall back to JSONL, parsing each non-empty line independently.
            file_data = []

            for line_number, line in enumerate(
                content.splitlines(), start=1
            ):
                if not line.strip():
                    continue

                try:
                    file_data.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSONL in {file_path}, "
                        f"line {line_number}: {error}"
                    ) from error

        data.extend(file_data)

    return data


def lowercase_entities(data):
    """
    Lowercase tokens inside annotated entity spans while preserving other tokens.

    GLiNER entity spans use inclusive start and end token indices. The input
    records are updated in place after validating each span.

    Args:
        data (list[dict]): GLiNER records containing tokenized_text and ner.

    Returns:
        list[dict]: Records with entity tokens converted to lowercase.

    Raises:
        ValueError: If an entity span has invalid or out-of-range indices.
    """
    for record_idx, record in enumerate(data):
        tokens = list(record["tokenized_text"])

        for start, end, label in record["ner"]:
            # Validate inclusive token boundaries before modifying the tokens.
            if not (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start <= end < len(tokens)
            ):
                raise ValueError(
                    f"Invalid entity span in record {record_idx}: "
                    f"{start}, {end}"
                )

            # Only the tokens covered by an entity annotation are lowercased.
            tokens[start:end + 1] = [
                token.lower()
                for token in tokens[start:end + 1]
            ]

        record["tokenized_text"] = tokens

    return data


def shuffle_and_fraction(data, fraction, seed):
    """
    Shuffle a dataset reproducibly and optionally retain a fraction of it.

    Args:
        data (list): Input training examples.
        fraction (float): Fraction to retain, in the interval (0, 1].
        seed (int): Random seed used for reproducible shuffling.

    Returns:
        list: Shuffled dataset, optionally reduced to the requested fraction.

    Raises:
        ValueError: If fraction is outside the permitted range.
    """
    if not 0 < fraction <= 1:
        raise ValueError(
            "Fraction must be in the range (0, 1]."
        )

    # Use a local random generator so the shuffle depends only on the seed.
    data = list(data)
    random.Random(seed).shuffle(data)

    if fraction < 1.0:
        # Keep at least one example when subsampling a non-empty dataset.
        target_size = max(1, int(len(data) * fraction))
        data = data[:target_size]

    return data


def split_train_eval(data, eval_fraction):
    """
    Split a shuffled dataset into training and evaluation subsets.

    When eval_fraction is zero or negative, all examples are used for training
    and no evaluation subset is created. Otherwise, both subsets must be non-empty.

    Args:
        data (list): Examples to split, already shuffled by the caller.
        eval_fraction (float): Fraction reserved for evaluation.

    Returns:
        tuple[list, list | None]: Training examples and optional evaluation examples.

    Raises:
        ValueError: If the fraction is invalid or too few examples are available.
    """
    if eval_fraction <= 0:
        return data, None

    if not 0 < eval_fraction < 1:
        raise ValueError(
            "--eval_fraction must be in the range [0, 1)."
        )

    if len(data) < 2:
        raise ValueError(
            "At least two examples are required "
            "for a train/eval split."
        )

    # Clamp the split index so both subsets contain at least one example.
    split_idx = int(len(data) * (1 - eval_fraction))
    split_idx = min(max(split_idx, 1), len(data) - 1)

    return data[:split_idx], data[split_idx:]


def run_training(
    model,
    train_dataset,
    eval_dataset,
    output_dir,
    args,
):
    """
    Fine-tune GLiNER using the configured optimization and training settings.

    Training uses a linear learning-rate schedule and automatically selects
    CPU execution when CUDA is unavailable.

    Args:
        model: Initialized GLiNER model.
        train_dataset (list[dict]): Training examples in GLiNER format.
        eval_dataset (list[dict] | None): Optional evaluation examples.
        output_dir (Path): Directory for checkpoints and model outputs.
        args (argparse.Namespace): Parsed training hyperparameters.

    Returns:
        Any: Result returned by the model's train_model method.
    """
    return model.train_model(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        others_lr=args.others_lr,
        others_weight_decay=args.others_weight_decay,
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        focal_loss_alpha=args.focal_loss_alpha,
        focal_loss_gamma=args.focal_loss_gamma,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=0,
        use_cpu=not torch.cuda.is_available(),
        report_to=args.report_to,
    )


def main():
    """
    Execute the full GLiNER fine-tuning pipeline.

    Real examples are optionally subsampled before synthetic examples are
    appended. The combined data is shuffled and either split for evaluation
    or paired with a separately provided evaluation dataset.
    """
    args = set_args()

    # Seed Python and PyTorch for more reproducible training behavior.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize the selected pretrained GLiNER model.
    model = GLiNER.from_pretrained(args.model_name)

    # Load real training examples and apply the requested real-data fraction.
    real_data = load_data_from_files(args.train_files)

    real_data = shuffle_and_fraction(
        real_data,
        args.train_fraction,
        args.seed,
    )

    # Synthetic examples are optional and are not subject to train_fraction.
    synthetic_data = []

    if args.synthetic_files:
        synthetic_data = load_data_from_files(
            args.synthetic_files
        )

    # Combine real and synthetic examples before creating the train/eval split.
    data = real_data + synthetic_data
    random.Random(args.seed).shuffle(data)

    # Prefer explicit evaluation files; otherwise split the combined dataset.
    if args.eval_files:
        train_dataset = data
        eval_dataset = load_data_from_files(
            args.eval_files
        )
    else:
        train_dataset, eval_dataset = split_train_eval(
            data,
            args.eval_fraction,
        )

    # Apply the same entity-token casing transformation to both subsets.
    train_dataset = lowercase_entities(train_dataset)

    if eval_dataset is not None:
        eval_dataset = lowercase_entities(eval_dataset)

    # Name the output directory using the model and source dataset directories.
    output_dir = infer_output_dir(
        base_output_dir=args.base_output_dir,
        model_name=args.model_name,
        train_files=args.train_files,
        synthetic_files=args.synthetic_files,
    )

    # Protect existing model artifacts unless --force explicitly permits deletion.
    if output_dir.exists():
        if not args.force:
            raise RuntimeError(
                f"Output directory {output_dir} already exists. "
                "Use --force to overwrite."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Start model fine-tuning with the prepared datasets and hyperparameters.
    run_training(
        model,
        train_dataset,
        eval_dataset,
        output_dir,
        args,
    )


if __name__ == "__main__":
    main()


