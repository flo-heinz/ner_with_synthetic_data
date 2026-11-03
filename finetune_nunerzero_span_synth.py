
from argparse import ArgumentParser
import json
import os
import random
import shutil
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from dotenv import load_dotenv
from gliner import GLiNER

load_dotenv()

# Configure training data paths, model settings, and hyperparameters.
def set_args():
    ap = ArgumentParser()

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

# Generate a descriptive output directory name from the model and datasets.
def infer_output_dir(
    base_output_dir,
    model_name,
    train_files,
    synthetic_files=None,
):
    model_name_clean = model_name.replace("/", "_")

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

# Load training examples from JSON or JSONL files.
def load_data_from_files(file_paths):
    data = []

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()

        if not content:
            continue

        try:
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

# Lowercase only tokens belonging to entity spans, preserving other tokens.
def lowercase_entities(data):
    for record_idx, record in enumerate(data):
        tokens = list(record["tokenized_text"])

        for start, end, label in record["ner"]:
            if not (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start <= end < len(tokens)
            ):
                raise ValueError(
                    f"Invalid entity span in record {record_idx}: "
                    f"{start}, {end}"
                )

            tokens[start:end + 1] = [
                token.lower()
                for token in tokens[start:end + 1]
            ]

        record["tokenized_text"] = tokens

    return data

# Shuffle reproducibly and optionally select a fraction of the dataset.
def shuffle_and_fraction(data, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError(
            "Fraction must be in the range (0, 1]."
        )

    data = list(data)
    random.Random(seed).shuffle(data)

    if fraction < 1.0:
        target_size = max(1, int(len(data) * fraction))
        data = data[:target_size]

    return data

# Split the shuffled dataset into training and evaluation subsets.
def split_train_eval(data, eval_fraction):
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

    split_idx = int(len(data) * (1 - eval_fraction))
    split_idx = min(max(split_idx, 1), len(data) - 1)

    return data[:split_idx], data[split_idx:]

# Fine-tune GLiNER using the configured training hyperparameters.
def run_training(
    model,
    train_dataset,
    eval_dataset,
    output_dir,
    args,
):
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
    args = set_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = GLiNER.from_pretrained(args.model_name)

    real_data = load_data_from_files(args.train_files)

    real_data = shuffle_and_fraction(
        real_data,
        args.train_fraction,
        args.seed,
    )

    synthetic_data = []

    if args.synthetic_files:
        synthetic_data = load_data_from_files(
            args.synthetic_files
        )

    # Combine real and synthetic examples before creating the train/eval split.
    data = real_data + synthetic_data
    random.Random(args.seed).shuffle(data)

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

    train_dataset = lowercase_entities(train_dataset)

    if eval_dataset is not None:
        eval_dataset = lowercase_entities(eval_dataset)

    output_dir = infer_output_dir(
        base_output_dir=args.base_output_dir,
        model_name=args.model_name,
        train_files=args.train_files,
        synthetic_files=args.synthetic_files,
    )

    if output_dir.exists():
        if not args.force:
            raise RuntimeError(
                f"Output directory {output_dir} already exists. "
                "Use --force to overwrite."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    run_training(
        model,
        train_dataset,
        eval_dataset,
        output_dir,
        args,
    )


if __name__ == "__main__":
    main()
