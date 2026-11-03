
from argparse import ArgumentParser
import json
import os
import random
import shutil
from pathlib import Path

import torch
from dotenv import load_dotenv
from gliner import GLiNER

os.environ["TOKENIZERS_PARALLELISM"] = "true"

load_dotenv()


def set_args():
    ap = ArgumentParser()

    ap.add_argument("--train_files", nargs="+", required=True)
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


def infer_output_dir(base_output_dir, model_name, train_files):
    model_name_clean = model_name.replace("/", "_")

    train_dirs = sorted(
        Path(file_path).parent.name.lower()
        for file_path in train_files
    )

    train_files_clean = "_".join(train_dirs)

    return base_output_dir / (
        f"{model_name_clean}--{train_files_clean}--lowercase-entities"
    )


def load_data_from_files(file_paths):
    data = []

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            first_char = file.read(1)
            file.seek(0)

            if first_char == "[":
                file_data = json.load(file)
            else:
                file_data = [
                    json.loads(line)
                    for line in file
                    if line.strip()
                ]

        data.extend(file_data)

    return data


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


def shuffle_and_fraction(data, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError(
            "Fraction must be in the range (0, 1]."
        )

    data = list(data)
    random.Random(seed).shuffle(data)

    if fraction < 1.0:
        size = max(1, int(len(data) * fraction))
        data = data[:size]

    return data


def split_train_eval(data, eval_fraction):
    if eval_fraction <= 0:
        return data, None

    if not 0 < eval_fraction < 1:
        raise ValueError(
            "--eval_fraction must be in the range [0, 1)."
        )

    split_idx = int(len(data) * (1 - eval_fraction))

    return data[:split_idx], data[split_idx:]


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
        others_lr=args.others_lr,
        weight_decay=args.weight_decay,
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

    model = GLiNER.from_pretrained(args.model_name)

    data = load_data_from_files(args.train_files)

    data = shuffle_and_fraction(
        data,
        args.train_fraction,
        args.seed,
    )

    if args.eval_files:
        train_dataset = data
        eval_dataset = load_data_from_files(args.eval_files)
    else:
        train_dataset, eval_dataset = split_train_eval(
            data,
            args.eval_fraction,
        )

    train_dataset = lowercase_entities(train_dataset)

    if eval_dataset is not None:
        eval_dataset = lowercase_entities(eval_dataset)

    output_dir = infer_output_dir(
        args.base_output_dir,
        args.model_name,
        args.train_files,
    )

    if output_dir.exists():
        if not args.force:
            raise RuntimeError(
                f"Output directory {output_dir} already exists."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    run_training(
        model,
        train_dataset,
        eval_dataset,from argparse import ArgumentParser
import json
import os
import random
import shutil
from pathlib import Path

import torch
from dotenv import load_dotenv
from gliner import GLiNER

os.environ["TOKENIZERS_PARALLELISM"] = "true"

load_dotenv()


# Configure input files and training hyperparameters.
def set_args():
    ap = ArgumentParser()

    ap.add_argument("--train_files", nargs="+", required=True)
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


# Name the output directory after the model and training datasets.
def infer_output_dir(base_output_dir, model_name, train_files):
    model_name_clean = model_name.replace("/", "_")

    train_dirs = sorted(
        Path(file_path).parent.name.lower()
        for file_path in train_files
    )

    train_files_clean = "_".join(train_dirs)

    return base_output_dir / (
        f"{model_name_clean}--{train_files_clean}--lowercase-entities"
    )


# Read GLiNER examples from JSON arrays or JSONL files.
def load_data_from_files(file_paths):
    data = []

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            first_char = file.read(1)
            file.seek(0)

            if first_char == "[":
                file_data = json.load(file)
            else:
                file_data = [
                    json.loads(line)
                    for line in file
                    if line.strip()
                ]

        data.extend(file_data)

    return data


# Lowercase only entity tokens, leaving other tokens unchanged.
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


# Shuffle reproducibly before selecting a training-data fraction.
def shuffle_and_fraction(data, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError(
            "Fraction must be in the range (0, 1]."
        )

    data = list(data)
    random.Random(seed).shuffle(data)

    if fraction < 1.0:
        size = max(1, int(len(data) * fraction))
        data = data[:size]

    return data


# Split the shuffled data when no separate evaluation files are provided.
def split_train_eval(data, eval_fraction):
    if eval_fraction <= 0:
        return data, None

    if not 0 < eval_fraction < 1:
        raise ValueError(
            "--eval_fraction must be in the range [0, 1)."
        )

    split_idx = int(len(data) * (1 - eval_fraction))

    return data[:split_idx], data[split_idx:]


# Fine-tune GLiNER with the configured hyperparameters.
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
        others_lr=args.others_lr,
        weight_decay=args.weight_decay,
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

    # Fix random seeds to make data selection and training more reproducible.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = GLiNER.from_pretrained(args.model_name)

    data = load_data_from_files(args.train_files)

    data = shuffle_and_fraction(
        data,
        args.train_fraction,
        args.seed,
    )

    if args.eval_files:
        train_dataset = data
        eval_dataset = load_data_from_files(args.eval_files)
    else:
        train_dataset, eval_dataset = split_train_eval(
            data,
            args.eval_fraction,
        )

    train_dataset = lowercase_entities(train_dataset)

    if eval_dataset is not None:
        eval_dataset = lowercase_entities(eval_dataset)

    output_dir = infer_output_dir(
        args.base_output_dir,
        args.model_name,
        args.train_files,
    )

    # Require --force before deleting an existing model directory.
    if output_dir.exists():
        if not args.force:
            raise RuntimeError(
                f"Output directory {output_dir} already exists."
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

        output_dir,
        args,
    )


if __name__ == "__main__":
    main()
