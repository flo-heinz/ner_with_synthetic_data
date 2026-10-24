# Balanced Synthetic NER Data Generation

Generate synthetic scientific sentences and named entity annotations for **renewable energy and biodiversity** using `Qwen/Qwen3-4B-Instruct-2507`, few-shot reference examples, and automatic validation. The data supports synthetic-only fine-tuning of **GLiNER** across 16 NER labels.

## Balancing

The revised `balanced_ner_generation.py` targets **300 saved entity annotations per label**, rather than 300 sentences. A sentence with three valid target-label entities contributes three annotations, so the number of accepted sentences may vary between labels.

Earlier thesis experiments instead balanced **accepted sentences per label** (200, 400, 600, and 800). Their reported results do **not** evaluate the revised entity-balanced script.

## Setup

Use Python 3.10+ and a locally running OpenAI-compatible inference server (default: `http://localhost:8000/v1`). Install dependencies:

```bash
pip install instructor openai pydantic tqdm
```

Reference JSON/JSONL files are read from `data/local/ner/outputs/engage-ft-auto-filtered/`. Each record needs a `text` field and an `entities` list containing entity text and label.

## Usage

```bash
python balanced_ner_generation.py \
  --synthetic_entities_per_label 300 \
  --few_shot_sentence_count 6 \
  --temperature 0.8 \
  --max_entity_occurrences 10 \
  --output_file balanced_ner_300.json
```

The output is saved to `data/local/ner/balanced_ner_300.json`. Existing output files are not overwritten.

## Validation and Output

The script rejects invalid or duplicate sentences, invalid or overlapping target-label spans, candidates exceeding the remaining annotation quota, and candidates violating the repetition limit. By default, each normalized entity expression can occur at most **10 times per label**.

Each saved JSON record contains `tokens` and `entities`, with `text`, `label`, `start_token`, and `end_token`. Token indices are **zero-based and end-exclusive**. Final annotation counts are checked against the requested quota for each generated label.

## Limitations and GLiNER

- Only annotations for the requested target label are saved; entities from other categories may remain unannotated.
- Automatic validation does not guarantee scientific accuracy or correct semantic labels.
- Strict quotas and repetition limits may require additional generation attempts.

A separate preprocessing step converts `tokens` to `tokenized_text` and entity spans to `ner` entries, changing end-exclusive indices to end-inclusive indices. The thesis fine-tunes `numind/NuNerZero_span` and evaluates against gold-standard annotations using precision, recall, and F1.

## GLiNER Fine-Tuning Arguments

The separate GLiNER training script accepts the following command-line arguments. It expects **preprocessed** records containing `tokenized_text` and `ner` (with inclusive end indices), not the generator's raw `tokens`/`entities` output.

### Input and output

| Argument | Default | Description |
| --- | --- | --- |
| `--train_files` | **Required** | One or more JSON/JSONL training files. These are loaded and optionally subsampled before synthetic data is added. |
| `--synthetic_files` | None | Optional additional JSON/JSONL training files, appended to the selected training data. |
| `--eval_files` | None | Optional separate evaluation files. If omitted, evaluation data is split from the combined training data. |
| `--model_name` | `numind/NuNerZero_span` | Pretrained GLiNER checkpoint to fine-tune. |
| `-o`, `--base_output_dir` | `data/local/ner/models` | Parent directory for the automatically named model output folder. |
| `-f`, `--force` | Off | Delete and replace an existing output folder. **Use with care.** |

### Training and evaluation

| Argument | Default | Description |
| --- | --- | --- |
| `--batch_size` | `2` | Per-device training and evaluation batch size. |
| `--accumulation_steps` | `1` | Number of gradient accumulation steps. |
| `--num_train_epochs` | `None` | Number of training epochs; when omitted, the training library determines its behavior. |
| `--train_fraction` | `1.0` | Fraction of data from `--train_files` to retain, before adding synthetic data. Must be greater than 0 and at most 1. |
| `--eval_fraction` | `0.1` | Fraction of combined data reserved for evaluation when `--eval_files` is absent. Set to `0` to disable this split. |
| `--learning_rate` | `5e-6` | Learning rate for the main model parameters. |
| `--others_lr` | `1e-5` | Learning rate for other parameter groups. |
| `--weight_decay` | `0.01` | Weight decay for the main parameter group. |
| `--others_weight_decay` | `0.01` | Weight decay for other parameter groups. |
| `--warmup_ratio` | `0.1` | Fraction of training steps used for learning-rate warmup. |
| `--focal_loss_alpha` | `0.75` | Focal-loss alpha parameter. |
| `--focal_loss_gamma` | `2.0` | Focal-loss gamma parameter. |
| `--save_steps` | `100` | Checkpoint saving interval in steps. |
| `--save_total_limit` | `1` | Maximum number of saved checkpoints to retain. |
| `--report_to` | `none` | Training-reporting integration passed to GLiNER. |
| `--seed` | `42` | Seed for dataset shuffling and PyTorch random number generation. |

### Training example

Assuming your training script is named `train_gliner.py`:

```bash
python train_gliner.py \
  --train_files data/local/ner/processed_train.json \
  --synthetic_files data/local/ner/processed_synthetic.json \
  --eval_files data/local/ner/processed_eval.json \
  --model_name numind/NuNerZero_span \
  --batch_size 2 \
  --accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 5e-6 \
  --seed 42
```

Replace the example paths and script name with the actual ones in your project. For **synthetic-only training**, provide the synthetic training dataset through `--train_files` and omit `--synthetic_files`.

The training script shuffles the data, lowercases tokens inside annotated entity spans, and uses CUDA if available (otherwise CPU). When `--eval_files` is supplied, it uses those files as the evaluation set instead of splitting the training data. Output folder names are derived from the model name and the **parent directory names** of the input files; distinct input filenames in the same directories may therefore produce the same output folder name.
