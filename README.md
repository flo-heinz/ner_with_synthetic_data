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
