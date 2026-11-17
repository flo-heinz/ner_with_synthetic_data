## Running the NER Pipeline

The project provides scripts for generating synthetic NER data, preparing GLiNER-compatible training datasets, fine-tuning models, and evaluating their performance.

The pipeline supports experiments using silver-annotated data, synthetic data, and combinations of both. Synthetic data generation is organized into two approaches: **with annotation guidelines** and **without annotation guidelines**.

Run all commands from the project's root directory after activating the Python environment. A local Qwen inference server must also be running when generating synthetic data.

### 1. Synthetic Data Generation

Two scripts are available for generating synthetic NER training examples:

- `synthetic_data.py` — generates synthetic scientific sentences and entity annotations using the baseline prompting approach without additional annotation guidelines.
- `improved_synthetic_data.py` — generates synthetic examples using an extended prompting approach that incorporates additional annotation guidelines.

The generated datasets are organized under `synthetic_data/`:

```text
synthetic_data/
├── with_guidelines/
│   ├── blabla.txt
│   ├── synthesized_data_6shot_250_guidelines.json
│   ├── synthesized_data_8shot_250_guidelines.json
│   └── synthesized_data_10shot_250_guidelines.json
│
└── without_guidelines/
    ├── example_size_experiment/
    └── few_shot_experiment/
```

**With guidelines**

The `with_guidelines/` directory contains synthetic datasets generated using annotation guidelines and different numbers of few-shot examples (6, 8, and 10 shots).

These datasets allow investigation of how the number of few-shot examples affects synthetic data generation when annotation guidelines are included.

**Without guidelines**

The `without_guidelines/` directory contains two experiment groups:

- `example_size_experiment/` — experiments investigating the effect of the number of generated synthetic examples.
- `few_shot_experiment/` — experiments investigating the effect of the number of few-shot examples used during generation.

These experiments allow the influence of dataset size and few-shot prompting to be studied without additional annotation guidelines.

To inspect the available generation arguments, run:

```bash
python synthetic_data.py --help
python improved_synthetic_data.py --help
```

For example, if supported by the selected script:

```bash
python synthetic_data.py \
  --synthetic_entities_per_label 300 \
  --few_shot_sentence_count 6 \
  --temperature 0.8 \
  --max_entity_occurrences 10 \
  --output_file synthetic_data/without_guidelines/example_size_experiment/balanced_ner_300.json
```

The exact command-line arguments should be verified against the selected script, as the two implementations may expose different options.

### 2. Preparing GLiNER Training Data

The project contains two scripts for preparing training datasets:

- `create_gliner_training_data.py` — converts annotated NER datasets into GLiNER-compatible training data.
- `create_gliner_training_data_with_syn.py` — prepares training data for experiments involving synthetic examples.

Inspect their available arguments using:

```bash
python create_gliner_training_data.py --help
python create_gliner_training_data_with_syn.py --help
```

For example, if the converter supports the corresponding arguments:

```bash
python create_gliner_training_data.py \
  --input-dir data/local/ner/outputs/engage-ft-auto-filtered \
  --output-file data/local/ner/gliner_training_data_filtered/data.json
```

The preprocessing step converts entity annotations into GLiNER-compatible records containing:

- `tokenized_text` — the tokenized input sentence.
- `ner` — entity annotations represented by token start indices, inclusive end indices, and entity labels.

When the original annotations use exclusive end indices, these are converted to inclusive indices.

For experiments involving synthetic data, the generated JSON files must be converted into the format expected by the fine-tuning scripts.

The synthetic input files can be selected from either:

```text
synthetic_data/with_guidelines/
synthetic_data/without_guidelines/
```

This enables training datasets to be prepared for different prompting strategies, few-shot configurations, and synthetic dataset sizes.

### 3. Fine-Tuning GLiNER

Two fine-tuning scripts are provided:

- `finetune_nunerzero_span.py` — fine-tunes GLiNER using prepared annotated training data.
- `finetune_nunerzero_span_synth.py` — fine-tunes GLiNER for experiments involving synthetic training data.

Both scripts use the pretrained `numind/NuNerZero_span` checkpoint as the starting point.

**Fine-tuning on silver-annotated data**

An example command is:

```bash
python finetune_nunerzero_span.py \
  --train_files data/local/ner/gliner_training_data_filtered/data.json \
  --model_name numind/NuNerZero_span \
  --train_fraction 0.2 \
  --eval_fraction 0.1 \
  --batch_size 2 \
  --learning_rate 5e-6 \
  --seed 42
```

**Fine-tuning with synthetic data**

The synthetic fine-tuning script supports experiments involving synthetic datasets generated with or without annotation guidelines.

For combined silver-annotated and synthetic training, an example command is:

```bash
python finetune_nunerzero_span_synth.py \
  --train_files data/local/ner/gliner_training_data_filtered/data.json \
  --synthetic_files data/local/ner/processed_synthetic.json \
  --eval_files data/local/ner/processed_eval.json \
  --model_name numind/NuNerZero_span \
  --train_fraction 0.2 \
  --batch_size 2 \
  --learning_rate 5e-6 \
  --seed 42
```

The `--synthetic_files` and `--eval_files` paths must point to the corresponding preprocessed datasets.

For synthetic-only fine-tuning, the preprocessed synthetic dataset can be supplied through `--train_files`, omitting `--synthetic_files`, provided the script supports this configuration.

The fine-tuning experiments can therefore compare:

1. **Silver-only training** — training using the existing annotated dataset.
2. **Synthetic-only training** — training exclusively on generated examples.
3. **Combined training** — training using both silver-annotated and synthetic examples.

Synthetic-data experiments can additionally compare datasets generated with and without annotation guidelines, as well as different few-shot configurations and dataset sizes.

The fine-tuned models are saved in their configured output directories.

### 4. Evaluating Fine-Tuned Models

The `evaluate.py` script is used to evaluate the trained NER models against a gold-standard evaluation dataset.

Inspect its supported arguments using:

```bash
python evaluate.py --help
```

Provide the fine-tuned model directory and evaluation dataset using the arguments supported by the script.

Model performance is assessed using standard NER evaluation metrics:

- **Precision** — the proportion of predicted entities that are correct.
- **Recall** — the proportion of gold-standard entities correctly identified.
- **F1 score** — the harmonic mean of precision and recall.

These metrics enable comparisons between the different training configurations.

In particular, evaluation can be used to investigate the influence of:

- Synthetic data generated with versus without annotation guidelines.
- Different numbers of few-shot examples.
- Different synthetic dataset sizes.
- Silver-only, synthetic-only, and combined training strategies.

All models should be evaluated using the same gold-standard dataset and evaluation procedure to ensure comparable results.

### 5. Recommended Execution Order

The recommended execution order depends on the selected experiment.

| Step | Script | Purpose |
|---|---|---|
| 1 | `synthetic_data.py` | Generate synthetic examples without additional guidelines |
| 2 | `improved_synthetic_data.py` | Generate synthetic examples with annotation guidelines |
| 3 | `create_gliner_training_data.py` | Prepare annotated training data |
| 4 | `create_gliner_training_data_with_syn.py` | Prepare datasets for synthetic-data experiments |
| 5 | `finetune_nunerzero_span.py` | Fine-tune GLiNER using annotated data |
| 6 | `finetune_nunerzero_span_synth.py` | Fine-tune GLiNER using synthetic or combined datasets |
| 7 | `evaluate.py` | Evaluate fine-tuned models on the gold-standard dataset |

Not all scripts are required for every experiment.

For example:

- **Silver-only experiments** require annotated data preparation, standard fine-tuning, and evaluation.
- **Synthetic-only experiments** require synthetic data generation, conversion into GLiNER format, synthetic fine-tuning, and evaluation.
- **Combined-data experiments** require preparation of both annotated and synthetic datasets before fine-tuning and evaluation.

### 6. Synthetic Data Experiment Organization

The repository separates synthetic generation experiments into the following categories:

| Experiment | Directory | Main variable |
|---|---|---|
| With guidelines | `synthetic_data/with_guidelines/` | Number of few-shot examples (6, 8, 10) |
| Without guidelines — example size | `synthetic_data/without_guidelines/example_size_experiment/` | Number of generated examples |
| Without guidelines — few-shot | `synthetic_data/without_guidelines/few_shot_experiment/` | Number of few-shot examples |

This organization supports systematic comparisons of the synthetic data generation strategies and their effects on downstream NER model performance.

**Note:** The repository structure and dataset filenames above reflect the provided screenshots. The example commands are illustrative and should be checked against the actual script interfaces. In particular, the available command-line arguments, preprocessing requirements, and output directories should be verified before running the pipeline.
