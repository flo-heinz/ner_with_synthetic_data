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
    │   ├── synthesized_data_8shot_200_base_prompt.json
    │   ├── synthesized_data_8shot_400_base_prompt.json
    │   ├── synthesized_data_8shot_600_base_prompt.json
    │   └── synthesized_data_8shot_800_base_prompt.json
    │
    └── few_shot_experiment/
        ├── synthesized_data_6shot_250_base_prompt.json
        ├── synthesized_data_8shot_250_base_prompt.json
        └── synthesized_data_10shot_250_base_prompt.json
```

#### 1.1 Synthetic Data With Guidelines

The `with_guidelines/` directory contains synthetic datasets generated using additional annotation guidelines.

Three configurations are available:

| Dataset | Few-shot examples | Generation size parameter |
|---|---|---|
| `synthesized_data_6shot_250_guidelines.json` | 6 | 250 |
| `synthesized_data_8shot_250_guidelines.json` | 8 | 250 |
| `synthesized_data_10shot_250_guidelines.json` | 10 | 250 |

The generation size parameter is kept constant at 250, while the number of few-shot examples varies.

This experiment investigates how different numbers of few-shot examples influence synthetic NER data generation when additional annotation guidelines are provided.

#### 1.2 Synthetic Data Without Guidelines

The `without_guidelines/` directory contains two experiment groups that use the baseline prompting approach.

**Example Size Experiment**

The `example_size_experiment/` directory contains datasets generated using a fixed number of 8 few-shot examples while varying the generation size parameter.

| Dataset | Few-shot examples | Generation size parameter |
|---|---|---|
| `synthesized_data_8shot_200_base_prompt.json` | 8 | 200 |
| `synthesized_data_8shot_400_base_prompt.json` | 8 | 400 |
| `synthesized_data_8shot_600_base_prompt.json` | 8 | 600 |
| `synthesized_data_8shot_800_base_prompt.json` | 8 | 800 |

This experiment investigates the effect of synthetic dataset size on downstream NER model performance while keeping the few-shot prompting configuration constant.

**Few-Shot Experiment**

The `few_shot_experiment/` directory contains datasets generated using different numbers of few-shot examples while keeping the generation size parameter fixed at 250.

| Dataset | Few-shot examples | Generation size parameter |
|---|---|---|
| `synthesized_data_6shot_250_base_prompt.json` | 6 | 250 |
| `synthesized_data_8shot_250_base_prompt.json` | 8 | 250 |
| `synthesized_data_10shot_250_base_prompt.json` | 10 | 250 |

This experiment investigates how the number of few-shot examples used during generation influences synthetic NER data quality and downstream model performance.

Together, these experiments enable comparisons of dataset size, few-shot prompting, and the use of additional annotation guidelines.

#### 1.3 Running Synthetic Data Generation

To inspect the available generation arguments, run:

```bash
python synthetic_data.py --help
python improved_synthetic_data.py --help
```

For example, if supported by the selected script:

```bash
python synthetic_data.py \
  --synthetic_entities_per_label 200 \
  --few_shot_sentence_count 8 \
  --temperature 0.8 \
  --max_entity_occurrences 10 \
  --output_file synthetic_data/without_guidelines/example_size_experiment/synthesized_data_8shot_200_base_prompt.json
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

Synthetic datasets can be selected from the three experiment directories:

```text
synthetic_data/with_guidelines/
synthetic_data/without_guidelines/example_size_experiment/
synthetic_data/without_guidelines/few_shot_experiment/
```

Each dataset can be prepared separately to allow comparisons between different synthetic data generation configurations.

### 3. Fine-Tuning GLiNER

Two fine-tuning scripts are provided:

- `finetune_nunerzero_span.py` — fine-tunes GLiNER using prepared annotated training data.
- `finetune_nunerzero_span_synth.py` — fine-tunes GLiNER for experiments involving synthetic training data.

Both scripts use the pretrained `numind/NuNerZero_span` checkpoint as the starting point.

#### 3.1 Fine-Tuning on Silver-Annotated Data

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

#### 3.2 Fine-Tuning With Synthetic Data

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

The fine-tuning experiments can compare:

1. **Silver-only training** — training using the existing annotated dataset.
2. **Synthetic-only training** — training exclusively on generated examples.
3. **Combined training** — training using both silver-annotated and synthetic examples.

Within the synthetic training configurations, additional comparisons can be performed across:

- Datasets generated with and without annotation guidelines.
- Few-shot configurations using 6, 8, or 10 examples.
- Generation size configurations of 200, 400, 600, and 800.

The fine-tuned models are saved in their configured output directories.

### 4. Evaluating Fine-Tuned Models

The `evaluate.py` script is used to evaluate trained NER models against a gold-standard evaluation dataset.

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

In particular, evaluation can be used to investigate:

- The influence of annotation guidelines on synthetic data generation.
- The effect of using 6, 8, or 10 few-shot examples.
- The effect of different synthetic dataset sizes.
- The performance of silver-only, synthetic-only, and combined training approaches.

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

### 6. Synthetic Data Experiment Overview

The synthetic datasets are organized into three experimental groups:

| Experiment | Prompting strategy | Few-shot examples | Generation size parameters |
|---|---|---|---|
| With guidelines | Extended prompt with annotation guidelines | 6, 8, 10 | 250 |
| Without guidelines — example size | Baseline prompt | 8 | 200, 400, 600, 800 |
| Without guidelines — few-shot | Baseline prompt | 6, 8, 10 | 250 |

The experiments are designed to isolate different aspects of synthetic data generation:

- **Guideline comparison:** Compare datasets generated with and without annotation guidelines under matching few-shot and generation size configurations.
- **Example size comparison:** Investigate the influence of the synthetic data generation size parameter while keeping the number of few-shot examples fixed at 8.
- **Few-shot comparison:** Investigate the influence of using 6, 8, or 10 few-shot examples while keeping the generation size parameter fixed at 250.

The resulting datasets can be used for downstream GLiNER fine-tuning experiments to assess how different synthetic data generation strategies affect NER performance.

**Note:** The repository structure and filenames reflect the provided screenshots. The numeric generation size parameters are inferred from filenames and should be checked against the generation scripts before interpreting them as exact record counts. The example commands are illustrative; command-line arguments, preprocessing requirements, and output paths should be verified against the actual Python implementations.
