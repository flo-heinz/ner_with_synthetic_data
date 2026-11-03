## Running the NER Pipeline

The project contains separate scripts for generating synthetic data, preparing GLiNER training datasets, fine-tuning models, and evaluating their performance.

Run all commands from the project's root directory after activating the Python environment and starting the local Qwen inference server when synthetic data generation is required.

### 1. Synthetic Data Generation

Two scripts are available for generating synthetic NER training data:

- `synthetic_data.py` — generates synthetic scientific sentences and entity annotations using the baseline prompting approach.
- `improved_synthetic_data.py` — generates synthetic examples using the extended prompting approach with additional annotation guidelines.

Run the scripts using:

```bash
python synthetic_data.py --help
python improved_synthetic_data.py --help
```

Then run the desired script with its supported arguments. For example, if the script supports the generation options described earlier:

```bash
python synthetic_data.py \
  --synthetic_entities_per_label 300 \
  --few_shot_sentence_count 6 \
  --temperature 0.8 \
  --max_entity_occurrences 10 \
  --output_file balanced_ner_300.json
```

The exact command-line options should be checked against the selected script because the two generation implementations may have different arguments.

### 2. Preparing GLiNER Training Data

The following scripts prepare annotated data for GLiNER fine-tuning:

- `create_gliner_training_data.py` — prepares annotated NER data for training.
- `create_gliner_training_data_with_syn.py` — prepares training data for experiments involving synthetic examples.

Run:

```bash
python create_gliner_training_data.py --help
python create_gliner_training_data_with_syn.py --help
```

For a converter supporting the input and output arguments described previously, an example is:

```bash
python create_gliner_training_data.py \
  --input-dir data/local/ner/outputs/engage-ft-auto-filtered \
  --output-file data/local/ner/gliner_training_data_filtered/data.json
```

The preprocessing stage converts the original entity annotations into GLiNER-compatible records containing `tokenized_text` and `ner`. Entity end indices are converted from exclusive to inclusive.

### 3. Fine-Tuning GLiNER

Two training scripts are available:

- `finetune_nunerzero_span.py` — fine-tunes GLiNER using the prepared training dataset.
- `finetune_nunerzero_span_synth.py` — fine-tunes GLiNER for experiments involving synthetic training data.

For fine-tuning on the prepared annotated dataset:

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

For combined real and synthetic training, if the synthetic training script supports `--synthetic_files`:

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

Replace the synthetic and evaluation paths with the actual preprocessed datasets.

For synthetic-only fine-tuning, provide the preprocessed synthetic dataset through `--train_files` and omit `--synthetic_files`.

The training scripts load the pretrained GLiNER checkpoint, prepare the training and evaluation data, and save the fine-tuned model in the configured output directory.

### 4. Evaluating Fine-Tuned Models

The `evaluate.py` script is used to evaluate the trained NER models.

Check its available arguments:

```bash
python evaluate.py --help
```

Provide the fine-tuned model directory and gold-standard evaluation dataset using the arguments supported by the script.

Model performance is assessed using precision, recall, and F1 score, allowing comparisons between silver-annotated, synthetic-only, and combined-data fine-tuning approaches.

### 5. Recommended Execution Order

Run the scripts in the following order, depending on the experiment:

| Step | Script | Purpose |
|---|---|---|
| 1 | `synthetic_data.py` or `improved_synthetic_data.py` | Generate synthetic NER examples |
| 2 | `create_gliner_training_data.py` | Prepare annotated training data |
| 3 | `create_gliner_training_data_with_syn.py` | Prepare data for synthetic-data experiments |
| 4 | `finetune_nunerzero_span.py` | Fine-tune on annotated data |
| 5 | `finetune_nunerzero_span_synth.py` | Fine-tune using synthetic data |
| 6 | `evaluate.py` | Evaluate models on the gold-standard dataset |

Not every script is required for every experiment. For example, synthetic-only fine-tuning does not require a separate silver-annotated training dataset.

**Note:** The commands above are based on the filenames shown and the script interfaces discussed previously. The actual argument names for each file should be verified using `--help` or by inspecting the script before execution.
