# Improving Domain-specific NER with Synthetic Data

This repository provides code and datasets for the master's thesis **Improving Domain-specific NER with Synthetic Data**.

The thesis investigates whether **synthetic training data** generated using `Qwen/Qwen3-4B-Instruct-2507` can improve **named entity recognition (NER)** in scientific publications related to renewable energy, biodiversity, and environmental sustainability.

Using GLiNER with the pretrained `numind/NuNerZero_span` checkpoint, three fine-tuning approaches are compared: **silver-annotated data**, **synthetic data**, and **combined training data**. The experiments cover 16 domain-specific entity categories and investigate different few-shot prompting strategies, dataset sizes, and annotation guidelines.

This work builds on the [Sustainability ENGAGE project](https://gitlab.uzh.ch/LiRI/projects/engage), which provides the underlying NLP framework and supporting components used in the experiments.

---

## Repository Structure

### Key Components

#### `codes/`
- **Description**: Python scripts for synthetic data generation, data conversion, fine-tuning, and evaluation.
- **Contents**:
  - `synthetic_data_generation/`
    - `synthetic_data-base_prompt.py` — Generates synthetic NER examples using baseline prompting.
    - `synthetic_data-guidelines_improved.py` — Generates synthetic examples with additional annotation guidelines.
  - `data_conversion/`
    - `create_gliner_training_data.py` — Converts silver-annotated data into GLiNER-compatible training examples.
    - `create_gliner_training_data_with_syn.py` — Converts synthetic data into GLiNER-compatible training examples.
  - `finetuning/`
    - `finetune_nunerzero_span.py` — Fine-tunes GLiNER using silver-annotated data.
    - `finetune_nunerzero_span_synth.py` — Fine-tunes GLiNER using synthetic or combined training data.
  - `evaluation/`
    - `evaluate.py` — Evaluates baseline and fine-tuned models using precision, recall, and F1 score.

#### `data/`
- **Description**: Contains raw data, silver-annotated training data, and gold-standard evaluation data.
- **Contents**:
  - `raw/` — Raw data used to prepare the silver-annotated dataset, along with a script for removing duplicates.
  - `silver/` — Silver-annotated scientific texts, stored as a compressed `.7z` archive.
  - `gold/` — Gold-standard annotations stored in Hugging Face Datasets format.

The datasets used in the thesis contain:

| Dataset | Publications | Sentence-level records | Entity mentions |
|---|---:|---:|---:|
| Silver-annotated | 963 | 262,794 | 287,369 |
| Gold-standard | 125 | 142 | 3,260 |

The silver-annotated dataset is used for training, while the gold-standard dataset is used exclusively for evaluation. DOI-based deduplication was performed to prevent publication-level overlap between the datasets.

#### `synthetic_data/`
- **Description**: Contains synthetic NER datasets generated using Qwen.
- **Contents**:
  - `with_guidelines/` — Datasets generated with annotation guidelines using 6, 8, and 10 few-shot examples.
  - `without_guidelines/few_shot_experiment/` — Datasets generated without guidelines using 6, 8, and 10 few-shot examples (250 examples per entity category).
  - `without_guidelines/example_size_experiment/` — Datasets generated using eight-shot prompting with 200, 400, 600, or 800 examples per entity category.

---

## How to Use

### 1. Clone the Repository

Clone this thesis repository:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Set Up the Environment

The experiments build on the [Sustainability ENGAGE project](https://gitlab.uzh.ch/LiRI/projects/engage).

Clone or obtain the original project and navigate to its root directory. Create the Conda environment and install the framework using its provided setup instructions:

```bash
conda create -n engage-ner python=3.11 -y
conda activate engage-ner
pip install -e .
```

The `pip install -e .` command must be executed from the original Sustainability ENGAGE project directory.

Additional dependencies may be required for synthetic data generation and fine-tuning.

### 3. Start the Qwen Inference Server

Synthetic data generation uses `Qwen/Qwen3-4B-Instruct-2507` through vLLM.

Start the inference server using Apptainer:

```bash
CUDA_VISIBLE_DEVICES=0 apptainer run --nv \
  docker://vllm/vllm-openai:v0.24.0-x86_64-cu129-ubuntu2404 \
  Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 1024 \
  --enforce-eager
```

Keep the server running while generating synthetic data.

### 4. Run Experiments

The following examples use scripts from this thesis repository. Configure input/output paths and experiment settings according to the local environment.

**Synthetic Data Generation**

```bash
python codes/synthetic_data_generation/synthetic_data-base_prompt.py
python codes/synthetic_data_generation/synthetic_data-guidelines_improved.py
```

**Data Conversion**

```bash
python codes/data_conversion/create_gliner_training_data.py
python codes/data_conversion/create_gliner_training_data_with_syn.py
```

**Silver-Annotated Fine-Tuning**

```bash
CUDA_VISIBLE_DEVICES=1 \
python codes/finetuning/finetune_nunerzero_span.py \
  --train_files data/local/ner/gliner_training_data_filtered/data.json \
  --model_name numind/NuNerZero_span \
  -o data/local/ner/models/silver_20 \
  -f \
  --batch_size 3 \
  --accumulation_steps 2 \
  --num_train_epochs 2 \
  --train_fraction 0.2
```

**Combined Silver and Synthetic Fine-Tuning**

```bash
CUDA_VISIBLE_DEVICES=1 \
python codes/finetuning/finetune_nunerzero_span_synth.py \
  --train_files data/local/ner/gliner_training_data_filtered/data.json \
  --synthetic_files data/local/ner/synthesized_data_filtered_ext.json \
  --model_name numind/NuNerZero_span \
  -o data/local/ner/models/with_syn_20 \
  -f \
  --batch_size 3 \
  --accumulation_steps 2 \
  --num_train_epochs 2 \
  --train_fraction 0.2
```

For synthetic-only fine-tuning, use `finetune_nunerzero_span_synth.py` with the converted synthetic dataset supplied as `--train_files`.

The main combined experiment uses **20% silver-annotated data and 200 synthetic examples per entity category**, generated with the eight-shot baseline prompt.

**Evaluation**

Evaluation uses the gold-standard dataset and the Sustainability ENGAGE evaluation pipeline.

From the original project's root directory:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 \
python -m engage_nlp.components.ner.evaluate \
  --config src/engage_nlp/components/ner/configs/evaluation_gliner.yaml \
  --model /path/to/finetuned_model
```

Use `--model baseline` to evaluate the baseline model.

The thesis reports mean overall evaluation metrics across five runs.

---

## Sample Data Format

GLiNER training examples contain tokenized text and entity annotations:

```json
{
  "tokenized_text": ["Wind", "turbines", "generate", "electricity", "."],
  "ner": [[0, 1, "wind_energy"]]
}
```

Each entity annotation specifies the starting token index, inclusive ending token index, and entity label.

---

## Notes

- This repository contains scripts developed or modified specifically for the thesis rather than the complete Sustainability ENGAGE framework.
- Unmodified framework components, configuration files, and supporting utilities can be obtained from the [original Sustainability ENGAGE project](https://gitlab.uzh.ch/LiRI/projects/engage).
- Some scripts require adjustments to dataset paths, dependencies, and GPU settings.
- The original experiments were conducted within the Sustainability ENGAGE framework; the standalone commands may require additional configuration.
