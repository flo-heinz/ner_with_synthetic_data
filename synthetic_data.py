import argparse
import json
import random
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm



synthetic_entities_per_label = 300
temperature = 0.8
few_shot_sentence_count = 8
max_workers = 20
max_entity_occurrences = 5

model = "Qwen/Qwen3-4B-Instruct-2507"
base_url = "http://localhost:8000/v1"

reference_dir = Path("data/local/ner/outputs/engage-ft-auto-filtered")
output_file = Path("data/local/ner/synthesized_data_6shot_initial_prompt.json")


ALLOWED_LABELS = {
    "biogas_energy",
    "biomass_and_biofuel",
    "ecosystem_service_process_function",
    "geographical_location",
    "geothermal_energy",
    "habitat_ecosystem",
    "hydropower_energy",
    "multiple_renewable_energy_sources",
    "organism_common_name",
    "organism_scientific_name",
    "scientific_method",
    "solar_photovoltaic_energy",
    "temporal_information",
    "tidal_energy",
    "wave_energy",
    "wind_energy",
}


PROMPT = """
Generate one new scientific sentence for a named entity recognition dataset.

Target NER label:
{label}

Real annotated examples:
{examples}

Requirements:
- Write exactly one scientifically plausible sentence.
- Infer the meaning of the target label from the examples.
- Do not copy or closely paraphrase the examples.
- Include at least one entity belonging to the target label.
- Do not use the label name itself as the entity text.
- Prefer varied entity expressions.
- Never use underscores in the sentence or entity text.
- Copy entity text exactly from the generated sentence.
""".strip()


TOKEN_PATTERN = re.compile(
    r"\d+(?:\.\d+)+|[^\W\d_]+(?:'[^\W\d_]+)?|\d+|[^\w\s]",
    re.UNICODE,
)



class GeneratedEntity(BaseModel):
    text: str
    label: str


class GeneratedExample(BaseModel):
    sentence: str
    entities: list[GeneratedEntity] = Field(min_length=1)



def normalize(text):
    return " ".join(text.casefold().split())


def label_to_text(label):
    return normalize(label.replace("_", " "))


def tokenize(text):
    return [
        (m.group(), m.start(), m.end())
        for m in TOKEN_PATTERN.finditer(text)
    ]



def load_examples():
    examples = defaultdict(list)
    seen = defaultdict(set)

    for path in reference_dir.iterdir():
        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        try:
            records = json.loads(text)
            if isinstance(records, dict):
                records = [records]
        except json.JSONDecodeError:
            records = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]

        for record in records:
            sentence = record.get("text", "").strip()

            for entity in record.get("entities", []):
                if not sentence or not isinstance(entity, dict):
                    continue

                label = entity.get("label")
                entity_text = entity.get("text", "").strip()

                if label not in ALLOWED_LABELS or not entity_text:
                    continue

                key = (normalize(sentence), normalize(entity_text))

                if key not in seen[label]:
                    seen[label].add(key)
                    examples[label].append(
                        {
                            "sentence": sentence,
                            "entity": entity_text,
                        }
                    )

    if not examples:
        raise ValueError(f"No examples found in {reference_dir}")

    return examples


def choose_examples(label, examples):
    selected = random.sample(
        examples[label],
        min(few_shot_sentence_count, len(examples[label])),
    )

    return "\n\n".join(
        f"Example {i}:\n"
        f"Sentence: {x['sentence']}\n"
        f"Target entity: {x['entity']}"
        for i, x in enumerate(selected, 1)
    )



def find_spans(text, entity_text, offsets):
    spans = []

    for match in re.finditer(re.escape(entity_text), text):
        start, end = match.span()

        indices = [
            i
            for i, (_, token_start, token_end) in enumerate(offsets)
            if token_start < end and token_end > start
        ]

        if not indices:
            continue

        start_token = indices[0]
        end_token = indices[-1] + 1

        if (
            offsets[start_token][1] == start
            and offsets[end_token - 1][2] == end
        ):
            spans.append((start_token, end_token))

    return spans


def overlaps(a, b):
    return (
        a["start_token"] < b["end_token"]
        and b["start_token"] < a["end_token"]
    )


def build_record(generated, target_label):
    text = generated.sentence.strip()

    if not text or "\n" in text or "_" in text:
        raise ValueError("Invalid sentence")

    offsets = tokenize(text)

    if not offsets:
        raise ValueError("Sentence contains no tokens")

    entities = []

    for item in generated.entities:
        entity_text = item.text.strip()
        label = item.label.strip()

        if label != target_label:
            continue

        if not entity_text or "_" in entity_text:
            continue

        # Avoid e.g. "biogas energy" for label "biogas_energy".
        if normalize(entity_text) == label_to_text(target_label):
            continue

        for start_token, end_token in find_spans(
            text,
            entity_text,
            offsets,
        ):
            entity = {
                "text": entity_text,
                "label": target_label,
                "start_token": start_token,
                "end_token": end_token,
            }

            if entity in entities:
                continue

            if any(overlaps(entity, other) for other in entities):
                continue

            entities.append(entity)

    if not entities:
        raise ValueError(f"No valid {target_label} entity")

    entities.sort(
        key=lambda x: (x["start_token"], x["end_token"])
    )

    return {
        "tokens": [token for token, _, _ in offsets],
        "entities": entities,
    }, text, entities

thread_local = threading.local()


def get_client():
    """
    Create one client per worker thread instead of one client
    for every generated sentence.
    """

    if not hasattr(thread_local, "client"):
        thread_local.client = instructor.from_openai(
            OpenAI(
                base_url=base_url,
                api_key="EMPTY",
            ),
            mode=instructor.Mode.JSON,
        )

    return thread_local.client


def generate_one(label, examples):
    prompt = PROMPT.format(
        label=label,
        examples=choose_examples(label, examples),
    )

    generated = get_client().chat.completions.create(
        model=model,
        response_model=GeneratedExample,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temperature,
    )

    return label, *build_record(generated, label)


def main():
    examples = load_examples()
    labels = sorted(set(examples) & ALLOWED_LABELS)

    total = len(labels) * synthetic_entities_per_label

    print(f"Labels: {len(labels)}")
    print(f"Entity annotations per label: {synthetic_entities_per_label}")
    print(f"Target total entity annotations: {total}")
    print(f"Workers: {max_workers}")

    records = []
    seen_sentences = set()

    entity_counts = defaultdict(Counter)
    accepted = Counter()  # accepted entity annotations per target label
    accepted_sentences = Counter()
    rejected = Counter()

    pending_per_label = 2

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        def submit(label):
            future = executor.submit(
                generate_one,
                label,
                examples,
            )
            futures[future] = label

    
        for label in labels:
            for _ in range(pending_per_label):
                submit(label)

        with tqdm(total=total, desc="Generating") as progress:

            while futures:
                done, _ = wait(
                    futures,
                    return_when=FIRST_COMPLETED,
                )

                for future in done:
                    label = futures.pop(future)

          
                    if accepted[label] >= synthetic_entities_per_label:
                        continue

                    try:
                        _, record, text, entities = future.result()

                        sentence_key = normalize(text)

                        if sentence_key in seen_sentences:
                            raise ValueError("Duplicate sentence")

                        
                        # Count every saved annotation, including repeated spans.
                        mention_counts = Counter(
                            normalize(entity["text"]) for entity in entities
                        )
                        n_entities = len(entities)
                        remaining = synthetic_entities_per_label - accepted[label]
                        if n_entities > remaining:
                            raise ValueError("Sentence exceeds remaining entity quota")

                        # Strict cap on each normalized entity expression.
                        if any(
                            entity_counts[label][key] + count > max_entity_occurrences
                            for key, count in mention_counts.items()
                        ):
                            raise ValueError("Entity occurrence limit reached")

                        seen_sentences.add(sentence_key)
                        records.append(record)

                        entity_counts[label].update(mention_counts)
                        accepted[label] += n_entities
                        accepted_sentences[label] += 1
                        progress.update(n_entities)

                    except Exception:
                        rejected[label] += 1

                    if accepted[label] < synthetic_entities_per_label:
                        submit(label)

                if all(
                    accepted[label] >= synthetic_entities_per_label
                    for label in labels
                ):
                    break

        for future in futures:
            future.cancel()


    for label in labels:
        if accepted[label] != synthetic_entities_per_label:
            raise RuntimeError(
                f"{label}: expected "
                f"{synthetic_entities_per_label}, "
                f"got {accepted[label]}"
            )

    # Verify the exact quota in the actual saved records.
    actual_counts = Counter(
        entity["label"] for record in records for entity in record["entities"]
    )
    for label in labels:
        if actual_counts[label] != synthetic_entities_per_label:
            raise RuntimeError(
                f"Saved annotation mismatch for {label}: {actual_counts[label]}"
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        raise FileExistsError(f"Output file already exists: {output_file}")

    output_file.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved {len(records)} sentences to {output_file}")

    for label in labels:
        counts = entity_counts[label]

        print(f"\n{label}")
        print(f"  Accepted entity annotations: {accepted[label]}")
        print(f"  Accepted sentences: {accepted_sentences[label]}")
        print(f"  Rejected: {rejected[label]}")
        print(f"  Different entities: {len(counts)}")

        for entity, count in counts.most_common(10):
            print(f"    {entity}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_entities_per_label", "--synthetic_examples_per_label", type=int, default=synthetic_entities_per_label)
    parser.add_argument("--temperature", type=float, default=temperature)
    parser.add_argument("--few_shot_sentence_count", type=int, default=few_shot_sentence_count)
    parser.add_argument("--max_entity_occurrences", type=int, default=max_entity_occurrences)
    parser.add_argument("--output_file", type=Path, required=True)
    args = parser.parse_args()

    synthetic_entities_per_label = args.synthetic_entities_per_label
    temperature = args.temperature
    few_shot_sentence_count = args.few_shot_sentence_count
    max_entity_occurrences = args.max_entity_occurrences
    output_file = Path("data/local/ner") / args.output_file.name

    main()
