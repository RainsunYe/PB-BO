"""BERT adapter boundary for deterministic PB-BO feature generation."""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import torch

from offline_data import PARAM_ORDER


BOTTLENECK_DIM = 128
ADAPTER_NAME = "hw_mlm_adapter"


def _load_description_map(path: Path) -> dict[tuple[str, int], str]:
    pattern = re.compile(r"^\s*When\s+(\w+)\s+is\s+(\d+)\s*,", re.IGNORECASE)
    descriptions = {}
    with path.open() as stream:
        for raw_line in stream:
            line = raw_line.strip()
            match = pattern.match(line)
            if match:
                descriptions[(match.group(1), int(match.group(2)))] = line
    return descriptions


def build_bert_embeddings(
    records: list[dict[str, str]],
    base_model: str,
    adapter_dir: Path,
    prompt_path: Path,
    device_name: str,
    seed: int,
) -> torch.Tensor:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    try:
        import adapters
    except ImportError as error:
        raise RuntimeError(
            "The adapters package is required; install requirements-data.txt"
        ) from error

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    descriptions = _load_description_map(prompt_path)
    unique_keys = sorted({
        (param, int(record[param]))
        for record in records
        for param in PARAM_ORDER
    })
    texts = [
        descriptions.get(
            key, f"When {key[0]} is {key[1]}, this parameter is set to {key[1]}."
        )
        for key in unique_keys
    ]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    encodings = tokenizer(
        texts,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    model = AutoModelForMaskedLM.from_pretrained(base_model)
    adapters.init(model)
    model.load_adapter(
        str(adapter_dir), load_as=ADAPTER_NAME, set_active=True
    )
    bert = model.bert.to(device).eval()

    down_projection = next(
        (
            module
            for _, module in bert.encoder.layer[-1].named_modules()
            if isinstance(module, torch.nn.Linear)
            and module.out_features == BOTTLENECK_DIM
        ),
        None,
    )
    if down_projection is None:
        raise RuntimeError("Could not find the trained 128-dim adapter bottleneck")

    vectors = []
    with torch.no_grad():
        for index in range(len(unique_keys)):
            captured: dict[str, torch.Tensor] = {}

            def capture(_module, _inputs, output):
                captured["bottleneck"] = output.detach()

            hook = down_projection.register_forward_hook(capture)
            bert(
                input_ids=encodings["input_ids"][index:index + 1].to(device),
                attention_mask=encodings["attention_mask"][index:index + 1].to(device),
            )
            hook.remove()
            vector = torch.nn.functional.gelu(captured["bottleneck"])[0, 0, :]
            vectors.append(vector.cpu())

    key_to_vector = {
        key: vectors[index] for index, key in enumerate(unique_keys)
    }
    embeddings = torch.zeros(
        len(records), len(PARAM_ORDER), BOTTLENECK_DIM, dtype=torch.float32
    )
    for row_index, record in enumerate(records):
        for param_index, param in enumerate(PARAM_ORDER):
            embeddings[row_index, param_index] = key_to_vector[
                (param, int(record[param]))
            ]
    return embeddings
