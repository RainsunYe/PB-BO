# PB-BO

Complete executable artifact for the PB-BO offline BOOM DSE baseline. The
`pb-bo` executable is unchanged. The repository also contains the minimal
inputs and scripts needed to rebuild its model-readable offline embeddings.

## Retained package contents

- `pb-bo`: executable entry point.
- `dataset/offline.csv`: 4,997 configurations, 26 parameters, nine benchmark
  IPC values, aggregate IPC, power, area, and runtime.
- `dataset/static_features/`: fixed AST and hierarchy inputs that a flat CSV
  cannot encode.
- `scripts/`: the three core data conversion modules.
- `configs/build_offline_bo.json`: reproducible build configuration.
- `data/bert/`: trained 128-dimensional BERT adapter and prompt.
- `data/offline_bo/`: ready-to-use model inputs.
- `data/raw/` and `data/checkpoints/`: runtime data and trained PB-BO model.
- `outputs/`: retained reference results.

## Environment

Use Python 3.10+ and install the complete runtime and data-builder
dependencies from the single requirements file:

```bash
pip install -r requirements-data.txt
```

The public `bert-base-uncased` base model is not duplicated because its weight
file is about 438 MB. Pass a local model directory or allow Transformers to
resolve the configured standard model name. The project-specific trained
adapter is retained under `data/bert/adapter/`.

## Rebuild embeddings

Build into a new directory; the builder refuses to overwrite a non-empty
output directory:

```bash
python scripts/build_offline_bo.py \
  --config configs/build_offline_bo.json \
  --output-dir rebuilt_offline_bo \
  --base-model /path/to/bert-base-uncased
```

The command reads `dataset/offline.csv`, `dataset/static_features/`, and
`data/bert/`. Build configuration, resolved paths, logs, seed, metrics, and
artifact hashes are saved under `runs/`.

## Run DSE

```bash
./pb-bo --seed 172 --alpha 0.2 --beta 0.6 --gpu 0
```

Run without LLM calls:

```bash
./pb-bo --seed 292 --alpha 0.2 --beta 0.6 --gpu 0 --no_llm
```

Compute metrics for a completed run:

```bash
./pb-bo metrics --result outputs/seed_172/result.json
```

The retained comparison results are `outputs/seed_172/` and
`outputs/seed_292/`.
