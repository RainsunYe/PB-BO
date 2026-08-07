# PB-BO BERT assets

`adapter/` contains the trained `hw_mlm_adapter` 128-dimensional bottleneck
adapter. `prompt/param_value_description.txt` contains the parameter-value text
used to generate embeddings.

The public `bert-base-uncased` base weights are intentionally not duplicated:
the local source file is about 438 MB, exceeding the normal per-file size for a
GitHub repository. Pass a local base-model directory with `--base-model`, or use
the configured standard model name when network model resolution is available.

The independently regenerated `bert_embeddings.pt` is verified byte-for-byte
against `data/offline_bo/embedding/bert_embeddings.pt`.

SHA-256:

```text
6c48ea226c49c491b95b97b6e7e05b5c29349b0283cd55afb6ec3839fe493534  adapter/adapter_config.json
72f830980420b989a102a9a0db991d9928afa54d26b407abc95be20b23ca786b  adapter/pytorch_adapter.bin
d6a213f5cb28d0a20f6b743a0ac5d9f85e67312b764e1029e48427474219711d  prompt/param_value_description.txt
```
