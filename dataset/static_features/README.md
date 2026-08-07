# Fixed structural features

These assets are required because a flat design-parameter CSV does not encode
elaborated RTL hierarchy or AST graphs. The full builder copies them unchanged;
the tiny builder slices only per-configuration tensors.

SHA-256:

```text
15a51fb0275a1bcbacee8aa89ff90234ab1a7cebfcc87341edfe56913cca6837  embedding/local_hierarchy_embeddings.pt
1f162ba4f22c9fd14c5a7fdab6cb92a739a3da37979afc16a7b8d688557002c5  embedding/module_graphs.pkl
7374d1b1d92bb63ecd3cdaa2dca9dfc343f2e954b876e023a2f6b6d6be9d77b2  embedding/param_ast_mask.pt
30f6b919b1652ef610bde35387087469804d271e6f3b76422bed14400aaca1dc  embedding/system_wl_embeddings.pt
```
