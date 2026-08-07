# Offline BOOM dataset

This directory contains the 4,997 OfflineSet2 design records and the compact
fixed structural inputs required to reproduce PB-BO model data.

## Files

- `offline.csv`: self-contained source table used by the conversion scripts.
- `static_features/embedding/`: fixed AST and hierarchy inputs omitted by a
  flat parameter/PPA table.

## `offline.csv` columns

The first column is `config_id`. It is followed by these 26 design parameters:

```text
fetchWidth
decodeWidth
MemDispatchWidth
MemIssueWidth
MemNumEntries
IntDispatchWidth
IntIssueWidth
IntNumEntries
FpDispatchWidth
FpIssueWidth
FpNumEntries
maxBrCount
numFetchBufferEntries
numRasEntries
numRobEntries
numIntPhysRegisters
numFpPhysRegisters
numLdqEntries
numStqEntries
numDCacheBanks
ICacheWay
ICacheFetchBytes
ICacheTLB
DCacheWay
DCacheMSHR
DCacheTLB
```

The nine benchmark columns are stored in the model's exact order:

```text
ipc_Dhrystone
ipc_median
ipc_mm
ipc_multiply
ipc_qsort
ipc_rsort
ipc_spmv
ipc_towers
ipc_vvadd
```

The remaining measurements are:

```text
avg_ipc     arithmetic mean IPC recorded by the source dataset
power_w     total power in watts
area_um2    total cell area in square micrometers
runtime_s   verilog + mem-comp + lib-comp + synth runtime in seconds
```

All IPC, power, and area values retain the source JSON precision needed to
reproduce the existing float32 tensors exactly.

## Static structural inputs

`offline.csv` does not contain elaborated RTL hierarchy or AST graphs. The
following compact assets are therefore retained explicitly:

- `local_hierarchy_embeddings.pt`: per-configuration local hierarchy features.
- `system_wl_embeddings.pt`: per-configuration 64-dimensional WL features.
- `module_graphs.pkl`: the 14 fixed AST module graphs shared by all records.
- `param_ast_mask.pt`: the fixed parameter-to-AST-module mask.

These files are copied unchanged by the builder and validated together with
the CSV-derived outputs.

## Provenance

The nine IPC columns and full-precision PPA values were taken from
`data/raw/offlineset/{performance,power,area}.json`. The structural assets were
derived by the original boom-dse-workspace AST and hierarchy scripts. The
runtime remains the sum of the recorded flow stages.
