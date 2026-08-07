"""Core data transformations for the PB-BO offline dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import shutil
from pathlib import Path
from typing import Any

import torch


PARAM_ORDER = [
    "fetchWidth", "numFetchBufferEntries", "numRasEntries", "maxBrCount",
    "ICacheWay", "ICacheFetchBytes", "ICacheTLB", "decodeWidth",
    "numRobEntries", "numIntPhysRegisters", "numFpPhysRegisters",
    "numLdqEntries", "numStqEntries", "numDCacheBanks", "DCacheWay",
    "DCacheMSHR", "DCacheTLB", "MemIssueWidth", "MemNumEntries",
    "MemDispatchWidth", "IntIssueWidth", "IntNumEntries",
    "IntDispatchWidth", "FpIssueWidth", "FpNumEntries", "FpDispatchWidth",
]

BENCHMARKS = [
    "Dhrystone", "median", "mm", "multiply", "qsort", "rsort", "spmv",
    "towers", "vvadd",
]

STATIC_FILES = [
    "local_hierarchy_embeddings.pt",
    "module_graphs.pkl",
    "param_ast_mask.pt",
    "system_wl_embeddings.pt",
]

REF_POINT_RAW = [0.7, 625.0, 2500000.0]
IDEAL_POINT_RAW = [1.5, 250.0, 800000.0]
REF_POINT_NORM = [1.0, 1.0, 1.0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(csv_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        records = list(reader)

    required = {
        "config_id", "power_w", "area_um2", *PARAM_ORDER,
        *(f"ipc_{name}" for name in BENCHMARKS),
    }
    missing = sorted(required - set(reader.fieldnames or []))
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {', '.join(missing)}. "
            "Use the complete dataset/offline.csv distributed with PB-BO."
        )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        records = records[:limit]
    if not records:
        raise ValueError(f"No records found in {csv_path}")

    config_ids = [int(record["config_id"]) for record in records]
    if len(config_ids) != len(set(config_ids)):
        raise ValueError("config_id values must be unique")
    if config_ids != sorted(config_ids):
        raise ValueError("offline.csv must be sorted by config_id")
    return records


def derive_value_vocab(records: list[dict[str, str]]) -> dict[str, Any]:
    param_values = {
        param: sorted({int(record[param]) for record in records})
        for param in PARAM_ORDER
    }
    param_offsets: dict[str, int] = {}
    offset = 0
    for param in PARAM_ORDER:
        param_offsets[param] = offset
        offset += len(param_values[param])
    return {
        "param_order": PARAM_ORDER,
        "param_values": param_values,
        "param_offsets": param_offsets,
        "total_unique_ids": offset,
    }


def build_value_data(records: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    vocab = derive_value_vocab(records)
    param_values = vocab["param_values"]
    param_offsets = vocab["param_offsets"]
    value_to_id = {
        param: {
            value: index + param_offsets[param]
            for index, value in enumerate(param_values[param])
        }
        for param in PARAM_ORDER
    }

    num_records = len(records)
    num_params = len(PARAM_ORDER)
    discrete_ids = torch.zeros(num_records, num_params, dtype=torch.long)
    norm_scalars = torch.zeros(num_records, num_params, 1, dtype=torch.float32)
    log2_scalars = torch.zeros(num_records, num_params, 1, dtype=torch.float32)

    for row_index, record in enumerate(records):
        for param_index, param in enumerate(PARAM_ORDER):
            value = int(record[param])
            values = param_values[param]
            discrete_ids[row_index, param_index] = value_to_id[param][value]
            if values[-1] > values[0]:
                norm_scalars[row_index, param_index, 0] = (
                    (value - values[0]) / (values[-1] - values[0])
                )
            log2_scalars[row_index, param_index, 0] = (
                math.log2(value) if value > 0 else 0.0
            )
    return {
        "discrete_ids": discrete_ids,
        "norm_scalars": norm_scalars,
        "log2_scalars": log2_scalars,
    }


def build_ppa_targets(records: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    perf = [
        [float(record[f"ipc_{benchmark}"]) for benchmark in BENCHMARKS]
        for record in records
    ]
    power = [[float(record["power_w"]) * 1000.0] for record in records]
    area = [[float(record["area_um2"])] for record in records]
    return {
        "perf": torch.tensor(perf, dtype=torch.float32),
        "power": torch.tensor(power, dtype=torch.float32),
        "area": torch.tensor(area, dtype=torch.float32),
    }


def build_tabular_artifacts(
    records: list[dict[str, str]], output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_ids = [int(record["config_id"]) for record in records]
    ppa_targets = build_ppa_targets(records)

    with (output_dir / "config_id_map.json").open("w") as stream:
        json.dump(config_ids, stream)
    torch.save(build_value_data(records), output_dir / "value_data.pt")
    torch.save(ppa_targets, output_dir / "ppa_targets.pt")
    torch.save(
        torch.ones(len(records), dtype=torch.bool), output_dir / "ppa_mask.pt"
    )
    torch.save(
        {
            "benchmark_order": BENCHMARKS,
            "perf_ref": ppa_targets["perf"].mean(dim=0),
        },
        output_dir / "perf_meta.pt",
    )
    return {"config_ids": config_ids, "ppa_targets": ppa_targets}


def install_static_features(
    static_dir: Path, output_dir: Path, num_records: int, full_size: int
) -> None:
    missing = [name for name in STATIC_FILES if not (static_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing static feature assets in {static_dir}: {', '.join(missing)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if num_records == full_size:
        for name in STATIC_FILES:
            shutil.copy2(static_dir / name, output_dir / name)
        return

    local_hierarchy = torch.load(
        static_dir / "local_hierarchy_embeddings.pt", weights_only=False
    )
    torch.save(
        local_hierarchy[:num_records],
        output_dir / "local_hierarchy_embeddings.pt",
    )

    with (static_dir / "module_graphs.pkl").open("rb") as stream:
        module_graphs = pickle.load(stream)
    with (output_dir / "module_graphs.pkl").open("wb") as stream:
        pickle.dump(module_graphs[:num_records], stream)

    shutil.copy2(static_dir / "param_ast_mask.pt", output_dir / "param_ast_mask.pt")

    wl = torch.load(static_dir / "system_wl_embeddings.pt", weights_only=False)
    wl_tiny = dict(wl)
    wl_tiny["embeddings"] = wl["embeddings"][:num_records]
    wl_tiny["config_ids"] = wl["config_ids"][:num_records]
    torch.save(wl_tiny, output_dir / "system_wl_embeddings.pt")


def build_pareto_artifacts(
    ppa_targets: dict[str, torch.Tensor],
    config_ids: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    from botorch.utils.multi_objective.pareto import is_non_dominated

    output_dir.mkdir(parents=True, exist_ok=True)
    perf = ppa_targets["perf"]
    power = ppa_targets["power"]
    area = ppa_targets["area"]
    ipc_geomean = perf.clamp(min=1e-8).log().mean(dim=1).exp()
    power_vec = power.squeeze(-1)
    area_vec = area.squeeze(-1)
    objectives = torch.stack([ipc_geomean, -power_vec, -area_vec], dim=1)
    is_pareto = is_non_dominated(objectives.double())
    pareto_indices = is_pareto.nonzero(as_tuple=True)[0].tolist()

    pareto_configs = []
    for index in pareto_indices:
        pareto_configs.append({
            "index": index,
            "config_id": config_ids[index],
            "ipc_geomean": round(ipc_geomean[index].item(), 6),
            "power_mW": round(power_vec[index].item(), 6),
            "area_um2": round(area_vec[index].item(), 6),
        })
    pareto_configs.sort(
        key=lambda item: (
            -item["ipc_geomean"], item["power_mW"], item["area_um2"]
        )
    )
    pareto_ppa = [
        [item["ipc_geomean"], item["power_mW"], item["area_um2"]]
        for item in pareto_configs
    ]
    pareto_ids = [str(item["config_id"]) for item in pareto_configs]

    ppa_tensor = torch.tensor(pareto_ppa, dtype=torch.double)
    ideal = torch.tensor(IDEAL_POINT_RAW, dtype=torch.double)
    reference = torch.tensor(REF_POINT_RAW, dtype=torch.double)
    ppa_normalized = (ppa_tensor - ideal) / (reference - ideal)
    hypervolume = Hypervolume(
        ref_point=torch.tensor([-1.0, -1.0, -1.0], dtype=torch.double)
    )
    hv_value = float(hypervolume.compute(-ppa_normalized))

    json_payload = {
        "total_configs": len(config_ids),
        "pareto_size": len(pareto_indices),
        "pareto_ids": pareto_ids,
        "pareto_ppa": pareto_ppa,
        "hypervolume": hv_value,
        "ref_point": REF_POINT_RAW,
        "ideal_point": IDEAL_POINT_RAW,
        "ref_point_norm": REF_POINT_NORM,
        "objectives": ["ipc_geomean", "power_mW", "area_um2"],
        "num_configs": len(config_ids),
        "num_pareto": len(pareto_indices),
        "pareto_front": pareto_configs,
    }
    with (output_dir / "pareto_front.json").open("w") as stream:
        json.dump(json_payload, stream, indent=2)

    torch.save({
        "pareto_indices": torch.tensor(pareto_indices, dtype=torch.long),
        "is_pareto": is_pareto,
        "objectives": objectives,
        "pareto_objectives": objectives[pareto_indices],
        "config_ids": config_ids,
        "ipc_geomean": ipc_geomean,
        "obj_names": ["ipc_geomean", "power_mW", "area_um2"],
    }, output_dir / "pareto_front.pt")
    return {"pareto_size": len(pareto_indices), "hypervolume": hv_value}


def artifact_hashes(output_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
