#!/usr/bin/env python3
"""Build all non-raw PB-BO offline artifacts from dataset inputs."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import torch

from bert_features import build_bert_embeddings
from offline_data import (
    artifact_hashes,
    build_pareto_artifacts,
    build_tabular_artifacts,
    install_static_features,
    load_records,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "build_offline_bo.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--reuse-bert", type=Path,
        help="Copy an existing BERT tensor; intended only for pipeline smoke tests.",
    )
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("build_offline_bo")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(run_dir / "log.txt")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def main() -> None:
    args = parse_args()
    with args.config.open() as stream:
        config = json.load(stream)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = args.run_dir or PROJECT_ROOT / "runs" / f"{timestamp}_build_offline_bo"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, run_dir / "config.json")
    logger = configure_logging(run_dir)

    input_csv = resolve_path(config["input_csv"])
    static_dir = resolve_path(config["static_features_dir"])
    output_dir = args.output_dir or resolve_path(config["output_dir"])
    base_model = args.base_model or config["bert"]["base_model"]
    base_model_path = Path(base_model)
    if not base_model_path.is_absolute() and (PROJECT_ROOT / base_model_path).exists():
        base_model = str(PROJECT_ROOT / base_model_path)
    limit = args.limit if args.limit is not None else config.get("limit")

    resolved_config = dict(config)
    resolved_config.update({
        "input_csv": str(input_csv),
        "static_features_dir": str(static_dir),
        "output_dir": str(output_dir),
        "limit": limit,
        "bert": dict(config["bert"], base_model=base_model),
    })
    with (run_dir / "resolved_config.json").open("w") as stream:
        json.dump(resolved_config, stream, indent=2)

    all_records = load_records(input_csv)
    records = all_records[:limit] if limit is not None else all_records
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    embedding_dir = output_dir / "embedding"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building tabular artifacts for %d configs", len(records))
    tabular = build_tabular_artifacts(records, embedding_dir)

    logger.info("Installing fixed AST and hierarchy features")
    install_static_features(
        static_dir, embedding_dir, len(records), len(all_records)
    )

    if args.reuse_bert:
        logger.info("Smoke mode: copying BERT tensor from %s", args.reuse_bert)
        bert = torch.load(args.reuse_bert, weights_only=False)[:len(records)]
    else:
        logger.info("Building BERT adapter embeddings with seed %d", config["seed"])
        bert = build_bert_embeddings(
            records=records,
            base_model=base_model,
            adapter_dir=resolve_path(config["bert"]["adapter_dir"]),
            prompt_path=resolve_path(config["bert"]["prompt_path"]),
            device_name=config["bert"]["device"],
            seed=int(config["seed"]),
        )
    torch.save(bert, embedding_dir / "bert_embeddings.pt")

    logger.info("Building Pareto artifacts")
    pareto_metrics = build_pareto_artifacts(
        tabular["ppa_targets"], tabular["config_ids"], output_dir
    )
    metrics = {
        "seed": int(config["seed"]),
        "num_configs": len(records),
        **pareto_metrics,
        "artifact_sha256": artifact_hashes(output_dir),
    }
    with (run_dir / "metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2)
    logger.info("Done: %s", output_dir)


if __name__ == "__main__":
    main()
