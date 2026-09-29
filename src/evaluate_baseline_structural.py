"""
Post-hoc structural (skeleton / endpoint / junction) F1 evaluation for the
baseline MV-TransUNet model.

The baseline has no dedicated topology heads, so its structural performance
has to be derived from its single segmentation output. This script:

  1. Loads the binary segmentation prediction already saved by
     evaluate_baseline_v2.py (predictions/segmentation/<key>.png). It does
     NOT re-run the model -- this is a pure post-processing pass over
     predictions the baseline evaluation already produced.
  2. Skeletonizes that prediction, then extracts endpoint/junction pixels
     from the skeleton using the SAME neighbour-count rule
     evaluate_topology_v2.py uses to build its ground-truth
     endpoint/junction targets (endpoint_target / junction_target,
     imported directly from that module, not reimplemented).
  3. Scores skeleton/endpoint/junction against ground truth with the
     IDENTICAL tolerant-matching function and tolerance radii
     (skeleton=1px, endpoint=2px, junction=2px) that evaluate_topology_v2.py
     uses to score the topology model's dedicated heads (tolerant_map_metrics,
     also imported directly, not reimplemented). This guarantees the
     baseline and topology structural numbers are computed by literally the
     same code path, so the comparison is apples-to-apples.

Prerequisite:
    Run evaluate_baseline_v2.py first for the checkpoint/config in question
    (predictions/segmentation/*.png must already exist under
    --predictions-dir). --save-probabilities is not required.

Example:
    python -m src.evaluate_baseline_structural ^
        --config config_baseline_matched_topology_scvam_seed123.yaml ^
        --predictions-dir evaluation_outputs/baseline_scvam_seed123_no_tta ^
        --datasets DRIVE_test STARE CHASE_DB1 HRF
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.evaluate_baseline_v2 import load_config, pair_files, resolve_datasets
from src.evaluate_topology_v2 import (
    binary_skeleton,
    endpoint_target,
    junction_target,
    tolerant_map_metrics,
)


# ============================================================
# ARGUMENTS
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc skeleton/endpoint/junction structural F1 for a "
            "baseline evaluate_baseline_v2.py prediction run."
        )
    )

    parser.add_argument("--config", type=str, default="config.yaml")

    parser.add_argument(
        "--predictions-dir",
        type=str,
        required=True,
        help=(
            "Output directory of a completed evaluate_baseline_v2.py run "
            "(must contain <dataset>/predictions/segmentation/*.png)."
        ),
    )

    parser.add_argument("--datasets", nargs="*", default=None)

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Defaults to '<predictions-dir>_structural'.",
    )

    return parser.parse_args()


# ============================================================
# OUTPUT HELPERS
# ============================================================


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            {key: json_safe(value) for key, value in payload.items()},
            stream,
            indent=2,
            sort_keys=True,
        )


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    excluded = {"dataset", "sample"}

    metric_names = sorted(
        {key for row in rows for key in row if key not in excluded}
    )

    summary: Dict[str, float] = {}

    for metric in metric_names:
        values = np.asarray(
            [float(row[metric]) for row in rows if metric in row],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]

        summary[metric] = float(values.mean()) if values.size else float("nan")
        summary[f"{metric}_std"] = (
            float(values.std(ddof=0)) if values.size else float("nan")
        )

    return summary


# ============================================================
# EVALUATION
# ============================================================


def evaluate_dataset_structural(
    predictions_root: Path,
    specification,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    pairs = pair_files(specification.image_dir, specification.mask_dir)

    segmentation_dir = (
        predictions_root
        / specification.name
        / "predictions"
        / "segmentation"
    )

    if not segmentation_dir.is_dir():
        raise FileNotFoundError(
            f"No saved baseline predictions found at "
            f"{segmentation_dir.resolve()}. Run evaluate_baseline_v2.py "
            "for this config/checkpoint first."
        )

    rows: List[Dict[str, Any]] = []

    progress = tqdm(pairs, desc=f"Structural (baseline) {specification.name}")

    for pair in progress:
        prediction_path = segmentation_dir / f"{pair.key}.png"

        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"Missing saved prediction for {pair.key}: "
                f"{prediction_path.resolve()}"
            )

        prediction_raw = cv2.imread(
            str(prediction_path), cv2.IMREAD_GRAYSCALE
        )
        if prediction_raw is None:
            raise IOError(
                f"Could not read prediction: {prediction_path.resolve()}"
            )
        # evaluate_baseline_v2.py's save_gray() writes uint8 binary masks
        # as raw {0, 1} values (it only rescales to {0, 255} for bool
        # arrays), so any nonzero pixel -- not just > 127 -- means
        # foreground here.
        prediction_binary = (prediction_raw > 0).astype(np.uint8)

        mask_raw = cv2.imread(str(pair.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise IOError(f"Could not read mask: {pair.mask_path.resolve()}")
        target = (mask_raw > 127).astype(np.uint8)

        if prediction_binary.shape != target.shape:
            raise RuntimeError(
                f"Shape mismatch for {pair.key}: "
                f"prediction={prediction_binary.shape}, "
                f"target={target.shape}"
            )

        predicted_skeleton = binary_skeleton(prediction_binary)
        predicted_endpoint = endpoint_target(predicted_skeleton)
        predicted_junction = junction_target(predicted_skeleton)

        target_skeleton = binary_skeleton(target)
        target_endpoint = endpoint_target(target_skeleton)
        target_junction = junction_target(target_skeleton)

        row: Dict[str, Any] = {
            "dataset": specification.name,
            "sample": pair.key,
        }

        row.update(
            {
                f"skeleton_{key}": value
                for key, value in tolerant_map_metrics(
                    probability=predicted_skeleton.astype(np.float32),
                    target=target_skeleton,
                    threshold=0.5,
                    tolerance_radius=1,
                ).items()
            }
        )

        row.update(
            {
                f"endpoint_{key}": value
                for key, value in tolerant_map_metrics(
                    probability=predicted_endpoint.astype(np.float32),
                    target=target_endpoint,
                    threshold=0.5,
                    tolerance_radius=2,
                ).items()
            }
        )

        row.update(
            {
                f"junction_{key}": value
                for key, value in tolerant_map_metrics(
                    probability=predicted_junction.astype(np.float32),
                    target=target_junction,
                    threshold=0.5,
                    tolerance_radius=2,
                ).items()
            }
        )

        rows.append(row)

        progress.set_postfix(
            skeleton_f1=f"{row['skeleton_f1']:.4f}",
            endpoint_f1=f"{row['endpoint_f1']:.4f}",
            junction_f1=f"{row['junction_f1']:.4f}",
        )

    summary = summarize_rows(rows)
    summary.update(
        {
            "dataset": specification.name,
            "number_of_images": len(rows),
            "source": "baseline_segmentation_posthoc_skeletonization",
            "skeleton_tolerance_px": 1,
            "endpoint_tolerance_px": 2,
            "junction_tolerance_px": 2,
        }
    )

    return summary, rows


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)

    predictions_root = Path(arguments.predictions_dir)
    if not predictions_root.is_dir():
        raise FileNotFoundError(
            f"Predictions directory not found: {predictions_root.resolve()}"
        )

    output_root = Path(
        arguments.output_dir
        if arguments.output_dir is not None
        else f"{arguments.predictions_dir}_structural"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = resolve_datasets(config=config, requested=arguments.datasets)

    print("=" * 78)
    print(
        "Baseline post-hoc structural "
        "(skeleton/endpoint/junction) evaluation"
    )
    print("=" * 78)
    print("Predictions directory:", predictions_root.resolve())
    print("Output directory:", output_root.resolve())
    print("Datasets:", ", ".join(datasets.keys()))
    print(
        "Tolerance radii: skeleton=1px, endpoint=2px, junction=2px "
        "(matches evaluate_topology_v2.py)"
    )

    summaries: List[Dict[str, Any]] = []

    for specification in datasets.values():
        summary, rows = evaluate_dataset_structural(
            predictions_root, specification
        )

        dataset_root = output_root / specification.name
        write_csv(
            dataset_root / "per_image_structural_metrics.csv", rows
        )
        write_json(
            dataset_root / "structural_summary_metrics.json", summary
        )
        summaries.append(summary)

        print()
        print("-" * 78)
        print(f"{specification.name} ({summary['number_of_images']} images)")
        for metric in ("skeleton_f1", "endpoint_f1", "junction_f1"):
            mean = summary.get(metric, float("nan"))
            std = summary.get(f"{metric}_std", float("nan"))
            print(f"{metric:16s}: {mean:.6f} +/- {std:.6f}")

    write_csv(
        output_root / "all_dataset_structural_summary.csv", summaries
    )

    with (
        output_root / "all_dataset_structural_summary.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(
            {
                str(summary["dataset"]): {
                    key: json_safe(value)
                    for key, value in summary.items()
                }
                for summary in summaries
            },
            stream,
            indent=2,
            sort_keys=True,
        )

    print()
    print("=" * 78)
    print("Baseline structural evaluation complete")
    print(
        "Combined summary:",
        (output_root / "all_dataset_structural_summary.csv").resolve(),
    )


if __name__ == "__main__":
    main()
