"""
Tolerance-sensitivity analysis for skeleton / endpoint / junction F1.

evaluate_topology_v2.py scores the topology model's structural heads with
fixed tolerance radii (skeleton=1px, endpoint=2px, junction=2px), and
evaluate_baseline_structural.py scores the baseline's post-hoc-skeletonized
segmentation output the same way. This script re-scores ALREADY-SAVED
predictions from either evaluator at tolerance radii of 1px, 2px, and 3px
for all three structural types, so the paper can report whether the
reported structural F1 gap is an artefact of the specific 1px/2px/2px
choice or holds across a range of tolerances. It does not re-run either
model -- it is a pure re-scoring pass over saved prediction PNGs.

Two prediction sources are supported:

  --source topology
      Reads the topology model's own predicted skeleton/endpoint/junction
      binary masks, saved unconditionally by evaluate_topology_v2.py at
      <predictions-dir>/<dataset>/predictions/{skeleton,endpoint,junction}/*.png

  --source baseline
      Derives skeleton/endpoint/junction post-hoc from the baseline's
      saved segmentation mask, saved by evaluate_baseline_v2.py at
      <predictions-dir>/<dataset>/predictions/segmentation/*.png
      (identical derivation to evaluate_baseline_structural.py).

Ground truth in both cases is skeletonized from the dataset's mask using
the same binary_skeleton/endpoint_target/junction_target functions
evaluate_topology_v2.py uses to build its own targets (imported directly,
not reimplemented).

Example:
    python -m src.tolerance_sensitivity_analysis ^
        --config config_topology_ablation_a8_scvam_seed123.yaml ^
        --predictions-dir evaluation_outputs/topology_a8_scvam_seed123_no_tta ^
        --source topology ^
        --datasets DRIVE_test STARE CHASE_DB1 HRF

    python -m src.tolerance_sensitivity_analysis ^
        --config config_baseline_matched_topology_scvam_seed123.yaml ^
        --predictions-dir evaluation_outputs/baseline_scvam_seed123_no_tta ^
        --source baseline ^
        --datasets DRIVE_test STARE CHASE_DB1 HRF
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

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

TOLERANCE_RADII_PX = (1, 2, 3)
STRUCTURE_TYPES = ("skeleton", "endpoint", "junction")


# ============================================================
# ARGUMENTS
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score saved skeleton/endpoint/junction predictions at "
            "tolerance radii of 1px, 2px, and 3px."
        )
    )

    parser.add_argument("--config", type=str, default="config.yaml")

    parser.add_argument(
        "--predictions-dir",
        type=str,
        required=True,
        help="Output directory of a completed evaluate_*.py run.",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=("topology", "baseline"),
        help=(
            "'topology' reads predictions/{skeleton,endpoint,junction}/*.png "
            "directly. 'baseline' derives them post-hoc from "
            "predictions/segmentation/*.png."
        ),
    )

    parser.add_argument("--datasets", nargs="*", default=None)

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Defaults to '<predictions-dir>_tolerance_sensitivity'.",
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


def load_binary_png(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise IOError(f"Could not read: {path.resolve()}")
    # Saved uint8 binary masks in this codebase are not consistently
    # rescaled to {0, 255}: evaluate_baseline_v2.py / evaluate_topology_v2.py
    # write raw {0, 1} values for arrays that were already uint8 before
    # save_gray() saw them (e.g. the "segmentation" folder), and {0, 255}
    # for arrays that were still bool (e.g. topology's skeleton/endpoint/
    # junction folders). Any nonzero pixel means foreground in both cases.
    return (raw > 0).astype(np.uint8)


# ============================================================
# EVALUATION
# ============================================================


def get_predicted_structures(
    predictions_root: Path,
    dataset_name: str,
    key: str,
    source: str,
) -> Dict[str, np.ndarray]:
    dataset_root = predictions_root / dataset_name / "predictions"

    if source == "topology":
        return {
            "skeleton": load_binary_png(
                dataset_root / "skeleton" / f"{key}.png"
            ),
            "endpoint": load_binary_png(
                dataset_root / "endpoint" / f"{key}.png"
            ),
            "junction": load_binary_png(
                dataset_root / "junction" / f"{key}.png"
            ),
        }

    # source == "baseline": derive post-hoc from the segmentation mask.
    segmentation = load_binary_png(
        dataset_root / "segmentation" / f"{key}.png"
    )
    predicted_skeleton = binary_skeleton(segmentation)
    return {
        "skeleton": predicted_skeleton,
        "endpoint": endpoint_target(predicted_skeleton),
        "junction": junction_target(predicted_skeleton),
    }


def evaluate_dataset_sensitivity(
    predictions_root: Path,
    specification,
    source: str,
) -> List[Dict[str, Any]]:
    pairs = pair_files(specification.image_dir, specification.mask_dir)

    rows: List[Dict[str, Any]] = []

    progress = tqdm(
        pairs,
        desc=f"Tolerance sensitivity ({source}) {specification.name}",
    )

    for pair in progress:
        mask_raw = cv2.imread(str(pair.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise IOError(f"Could not read mask: {pair.mask_path.resolve()}")
        target = (mask_raw > 127).astype(np.uint8)

        target_skeleton = binary_skeleton(target)
        target_structures = {
            "skeleton": target_skeleton,
            "endpoint": endpoint_target(target_skeleton),
            "junction": junction_target(target_skeleton),
        }

        predicted_structures = get_predicted_structures(
            predictions_root=predictions_root,
            dataset_name=specification.name,
            key=pair.key,
            source=source,
        )

        for structure_type in STRUCTURE_TYPES:
            prediction = predicted_structures[structure_type]
            target_map = target_structures[structure_type]

            if prediction.shape != target_map.shape:
                raise RuntimeError(
                    f"Shape mismatch for {pair.key} ({structure_type}): "
                    f"prediction={prediction.shape}, "
                    f"target={target_map.shape}"
                )

            for tolerance_radius in TOLERANCE_RADII_PX:
                metrics = tolerant_map_metrics(
                    probability=prediction.astype(np.float32),
                    target=target_map,
                    threshold=0.5,
                    tolerance_radius=tolerance_radius,
                )

                rows.append(
                    {
                        "dataset": specification.name,
                        "sample": pair.key,
                        "structure_type": structure_type,
                        "tolerance_px": tolerance_radius,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                    }
                )

    return rows


def summarize_sensitivity(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Mapping[str, Any]]] = {}

    for row in rows:
        key = (row["dataset"], row["structure_type"], row["tolerance_px"])
        groups.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []

    for (dataset, structure_type, tolerance_px), group_rows in sorted(
        groups.items()
    ):
        for metric in ("precision", "recall", "f1"):
            values = np.asarray(
                [float(row[metric]) for row in group_rows],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]

            summary_rows.append(
                {
                    "dataset": dataset,
                    "structure_type": structure_type,
                    "tolerance_px": tolerance_px,
                    "metric": metric,
                    "mean": float(values.mean()) if values.size else float("nan"),
                    "std": float(values.std(ddof=0)) if values.size else float("nan"),
                    "number_of_images": int(values.size),
                }
            )

    return summary_rows


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
        else f"{arguments.predictions_dir}_tolerance_sensitivity"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = resolve_datasets(config=config, requested=arguments.datasets)

    print("=" * 78)
    print("Tolerance-sensitivity analysis (1px / 2px / 3px)")
    print("=" * 78)
    print("Source:", arguments.source)
    print("Predictions directory:", predictions_root.resolve())
    print("Output directory:", output_root.resolve())
    print("Datasets:", ", ".join(datasets.keys()))

    all_rows: List[Dict[str, Any]] = []

    for specification in datasets.values():
        rows = evaluate_dataset_sensitivity(
            predictions_root=predictions_root,
            specification=specification,
            source=arguments.source,
        )
        all_rows.extend(rows)

    write_csv(output_root / "per_image_tolerance_sensitivity.csv", all_rows)

    summary_rows = summarize_sensitivity(all_rows)
    write_csv(output_root / "tolerance_sensitivity_summary.csv", summary_rows)

    with (output_root / "tolerance_sensitivity_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            [
                {key: json_safe(value) for key, value in row.items()}
                for row in summary_rows
            ],
            stream,
            indent=2,
            sort_keys=True,
        )

    print()
    print("-" * 78)
    print("F1 by dataset / structure type / tolerance radius")
    print("-" * 78)
    for row in summary_rows:
        if row["metric"] != "f1":
            continue
        print(
            f"{row['dataset']:12s} {row['structure_type']:9s} "
            f"{row['tolerance_px']}px: "
            f"{row['mean']:.4f} +/- {row['std']:.4f}"
        )

    print()
    print("=" * 78)
    print("Tolerance-sensitivity analysis complete")
    print(
        "Combined summary:",
        (output_root / "tolerance_sensitivity_summary.csv").resolve(),
    )


if __name__ == "__main__":
    main()
