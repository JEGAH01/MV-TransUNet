#!/usr/bin/env python
"""
Publication-quality aggregation for MV-TransUNet experiments.

Supported result files:
    native_evaluation_results.yaml
    resolution_normalization_results.yaml
    adaptive_resolution_normalization_results.yaml

The parser supports:
1. Native/D2-AR files with one metric block per dataset.
2. D2-R files structured as:
       results:
         DRIVE_CONTROL:
           short_side_565:
             dice: ...
             ...
           short_side_768:
             dice: ...
             ...

Run from the project root:
    python -m src.analyze_experiments

Optional:
    python -m src.analyze_experiments \
        --experiments-dir experiments \
        --output-dir analysis_outputs
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon


RESULT_FILENAMES: Dict[str, str] = {
    "native_evaluation_results.yaml": "Native",
    "resolution_normalization_results.yaml": "D2-R",
    "adaptive_resolution_normalization_results.yaml": "D2-AR",
}

METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "dice": ("dice", "dice_score", "mean_dice"),
    "thin_dice": (
        "thin_dice",
        "thin_vessel_dice",
        "skeleton_dice",
        "thin",
    ),
    "sensitivity": (
        "sensitivity",
        "recall",
        "true_positive_rate",
        "tpr",
        "se",
    ),
    "specificity": (
        "specificity",
        "true_negative_rate",
        "tnr",
        "sp",
    ),
    "accuracy": ("accuracy", "acc"),
    "auc": ("auc", "roc_auc", "auroc"),
}

DATASET_ALIASES: Dict[str, str] = {
    "DRIVE_CONTROL": "DRIVE",
    "DRIVE": "DRIVE",
    "STARE": "STARE",
    "CHASE_DB1": "CHASE_DB1",
    "CHASE": "CHASE_DB1",
    "HRF": "HRF",
    "HFR": "HRF",
}

SEED_PATTERN = re.compile(r"(?:seed[_-]?)(\d+)", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")
SHORT_SIDE_PATTERN = re.compile(
    r"(?:short[_ -]?side|resolution|size)[_ -]?(\d{2,4})",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate MV-TransUNet Native, D2-R, and D2-AR results."
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory containing experiment result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory in which analysis outputs will be created.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_key(value: Any) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def normalize_dataset_name(value: Any) -> Optional[str]:
    name = (
        str(value)
        .strip()
        .upper()
        .replace("-", "_")
        .replace(" ", "_")
    )
    return DATASET_ALIASES.get(name)


def to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, str):
        stripped = value.strip()
        if NUMBER_PATTERN.match(stripped):
            number = float(stripped)
            return number if math.isfinite(number) else None

    return None


def extract_seed(path: Path, data: Any) -> str:
    for part in reversed(path.parts):
        match = SEED_PATTERN.search(part)
        if match:
            return match.group(1)

    if isinstance(data, Mapping):
        for key in ("seed", "random_seed", "split_seed"):
            if key in data:
                return str(data[key])

        for section_name in ("metadata", "protocol", "experiment"):
            section = data.get(section_name)
            if isinstance(section, Mapping):
                for key in ("seed", "random_seed", "split_seed"):
                    if key in section:
                        return str(section[key])

    return "unknown"


def extract_metrics(mapping: Mapping[str, Any]) -> Dict[str, float]:
    normalized_mapping = {
        normalize_key(key): value
        for key, value in mapping.items()
    }

    metrics: Dict[str, float] = {}

    for canonical_name, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            normalized_alias = normalize_key(alias)
            if normalized_alias not in normalized_mapping:
                continue

            value = to_float(normalized_mapping[normalized_alias])
            if value is not None:
                metrics[canonical_name] = value
                break

    return metrics


def extract_resolution_from_key(value: Any) -> Optional[int]:
    text = str(value).strip()

    explicit_match = SHORT_SIDE_PATTERN.search(text)
    if explicit_match:
        return int(explicit_match.group(1))

    if text.isdigit():
        numeric = int(text)
        if 32 <= numeric <= 8192:
            return numeric

    return None


def infer_resolution(
    mapping: Mapping[str, Any],
    path: Sequence[str],
) -> Optional[int]:
    normalized_mapping = {
        normalize_key(key): value
        for key, value in mapping.items()
    }

    for key in (
        "target_short_side",
        "short_side",
        "resolution",
        "size",
        "adaptive_short_side",
        "median_short_side",
    ):
        if key in normalized_mapping:
            value = to_float(normalized_mapping[key])
            if value is not None:
                return int(round(value))

    for part in reversed(path):
        resolution = extract_resolution_from_key(part)
        if resolution is not None:
            return resolution

    return None


def recursively_find_dataset_blocks(
    node: Any,
    path: Tuple[str, ...] = (),
) -> List[Tuple[str, Mapping[str, Any], Tuple[str, ...]]]:
    """
    Find metric blocks for Native and D2-AR files.

    This generic parser recognizes:
      dataset_name:
        dice: ...
        sensitivity: ...
    and list entries containing an explicit dataset field.
    """
    found: List[
        Tuple[str, Mapping[str, Any], Tuple[str, ...]]
    ] = []

    if isinstance(node, Mapping):
        for key, value in node.items():
            dataset = normalize_dataset_name(key)

            if (
                dataset is not None
                and isinstance(value, Mapping)
                and extract_metrics(value)
            ):
                found.append(
                    (
                        dataset,
                        value,
                        path + (str(key),),
                    )
                )

            found.extend(
                recursively_find_dataset_blocks(
                    value,
                    path + (str(key),),
                )
            )

    elif isinstance(node, Sequence) and not isinstance(
        node,
        (str, bytes),
    ):
        for index, value in enumerate(node):
            if isinstance(value, Mapping):
                dataset: Optional[str] = None

                for candidate in (
                    "dataset",
                    "dataset_name",
                    "name",
                ):
                    if candidate in value:
                        dataset = normalize_dataset_name(
                            value[candidate]
                        )
                        if dataset is not None:
                            break

                if dataset is not None and extract_metrics(value):
                    found.append(
                        (
                            dataset,
                            value,
                            path + (str(index),),
                        )
                    )

            found.extend(
                recursively_find_dataset_blocks(
                    value,
                    path + (str(index),),
                )
            )

    return found


def parse_d2r_results(
    data: Any,
    path: Path,
) -> List[Dict[str, Any]]:
    """
    Parse the nested D2-R resolution-sweep structure.

    Expected canonical structure:
        results:
          DATASET:
            short_side_565:
              dice: ...
              thin_vessel_dice: ...
              ...
    """
    if not isinstance(data, Mapping):
        return []

    results = data.get("results")
    if not isinstance(results, Mapping):
        return []

    seed = extract_seed(path, data)
    records: List[Dict[str, Any]] = []

    for raw_dataset_name, dataset_block in results.items():
        dataset = normalize_dataset_name(raw_dataset_name)
        if dataset is None or not isinstance(dataset_block, Mapping):
            continue

        # Support the rare case where the dataset itself is already
        # a single metric block.
        direct_metrics = extract_metrics(dataset_block)
        if direct_metrics:
            direct_resolution = infer_resolution(
                dataset_block,
                ("results", str(raw_dataset_name)),
            )
            record: Dict[str, Any] = {
                "seed": seed,
                "method": "D2-R",
                "dataset": dataset,
                "resolution": direct_resolution,
                "source_file": str(path),
                "yaml_path": f"results.{raw_dataset_name}",
            }
            record.update(direct_metrics)
            records.append(record)

        # Canonical D2-R sweep: one child mapping per short side.
        for resolution_key, resolution_block in dataset_block.items():
            if not isinstance(resolution_block, Mapping):
                continue

            metrics = extract_metrics(resolution_block)
            if not metrics:
                continue

            resolution = extract_resolution_from_key(resolution_key)
            if resolution is None:
                resolution = infer_resolution(
                    resolution_block,
                    (
                        "results",
                        str(raw_dataset_name),
                        str(resolution_key),
                    ),
                )

            record = {
                "seed": seed,
                "method": "D2-R",
                "dataset": dataset,
                "resolution": resolution,
                "source_file": str(path),
                "yaml_path": (
                    f"results.{raw_dataset_name}.{resolution_key}"
                ),
            }
            record.update(metrics)
            records.append(record)

    return records


def parse_generic_results(
    data: Any,
    path: Path,
    method: str,
) -> List[Dict[str, Any]]:
    seed = extract_seed(path, data)
    records: List[Dict[str, Any]] = []
    seen = set()

    for dataset, mapping, yaml_path in (
        recursively_find_dataset_blocks(data)
    ):
        metrics = extract_metrics(mapping)
        if not metrics:
            continue

        resolution = (
            infer_resolution(mapping, yaml_path)
            if method == "D2-R"
            else None
        )

        signature = (
            dataset,
            resolution,
            tuple(sorted(metrics.items())),
        )
        if signature in seen:
            continue
        seen.add(signature)

        record: Dict[str, Any] = {
            "seed": seed,
            "method": method,
            "dataset": dataset,
            "resolution": resolution,
            "source_file": str(path),
            "yaml_path": ".".join(yaml_path),
        }
        record.update(metrics)
        records.append(record)

    return records


def parse_result_file(
    path: Path,
    method: str,
) -> List[Dict[str, Any]]:
    data = load_yaml(path)

    if method == "D2-R":
        records = parse_d2r_results(data, path)

        # Fallback for older or alternative D2-R output structures.
        if not records:
            records = parse_generic_results(
                data,
                path,
                method,
            )

        return records

    return parse_generic_results(
        data,
        path,
        method,
    )


def discover_result_files(
    experiments_dir: Path,
) -> List[Tuple[Path, str]]:
    if not experiments_dir.exists():
        raise FileNotFoundError(
            "Experiments directory does not exist: "
            f"{experiments_dir.resolve()}"
        )

    discovered: List[Tuple[Path, str]] = []

    for filename, method in RESULT_FILENAMES.items():
        for path in experiments_dir.rglob(filename):
            discovered.append((path, method))

    return sorted(
        discovered,
        key=lambda item: str(item[0]).lower(),
    )


def select_best_d2r_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    non_d2r = frame[
        frame["method"] != "D2-R"
    ].copy()

    d2r = frame[
        (frame["method"] == "D2-R")
        & frame["dice"].notna()
    ].copy()

    if d2r.empty:
        return non_d2r

    best_indices = (
        d2r.groupby(
            ["seed", "dataset"],
            dropna=False,
        )["dice"]
        .idxmax()
    )

    selected = d2r.loc[best_indices].copy()
    selected["method"] = "D2-R (best tested)"

    return pd.concat(
        [non_d2r, selected],
        ignore_index=True,
    )


def build_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for (method, dataset), group in frame.groupby(
        ["method", "dataset"],
        dropna=False,
    ):
        for metric in METRIC_ALIASES:
            values = (
                group[metric]
                .dropna()
                .astype(float)
            )

            if values.empty:
                continue

            n = len(values)
            mean = float(values.mean())
            std = (
                float(values.std(ddof=1))
                if n > 1
                else float("nan")
            )
            ci_half = (
                1.96 * std / math.sqrt(n)
                if n > 1
                else float("nan")
            )

            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "ci_low": (
                        mean - ci_half
                        if n > 1
                        else float("nan")
                    ),
                    "ci_high": (
                        mean + ci_half
                        if n > 1
                        else float("nan")
                    ),
                    "mean_std": (
                        f"{mean:.4f} ± {std:.4f}"
                        if n > 1
                        else f"{mean:.4f}"
                    ),
                }
            )

    return pd.DataFrame(rows)


def rank_biserial(
    differences: np.ndarray,
) -> float:
    differences = differences[
        np.isfinite(differences)
    ]
    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:
        return 0.0

    order = np.argsort(
        np.abs(differences)
    )
    ranks = np.empty(
        len(differences),
        dtype=float,
    )
    ranks[order] = np.arange(
        1,
        len(differences) + 1,
        dtype=float,
    )

    positive = ranks[
        differences > 0
    ].sum()
    negative = ranks[
        differences < 0
    ].sum()

    denominator = positive + negative
    if denominator == 0:
        return 0.0

    return float(
        (positive - negative) / denominator
    )


def paired_statistics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    comparisons = (
        ("Native", "D2-AR"),
        ("Native", "D2-R (best tested)"),
        ("D2-AR", "D2-R (best tested)"),
    )

    rows: List[Dict[str, Any]] = []

    for dataset in sorted(
        frame["dataset"].dropna().unique()
    ):
        dataset_frame = frame[
            frame["dataset"] == dataset
        ]

        for method_a, method_b in comparisons:
            for metric in METRIC_ALIASES:
                method_a_values = (
                    dataset_frame[
                        dataset_frame["method"] == method_a
                    ][["seed", metric]]
                    .dropna()
                    .drop_duplicates("seed")
                    .rename(columns={metric: "a"})
                )

                method_b_values = (
                    dataset_frame[
                        dataset_frame["method"] == method_b
                    ][["seed", metric]]
                    .dropna()
                    .drop_duplicates("seed")
                    .rename(columns={metric: "b"})
                )

                paired = method_a_values.merge(
                    method_b_values,
                    on="seed",
                    how="inner",
                )

                if paired.empty:
                    continue

                differences = (
                    paired["b"].to_numpy()
                    - paired["a"].to_numpy()
                )

                if (
                    len(paired) >= 2
                    and np.any(differences != 0)
                ):
                    try:
                        test = wilcoxon(
                            paired["b"],
                            paired["a"],
                            zero_method="wilcox",
                            alternative="two-sided",
                            method="auto",
                        )
                        statistic = float(
                            test.statistic
                        )
                        p_value = float(
                            test.pvalue
                        )
                    except ValueError:
                        statistic = float("nan")
                        p_value = float("nan")
                else:
                    statistic = float("nan")
                    p_value = float("nan")

                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "method_a": method_a,
                        "method_b": method_b,
                        "n_pairs": len(paired),
                        "mean_difference_b_minus_a": float(
                            np.mean(differences)
                        ),
                        "median_difference_b_minus_a": float(
                            np.median(differences)
                        ),
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                        "rank_biserial_effect": rank_biserial(
                            differences
                        ),
                        "significant_0_05": (
                            bool(p_value < 0.05)
                            if math.isfinite(p_value)
                            else False
                        ),
                    }
                )

    return pd.DataFrame(rows)


def create_metric_chart(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    metric_data = summary[
        summary["metric"] == metric
    ].copy()

    if metric_data.empty:
        return

    datasets = [
        dataset
        for dataset in (
            "DRIVE",
            "STARE",
            "CHASE_DB1",
            "HRF",
        )
        if dataset in metric_data["dataset"].unique()
    ]

    methods = [
        method
        for method in (
            "Native",
            "D2-AR",
            "D2-R (best tested)",
        )
        if method in metric_data["method"].unique()
    ]

    if not datasets or not methods:
        return

    x_positions = np.arange(
        len(datasets),
        dtype=float,
    )
    bar_width = 0.8 / len(methods)

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    for method_index, method in enumerate(methods):
        means: List[float] = []
        errors: List[float] = []

        for dataset in datasets:
            row = metric_data[
                (metric_data["method"] == method)
                & (metric_data["dataset"] == dataset)
            ]

            if row.empty:
                means.append(float("nan"))
                errors.append(0.0)
                continue

            means.append(
                float(row.iloc[0]["mean"])
            )

            standard_deviation = float(
                row.iloc[0]["std"]
            )
            errors.append(
                standard_deviation
                if math.isfinite(standard_deviation)
                else 0.0
            )

        positions = (
            x_positions
            + (
                method_index
                - (len(methods) - 1) / 2
            )
            * bar_width
        )

        axis.bar(
            positions,
            means,
            width=bar_width,
            yerr=errors,
            capsize=4,
            label=method,
        )

    axis.set_xlabel("Dataset")
    axis.set_ylabel(
        metric.replace("_", " ").title()
    )
    axis.set_title(
        f"{metric.replace('_', ' ').title()} "
        "by Dataset and Method"
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(datasets)
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_resolution_sweeps(
    full_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    d2r = full_frame[
        (full_frame["method"] == "D2-R")
        & full_frame["resolution"].notna()
        & full_frame["dice"].notna()
    ].copy()

    if d2r.empty:
        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for dataset in sorted(
        d2r["dataset"].unique()
    ):
        dataset_rows = d2r[
            d2r["dataset"] == dataset
        ]

        grouped = (
            dataset_rows
            .groupby("resolution")["dice"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("resolution")
        )

        figure, axis = plt.subplots(
            figsize=(8, 5)
        )

        axis.errorbar(
            grouped["resolution"],
            grouped["mean"],
            yerr=grouped["std"].fillna(0.0),
            marker="o",
            capsize=4,
        )

        best_row = grouped.loc[
            grouped["mean"].idxmax()
        ]

        axis.scatter(
            [best_row["resolution"]],
            [best_row["mean"]],
            s=90,
            marker="*",
            label=(
                "Best tested: "
                f"{int(best_row['resolution'])} px"
            ),
        )

        axis.set_xlabel(
            "Target short side (pixels)"
        )
        axis.set_ylabel("Dice")
        axis.set_title(
            f"{dataset}: D2-R Resolution Sweep"
        )
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend()

        figure.tight_layout()
        figure.savefig(
            output_dir
            / f"{dataset.lower()}_dice_vs_resolution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)


def create_latex_table(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    dice_summary = summary[
        summary["metric"] == "dice"
    ].copy()

    if dice_summary.empty:
        output_path.write_text(
            "% No Dice results were available.\n",
            encoding="utf-8",
        )
        return

    pivot = dice_summary.pivot(
        index="method",
        columns="dataset",
        values="mean_std",
    )

    preferred_datasets = [
        dataset
        for dataset in (
            "DRIVE",
            "STARE",
            "CHASE_DB1",
            "HRF",
        )
        if dataset in pivot.columns
    ]

    pivot = pivot.reindex(
        columns=preferred_datasets
    )

    latex = pivot.to_latex(
        escape=True,
        na_rep="--",
        caption=(
            "Cross-dataset Dice performance reported as "
            "mean $\\pm$ standard deviation across training seeds."
        ),
        label="tab:cross_dataset_dice",
    )

    output_path.write_text(
        latex,
        encoding="utf-8",
    )


def print_dice_summary(
    summary: pd.DataFrame,
) -> None:
    dice_summary = summary[
        summary["metric"] == "dice"
    ].copy()

    if dice_summary.empty:
        print(
            "\nNo Dice summary could be generated."
        )
        return

    pivot = dice_summary.pivot(
        index="method",
        columns="dataset",
        values="mean_std",
    )

    preferred_datasets = [
        dataset
        for dataset in (
            "DRIVE",
            "STARE",
            "CHASE_DB1",
            "HRF",
        )
        if dataset in pivot.columns
    ]

    pivot = pivot.reindex(
        columns=preferred_datasets
    )

    print("\nDice summary")
    print("=" * 80)
    print(pivot.to_string())
    print("=" * 80)


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    discovered = discover_result_files(
        args.experiments_dir
    )

    if not discovered:
        raise FileNotFoundError(
            "No recognized result YAML files were found under "
            f"{args.experiments_dir.resolve()}.\n"
            "Expected one or more of:\n"
            "  native_evaluation_results.yaml\n"
            "  resolution_normalization_results.yaml\n"
            "  adaptive_resolution_normalization_results.yaml"
        )

    all_records: List[Dict[str, Any]] = []

    print(
        f"Discovered {len(discovered)} result file(s)."
    )

    for path, method in discovered:
        parsed_records = parse_result_file(
            path,
            method,
        )

        print(
            f"  {method:<6} | {path} | "
            f"{len(parsed_records)} result block(s)"
        )

        all_records.extend(parsed_records)

    if not all_records:
        raise RuntimeError(
            "Result files were found, but no recognized "
            "dataset metric blocks could be parsed."
        )

    full_frame = pd.DataFrame(
        all_records
    )

    for metric in METRIC_ALIASES:
        if metric not in full_frame.columns:
            full_frame[metric] = np.nan

    preferred_columns = [
        "seed",
        "method",
        "dataset",
        "resolution",
        *METRIC_ALIASES.keys(),
        "source_file",
        "yaml_path",
    ]

    full_frame = full_frame[
        [
            column
            for column in preferred_columns
            if column in full_frame.columns
        ]
    ].sort_values(
        [
            "seed",
            "method",
            "dataset",
            "resolution",
        ],
        na_position="last",
    )

    all_results_path = (
        args.output_dir
        / "all_results_long.csv"
    )
    full_frame.to_csv(
        all_results_path,
        index=False,
    )

    comparison_frame = select_best_d2r_rows(
        full_frame
    )
    summary = build_summary(
        comparison_frame
    )
    statistics = paired_statistics(
        comparison_frame
    )

    summary_path = (
        args.output_dir
        / "summary_mean_std.csv"
    )
    statistics_path = (
        args.output_dir
        / "paired_statistics.csv"
    )
    latex_path = (
        args.output_dir
        / "summary_mean_std.tex"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )
    statistics.to_csv(
        statistics_path,
        index=False,
    )
    create_latex_table(
        summary,
        latex_path,
    )

    create_metric_chart(
        summary,
        "dice",
        args.output_dir
        / "dice_bar_chart.png",
    )
    create_metric_chart(
        summary,
        "thin_dice",
        args.output_dir
        / "thin_dice_bar_chart.png",
    )
    create_metric_chart(
        summary,
        "auc",
        args.output_dir
        / "auc_bar_chart.png",
    )
    create_resolution_sweeps(
        full_frame,
        args.output_dir
        / "resolution_sweeps",
    )

    print_dice_summary(summary)

    print(
        "\nOutputs saved to: "
        f"{args.output_dir.resolve()}"
    )

    d2r_rows = full_frame[
        full_frame["method"] == "D2-R"
    ]
    if not d2r_rows.empty:
        print("\nD2-R sweep verification")
        print("-" * 80)

        for dataset in sorted(
            d2r_rows["dataset"].unique()
        ):
            dataset_rows = d2r_rows[
                d2r_rows["dataset"] == dataset
            ].sort_values("resolution")

            resolutions = [
                int(value)
                for value in dataset_rows[
                    "resolution"
                ].dropna().tolist()
            ]

            best_row = dataset_rows.loc[
                dataset_rows["dice"].idxmax()
            ]

            print(
                f"{dataset}: parsed resolutions={resolutions} | "
                f"best tested={int(best_row['resolution'])} | "
                f"Dice={float(best_row['dice']):.4f}"
            )

    recognized_seeds = sorted(
        seed
        for seed in comparison_frame[
            "seed"
        ].astype(str).unique()
        if seed != "unknown"
    )

    if len(recognized_seeds) < 3:
        print(
            "\nWARNING: fewer than three recognized "
            "training seeds were found."
        )

    print(
        "Note: Wilcoxon testing across only three seeds has "
        "low power; per-image paired testing should be added later."
    )


if __name__ == "__main__":
    main()