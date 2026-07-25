"""
Repair DRIVE_train topology filenames.

Current incorrect naming:
    001.png, 002.png, ..., 020.png

Correct DRIVE training identifiers:
    21.png, 22.png, ..., 40.png

Mapping:
    001 -> 21
    002 -> 22
    ...
    020 -> 40
"""

from pathlib import Path
from typing import List


TOPOLOGY_ROOT = Path(
    "./datasets_topology/DRIVE_train"
)

# Rename every generated target type that uses the sequential naming.
POSSIBLE_TARGET_DIRECTORIES = [
    "skeletons",
    "endpoints",
    "junctions",
    "radius",
    "radii",
    "radius_maps",
    "width",
    "widths",
    "width_maps",
    "thin_vessels",
    "thin_vessel",
    "thin_masks",
]


def discover_target_directories(
    topology_root: Path,
) -> List[Path]:
    """
    Return existing topology-target directories.
    """

    directories: List[Path] = []

    for directory_name in POSSIBLE_TARGET_DIRECTORIES:
        directory = topology_root / directory_name

        if directory.is_dir():
            directories.append(directory)

    return directories


def rename_directory_targets(
    directory: Path,
) -> None:
    """
    Rename 001-020 to 21-40 inside one target directory.

    Temporary names are used first to prevent accidental filename
    collisions during the operation.
    """

    expected_sources = [
        directory / f"{index:03d}.png"
        for index in range(1, 21)
    ]

    existing_sources = [
        path
        for path in expected_sources
        if path.is_file()
    ]

    if not existing_sources:
        print(
            f"Skipping {directory.name}: "
            "no sequential 001-020 files found."
        )
        return

    if len(existing_sources) != 20:
        missing = [
            path.name
            for path in expected_sources
            if not path.is_file()
        ]

        raise RuntimeError(
            f"{directory} contains only "
            f"{len(existing_sources)} of the expected 20 files.\n"
            f"Missing files: {missing}"
        )

    destination_paths = [
        directory / f"{drive_id}.png"
        for drive_id in range(21, 41)
    ]

    conflicting_destinations = [
        path.name
        for path in destination_paths
        if path.exists()
    ]

    if conflicting_destinations:
        raise FileExistsError(
            f"Cannot rename files in {directory} because the "
            "following destination files already exist:\n"
            f"{conflicting_destinations}"
        )

    temporary_paths = []

    # Phase 1: move every source to an unambiguous temporary name.
    for sequence_index in range(1, 21):
        source_path = (
            directory / f"{sequence_index:03d}.png"
        )

        temporary_path = (
            directory
            / f"__temporary_drive_{sequence_index:03d}.png"
        )

        source_path.rename(temporary_path)
        temporary_paths.append(temporary_path)

    # Phase 2: assign the official DRIVE training identifiers.
    for sequence_index, temporary_path in enumerate(
        temporary_paths,
        start=1,
    ):
        drive_id = sequence_index + 20

        destination_path = (
            directory / f"{drive_id}.png"
        )

        temporary_path.rename(destination_path)

    final_files = [
        directory / f"{drive_id}.png"
        for drive_id in range(21, 41)
    ]

    if not all(path.is_file() for path in final_files):
        raise RuntimeError(
            f"Post-rename validation failed for {directory}."
        )

    print(
        f"Repaired {directory.name}: "
        "001-020 -> 21-40"
    )


def main() -> None:
    """
    Repair and validate DRIVE_train topology target filenames.
    """

    if not TOPOLOGY_ROOT.is_dir():
        raise FileNotFoundError(
            "Topology root was not found:\n"
            f"{TOPOLOGY_ROOT.resolve()}"
        )

    target_directories = discover_target_directories(
        TOPOLOGY_ROOT
    )

    if not target_directories:
        raise RuntimeError(
            "No recognized topology-target directories were found "
            f"inside:\n{TOPOLOGY_ROOT.resolve()}"
        )

    print("=" * 72)
    print("DRIVE Topology Filename Repair")
    print("=" * 72)
    print(f"Root: {TOPOLOGY_ROOT.resolve()}")
    print()

    for directory in target_directories:
        rename_directory_targets(directory)

    print()
    print("=" * 72)
    print("DRIVE topology filename repair completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()