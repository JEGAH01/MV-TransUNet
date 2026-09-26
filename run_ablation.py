"""Generic runner so ablation configs can be trained without editing train_topology.py.

Usage:
    python run_ablation.py <config.yaml>

The config argument is REQUIRED and has no default -- this script previously
defaulted to config_topology.yaml when no argument was received, which caused
an accidental retrain that overwrote checkpoints/topology_full_seed42 on
2026-09-20. It now refuses to run without an explicit, existing config path.
"""
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "ERROR: run_ablation.py requires exactly one argument: the config "
            "path.\nExample: python run_ablation.py "
            "config_topology_ablation_branch_only.yaml",
            file=sys.stderr,
        )
        sys.exit(2)

    config_path = sys.argv[1]
    if not Path(config_path).is_file():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(2)

    print(f"[run_ablation] Using config: {config_path}", flush=True)

    from src.train_topology import main as train_main
    train_main(config_path)


if __name__ == "__main__":
    main()
