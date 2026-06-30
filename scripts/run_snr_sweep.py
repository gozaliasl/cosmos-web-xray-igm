#!/usr/bin/env python3
"""
Run the X-ray analysis pipeline for a grid of detection SNR thresholds.

For each requested SNR value this script:
  1. Writes a temporary configuration file with the updated SNR threshold
     and unique output directories.
  2. Executes `main_analysis.py`.
  3. Runs `analyze_stacking.py` for every catalog defined in the config.
  4. Generates comparison plots via `compare_stacking.py` (when >=2 catalogs).

Results, figures, stacking products, and the configuration snapshot are saved
under `outputs/` with directory names that encode the SNR threshold
(`snr1p5`, `snr2p0`, ...).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_THRESHOLDS = (1.5, 2.0, 2.5, 3.0)


def load_config(path: Path) -> Dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def write_config(config: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)


def format_threshold(threshold: float) -> str:
    """Create a filesystem-friendly suffix like 'snr1p5'."""
    return f"snr{threshold:.1f}".replace(".", "p")


def get_catalog_names(config: Dict) -> List[str]:
    data_cfg = config.get("data", {})
    catalogs = data_cfg.get("catalogs")
    if catalogs:
        names = []
        for entry in catalogs:
            if not entry:
                continue
            name = entry.get("name")
            if not name:
                path = entry.get("group_catalog")
                if path:
                    name = Path(path).stem
            if name:
                names.append(name)
        return names

    single_catalog = data_cfg.get("catalog_name")
    if single_catalog:
        return [single_catalog]
    if data_cfg.get("group_catalog"):
        return [Path(data_cfg["group_catalog"]).stem]

    raise ValueError("No catalog definitions found in configuration.")


def run_command(cmd: Iterable[str], workdir: Path) -> None:
    cmd = list(cmd)
    print(f"[run_snr_sweep] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=workdir, check=True)


def update_output_paths(config: Dict, suffix: str) -> None:
    output_cfg = config.setdefault("output", {})
    output_cfg["results_dir"] = f"outputs/results_{suffix}"
    output_cfg["figures_dir"] = f"outputs/figures_{suffix}"
    output_cfg["stacking_dir"] = f"outputs/stacking_{suffix}"


def snapshot_config(original_path: Path, temp_path: Path, suffix: str) -> None:
    target_dir = REPO_ROOT / "outputs" / "configs"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{original_path.stem}_{suffix}.yaml"
    shutil.copy2(temp_path, target_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SNR-threshold sweep for the X-ray pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Base configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="List of detection SNR thresholds to evaluate (default: 1.5 2.0 2.5 3.0)",
    )
    args = parser.parse_args()

    base_config_path = (REPO_ROOT / args.config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {base_config_path}")

    base_config = load_config(base_config_path)
    catalog_names = get_catalog_names(base_config)
    if not catalog_names:
        raise RuntimeError("No catalogs found in configuration.")

    compare_enabled = len(catalog_names) >= 2

    for threshold in args.thresholds:
        suffix = format_threshold(threshold)
        print(f"\n[run_snr_sweep] === Processing SNR threshold {threshold:.1f} ({suffix}) ===")

        config_copy = yaml.safe_load(yaml.safe_dump(base_config))  # deep copy
        config_copy.setdefault("detection", {})["snr_threshold"] = float(threshold)
        update_output_paths(config_copy, suffix)

        temp_config_path = REPO_ROOT / f"config_{suffix}.yaml"
        write_config(config_copy, temp_config_path)

        try:
            # Step 1: main analysis
            run_command(
                [sys.executable, "main_analysis.py", "--config", str(temp_config_path)],
                REPO_ROOT,
            )

            # Step 2: stacking analysis per catalog
            for catalog in catalog_names:
                run_command(
                    [
                        sys.executable,
                        "analyze_stacking.py",
                        "--config",
                        str(temp_config_path),
                        "--catalog",
                        catalog,
                    ],
                    REPO_ROOT,
                )

            # Step 3: multi-catalog comparison plots
            if compare_enabled:
                run_command(
                    [sys.executable, "compare_stacking.py", "--config", str(temp_config_path)],
                    REPO_ROOT,
                )

            # Preserve a snapshot of the config that produced these outputs
            snapshot_config(base_config_path, temp_config_path, suffix)

        finally:
            if temp_config_path.exists():
                temp_config_path.unlink()

    print("\n[run_snr_sweep] Completed SNR sweep.")


if __name__ == "__main__":
    main()
