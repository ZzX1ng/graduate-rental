#!/usr/bin/env python3
"""Archive small result artifacts for probing-guided W4A4/outlier8 runs."""

import argparse
import csv
import shutil
from pathlib import Path


RESULT_FILES = (
    "val_metrics.json",
    "quant_summary.json",
    "outlier_runtime_stats.json",
    "activation_calibration_stats.json",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--status",
        action="append",
        nargs=2,
        metavar=("PHASE", "STATUS_TSV"),
        required=True,
    )
    args = parser.parse_args()

    manifest = []
    for phase, status_path_value in args.status:
        status_path = Path(status_path_value)
        phase_dir = args.output / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(status_path, phase_dir / "status.tsv")
        with status_path.open(newline="", encoding="utf-8") as reader_file:
            for row in csv.DictReader(reader_file, delimiter="\t"):
                if row["status"] not in ("completed", "skipped_existing"):
                    continue
                run_name = row["run_name"]
                source_dir = args.run_root / run_name
                destination_dir = phase_dir / run_name
                destination_dir.mkdir(parents=True, exist_ok=True)
                for file_name in RESULT_FILES:
                    source = source_dir / file_name
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    destination = destination_dir / file_name
                    shutil.copy2(source, destination)
                    manifest.append(
                        {
                            "phase": phase,
                            "run_name": run_name,
                            "file": file_name,
                            "size_bytes": destination.stat().st_size,
                        }
                    )

    manifest_path = args.output / "manifest.tsv"
    with manifest_path.open("w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(
            writer_file,
            fieldnames=("phase", "run_name", "file", "size_bytes"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Archived {len(manifest)} files to {args.output}")


if __name__ == "__main__":
    main()
