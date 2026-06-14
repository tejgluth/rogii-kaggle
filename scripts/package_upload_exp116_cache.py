#!/usr/bin/env python3
"""Package a completed exp116 local run as a private Kaggle Dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def copy_required(run_dir: Path, package_dir: Path) -> None:
    required = [
        "models",
        "exp116_run_metadata.json",
        "submission.csv",
        "validation_report.json",
    ]
    package_dir.mkdir(parents=True, exist_ok=True)
    for name in required:
        src = run_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required run output: {src}")
        dst = package_dir / name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # Keep the Kaggle Dataset small. The cached inference notebook rebuilds
    # test features from the official competition input, so generated train
    # feature CSVs and train metadata are not needed.
    for optional in ["test_preds.csv", "test_meta.csv"]:
        src = run_dir / optional
        if src.exists():
            shutil.copy2(src, package_dir / optional)


def write_dataset_metadata(package_dir: Path, dataset_id: str, title: str) -> None:
    metadata = {
        "id": dataset_id,
        "title": title,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (package_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="local_runs/exp116_fast_cache")
    parser.add_argument("--package-dir", default="kaggle_datasets/rogii-exp116-fast-model-cache")
    parser.add_argument("--dataset-id", default="tejasgluth/rogii-exp116-fast-model-cache")
    parser.add_argument("--title", default="ROGII exp116 fast model cache")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    package_dir = Path(args.package_dir).resolve()
    copy_required(run_dir, package_dir)
    write_dataset_metadata(package_dir, args.dataset_id, args.title)

    print(f"packaged: {package_dir}")
    for path in sorted(package_dir.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(package_dir)}\t{path.stat().st_size}")

    if args.upload:
        api = KaggleApi()
        api.authenticate()
        try:
            api.dataset_create_new(str(package_dir), public=False, quiet=False)
        except Exception as exc:
            message = str(exc)
            if "already exists" not in message.lower():
                raise
            api.dataset_create_version(
                str(package_dir),
                version_notes="Update exp116 fast model cache",
                quiet=False,
            )


if __name__ == "__main__":
    main()
