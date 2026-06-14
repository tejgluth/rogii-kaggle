#!/usr/bin/env python3
"""Audit and compare ROGII submission candidates.

This is intentionally output-only: it validates candidate CSVs against the
sample submission, computes stable hashes, and reports pairwise visible-test
distances. The visible test has no labels, so distances are only a diversity
diagnostic, not a score estimate.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data/rogii-wellbore-geology-prediction/sample_submission.csv"
OUT_DIR = ROOT / "experiments/results"
SUMMARY_CSV = OUT_DIR / "submission_candidate_audit_2026-06-14.csv"
SUMMARY_MD = OUT_DIR / "submission_candidate_audit_2026-06-14.md"


PRIORITY_PATHS = {
    "LB7.591_lightning_guarded": "local_runs/rogii-dual-pipeline-self-verifying-fixed_output_latest/submission.csv",
    "LB7.580_lightning_w55": "local_runs/rogii-lightning-preoverride-w55_output/submission.csv",
    "LB7.633_pilkwang_w55": "local_runs/rogii-target-free-tvt-geosteering-fork_output/submission.csv",
    "submitted_yaroslav_d4": "local_runs/rogii-ridge-artifact-projection-d4-fork_output/submission.csv",
    "v5f_probe_private": "local_runs/rogii-fle3n-v5f-probe-fork_output/submission.csv",
    "v5f_xfer045": "local_runs/rogii-fle3n-v5f-xfer-0p45_output/submission.csv",
    "v5f_xfer055": "local_runs/rogii-fle3n-v5f-xfer-0p55_output/submission.csv",
    "pixiux_private": "local_runs/rogii-pixiux-dual-pipeline-blend-fork_output/submission.csv",
    "hamatz_private": "local_runs/rogii-hamatz-dual-pipeline-reproduced-fork_output/submission.csv",
    "rikuter_private": "local_runs/rogii-rikuter-pilkwang-ridge-artifact-fork_output/submission.csv",
    "iaztec_private": "local_runs/rogii-iaztec-sp45-fleongg-eda-fork_output/submission.csv",
    "public_v8_gold": "kaggle_public_v8_gold/output_latest/submission.csv",
    "public_zeyufeng_final": "kaggle_public_zeyufeng_final/output/submission.csv",
    "public_hongweiluan_v6_e2e": "kaggle_public_hongweiluan_v6_e2e/output/submission.csv",
    "public_quanzhong_pf_fleongg": "local_runs/public_quanzhong_pf_fleongg_blend_output/submission.csv",
    "public_quanzhong_sp45_component": "local_runs/public_quanzhong_pf_fleongg_blend_output/sp45_projection_submission.csv",
    "public_quanzhong_fleongg_component": "local_runs/public_quanzhong_pf_fleongg_blend_output/fleongg_pretrained_submission.csv",
    "public_shuaixu_exp100": "local_runs/public_shuaixu_exp100_yaroslav_output/submission.csv",
    "public_shuaixu_exp100_sp45_component": "local_runs/public_shuaixu_exp100_yaroslav_output/sp45_projection_submission.csv",
    "public_shuaixu_exp100_fleongg_component": "local_runs/public_shuaixu_exp100_yaroslav_output/fleongg_pretrained_submission.csv",
    "public_hongweiluan_v6_inference": "local_runs/public_hongweiluan_v6_inference_output/submission.csv",
}


def stable_hash(df: pd.DataFrame) -> str:
    view = df[["id", "tvt"]].copy()
    view["tvt"] = view["tvt"].astype(np.float64)
    values = pd.util.hash_pandas_object(view, index=False).values.tobytes()
    return hashlib.sha256(values).hexdigest()[:16]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt(np.mean(d * d)))


def load_candidates(sample: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    preds: dict[str, np.ndarray] = {}

    candidates = dict(PRIORITY_PATHS)
    for path in sorted((ROOT / "local_runs").glob("*/submission.csv")):
        rel = str(path.relative_to(ROOT))
        name = path.parent.name
        candidates.setdefault(name, rel)
    for path in sorted(ROOT.glob("kaggle_public_*/output*/submission.csv")):
        rel = str(path.relative_to(ROOT))
        name = path.parent.parent.name + "_" + path.parent.name
        candidates.setdefault(name, rel)

    for name, rel in sorted(candidates.items()):
        path = ROOT / rel
        row: dict[str, object] = {"name": name, "path": rel, "exists": path.exists()}
        if not path.exists():
            rows.append(row)
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - diagnostic path
            row.update({"valid": False, "error": repr(exc)})
            rows.append(row)
            continue
        valid = (
            list(df.columns[:2]) == ["id", "tvt"]
            and len(df) == len(sample)
            and df["id"].equals(sample["id"])
            and np.isfinite(pd.to_numeric(df["tvt"], errors="coerce").to_numpy(dtype=float)).all()
        )
        row.update(
            {
                "valid": valid,
                "rows": len(df),
                "sha16": stable_hash(df) if valid else "",
            }
        )
        if valid:
            tvt = df["tvt"].to_numpy(dtype=float)
            preds[name] = tvt
            row.update(
                {
                    "mean": float(np.mean(tvt)),
                    "std": float(np.std(tvt)),
                    "min": float(np.min(tvt)),
                    "max": float(np.max(tvt)),
                    "q01": float(np.quantile(tvt, 0.01)),
                    "q50": float(np.quantile(tvt, 0.50)),
                    "q99": float(np.quantile(tvt, 0.99)),
                }
            )
        rows.append(row)
    return rows, preds


def nearest_neighbors(
    preds: dict[str, np.ndarray],
    hashes: dict[str, str],
    *,
    distinct_hash_only: bool,
) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    names = sorted(preds)
    for i, name in enumerate(names):
        best_name = ""
        best_dist = math.inf
        for j, other in enumerate(names):
            if i == j:
                continue
            if distinct_hash_only and hashes.get(name) == hashes.get(other):
                continue
            dist = rmse(preds[name], preds[other])
            if dist < best_dist:
                best_dist = dist
                best_name = other
        out[name] = (best_name, best_dist)
    return out


def pairwise_to_anchors(preds: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    anchors = [
        "LB7.591_lightning_guarded",
        "LB7.580_lightning_w55",
        "LB7.633_pilkwang_w55",
        "v5f_probe_private",
        "public_v8_gold",
        "public_hongweiluan_v6_e2e",
    ]
    out: dict[str, dict[str, float]] = {}
    for name, values in preds.items():
        out[name] = {}
        for anchor in anchors:
            if anchor in preds and anchor != name:
                out[name][f"rmse_to_{anchor}"] = rmse(values, preds[anchor])
    return out


def write_outputs(rows: list[dict[str, object]], preds: dict[str, np.ndarray]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hashes = {str(row["name"]): str(row.get("sha16", "")) for row in rows if row.get("valid")}
    nearest = nearest_neighbors(preds, hashes, distinct_hash_only=False)
    nearest_distinct = nearest_neighbors(preds, hashes, distinct_hash_only=True)
    anchors = pairwise_to_anchors(preds)
    for row in rows:
        name = str(row["name"])
        if name in nearest:
            row["nearest"] = nearest[name][0]
            row["nearest_rmse"] = nearest[name][1]
        if name in nearest_distinct and math.isfinite(nearest_distinct[name][1]):
            row["nearest_distinct"] = nearest_distinct[name][0]
            row["nearest_distinct_rmse"] = nearest_distinct[name][1]
        row.update(anchors.get(name, {}))

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [r for r in rows if r.get("valid")]
    valid_rows.sort(key=lambda r: float(r.get("nearest_distinct_rmse", math.inf)), reverse=True)

    duplicate_rows = [r for r in valid_rows if float(r.get("nearest_rmse", math.inf)) < 0.02]
    diverse_rows = [r for r in valid_rows if float(r.get("nearest_distinct_rmse", 0.0)) >= 1.0]

    lines = [
        "# Submission Candidate Audit - 2026-06-14",
        "",
        "Visible-test distances are diversity diagnostics only; they are not RMSE estimates.",
        "",
        f"Valid files: {len(valid_rows)}",
        "",
        "## Most Diverse Valid Outputs",
        "",
        "| candidate | sha16 | mean | std | nearest distinct | distinct RMSE |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for row in valid_rows[:25]:
        lines.append(
            f"| {row['name']} | {row.get('sha16','')} | "
            f"{float(row.get('mean', 0.0)):.4f} | {float(row.get('std', 0.0)):.4f} | "
            f"{row.get('nearest_distinct','')} | {float(row.get('nearest_distinct_rmse', 0.0)):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Near-Duplicate Outputs",
            "",
            "| candidate | sha16 | nearest | nearest RMSE |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for row in duplicate_rows:
        lines.append(
            f"| {row['name']} | {row.get('sha16','')} | "
            f"{row.get('nearest','')} | {float(row.get('nearest_rmse', 0.0)):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Rerunnable Diversity Pool",
            "",
            "These candidates are algorithmic/public-kernel outputs that are separated from the already scored Lightning/Pilkwang band on the visible rows.",
            "",
        "| candidate | nearest distinct RMSE | to Lightning guarded | to Pilkwang | to v5f |",
        "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in diverse_rows:
        name = str(row["name"])
        if not (
            name.startswith("public_")
            or "v5f" in name
            or "shuaixu" in name
            or "hongwei" in name
            or "quanzhong" in name
        ):
            continue
        lines.append(
            f"| {name} | {float(row.get('nearest_distinct_rmse', 0.0)):.4f} | "
            f"{float(row.get('rmse_to_LB7.591_lightning_guarded', math.nan)):.4f} | "
            f"{float(row.get('rmse_to_LB7.633_pilkwang_w55', math.nan)):.4f} | "
            f"{float(row.get('rmse_to_v5f_probe_private', math.nan)):.4f} |"
        )
    lines.append("")
    SUMMARY_MD.write_text("\n".join(lines))


def main() -> None:
    sample = pd.read_csv(SAMPLE_PATH)
    rows, preds = load_candidates(sample)
    write_outputs(rows, preds)
    print(f"wrote {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"wrote {SUMMARY_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
