"""
Experiment Tracker
==================
Single source of truth for all experiment results.
Claude Code calls this after every completed experiment.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOG_PATH = ROOT / "experiments" / "experiment_log.csv"
RESULTS_DIR = ROOT / "experiments" / "results"

COLUMNS = [
    "experiment_id", "timestamp", "phase", "model",
    "cv_rmse", "cv_rmse_std", "n_features",
    "base_experiment", "description", "notes",
    "oof_path", "test_path", "training_time_seconds",
]


def _ensure_log_exists():
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()


def log_experiment(
    experiment_id: str,
    cv_rmse: float,
    model: str,
    phase: str = "feature_engineering",
    cv_rmse_std: float = None,
    n_features: int = None,
    base_experiment: str = None,
    description: str = "",
    notes: str = "",
    training_time_seconds: float = None,
):
    """Log a completed experiment to the CSV tracker."""
    _ensure_log_exists()

    oof_path = f"experiments/oof/oof_{model}_{experiment_id}.npy"
    test_path = f"experiments/test_preds/test_{model}_{experiment_id}.npy"

    row = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "model": model,
        "cv_rmse": round(cv_rmse, 4) if cv_rmse else "",
        "cv_rmse_std": round(cv_rmse_std, 4) if cv_rmse_std else "",
        "n_features": n_features or "",
        "base_experiment": base_experiment or "",
        "description": description,
        "notes": notes,
        "oof_path": oof_path if Path(ROOT / oof_path).exists() else "MISSING",
        "test_path": test_path if Path(ROOT / test_path).exists() else "MISSING",
        "training_time_seconds": training_time_seconds or "",
    }

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writerow(row)

    print(f"[Tracker] Logged {experiment_id}: cv_rmse={cv_rmse}")


def log_from_result_file(result_json_path: str):
    """Auto-log from a result JSON file written by a Codex worker."""
    with open(result_json_path) as f:
        result = json.load(f)

    log_experiment(
        experiment_id=result.get("experiment_id", Path(result_json_path).stem),
        cv_rmse=result.get("cv_rmse"),
        model=result.get("model", "unknown"),
        phase=result.get("phase", "feature_engineering"),
        cv_rmse_std=result.get("cv_rmse_std"),
        n_features=result.get("n_features"),
        base_experiment=result.get("base_experiment"),
        description=result.get("description", ""),
        notes=result.get("notes", ""),
        training_time_seconds=result.get("training_time_seconds"),
    )


def get_best_experiments(n: int = 10, phase: str = None) -> list[dict]:
    """Return top N experiments by CV RMSE."""
    _ensure_log_exists()
    rows = []
    with open(LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if phase and row["phase"] != phase:
                continue
            try:
                row["cv_rmse"] = float(row["cv_rmse"])
                rows.append(row)
            except (ValueError, KeyError):
                continue

    return sorted(rows, key=lambda r: r["cv_rmse"])[:n]


def get_all_oof_paths() -> list[str]:
    """Return paths to all OOF files for experiments with valid results."""
    best = get_best_experiments(n=999)
    paths = []
    for row in best:
        p = ROOT / row.get("oof_path", "")
        if p.exists():
            paths.append(str(p))
    return paths


def print_summary():
    """Print a readable summary table of all experiments."""
    _ensure_log_exists()
    rows = []
    with open(LOG_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No experiments logged yet.")
        return

    print(f"\n{'ID':<15} {'Phase':<20} {'Model':<12} {'CV RMSE':<10} {'Notes'}")
    print("-" * 80)
    for row in sorted(rows, key=lambda r: r.get("cv_rmse") or "99"):
        print(f"{row['experiment_id']:<15} {row['phase']:<20} "
              f"{row['model']:<12} {row['cv_rmse']:<10} {row['notes'][:40]}")


if __name__ == "__main__":
    print_summary()
