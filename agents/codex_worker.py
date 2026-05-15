"""
Codex Worker Spawner
====================
Spawns Codex CLI subprocesses to run ML experiments in parallel.
Called by the Claude Code orchestrator.
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROMPTS_DIR = ROOT / "prompts"
SPECS_DIR = ROOT / "experiments" / "specs"
RESULTS_DIR = ROOT / "experiments" / "results"


def _build_codex_prompt(spec: dict) -> str:
    """Build the full prompt string to pass to the Codex CLI."""
    system_context = (PROMPTS_DIR / "00_system_context.md").read_text()
    worker_template = (PROMPTS_DIR / "codex_worker_template.md").read_text()

    phase = spec.get("phase", "feature_engineering")
    phase_prompt_file = {
        "eda": "01_eda.md",
        "baseline": "02_baseline.md",
        "feature_engineering": "03_feature_engineering.md",
        "stacking": "04_stacking.md",
    }.get(phase, "03_feature_engineering.md")

    phase_prompt = (PROMPTS_DIR / phase_prompt_file).read_text()

    prompt = f"""
{system_context}

---

## Your Specific Task

{phase_prompt}

---

## Experiment Spec

{json.dumps(spec, indent=2)}

---

## Worker Instructions

{worker_template}

---

## Required Output Files

You MUST create ALL of these files before finishing:

1. OOF predictions: {spec.get('save_oof_path', 'experiments/oof/oof_MISSING.npy')}
   - Shape: (n_train_rows,) — one prediction per training row
   - Format: numpy float32 array

2. Test predictions: {spec.get('save_test_path', 'experiments/test_preds/test_MISSING.npy')}
   - Shape: (n_test_rows,) — one prediction per test row
   - Format: numpy float32 array

3. Result JSON: {spec.get('save_result_path', 'experiments/results/MISSING.json')}
   - Must follow the exact schema in CLAUDE.md
   - Include cv_rmse, cv_rmse_std, features_used, training_time_seconds, notes

Verify all three files exist and are non-empty before exiting.
Print "EXPERIMENT COMPLETE: cv_rmse=X.XX" as your final output line.
"""
    return prompt.strip()


def spawn_codex_worker(spec_path: str, timeout_seconds: int = 1200) -> dict:
    """
    Spawn a single Codex CLI worker for one experiment.

    Args:
        spec_path: Path to the JSON experiment spec file
        timeout_seconds: Kill worker if it exceeds this (default 20 min)

    Returns:
        dict with keys: experiment_id, success, cv_rmse, error_message
    """
    spec_path = Path(spec_path)
    if not spec_path.exists():
        return {"success": False, "error_message": f"Spec file not found: {spec_path}"}

    with open(spec_path) as f:
        spec = json.load(f)

    exp_id = spec.get("experiment_id", spec_path.stem)
    prompt = _build_codex_prompt(spec)

    # Write prompt to a temp file so it doesn't get shell-escaped
    prompt_file = SPECS_DIR / f"{exp_id}_prompt.txt"
    prompt_file.write_text(prompt)

    error_file = RESULTS_DIR / f"{exp_id}_error.txt"

    print(f"[Worker] Spawning Codex for {exp_id}...")
    start_time = time.time()

    try:
        # Codex CLI 0.130+: use `codex exec` non-interactive mode.
        # - workspace-write sandbox + bypass approvals = unattended full-auto
        # - --skip-git-repo-check because this project isn't a git repo
        # - prompt is piped via stdin to avoid argv length / escaping issues
        cmd = [
            "codex",
            "exec",
            "--sandbox", "danger-full-access",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--color", "never",
            "-",
        ]

        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            error_file.write_text(result.stderr or result.stdout)
            print(f"[Worker] {exp_id} FAILED (exit {result.returncode}) after {elapsed:.0f}s")
            return {
                "experiment_id": exp_id,
                "success": False,
                "error_message": result.stderr[:500] if result.stderr else "Non-zero exit",
            }

        # Check if result file was written
        result_path = Path(spec.get("save_result_path", ""))
        if result_path.exists():
            with open(result_path) as f:
                exp_result = json.load(f)
            cv = exp_result.get("cv_rmse", None)
            print(f"[Worker] {exp_id} DONE — cv_rmse={cv} ({elapsed:.0f}s)")
            return {
                "experiment_id": exp_id,
                "success": True,
                "cv_rmse": cv,
                "elapsed_seconds": elapsed,
            }
        else:
            msg = f"Result file not found: {result_path}"
            error_file.write_text(msg + "\n\nSTDOUT:\n" + result.stdout)
            print(f"[Worker] {exp_id} FAILED — {msg}")
            return {"experiment_id": exp_id, "success": False, "error_message": msg}

    except subprocess.TimeoutExpired:
        msg = f"Codex worker timed out after {timeout_seconds}s"
        error_file.write_text(msg)
        print(f"[Worker] {exp_id} TIMEOUT")
        return {"experiment_id": exp_id, "success": False, "error_message": msg}

    except FileNotFoundError:
        msg = ("'codex' command not found. "
               "Install with: npm install -g @openai/codex\n"
               "Then authenticate: codex auth")
        print(f"[Worker] ERROR — {msg}")
        return {"experiment_id": exp_id, "success": False, "error_message": msg}

    finally:
        # Clean up temp prompt file
        if prompt_file.exists():
            prompt_file.unlink()


def spawn_parallel_workers(
    spec_paths: list[str],
    max_workers: int = 2,
    timeout_seconds: int = 1200,
) -> list[dict]:
    """
    Spawn multiple Codex workers in parallel.

    Args:
        spec_paths: List of paths to JSON spec files
        max_workers: Max concurrent Codex processes (2 = safe for Plus plan)
        timeout_seconds: Per-worker timeout

    Returns:
        List of result dicts sorted by experiment_id
    """
    print(f"[Orchestrator] Spawning {len(spec_paths)} workers "
          f"(max_parallel={max_workers})")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_spec = {
            executor.submit(spawn_codex_worker, sp, timeout_seconds): sp
            for sp in spec_paths
        }
        for future in as_completed(future_to_spec):
            spec_path = future_to_spec[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"[Orchestrator] Unexpected error for {spec_path}: {exc}")
                results.append({
                    "experiment_id": Path(spec_path).stem,
                    "success": False,
                    "error_message": str(exc),
                })

    # Summary
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    print(f"\n[Orchestrator] Batch complete: "
          f"{len(successful)} succeeded, {len(failed)} failed")

    if successful:
        best = min(successful, key=lambda r: r.get("cv_rmse") or float("inf"))
        print(f"[Orchestrator] Best this batch: "
              f"{best['experiment_id']} cv_rmse={best.get('cv_rmse')}")

    return sorted(results, key=lambda r: r.get("experiment_id", ""))


def create_spec(
    experiment_id: str,
    phase: str,
    description: str,
    model: str,
    features_to_add: list[str] = None,
    base_experiment: str = None,
    extra_params: dict = None,
) -> str:
    """
    Helper to create a spec JSON file and return its path.
    Handles path construction automatically.
    """
    spec = {
        "experiment_id": experiment_id,
        "phase": phase,
        "description": description,
        "model": model,
        "features_to_add": features_to_add or [],
        "base_experiment": base_experiment,
        "save_oof_path": f"experiments/oof/oof_{model}_{experiment_id}.npy",
        "save_test_path": f"experiments/test_preds/test_{model}_{experiment_id}.npy",
        "save_result_path": f"experiments/results/{experiment_id}.json",
        **(extra_params or {}),
    }

    spec_path = SPECS_DIR / f"{experiment_id}.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    print(f"[Orchestrator] Created spec: {spec_path}")
    return str(spec_path)


if __name__ == "__main__":
    # Quick test: verify Codex CLI is installed
    try:
        result = subprocess.run(
            ["codex", "--version"],
            capture_output=True, text=True, timeout=10
        )
        print(f"Codex CLI found: {result.stdout.strip()}")
    except FileNotFoundError:
        print("ERROR: Codex CLI not installed.")
        print("Install: npm install -g @openai/codex")
        print("Auth:    codex auth")
        sys.exit(1)
