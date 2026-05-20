"""exp112: refined non-heldout OOF smoother search around exp111.

The exp111 winner was selected on non-heldout OOF rows:
`savgol_w501_p2_s1`.  This script refines that smoother family using the same
selection protocol and keeps the sample-submission wells audit-only.
"""
from __future__ import annotations

import run_exp111_exp109_oof_postprocess as exp111


exp111.RESULT = exp111.ROOT / "experiments/results/exp112.json"
exp111.SUBMISSION = exp111.ROOT / "submissions/exp112_exp109_savgol_refine.csv"
exp111.EXPERIMENT_ID = "exp112"
exp111.PHASE = "exp109_refined_savgol_postprocess"
exp111.DEV_IMPROVEMENT_EPS = 0.001


def refined_candidates() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = [{"name": "none", "kind": "none"}]
    for window in (351, 401, 451, 501, 551, 601, 651, 701, 751):
        for polyorder in (1, 2, 3):
            for shrink in (0.75, 0.875, 1.0, 1.125):
                candidates.append(
                    {
                        "name": f"savgol_w{window}_p{polyorder}_s{shrink:g}",
                        "kind": "savgol",
                        "window": window,
                        "polyorder": polyorder,
                        "shrink": shrink,
                    }
                )
    return candidates


exp111.postprocess_candidates = refined_candidates


if __name__ == "__main__":
    exp111.main()
