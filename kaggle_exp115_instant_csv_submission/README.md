# ROGII exp115 instant CSV submission

Diagnostic submission notebook that writes the cached `submissions/exp115_separate_component_smoothers.csv` in seconds. It validates that the runtime `sample_submission.csv` IDs exactly match the cached CSV before writing `/kaggle/working/submission.csv`; if Kaggle uses a different hidden sample, it fails immediately instead of timing out.

Use this to test whether the timeout is caused by model runtime or Kaggle scoring/version selection.
