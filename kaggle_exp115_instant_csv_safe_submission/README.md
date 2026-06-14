# ROGII exp115 instant CSV safe diagnostic submission

Diagnostic notebook. It writes cached exp115 predictions if runtime sample IDs match. If they do not match, it writes a valid constant fallback submission instead of raising, and saves `/kaggle/working/debug_submission_info.json`.
