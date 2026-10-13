---
name: acceptance-summarize
description: Summarize a HIP/Verbs Acceptance Suite report.json — overall verdict, per-category breakdown, and the gate failures with their reason/ticket. Use when the user asks to explain acceptance results, summarize a run, or see why the gate failed.
---

# Summarize a run

Parses a `report.json` and prints a high-level pass/fail summary, a per-category
breakdown, and every gate failure with its reason/ticket.

## Steps

1. Run the summarizer on a report produced by `acceptance-run`:

```bash
python3 .cursor/skills/acceptance-summarize/scripts/summarize.py report.json
```

2. Relay to the user:
   - the mode and overall counts, plus the per-category breakdown;
   - each gate failure with its reason and ticket (if any);
   - any flagged tests.

## Notes

- `--kind perf` runs also populate each test's `perf` block in the report if the
  user wants latency/bandwidth alongside the pass/fail summary.
