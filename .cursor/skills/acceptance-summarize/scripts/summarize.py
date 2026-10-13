#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Summarize a HIP/Verbs Acceptance Suite report.json.

Prints:
  * the mode + overall counts,
  * a per-category breakdown,
  * gate failures and flagged tests (with reason/ticket).

Usage:
  summarize.py <report.json>
"""
import json
import sys
from collections import Counter, defaultdict


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    if not args:
        sys.stderr.write(__doc__)
        return 2

    with open(args[0]) as f:
        d = json.load(f)
    tests = d.get("tests", [])
    s = d.get("summary", {})

    print(f"# Acceptance summary ({args[0]})")
    print(f"mode={d.get('mode')}  "
          f"pass={s.get('passed',0)} fail={s.get('failed',0)} "
          f"skip={s.get('skipped',0)} na={s.get('not_applicable',0)}")

    by_cat = defaultdict(Counter)
    for t in tests:
        by_cat[t.get("category", "?")][t.get("status", "?")] += 1
    print("\n## By category")
    for cat in sorted(by_cat):
        c = by_cat[cat]
        print(f"  {cat:24} " + " ".join(f"{k}={v}" for k, v in sorted(c.items())))

    fails = [t for t in tests if t.get("gate_fail")]
    flagged = [t for t in tests if t.get("flagged")]

    print(f"\n## Gate failures ({len(fails)})")
    if not fails:
        print("  none — gate is green")
    for t in fails:
        tick = f" [{t['ticket']}]" if t.get("ticket") else ""
        reason = (t.get("reason") or t.get("note") or "").split("\n")[0]
        print(f"  {t['id']}{tick}: {reason}")

    if flagged:
        print(f"\n## Flagged ({len(flagged)})")
        for t in flagged:
            print(f"  {t['id']}: {t.get('note') or 'flagged'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
