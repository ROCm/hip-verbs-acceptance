#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Render the performance metrics from a HIP/Verbs Acceptance Suite report.json.

The suite prints a performance block at the end of a --kind perf run, but the
report holds more than fits on a console (raw per-iteration samples, every
rank). This renders the report, and -- with --baseline -- answers the question
the console block cannot: did anything get slower?

    ./build/hva --kind all --report run.json ...
    python3 scripts/perf_report.py run.json
    python3 scripts/perf_report.py run.json --baseline old.json

Multiple reports can be passed (e.g. one per suite). --format md emits a
Markdown table instead of aligned text, for pasting into a ticket or PR.

Repeat --baseline to fold several runs into one baseline. Do this: a single
baseline run only knows how much a metric wobbled WITHIN that run, which on a
shared machine is far less than it wobbles between runs -- so a one-run baseline
reports ordinary drift as a regression. Two or three runs give the gate a real
envelope to compare against:

    python3 ... perf_report.py new.json --baseline r1.json --baseline r2.json \
                                        --baseline r3.json

A comparison never fails on its own: --fail-on-regression makes THIS SCRIPT exit
1 when a confirmed regression is found, for CI to act on. The acceptance gate
itself stays functional-only -- perf never changes the suite's exit code.

Perf keys are flat and self-describing (see common/Perf.hpp):

    <metric>.<stat>[.<size>]              e.g. rdma_write.bw.median.4KiB
    <metric>.unit[.<size>]                e.g. rdma_write.bw.unit.4KiB = GB/s
    <metric>.iters[.<size>]
    <metric>.samples[.<size>].<unit>      raw per-iteration array

`size` appears only on swept metrics. The unit also encodes direction
(GB/s = higher-is-better, us/op = lower-is-better).
"""

import argparse
import json
import re
import sys

# Stat tokens emitted by perf_detail::emit. Metric names never contain one, so
# the first match scanning a dotted key left-to-right splits "<metric>" from
# "<stat>[.<size>][.<unit>]".
STATS = ("median", "min", "max", "avg", "unit", "iters", "samples")

# human_bytes() output: an integer followed by a binary unit.
SIZE_RE = re.compile(r"^(\d+)(B|KiB|MiB|GiB|TiB)$")
SCALE = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2,
         "GiB": 1024 ** 3, "TiB": 1024 ** 4}

# A metric whose max exceeds its own median by this factor: one pass ran long.
SPREAD_RATIO = 1.5

# A test whose median is this far off the median of its peers (same metric name
# measured by other tests) is an outlier among them. Catches the cases that
# SPREAD_RATIO cannot -- notably single-sample metrics, which have no spread.
PEER_RATIO = 3.0

# Default --threshold: a median must move this much (percent) before a
# comparison calls it anything but noise.
DEFAULT_THRESHOLD = 10.0

# Which direction is an improvement, by unit. A unit that is not listed gets a
# delta but no verdict -- guessing would be worse than saying nothing.
HIGHER_IS_BETTER = {"GB/s": True, "us/op": False}


def split_key(key):
    """'rdma_write.bw.median.4KiB'     -> ('rdma_write.bw', 'median', '4KiB', None)
       'reg_mr.lat.samples.4KiB.us/op' -> ('reg_mr.lat', 'samples', '4KiB', 'us/op')
       'rdma_read.lat.median'          -> ('rdma_read.lat', 'median', None, None)
    None when the key carries no recognised stat token."""
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part not in STATS:
            continue
        rest = parts[i + 1:]
        size = None
        if rest and SIZE_RE.match(rest[0]):
            size = rest[0]
            rest = rest[1:]
        # A key that is only a stat token ("median") names no metric. The C++
        # splitter rejects it; reject it here too, or the two halves of the
        # same grammar drift and this renders a blank metric row.
        metric = ".".join(parts[:i])
        if not metric:
            return None
        # Whatever trails is the unit (units contain '/' but never '.').
        return metric, part, size, (".".join(rest) or None)
    return None


def parse_perf(perf):
    """Flat perf dict -> {(metric, size): {stat: value}}.

    Unparseable keys are skipped: the report comes from another tool and a
    surprise in it must not break the rendering."""
    out = {}
    for key, value in (perf or {}).items():
        parsed = split_key(key)
        if parsed is None:
            continue
        metric, stat, size, unit = parsed
        slot = out.setdefault((metric, size), {})
        if stat == "unit":
            slot["unit"] = value
        elif stat == "samples":
            slot.setdefault("unit", unit)   # unit rides in the key here
            slot["samples"] = value if isinstance(value, list) else []
        else:
            slot[stat] = value
    return out


def size_key(size):
    """Order sizes numerically (64B < 4KiB < 1MiB), not lexically."""
    if not size:
        return (0, 0)
    m = SIZE_RE.match(size)
    return (1, int(m.group(1)) * SCALE[m.group(2)]) if m else (1, 0)


def collect(docs):
    """[(label, doc)] -> [(category, test_id, metric, size, stats)] sorted."""
    records = []
    for _label, doc in docs:
        for test in doc.get("tests", []) or []:
            for (metric, size), stats in parse_perf(test.get("perf")).items():
                records.append((test.get("category") or "(uncategorised)",
                                test.get("id", "?"), metric, size, stats))
    records.sort(key=lambda r: (r[0], r[1], r[2], size_key(r[3])))
    return records


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return None
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def anomalies(records):
    """Things worth a human look, as (label, detail, test_id, ratio).

    Two independent checks, because they catch different failures:

    * spread -- max >= SPREAD_RATIO x this metric's own median. One pass ran
      far longer than typical: a cold path, a TLB/page-table effect,
      contention. The median hides it by design.
    * peer   -- a test's median is >= PEER_RATIO off the median of every other
      test reporting the same metric name. This is the only check that works
      for single-sample metrics, which have no spread of their own.
    """
    found = []

    for _category, test_id, metric, size, stats in records:
        median, mx = stats.get("median"), stats.get("max")
        if not isinstance(median, (int, float)) or not isinstance(mx, (int, float)):
            continue
        if median > 0 and stats.get("iters", 0) != 1 and mx / median >= SPREAD_RATIO:
            label = metric + (f" {size}" if size else "")
            unit = stats.get("unit") or ""
            found.append((label,
                          f"median {num(median)} → max {num(mx)} {unit}".rstrip(),
                          test_id, mx / median, "one pass ran long"))

    # Group medians by (metric, size) across tests to find peer outliers.
    peers = {}
    for _category, test_id, metric, size, stats in records:
        median = stats.get("median")
        if isinstance(median, (int, float)) and median > 0:
            peers.setdefault((metric, size), []).append(
                (test_id, median, stats.get("unit") or ""))
    for (metric, size), entries in peers.items():
        if len(entries) < 3:      # need a meaningful peer group
            continue
        baseline = _median([m for _t, m, _u in entries])
        if not baseline:
            continue
        for test_id, median, unit in entries:
            ratio = max(median / baseline, baseline / median)
            if ratio >= PEER_RATIO:
                label = metric + (f" {size}" if size else "")
                found.append((label,
                              f"{num(median)} {unit} vs peer median "
                              f"{num(baseline)}".rstrip(),
                              test_id, ratio, "far off its peers"))

    found.sort(key=lambda r: -r[3])
    return found


def index(records):
    """[(category, test, metric, size, stats)] -> {(test, metric, size): rec}.

    First occurrence wins, so passing the same report twice is harmless."""
    out = {}
    for rec in records:
        out.setdefault((rec[1], rec[2], rec[3]), rec)
    return out


def fold_baseline(docs):
    """[(label, doc)] -> {(test, metric, size): rec} spanning every run.

    One report is a point estimate: its [min, max] only covers the spread WITHIN
    that run, and on a shared machine a metric routinely shifts far more than
    that between runs of the same binary (measured on MI300X: d2d.sync 16MiB
    moved 1071 -> 603 GB/s with both runs internally tight). Folding N
    baseline runs takes the median of their medians and the envelope of their
    bands, so the noise gate finally sees run-to-run variance. `runs` records
    how many contributed -- band() needs it to tell a real envelope apart from a
    single-sample point."""
    merged = {}
    for doc in docs:
        for key, rec in index(collect([doc])).items():
            merged.setdefault(key, []).append(rec)

    out = {}
    for key, recs in merged.items():
        stats = dict(recs[0][4])
        values = [r[4] for r in recs]
        medians = [s["median"] for s in values
                   if isinstance(s.get("median"), (int, float))]
        lows = [s["min"] for s in values if isinstance(s.get("min"), (int, float))]
        highs = [s["max"] for s in values if isinstance(s.get("max"), (int, float))]
        if medians:
            stats["median"] = _median(medians)
        if lows:
            stats["min"] = min(lows + medians)
        if highs:
            stats["max"] = max(highs + medians)
        stats["runs"] = len(recs)
        out[key] = recs[0][:4] + (stats,)
    return out


def band(stats):
    """(min, max) when the metric has a real spread, else None.

    A single-sample metric reports min == max == median, which is a point, not
    a band -- treating it as one would make every comparison 'within noise'.
    A folded baseline is exempt: several runs that each measured once still
    describe a genuine spread (`runs` > 1)."""
    lo, hi = stats.get("min"), stats.get("max")
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        return None
    if hi <= lo:
        return None
    if stats.get("runs", 1) > 1:
        return (lo, hi)
    if stats.get("iters", 0) in (0, 1):
        return None
    return (lo, hi)


def compare(current, baseline, threshold):
    """Join two record sets on (test, metric, size).

    Returns (rows, new_keys, missing_keys). Each row is a dict carrying both
    medians, the signed delta in percent, and a verdict:

      better / worse   -- moved past `threshold` AND the two runs' [min, max]
                          bands do not overlap
      ~                -- everything else: either a small move, or a big one
                          that the run-to-run spread already explains
      ?                -- unknown unit, so 'better' has no meaning here

    The band test is what keeps this honest. A 10-iteration latency metric
    routinely swings 15% between runs of the SAME binary; calling that a
    regression trains people to ignore the tool."""
    rows, new_keys = [], []
    for key, rec in sorted(current.items()):
        base = baseline.get(key)
        if base is None:
            new_keys.append((key, rec))
            continue

        cur_stats, base_stats = rec[4], base[4]
        cur, old = cur_stats.get("median"), base_stats.get("median")
        if not isinstance(cur, (int, float)) or not isinstance(old, (int, float)):
            continue
        if old == 0:
            continue

        delta = (cur - old) / abs(old) * 100.0
        unit = cur_stats.get("unit") or base_stats.get("unit") or ""
        higher_better = HIGHER_IS_BETTER.get(unit)

        cur_band, base_band = band(cur_stats), band(base_stats)
        overlap = (cur_band is not None and base_band is not None and
                   cur_band[0] <= base_band[1] and base_band[0] <= cur_band[1])

        if higher_better is None:
            verdict = "?"
        elif abs(delta) < threshold or overlap:
            verdict = "~"
        else:
            improved = (delta > 0) == higher_better
            verdict = "better" if improved else "worse"

        rows.append({"category": rec[0], "test": rec[1], "metric": rec[2],
                     "size": rec[3], "unit": unit, "baseline": old,
                     "current": cur, "delta": delta, "verdict": verdict,
                     "overlap": overlap})

    missing = [(key, rec) for key, rec in sorted(baseline.items())
               if key not in current]
    # Worst regressions first; improvements and noise after.
    rows.sort(key=lambda r: (r["verdict"] != "worse", -abs(r["delta"])))
    return rows, new_keys, missing


def regressions(rows):
    return [r for r in rows if r["verdict"] == "worse"]


def signed(value):
    """'+4.2%' / '-11.8%' -- the sign carries the direction of the move."""
    return f"{value:+.1f}%"


def label(key):
    test, metric, size = key
    return f"{metric}{' ' + size if size else ''} ({test})"


# Printed when a regression is called against a one-run baseline -- the case
# where the gate is at its weakest and the reader deserves to know.
THIN_BASELINE_NOTE = (
    "note: the baseline is a single run, so its band only covers within-run "
    "spread.\n      Repeat --baseline with 2-3 runs to let run-to-run drift "
    "count as noise.")


def render_comparison_text(rows, new_keys, missing, base_labels, threshold,
                           base_runs=1, show_all=False):
    shown = rows if show_all else [r for r in rows if r["verdict"] in ("worse", "better")]
    out = [f"Comparison — baseline: {', '.join(base_labels)}"
           f"{f' ({base_runs} runs folded)' if base_runs > 1 else ''}",
           f"{len(rows)} shared metric(s) · threshold {threshold:g}% · "
           f"'~' = within run-to-run spread", ""]
    if rows and not show_all:
        hidden = len(rows) - len(shown)
        note = (f"{len(shown)} regression(s)/progression(s) shown"
                f"{f', {hidden} unchanged hidden (--show-all to see everything)' if hidden else ''}")
        out += [note, ""]
    if shown:
        # A rank-prefixed metric ("rank1.foo.bar") can be longer than the base
        # width; size the column to the widest name actually shown, or every
        # row but that one silently drifts out of alignment.
        metric_width = max([24] + [len(r["metric"]) for r in shown])
        out.append(f"  {'metric':<{metric_width}} {'size':>7} {'baseline':>10} "
                   f"{'current':>10} {'delta':>8}  {'verdict':<7} test")
        for r in shown:
            out.append(f"  {r['metric']:<{metric_width}} {r['size'] or '':>7} "
                       f"{num(r['baseline']):>10} {num(r['current']):>10} "
                       f"{signed(r['delta']):>8}  {r['verdict']:<7} {r['test']}")
        out.append("")
    elif rows:
        out.append("(nothing moved past the threshold)")
        out.append("")

    worse = regressions(rows)
    if worse:
        out.append(f"{len(worse)} regression(s) beyond {threshold:g}% and "
                   f"outside the measured spread:")
        for r in worse:
            out.append(f"  {r['metric']}{' ' + r['size'] if r['size'] else ''}"
                       f"  {num(r['baseline'])} → {num(r['current'])} "
                       f"{r['unit']}  ({signed(r['delta'])})  {r['test']}")
        if base_runs < 2:
            out.append(THIN_BASELINE_NOTE)
        out.append("")

    for title, entries in (("Only in this run", new_keys),
                           ("Only in the baseline", missing)):
        if entries:
            out.append(f"{title} ({len(entries)}):")
            out += [f"  {label(key)}" for key, _rec in entries]
            out.append("")
    return out


def render_comparison_md(rows, new_keys, missing, base_labels, threshold,
                         base_runs=1, show_all=False):
    shown = rows if show_all else [r for r in rows if r["verdict"] in ("worse", "better")]
    out = [f"## Comparison — baseline: {', '.join(base_labels)}"
           f"{f' ({base_runs} runs folded)' if base_runs > 1 else ''}", "",
           f"_{len(rows)} shared metric(s), threshold {threshold:g}%. "
           f"`~` means the move is inside the run-to-run spread._", ""]
    if rows and not show_all:
        hidden = len(rows) - len(shown)
        out += [f"_{len(shown)} regression(s)/progression(s) shown"
                f"{f', {hidden} unchanged hidden (`--show-all` to see everything)' if hidden else ''}._",
                ""]
    if base_runs < 2 and regressions(rows):
        out += ["> " + THIN_BASELINE_NOTE.replace("\n      ", " "), ""]
    if shown:
        out += ["| Metric | Size | Baseline | Current | Delta | Verdict | Unit | Test |",
                "|---|---|---:|---:|---:|---|---|---|"]
        for r in shown:
            verdict = f"**{r['verdict']}**" if r["verdict"] == "worse" else r["verdict"]
            out.append("| " + " | ".join([
                r["metric"], r["size"] or "-", num(r["baseline"]),
                num(r["current"]), signed(r["delta"]), verdict,
                r["unit"] or "-", f"`{r['test']}`"]) + " |")
        out.append("")
    elif rows:
        out += ["_(nothing moved past the threshold)_", ""]
    for title, entries in (("Only in this run", new_keys),
                           ("Only in the baseline", missing)):
        if entries:
            out += [f"**{title}** ({len(entries)})", ""]
            out += [f"- {label(key)}" for key, _rec in entries]
            out.append("")
    return out


def num(value):
    """Compact fixed-point: keeps small values readable, drops noise digits."""
    if not isinstance(value, (int, float)):
        return "-"
    if value >= 1000:
        return f"{value:,.0f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def env_line(docs):
    for _label, doc in docs:
        nodes = doc.get("nodes") or []
        env = (nodes[0].get("environment") if nodes else None) or {}
        bits = [env.get("vendor"), env.get("gpu_model")]
        if env.get("rocm_version"):
            bits.append("ROCm " + env["rocm_version"])
        bits = [b for b in bits if b]
        if bits:
            return " / ".join(bits)
    return None


def render_text(records, docs, labels):
    out = []
    env = env_line(docs)
    out.append("Performance" + (f" — {env}" if env else ""))
    tests = len({(r[0], r[1]) for r in records})
    out.append(f"{len(records)} metric(s) · {tests} test(s) · from {', '.join(labels)}")
    out.append("")
    # A rank-prefixed metric ("rank1.foo.bar") can be longer than the base
    # width; size the column to the widest name in this render, or every row
    # but that one silently drifts out of alignment.
    metric_width = max([24] + [len(r[2]) for r in records])
    out.append(f"    {'metric':<{metric_width}} {'size':>7}  {'median':>10} "
               f"{'min':>10} {'max':>10} {'unit':<6} {'n':>4}")

    category = test_id = None
    for rec_category, rec_test, metric, size, stats in records:
        if rec_category != category:
            category, test_id = rec_category, None
            out.append("")
            out.append(f"--- {rec_category} ---")
        if rec_test != test_id:
            if test_id is not None:
                out.append("")  # separates this test's group from the previous one
            test_id = rec_test
            out.append(f"  {rec_test}")
        iters = stats.get("iters")
        out.append(f"    {metric:<{metric_width}} {size or '':>7}  "
                   f"{num(stats.get('median')):>10} {num(stats.get('min')):>10} "
                   f"{num(stats.get('max')):>10} {stats.get('unit') or '':<6} "
                   f"{iters if iters else '-':>4}")
    out.append("")

    flagged = anomalies(records)
    if flagged:
        # The detail wording already says which check fired ("vs peer median"
        # = cross-test, "median -> max" = within-run spread), so no extra column.
        out.append("Anomalies")
        for label, detail, test_id, ratio, _why in flagged:
            out.append(f"  {ratio:>8.1f}x  {label:<24} {detail:<44} {test_id}")
        out.append("")
    return out


def render_md(records, docs, labels):
    out = []
    env = env_line(docs)
    out.append("## Performance" + (f" — {env}" if env else ""))
    tests = len({(r[0], r[1]) for r in records})
    out += ["", f"_{len(records)} metric(s) across {tests} test(s) "
                f"— from {', '.join(labels)}._", ""]
    category = None
    for rec_category, test_id, metric, size, stats in records:
        if rec_category != category:
            if category is not None:
                out.append("")
            category = rec_category
            out += [f"### {rec_category}", "",
                    "| Test | Metric | Size | Median | Min | Max | Unit | Iters |",
                    "|---|---|---|---:|---:|---:|---|---:|"]
        out.append("| " + " | ".join([
            f"`{test_id}`", metric, size or "-",
            num(stats.get("median")), num(stats.get("min")),
            num(stats.get("max")), stats.get("unit") or "-",
            str(stats.get("iters", "-"))]) + " |")
    out.append("")

    flagged = anomalies(records)
    if flagged:
        out += ["**Anomalies**", "",
                "| Ratio | Metric | Detail | Why | Test |",
                "|---:|---|---|---|---|"]
        for label, detail, test_id, ratio, why in flagged:
            out.append(f"| **{ratio:.1f}×** | {label} | {detail} | {why} "
                       f"| `{test_id}` |")
        out.append("")
    return out


def load(paths):
    """[path] -> ([(path, doc)], [basename]); raises on an unreadable file."""
    docs, labels = [], []
    for path in paths:
        with open(path) as handle:
            docs.append((path, json.load(handle)))
        labels.append(path.rsplit("/", 1)[-1])
    return docs, labels


def main(argv):
    parser = argparse.ArgumentParser(
        prog="perf_report.py",
        description="Render (and optionally diff) HIP/Verbs Acceptance Suite perf metrics.")
    parser.add_argument("reports", nargs="+", metavar="REPORT.json",
                        help="report(s) from a --kind perf|all run")
    parser.add_argument("--format", choices=("text", "md"), default="text")
    parser.add_argument("--baseline", action="append", default=[],
                        metavar="REPORT.json",
                        help="compare against this earlier report (repeatable)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        metavar="PCT",
                        help=f"median move to call a regression "
                             f"(default: {DEFAULT_THRESHOLD:g}%%)")
    parser.add_argument("--show-all", action="store_true",
                        help="show every shared metric in the comparison, not "
                             "just regressions/progressions")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="exit 1 when a confirmed regression is found")
    args = parser.parse_args(argv[1:])
    # A negative threshold makes `abs(delta) < threshold` unsatisfiable, so
    # every non-overlapping metric reaches the directional branch and a 0%
    # move gets labelled worse.
    if args.threshold < 0:
        parser.error("--threshold must be >= 0")

    try:
        docs, labels = load(args.reports)
        base_docs, base_labels = load(args.baseline)
    except Exception as exc:
        sys.stderr.write(f"perf_report: cannot read report: {exc}\n")
        return 2

    records = collect(docs)
    if not records:
        sys.stderr.write(
            "perf_report: no performance metrics in the report(s).\n"
            "Re-run the suite with --kind perf (or --kind all) to collect them.\n")
        return 1

    render = render_md if args.format == "md" else render_text
    lines = render(records, docs, labels)

    worse = []
    if base_docs:
        base_records = collect(base_docs)
        if not base_records:
            sys.stderr.write(
                "perf_report: the baseline holds no metrics; nothing to compare.\n")
        else:
            rows, new_keys, missing = compare(
                index(records), fold_baseline(base_docs), args.threshold)
            worse = regressions(rows)
            compare_render = (render_comparison_md if args.format == "md"
                              else render_comparison_text)
            lines += compare_render(rows, new_keys, missing, base_labels,
                                    args.threshold, len(base_docs), args.show_all)

    sys.stdout.write("\n".join(lines) + "\n")
    # Report-only by default: a regression is information, not a verdict, unless
    # the caller explicitly asked for it to be one.
    return 1 if (worse and args.fail_on_regression) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
