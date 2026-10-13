---
name: acceptance-run
description: Build and run the HIP/Verbs Acceptance Suite and report the verdict. Use when the user asks to run acceptance tests, run the HIP or Verbs tests, gate a machine, smoke-test the harness, or bring the suite up on new hardware.
---

# Run the acceptance suite

Builds `hva` if needed, runs the requested tests, and reports the
`ACCEPTANCE:` verdict. Details of flags and output live in
[../../../README.md](../../../README.md); do not restate them to the user.

## Steps

1. Ensure a build exists (build once if `build/hva` is missing):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DACCEPT_WITH_HIP=ON -DACCEPT_WITH_IBVERBS=ON -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j
```

2. Pick the run based on what the user wants:

| Goal | Command |
|---|---|
| HIP tests (1 node) | `./build/hva --filter-category 'hip/*' --kind func --report report.json` |
| Verbs tests (2 ranks, 1 host) | `./build/hva --local-ranks 2 --filter-category 'verbs/*' --kind func --report report.json` |
| Verbs tests (2 real nodes) | `./build/hva --nodes h1,h2 --filter-category 'verbs/*' --kind func --report report.json` |
| With perf metrics | add `--kind perf` (or `--kind all`) |
| Gate against a policy | add `--vendor <name> --policy policy/<name>.csv` |
| New platform (no policy) | omit `--policy`/`--vendor` -> discovery mode (see [../../../docs/NEW_PLATFORM.md](../../../docs/NEW_PLATFORM.md)) |

3. Report the verdict line, the summary counts, and the `report.json` path. For a
   deeper explanation or failure-to-RCCL-feature mapping, use the
   `acceptance-summarize` skill on the produced `report.json`.

## Notes

- Exit codes: `0` gate pass, `1` gate fail, `2` usage error, `4` launch failure.
- `--nodes` needs the binary and `--policy` dir at the same absolute path on both
  nodes (shared FS); under Slurm, allocate two nodes and pass their hostnames.
- Harness-only smoke (no GPU/NIC): `./build/hva --vendor synthetic
  --policy policy/synthetic.csv --filter 'syn_*' --report report.json` (exits 1 on
  purpose).
