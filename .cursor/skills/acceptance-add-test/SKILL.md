---
name: acceptance-add-test
description: Scaffold a new HIP or Verbs test in the HIP/Verbs Acceptance Suite and wire it into the policies. Use when the user asks to add an acceptance test, add a HIP/Verbs test, cover a new primitive, or extend the suite.
---

# Add an acceptance test

Follow the full guide in [../../../docs/ADDING_A_TEST.md](../../../docs/ADDING_A_TEST.md).
This skill is the execution checklist.

## Input: a description, or a proposed test

Start from a plain description, or from a row emitted by the
[`extract-hip-requirements`](../extract-hip-requirements/SKILL.md) /
[`extract-verbs-requirements`](../extract-verbs-requirements/SKILL.md) skills. Those
write `proposed_tests.json`; each `PROPOSE` item carries everything you need:

```bash
python3 -c 'import json,glob; d=json.load(open(sorted(glob.glob("*-req-*/proposed_tests.json"))[0]));
[print(i["id"], i["title"], "|", i["class"], "|", i.get("gate") or "-") for i in d["items"] if i["status"]=="PROPOSE"]'
```

Map an item onto the checklist below: `title` → test id, `sequence`+`verify` → the
body, `class`+`gate` → the policy row (`when` from `gate`), `signal_apis` → which
HIP/Verbs calls to make, `impl_hint`/`call_sites`/`provenance_hint` → reference code
to open and read before writing the sequence.

## Workflow

```
- [ ] 1. Choose category + file
- [ ] 2. Write the ACCEPT_TEST body
- [ ] 3. Add perf (PerfLoop + pass_with_latency) from the start
- [ ] 4. Register in every relevant policy/<gpu>_<nic>.csv
- [ ] 5. Build + verify (filter, perf present, lint, selftest)
```

## Steps

1. **File**: add to the existing category file (e.g. `hip/device_management.cc`,
   `verbs/foundation.cc`) for no build change; a new `.cc` must be added to
   `target_sources` in [../../../CMakeLists.txt](../../../CMakeLists.txt) under the
   `ACCEPT_WITH_HIP` or `ACCEPT_WITH_IBVERBS` block.

2. **Body**: use `ACCEPT_TEST(Id, "category", Topology, Kind, timeout)` returning
   `pass()` / `fail("reason")` / `skip("reason")`. Use `HIP_CHECK` (HIP) or
   `IBV_CHECK` / `IBV_CHECK_PTR` (Verbs). Copy the template from the guide.

3. **Perf**: wrap the real operation in a `PerfLoop` and `return
   pass_with_latency(loop, "<metric>")` so the test emits metrics under
   `--kind perf`. Do not skip this - untimed tests (like the current
   `verbs/datapath`) produce no perf.

4. **Policy**: add a row to each `policy/<gpu>_<nic>.csv` the test targets. Pick the
   class (`MUST_PASS` / `SKIP` + `when` / `NA` / `KNOWN_FAILURE` + ticket). Format:
   [../../../docs/POLICY_REFERENCE.md](../../../docs/POLICY_REFERENCE.md).

5. **Verify**:

```bash
cmake --build build -j
./build/hva --filter '<Id>' --kind func --report r.json
./build/hva --filter '<Id>' --kind perf --report r.json   # perf present?
./build/hva --lint --vendor mi350_ionic --policy policy/mi350_ionic.csv  # no UNCLASSIFIED
./build/unit_tests && bash scripts/selftest.sh ./build/hva .
```
