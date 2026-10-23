# Adding a test

Every test is a single `ACCEPT_TEST(...)` block that self-registers at startup.
Tests are grouped by category into `.cc` files under `hip/` and `verbs/`. This
guide covers the HIP and Verbs skeletons, how to ship perf from the start, and how
to wire the test into the per-vendor policies.

## Checklist

```
- [ ] 1. Pick the file (existing category file, or a new .cc)
- [ ] 2. Write the test body (ACCEPT_TEST + pass/fail/skip)
- [ ] 3. Add perf (wrap the real op in PerfLoop, return pass_with_latency)
- [ ] 4. Register it in every policy/<gpu>_<nic>.csv
- [ ] 5. Build and verify locally
```

## 1. Pick the file

- Adding to an **existing category** (e.g. another `hip/device` test): put the
  block in the matching file (`hip/device_management.cc`) — no build change needed.
- A **new file**: create it under `hip/` or `verbs/` and add it to
  `target_sources(hva ...)` in
  [../CMakeLists.txt](../CMakeLists.txt) (the source list is explicit, not globbed).
  HIP tests go under the `ACCEPT_WITH_HIP` block, Verbs under `ACCEPT_WITH_IBVERBS`.

## 2. Write the test body

The macro is `ACCEPT_TEST(Id, "category", Topology, Kind, timeout_sec)`. The body
receives `ctx` (rank/size/bootstrap) and returns `pass()`, `fail("reason")`, or
`skip("reason")`. Choose:

- `Topology::Single` for one process; `Topology::TwoNode` for tests that need two
  ranks (RDMA data path). `Kind::Functional` for correctness.
- `skip()` for a software-gated absence (missing symbol / old ROCm); `fail()` for a
  real defect. Crash/hang/timeout are detected by the runner — do not catch them.

### HIP template

```cpp
// in hip/<category>.cc, inside namespace acceptance, guarded by ACCEPT_HAVE_HIP
ACCEPT_TEST(MyFeature_Works, "hip/device", Topology::Single, Kind::Functional, 30) {
  (void)ctx;
  HIP_CHECK(hipSetDevice(0));

  DeviceBuffer buf;
  HIP_CHECK(buf.allocate(4096));
  // ... exercise the primitive, verify the result ...
  if (/* wrong */ false) return fail("MyFeature produced wrong result");
  return pass();
}
```

`HIP_CHECK(expr)` returns `fail()` with the call text + `hipGetErrorString` on a
non-`hipSuccess` return, and (under `--kind perf`) records the call's latency.

### Verbs template

```cpp
// in verbs/<category>.cc, inside namespace acceptance, guarded by ACCEPT_HAVE_IBVERBS
ACCEPT_TEST(vat_my_feature, "verbs/foundation", Topology::Single, Kind::Functional, 30) {
  (void)ctx;
  OpenedDevice d;
  std::string err;
  bool no_devices = false;
  if (!open_selected_device(d, err, no_devices))
    return no_devices ? skip(err) : fail(err);

  struct ibv_pd* pd = ibv_alloc_pd(d.ctx);
  IBV_CHECK_PTR(pd, "ibv_alloc_pd");
  // ... exercise the primitive ...
  (void)ibv_dealloc_pd(pd);
  return pass();
}
```

`IBV_CHECK(expr)` is for calls returning `0`/errno (also auto-timed under perf);
`IBV_CHECK_PTR(ptr, "name")` is for constructor calls returning a pointer/null.

## 3. Add perf (do this from the start)

Wrap the real operation in a `PerfLoop` and return `pass_with_latency`. In `func`
mode the loop runs once (functional check); in `perf` mode it runs warmup+N and the
metric lands in `report.json`. This is why datapath tests without a `PerfLoop`
emit no metrics — new tests should not repeat that.

```cpp
#include "common/Perf.hpp"

PerfLoop loop;
std::string failure;
for (auto pass : loop) {
  // the timed region: the real op, once per pass
  if (/* op failed */ false) { failure = "..."; break; }
  if (pass.first()) {
    // functional verification runs only on the first (untimed warmup) pass
  }
}
if (!failure.empty()) return fail(failure);
return pass_with_latency(loop, "myfeature.op");   // metric key in report.json
```

For a throughput metric use `loop.emit_bandwidth(sink, "myfeature.copy", bytes)`
with a `PerfSink sink;` and return a `TestResult` carrying that sink (see
`hip/data_movement.cc` for the bandwidth pattern).

## 4. Register it in every policy

For each `policy/<gpu>_<nic>.csv` add a row so the gate knows how to treat the test on
that platform. Columns: `test,class,when,expected,ticket,reason`.

```csv
# always required to pass on this platform:
MyFeature_Works,MUST_PASS,,,,
# gated on a runtime capability (skips cleanly when absent):
MyFeature_Works,SKIP,rocm>=7.1,,,requires <symbol> (ROCm >= 7.1)
# permanently not applicable on this hardware:
MyFeature_Works,NA,,,,no image hw on CDNA3+
# a tracked bug that still reproduces (gate stays red until fixed):
MyFeature_Works,KNOWN_FAILURE,,fail,ROCM-12345,short description
```

Guidance:
- `MUST_PASS` is the default for a deterministic functional check.
- Use the `when` column to gate on `rocm>=X` / `driver<Y` / `gpu<N` /
  `sym:<name>` / `!sym:<name>` (`;`-join for AND). See
  [POLICY_REFERENCE.md](POLICY_REFERENCE.md).
- Platforms currently shipped (one file per platform, named `<gpu>_<nic>.csv`,
  covering **both** verbs and HIP tests): `policy/mi350_ionic.csv` (MI350 + ionic
  ionic NIC) and `policy/mi350_thor.csv` (MI350 + Thor bnxt_re NIC), plus
  `synthetic.csv`. Add your row to the platform(s) your test targets.
- If a platform has no policy file, discovery mode runs the test as `MUST_PASS`
  anyway (see [NEW_PLATFORM.md](NEW_PLATFORM.md)) — the policy row records the
  intended classification once you know it.

## 5. Build and verify

```bash
cmake --build build -j
# run just your test:
./build/hva --filter 'MyFeature_Works' --kind func --report r.json
# perf metrics present?
./build/hva --filter 'MyFeature_Works' --kind perf --report r.json
python3 -c 'import json;print([t["perf"] for t in json.load(open("r.json"))["tests"]])'
# harness still self-consistent:
./build/unit_tests && bash scripts/selftest.sh ./build/hva .
# confirm the test is listed in the policy (no UNCLASSIFIED):
./build/hva --lint --vendor mi350_ionic --policy policy/mi350_ionic.csv
```
