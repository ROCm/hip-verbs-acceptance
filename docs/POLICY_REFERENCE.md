# Policy reference

A policy is a per-platform CSV (`policy/<gpu>_<nic>.csv`, e.g. `mi350_thor.csv`)
that tells the gate how to treat each test on that platform. It decouples "what the test observed" (pass/fail/
skip) from "what that means here" (required, not-applicable, known-bug).

## Format

```
test,class,when,expected,ticket,reason
```

- Split on the **first 5 commas**; `reason` is the free-form remainder and may
  contain commas (no quoting).
- `#` lines and blank lines are ignored.
- `test` may be an exact id or an fnmatch glob (`Vmm_*`). On overlap, the longest
  literal prefix wins; exact ids beat globs.
- A test may appear on multiple rows with different `when` guards; the first row
  whose `when` matches the environment applies, so put specific rows before a
  catch-all row.

Example (from `policy/mi350_ionic.csv`):

```csv
Memcpy_H2D_Sync,MUST_PASS,,,,
StreamGetId_Stable,MUST_PASS,rocm>=7.1,,,requires hipStreamGetId (ROCm >= 7.1)
Texture_Object_Lifecycle,NA,,,,no image hw on CDNA3+ (gfx950)
Vmm_Map_NoDeadlock_KernelInFlight,KNOWN_FAILURE,,fail,ROCM-25387,hipMemMap deadlock behind in-flight kernel
```

## Columns

### class (required)

| class | meaning | gate on Fail | gate on Skip | gate on Pass |
|---|---|---|---|---|
| `MUST_PASS` | must pass on this platform | fails | fails | ok |
| `SKIP` | software-gated absence is acceptable | (n/a, reported as skip) | neutral | ok |
| `NA` | permanently not applicable on this hardware | (n/a, reported as na) | neutral | ok |
| `KNOWN_FAILURE` | tracked bug | reproduced -> **still fails** (fix pressure) | - | unexpected pass -> **fails**, flagged "possibly fixed" |

A registered test that is not listed in the (present) policy is **not run at
all** (`classify_and_partition` silently excludes it — the policy is the
authoritative run list, there is no `UNCLASSIFIED` class and no gate-neutral
"ran but doesn't count" outcome); use `--lint` to find tests missing from the
policy. Discovery mode (no policy file at all) is different: it treats every
test as `MUST_PASS` and runs all of them.

`KNOWN_FAILURE` always surfaces as `FAIL` by design to keep the bug
visible; the gate goes green only once the bug is fixed (reclassify to `MUST_PASS`)
or the behavior becomes permanent (reclassify to `NA`).

### when (optional)

Gates the row on the runtime environment. Multiple terms join with `;` (AND). If
`when` does not match, the next row for that test id (by declaration order) is
tried. If no row's `when` matches but at least one row exists for the test id,
the test is **not run**. Its class is synthesized from the first row's `when`,
not copied from that row's class: `SKIP` when every term of that predicate is
software-gated (`rocm`, `driver`, `fw`, `sym:`, `!sym:`), and `NA` otherwise
(e.g. a `gpu` term). The first row's ticket and reason are kept; an empty
reason becomes "no predicate matched (environment does not satisfy any when
clause)". `when_matched` is left unset. So `test_x,MUST_PASS,rocm>=7.1,,,` on a
ROCm 7.0 host reports `skip`, not a `MUST_PASS` failure. Add a catch-all row
(empty `when`) to choose the class explicitly. Only a test id with **no row at
all** (no exact match, no matching glob) is unlisted and not run (see above).

| term | true when |
|---|---|
| `rocm>=7.1`, `rocm<7.0`, `rocm>7.0.2` | ROCm version comparison (dotted, zero-padded) |
| `driver<6.0` | driver version comparison |
| `fw<100.0` | RDMA firmware version comparison |
| `gpu<2`, `gpu>=2` | visible GPU count |
| `sym:hipStreamGetId` | the runtime exports this symbol |
| `!sym:libionic` | the runtime does NOT export this symbol |

Common pattern — require on new stacks, mark NA on old ones:

```csv
test_x,NA,rocm<7.0,,T-1,needs ROCm 7.0+
test_x,MUST_PASS,,,,
```

### expected (KNOWN_FAILURE only)

The expected reduced status: `fail` (default), `pass`, `skip`, or `none`. When the
observed reduced status matches, the note is "known failure" (reproduced); on a
mismatch (e.g. it now passes) the test is flagged "possibly fixed".

### ticket / reason

`ticket` is a tracker id (e.g. `ROCM-25387`), surfaced in the report. `reason` is
free text shown next to `SKIP`/`NA`/`KNOWN_FAILURE` lines and in the report.

## Vendor selection

The policy file is chosen by this priority:

1. `--policy <file>` — used directly (highest priority).
2. `--vendor <name>` — selects `<binary_dir>/policy/<name>.csv`.
3. auto-detected platform — prefer `<binary_dir>/policy/<gpu>_<nic>.csv`
   (e.g. `mi350_thor.csv`), else fall back to the bare NIC file
   `<binary_dir>/policy/<nic>.csv`, else discovery. `<gpu>` comes from a
   gfx->marketing map (`gfx950`->`mi350`, `gfx942`->`mi300`, `gfx90a`->`mi250`;
   unmapped arch => no prefix, straight to `<nic>.csv`); `<nic>` is the detected
   NIC vendor (`connectx` / `thor` / `ionic`, else `synthetic`).

If the resolved file is missing **and it was auto-selected** (cases 2-3 by default
dir), the suite enters discovery mode: a `WARN`, then all tests run as `MUST_PASS`.
If `--policy <file>` names a file that does not exist, that is a hard error (typo
guard), not discovery.

## Shipped policies

Each platform has **one** file covering both its verbs and HIP tests (verbs test
ids are `vat_*`, HIP tests are CamelCase, so they never collide):

- `policy/mi350_ionic.csv` — MI350 (gfx950) + the ionic NIC: verbs + HIP.
- `policy/mi350_thor.csv` — MI350 (gfx950) + Thor (bnxt_re NIC): verbs + HIP.
- `policy/synthetic.csv` — the harness synthetic set.

See [NEW_PLATFORM.md](NEW_PLATFORM.md) for authoring a new one from a discovery run,
and [ADDING_A_TEST.md](ADDING_A_TEST.md) for adding a row when you add a test.
