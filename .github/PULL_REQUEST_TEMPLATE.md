## Motivation

<!-- What does this PR do and why? -->

## Changes

<!-- Summary of the changes. -->

## Testing

<!-- How did you verify this? e.g. build flags, which suites/filters you ran. -->

```bash
cmake -S . -B build -DACCEPT_WITH_HIP=OFF -DACCEPT_WITH_IBVERBS=OFF
cmake --build build -j
./build/unit_tests
bash scripts/selftest.sh ./build/hva .
```

## Checklist

- [ ] New source files carry the `SPDX-License-Identifier: MIT` header.
- [ ] No build artifacts/binaries committed.
- [ ] Third-party additions recorded in `THIRD_PARTY_NOTICES.md`.
- [ ] Commits are signed off (DCO, `git commit -s`).
- [ ] Harness self-test passes.
