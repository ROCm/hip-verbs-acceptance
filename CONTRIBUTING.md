# Contributing

Thanks for your interest in contributing to the HIP/Verbs Acceptance Suite.

## Getting started

- Build and run the suite as described in [README.md](README.md). With both
  backends off (`-DACCEPT_WITH_HIP=OFF -DACCEPT_WITH_IBVERBS=OFF`) the harness
  builds anywhere and runs the synthetic self-test — a good first check.
- Adding a test? Follow [docs/ADDING_A_TEST.md](docs/ADDING_A_TEST.md).
- Design context lives in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Development workflow

1. Fork the repository and create a topic branch from `main`.
2. Make your change. Keep it focused; prefer small, reviewable PRs.
3. Ensure the suite still builds and the harness self-test passes:
   ```bash
   cmake -S . -B build -DACCEPT_WITH_HIP=OFF -DACCEPT_WITH_IBVERBS=OFF
   cmake --build build -j
   ./build/unit_tests
   bash scripts/selftest.sh ./build/hva .
   ```
4. Open a pull request describing the motivation and the testing you did.

## Coding conventions

- C++20; keep to the style of the surrounding code.
- Every new source file must carry the license header:
  ```
  Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
  SPDX-License-Identifier: MIT
  ```
- Do not vendor third-party code without recording it in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Do not commit build artifacts or binaries.

## Developer Certificate of Origin (DCO)

All commits must be signed off, certifying you wrote the change or have the
right to submit it under the project's MIT license (see
https://developercertificate.org/). Add a sign-off line to each commit:

```
Signed-off-by: Your Name <your.email@example.com>
```

Use `git commit -s` to add it automatically.

## Questions

For questions that don't fit a GitHub issue, contact the maintainers:
nikita.gusev2@amd.com or Vadim.Kutovoi@amd.com.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License found in [LICENSE](LICENSE).
