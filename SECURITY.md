# Security Policy

## Reporting a vulnerability

Please do **not** report security vulnerabilities through public GitHub issues.

Report suspected vulnerabilities through one of:

- GitHub's private vulnerability reporting ("Report a vulnerability" under the
  repository's **Security** tab),
- the maintainers directly at nikita.gusev2@amd.com or Vadim.Kutovoi@amd.com, or
- AMD Product Security: https://www.amd.com/en/resources/product-security.html

Please include:

- a description of the issue and its impact,
- steps to reproduce (a minimal test case if possible),
- affected versions / commit, and any known mitigations.

We will acknowledge receipt, investigate, and coordinate a fix and disclosure
timeline with you.

## Scope

This repository is a host-side acceptance-test harness. It links no third-party
libraries at build time (HIP/HSA/verbs entry points are resolved at run time via
`dlopen`/`dlsym`). Reports about the harness, its build, or its test bodies are
in scope; issues in the underlying ROCm/rdma-core stacks should be reported to
those projects.
