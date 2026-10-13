# Vendored rdma-core (libibverbs) headers

These headers are vendored **only for compilation** of the Verbs acceptance
tests. No RDMA library is linked at build time; every `ibv_*` entry point is
resolved dynamically at run time through the `dlopen`/`dlsym` wrapper
(`wrapper/verbs/*`, `wrapper/DynLib.*`). The headers therefore provide
declarations and types only — the login-node build needs no `libibverbs-dev`.

## Minimal set

The build pulls in only the `#include` closure of `<infiniband/verbs.h>`
(computed with `g++ -M`): four headers. Everything else it needs
(`<linux/types.h>`, `<stdint.h>`, `<pthread.h>`, `<limits>`, ...) is standard
C library / libstdc++ and is present on every build host.

| Vendored path                         | Upstream (rdma-core) path            |
| ------------------------------------- | ------------------------------------ |
| `infiniband/verbs.h`                  | `libibverbs/verbs.h`                 |
| `infiniband/verbs_api.h`              | `libibverbs/verbs_api.h`             |
| `infiniband/ib_user_ioctl_verbs.h`    | `kernel-headers/rdma/ib_user_ioctl_verbs.h` |
| `rdma/ib_user_verbs.h`                | `kernel-headers/rdma/ib_user_verbs.h`       |

## Source

Vendored from the Ubuntu `libibverbs-dev` package (rdma-core):

- Package: `libibverbs-dev:amd64`
- Version: `50.0-2ubuntu0.2` (rdma-core v50.0)
- Origin: Ubuntu build host `/usr/include/{infiniband,rdma}/`

## Data-path caveat

`ibv_post_send` / `ibv_post_recv` / `ibv_poll_cq` are `static inline` in
`verbs.h` and dispatch through `qp->context->ops.*` — they are **not** exported
symbols, so `dlsym` cannot resolve them. The Foundation suite does not use them;
the data-path suite will call through `context->ops` (or the `ibv_wr_*` API)
directly. Every Foundation entry point (`ibv_get_device_list`, `ibv_open_device`,
`ibv_query_device/port/gid`, `ibv_alloc_pd`, `ibv_create_cq`, `ibv_reg_mr`,
`ibv_create_qp`, `ibv_fork_init`, ...) is a real exported symbol.

Note: `ibv_query_port` and `ibv_reg_mr` are preprocessor macros in `verbs.h`
(wrapping `___ibv_query_port` / an `iova2` picker). The real exported symbols
exist; the wrapper takes explicit function-pointer typedefs for these two rather
than `decltype` of the macro'd identifier.

## Updating

Re-copy the four headers from a matching `libibverbs-dev` install and update the
**Source** version above. Keep the set to the `g++ -M` closure of
`<infiniband/verbs.h>`; if a new rdma-core reorganizes includes, recompute it:

```sh
g++ -std=c++20 -M -I/usr/include /usr/include/infiniband/verbs.h \
  | tr ' ' '\n' | grep -E 'infiniband|rdma'
```
