/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
#pragma once
// HipHelpers — shared host-side helpers for the HIP acceptance tests: the
// HIP_CHECK error macro and host-only verification helpers. The HIP API symbols
// it uses are resolved through the dynamic wrapper (HipApi.hpp), so this header
// compiles under plain g++ on a login node with no HIP toolchain. The trivial
// device kernels live in HipKernels.hpp (HIP-compiler only). Gated on
// ACCEPT_HAVE_HIP so the header is a no-op in non-HIP builds.

#ifdef ACCEPT_HAVE_HIP

#include "hip/utils/HipApi.hpp"
#include "hip/utils/HipKernels.hpp"

#include <string>
#include <vector>

#include "common/Perf.hpp"
#include "common/TestRegistry.hpp"

// HIP_CHECK — evaluate a HIP call; on non-success return fail() from the enclosing
// ACCEPT_TEST body. Under perf mode (--kind perf|all, PerfProbeScope active) it
// also records the call's latency; otherwise PerfTimer is a no-op.
#define HIP_CHECK(expr)                                                       \
  do {                                                                        \
    ::acceptance::PerfTimer _hip_pt(#expr);                                   \
    hipError_t _hip_err = (expr);                                             \
    _hip_pt.stop();                                                           \
    if (_hip_err != hipSuccess)                                               \
      return ::acceptance::fail(std::string(#expr) + ": " +                      \
                            hipGetErrorString(_hip_err));                     \
  } while (0)

namespace acceptance {

// Copy `bytes` source->destination once (perf off) or warmup+N (perf on),
// emitting GB/s under `metric` in perf mode. peer>=0 selects a P2P copy from
// `src_device` to `peer` (source device is explicit, not assumed GPU 0);
// stream==nullptr forces a synchronous copy. Returns the first failing
// hipError_t (hipSuccess when all copies succeed).
inline hipError_t perf_memcpy(PerfSink& sink, const std::string& metric,
                              void* destination, const void* source,
                              std::size_t bytes, hipMemcpyKind kind,
                              hipStream_t stream, int peer = -1,
                              int src_device = 0, MeasureOpts options = {}) {
  PerfLoop loop(options);
  for (auto pass : loop) {
    hipError_t status;
    if (peer >= 0) {
      status = stream
                   ? hipMemcpyPeerAsync(destination, peer, source, src_device,
                                        bytes, stream)
                   : hipMemcpyPeer(destination, peer, source, src_device,
                                   bytes);
    } else {
      status = stream
                   ? hipMemcpyAsync(destination, source, bytes, kind, stream)
                   : hipMemcpy(destination, source, bytes, kind);
    }
    if (status == hipSuccess)
      status = stream ? hipStreamSynchronize(stream) : hipDeviceSynchronize();
    if (status != hipSuccess)
      return status;
  }
  loop.emit_bandwidth(sink, metric, bytes);
  return hipSuccess;
}

static inline bool hip_verify_indices(const int* d_buf, int n, int* bad_index,
                                      int* bad_value) {
  std::vector<int> host(static_cast<size_t>(n), -1);
  if (hipMemcpy(host.data(), d_buf, sizeof(int) * static_cast<size_t>(n),
                hipMemcpyDeviceToHost) != hipSuccess) {
    if (bad_index) *bad_index = -1;
    if (bad_value) *bad_value = -1;
    return false;
  }
  for (int i = 0; i < n; ++i) {
    if (host[static_cast<size_t>(i)] != i) {
      if (bad_index) *bad_index = i;
      if (bad_value) *bad_value = host[static_cast<size_t>(i)];
      return false;
    }
  }
  return true;
}

}  // namespace acceptance

#endif  // ACCEPT_HAVE_HIP
