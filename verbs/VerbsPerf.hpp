/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// VerbsPerf — region-timing helper for the RDMA data path, mirroring perf_memcpy
// in hip/utils/HipHelpers.hpp. perf_rdma posts one signaled RDMA op + CQ poll per
// pass (one pass in functional mode, warmup+N under --kind perf|all) and emits the
// per-op latency (us/op) -- the meaningful metric for these small loopback ops,
// where bandwidth would round to ~0. Per-call verbs latency for control-path calls
// is captured separately by IBV_CHECK. No-op unless ibverbs is available.

#ifdef ACCEPT_HAVE_IBVERBS

#include <cstdint>
#include <string>

#include "common/Perf.hpp"
#include "verbs/RcConn.hpp"

namespace acceptance {

// One measured pass for single-shot, barrier-ordered 2-rank protocols (coherence,
// alloc-matrix): warmup+N would desync the barriers, so this runs exactly one pass
// in both functional and perf mode and records a single latency sample.
constexpr MeasureOpts kSinglePass{0, 1};

// Which RDMA data-path op perf_rdma drives.
enum class RdmaOp { Write, Read };

// Post `payload_bytes` over `conn` once (perf off) or warmup+N (perf on), reaping
// each completion, and emit per-op latency (us/op) under `metric` in perf mode.
// Returns "" when all passes complete, or the first failure reason (the loop stops
// on the first error).
inline std::string perf_rdma(PerfSink& sink, const std::string& metric,
                             verbsx::RcConn& connection, RdmaOp op,
                             uint32_t local_key, void* local_addr,
                             size_t payload_bytes,
                             const verbsx::RemoteBuf& remote,
                             int poll_timeout_ms, MeasureOpts options = {}) {
  PerfLoop loop(options);
  for (auto pass : loop) {
    const uint32_t length = static_cast<uint32_t>(payload_bytes);
    std::string reason =
        (op == RdmaOp::Write)
            ? connection.post_write(local_key, local_addr, length, remote,
                                    /*wr_id=*/1)
            : connection.post_read(local_key, local_addr, length, remote,
                                   /*wr_id=*/1);
    if (reason.empty()) {
      reason = connection.poll(poll_timeout_ms, nullptr);
    }
    if (!reason.empty()) {
      return reason;
    }
  }

  loop.emit_latency(sink, metric);
  return "";
}

}  // namespace acceptance

#endif  // ACCEPT_HAVE_IBVERBS
