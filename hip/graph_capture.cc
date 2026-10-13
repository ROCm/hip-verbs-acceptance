/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
// HIP acceptance tests — Graph Capture group (RCCL graph capture path).
//
// Mirrors real RCCL code:
//   * Capture_Dependency_Update_Chain (ncclStrongStreamWaitStream): under
//     capture, read the frontier with hipStreamGetCaptureInfo_v2, interpose an
//     event-record node, wire frontier->recordNode, reroute future deps through
//     it, then verify the edges structurally with hipGraphGetEdges.
//   * Graph_Capture_AddHostNode (Recorder::record): inject a host node via
//     hipGraphAddHostNode(..., nullptr, 0, &params) under capture.
// RCCL prefers the _v3/_v2 variants and falls back to the base/_v2 APIs used
// here. Built only when ACCEPT_WITH_HIP=ON.

#ifndef ACCEPT_HAVE_HIP
// No HIP — nothing to register.
#else

#include "hip/utils/HipApi.hpp"

#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

#include "common/TestRegistry.hpp"
#include "hip/utils/HipHelpers.hpp"
#include "hip/utils/HipModule.hpp"
#include "hip/utils/HipTest.hpp"

namespace acceptance {

namespace {

// Host function invoked by a captured host node. Sets the int pointed to by
// userData to 1 so the test can prove the host node actually ran on launch.
void graph_host_flag_fn(void* userData) {
  static_cast<std::atomic<int>*>(userData)->store(1, std::memory_order_relaxed);
}

// Copy `n` ints from the device buffer to the host and check buf[i] == i +
// expected_offset. Returns "" on match, else a reason naming the first offender.
std::string verify_offset_indices(const int* d_buf, int n, int expected_offset) {
  std::vector<int> host(static_cast<size_t>(n), -1);
  if (hipMemcpy(host.data(), d_buf, sizeof(int) * static_cast<size_t>(n),
                hipMemcpyDeviceToHost) != hipSuccess)
    return "hipMemcpy D2H failed";
  for (int i = 0; i < n; ++i)
    if (host[static_cast<size_t>(i)] != i + expected_offset)
      return "wrong at index " + std::to_string(i) + ": got " +
             std::to_string(host[static_cast<size_t>(i)]) + ", expected " +
             std::to_string(i + expected_offset);
  return "";
}

// Pre-capture warm-up: a throwaway k_noop launch forces the hiprtc module to
// compile/load (module load is not a capturable stream op), then sync. Returns
// "" on success or a reason.
std::string preload_kernels(hipStream_t stream) {
  std::string launch_err;
  if (hip_kernel_launch("k_noop", dim3(1), dim3(1), 0, stream, nullptr, 0,
                        &launch_err) != hipSuccess)
    return launch_err;
  const hipError_t sync = hipStreamSynchronize(stream);
  if (sync != hipSuccess)
    return std::string("hipStreamSynchronize: ") + hipGetErrorString(sync);
  return "";
}

// End the in-progress capture (finalizing the graph into `graph`) and hand it to
// `owner` so RAII destroys it. Used on capture-phase error paths.
void finalize_capture(hipStream_t stream, hipGraph_t& graph, Graph& owner) {
  (void)hipStreamEndCapture(stream, &graph);
  owner.adopt(graph);
}

// Instantiate `graph`, launch it on `stream`, and sync. Returns "" on success or
// a reason naming the failing HIP call.
std::string instantiate_launch_sync(GraphExec& exec, hipGraph_t graph,
                                    hipStream_t stream) {
  hipError_t status =
      hipGraphInstantiate(exec.receive(), graph, nullptr, nullptr, 0);
  if (status != hipSuccess)
    return std::string("hipGraphInstantiate: ") + hipGetErrorString(status);
  status = hipGraphLaunch(exec.get(), stream);
  if (status != hipSuccess)
    return std::string("hipGraphLaunch: ") + hipGetErrorString(status);
  status = hipStreamSynchronize(stream);
  if (status != hipSuccess)
    return std::string("hipStreamSynchronize: ") + hipGetErrorString(status);
  return "";
}

// The two graph nodes whose injected edges the test asserts on afterward.
struct DependencyChainNodes {
  hipGraphNode_t node_a = nullptr;    // captured fill kernel (the frontier)
  hipGraphNode_t rec_node = nullptr;  // interposed event-record node
};

// Capture the A -> recordNode -> B dependency chain on `stream`: launch fill
// (node A), read the capture frontier, interpose an event-record node wired
// after the frontier, reroute future captured work through it via
// hipStreamUpdateCaptureDependencies, launch add-constant (node B), then end
// capture and hand the finalized graph to `graph_owner`. On any capture-phase
// error the in-progress capture is finalized (so no capture dangles) and a
// non-empty reason is returned; `out` is populated only on success.
std::string capture_dependency_chain(hipStream_t stream, DeviceBuffer& buf,
                                     int n, int addend, Event& record_event,
                                     Graph& graph_owner,
                                     DependencyChainNodes& out) {
  // hipStreamGetCaptureInfo_v2 / hipStreamEndCapture write the in-progress
  // graph into this handle; abort_capture finalizes it into graph_owner so RAII
  // destroys it on any capture-phase error.
  hipGraph_t graph = nullptr;
  auto abort_capture = [&](const std::string& reason) {
    finalize_capture(stream, graph, graph_owner);
    return reason;
  };

  hipError_t begin_err =
      hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed);
  if (begin_err != hipSuccess)
    return std::string("hipStreamBeginCapture: ") + hipGetErrorString(begin_err);

  // Node A: buf[i] = i (the captured work whose frontier we read).
  std::string launch_err;
  FillArgs fill_args{buf.as<int>(), n};
  if (hip_kernel_launch("k_fill_indices", grid_for(n), dim3(kBlockSize), 0,
                        stream, &fill_args, sizeof(fill_args),
                        &launch_err) != hipSuccess)
    return abort_capture(launch_err);

  // Read the work stream's current frontier (expect Active + graph + node A).
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  unsigned long long id = 0;
  const hipGraphNode_t* deps = nullptr;
  size_t num_deps = 0;
  hipError_t info_err = hipStreamGetCaptureInfo_v2(stream, &status, &id, &graph,
                                                   &deps, &num_deps);
  if (info_err != hipSuccess)
    return abort_capture(std::string("hipStreamGetCaptureInfo_v2: ") +
                         hipGetErrorString(info_err));
  if (status != hipStreamCaptureStatusActive || graph == nullptr ||
      num_deps < 1)
    return abort_capture("unexpected capture state: status=" +
                         std::to_string(static_cast<int>(status)) +
                         " graph=" + (graph ? "set" : "null") +
                         " num_deps=" + std::to_string(num_deps));
  const hipGraphNode_t node_a = deps[0];

  // Interpose an event-record node (RCCL's serialEvent record), depending on
  // nothing yet; we wire the frontier -> recordNode edges explicitly next.
  hipGraphNode_t rec_node = nullptr;
  hipError_t add_rec_err = hipGraphAddEventRecordNode(&rec_node, graph, nullptr,
                                                      0, record_event.get());
  if (add_rec_err != hipSuccess)
    return abort_capture(std::string("hipGraphAddEventRecordNode: ") +
                         hipGetErrorString(add_rec_err));

  // Wire frontier -> recordNode: for each current dependency add edge dep ->
  // recordNode (this is RCCL's per-node hipGraphAddDependencies loop).
  for (size_t i = 0; i < num_deps; ++i) {
    hipError_t add_dep_err =
        hipGraphAddDependencies(graph, &deps[i], &rec_node, 1);
    if (add_dep_err != hipSuccess)
      return abort_capture(std::string("hipGraphAddDependencies: ") +
                           hipGetErrorString(add_dep_err));
  }

  // Make every future captured op depend on recordNode.
  hipError_t update_err = hipStreamUpdateCaptureDependencies(
      stream, &rec_node, 1, hipStreamSetCaptureDependencies);
  if (update_err != hipSuccess)
    return abort_capture(std::string("hipStreamUpdateCaptureDependencies: ") +
                         hipGetErrorString(update_err));

  // Node B: buf[i] += addend. Captured after the update, so it must depend on
  // recordNode (transitively on A).
  AddArgs add_args{buf.as<int>(), n, addend};
  if (hip_kernel_launch("k_add_constant", grid_for(n), dim3(kBlockSize), 0,
                        stream, &add_args, sizeof(add_args),
                        &launch_err) != hipSuccess)
    return abort_capture(launch_err);

  hipError_t end_err = hipStreamEndCapture(stream, &graph);
  if (end_err != hipSuccess)
    return std::string("hipStreamEndCapture: ") + hipGetErrorString(end_err);
  if (graph == nullptr) return "hipStreamEndCapture returned a null graph";
  graph_owner.adopt(graph);

  out.node_a = node_a;
  out.rec_node = rec_node;
  return "";
}

// Structural check (load-bearing): the finalized graph must contain
// A -> recordNode and recordNode -> B. The latter only exists because the
// capture-dependency update routed B through recordNode. Returns "" on success.
std::string verify_injected_edges(hipGraph_t graph, hipGraphNode_t node_a,
                                  hipGraphNode_t rec_node) {
  size_t num_edges = 0;
  if (hipGraphGetEdges(graph, nullptr, nullptr, &num_edges) != hipSuccess)
    return "hipGraphGetEdges (count) failed";
  std::vector<hipGraphNode_t> from(num_edges), to(num_edges);
  if (hipGraphGetEdges(graph, from.data(), to.data(), &num_edges) != hipSuccess)
    return "hipGraphGetEdges (fill) failed";
  bool a_to_rec = false, rec_to_b = false;
  for (size_t i = 0; i < num_edges; ++i) {
    if (from[i] == node_a && to[i] == rec_node) a_to_rec = true;
    if (from[i] == rec_node) rec_to_b = true;
  }
  if (!a_to_rec || !rec_to_b)
    return "injected dependency edges missing: A->rec=" +
           std::string(a_to_rec ? "yes" : "no") + " rec->B=" +
           std::string(rec_to_b ? "yes" : "no") +
           " (num_edges=" + std::to_string(num_edges) + ")";
  return "";
}

}  // namespace

// Under capture, inject an event-record node and reroute dependencies; verify edges and result.
ACCEPT_TEST(Capture_Dependency_Update_Chain, "hip/graph", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  const int n = kIndexCount;
  const int addend = 100;  // constant node B adds to every element
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);

  DeviceBuffer buf;
  Stream stream;
  Event record_event;
  Graph graph_owner;
  GraphExec exec;
  HIP_CHECK(buf.allocate(bytes));
  HIP_CHECK(stream.create());
  HIP_CHECK(record_event.create_with_flags(hipEventDisableTiming));
  HIP_CHECK(hipMemset(buf.get(), kPoisonByte, bytes));
  HIP_CHECK(hipDeviceSynchronize());

  if (const std::string e = preload_kernels(stream.get()); !e.empty())
    return fail(e);

  // Capture the A -> recordNode -> B chain and finalize it into graph_owner.
  DependencyChainNodes nodes;
  if (const std::string e = capture_dependency_chain(
          stream.get(), buf, n, addend, record_event, graph_owner, nodes);
      !e.empty())
    return fail(e);
  hipGraph_t graph = graph_owner.get();

  // Structural check: the injected A->recordNode and recordNode->B edges exist.
  if (const std::string e =
          verify_injected_edges(graph, nodes.node_a, nodes.rec_node);
      !e.empty())
    return fail(e);

  // Functional check: instantiate + launch + verify buf[i] == i + addend.
  if (const std::string e = instantiate_launch_sync(exec, graph, stream.get());
      !e.empty())
    return fail(e);

  const std::string err = verify_offset_indices(buf.as<int>(), n, addend);
  if (!err.empty()) return fail("captured dependency chain " + err);
  return pass();
}

// Capture a graph with a host node plus kernel work; launch and verify both ran.
ACCEPT_TEST(Graph_Capture_AddHostNode, "hip/graph", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  const int n = kIndexCount;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);

  static std::atomic<int> host_ran{0};
  host_ran.store(0, std::memory_order_relaxed);

  DeviceBuffer buf;
  Stream stream;
  Graph graph_owner;
  GraphExec exec;
  HIP_CHECK(buf.allocate(bytes));
  HIP_CHECK(stream.create());
  HIP_CHECK(hipMemset(buf.get(), kPoisonByte, bytes));
  HIP_CHECK(hipDeviceSynchronize());

  hipGraph_t graph = nullptr;
  auto abort_capture = [&](const std::string& reason) {
    finalize_capture(stream.get(), graph, graph_owner);
    return fail(reason);
  };

  if (const std::string e = preload_kernels(stream.get()); !e.empty())
    return fail(e);

  HIP_CHECK(hipStreamBeginCapture(stream.get(), hipStreamCaptureModeRelaxed));

  // Node A: real captured work being "logged" by the recorder host node.
  std::string launch_err;
  FillArgs fill_args{buf.as<int>(), n};
  if (hip_kernel_launch("k_fill_indices", grid_for(n), dim3(kBlockSize), 0,
                        stream.get(), &fill_args, sizeof(fill_args),
                        &launch_err) != hipSuccess)
    return abort_capture(launch_err);

  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  unsigned long long id = 0;
  const hipGraphNode_t* deps = nullptr;
  size_t num_deps = 0;
  hipError_t info_err = hipStreamGetCaptureInfo_v2(stream.get(), &status, &id,
                                                   &graph, &deps, &num_deps);
  if (info_err != hipSuccess)
    return abort_capture(std::string("hipStreamGetCaptureInfo_v2: ") +
                         hipGetErrorString(info_err));
  if (status != hipStreamCaptureStatusActive || graph == nullptr)
    return abort_capture("unexpected capture state: status=" +
                         std::to_string(static_cast<int>(status)) +
                         " graph=" + (graph ? "set" : "null"));

  // Inject the host node exactly as RCCL does: nullptr deps, count 0.
  hipHostNodeParams params{};
  params.fn = &graph_host_flag_fn;
  params.userData = &host_ran;
  hipGraphNode_t host_node = nullptr;
  hipError_t add_host_err =
      hipGraphAddHostNode(&host_node, graph, nullptr, 0, &params);
  if (add_host_err != hipSuccess)
    return abort_capture(std::string("hipGraphAddHostNode: ") +
                         hipGetErrorString(add_host_err));

  hipError_t end_err = hipStreamEndCapture(stream.get(), &graph);
  if (end_err != hipSuccess)
    return fail(std::string("hipStreamEndCapture: ") +
                hipGetErrorString(end_err));
  if (graph == nullptr) return fail("hipStreamEndCapture returned a null graph");
  graph_owner.adopt(graph);

  if (const std::string e = instantiate_launch_sync(exec, graph, stream.get());
      !e.empty())
    return fail(e);

  const int ran = host_ran.load(std::memory_order_relaxed);
  if (ran != 1)
    return fail("captured host node did not run (flag=" + std::to_string(ran) +
                ")");
  int bad_index = -1, bad_value = -1;
  if (!hip_verify_indices(buf.as<int>(), n, &bad_index, &bad_value))
    return fail("captured work wrong at index " + std::to_string(bad_index) +
                ": got " + std::to_string(bad_value));
  return pass();
}

}  // namespace acceptance
#endif  // ACCEPT_HAVE_HIP
