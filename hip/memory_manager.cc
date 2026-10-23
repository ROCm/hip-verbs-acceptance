/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
// HIP acceptance tests — Memory Manager group (RCCL mem-manager / VMM path).
//
// Covers the allocation floor RCCL builds on: baseline device/pinned host alloc,
// the full VMM lifecycle (granularity, create/map/setaccess/unmap/release,
// ROCM-2550 RDMA+VMM attribute and multi-map, negative args, POSIX-fd/fabric
// export-import), and the stream-ordered/pooled/managed/fine-grained allocators.
// Device-capability tests skip() at runtime (TODO: move those gates into policy).
// Built only when ACCEPT_WITH_HIP=ON.

#ifndef ACCEPT_HAVE_HIP
// No HIP — nothing to register.
#else

#include "hip/utils/HipApi.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#include "common/Perf.hpp"
#include "common/PosixUtil.hpp"
#include "common/SockCompat.hpp"
#include "common/TestRegistry.hpp"
#include "hip/utils/HipHelpers.hpp"
#include "hip/utils/HipModule.hpp"
#include "hip/utils/HipTest.hpp"

namespace acceptance {

namespace {

// One buffer per reclaim cycle, large enough that a per-cycle leak dwarfs the
// tolerance and small enough to fit beside other work.
constexpr size_t kMiB = 1024 * 1024;
constexpr size_t kReclaimBytes = 256 * kMiB;
constexpr int kReclaimCycles = 6;
constexpr size_t kReclaimToleranceBytes = 16 * kMiB;
// Device-wide accounting lags the release even after a synchronize.
constexpr int kReclaimSettleMs = 500;
// It also disagrees with itself while a release is landing: on some runtimes
// consecutive reads differ by hundreds of MiB in both directions. The median of
// several samples rejects those; the minimum must not be used, because a dip
// below the true value would mask the very leak this measures.
constexpr int kReclaimSamples = 5;
constexpr int kReclaimSampleGapMs = 200;
// A wedged peer must fail the test, not hang it.
constexpr int kReclaimPeerTimeoutSec = 15;
// Let the importer reach its own exit before SIGKILL replaces the exit code.
constexpr long kReclaimReapGraceMs = 2000;

// Lockstep handshake, one cycle: owner sends the fd -> importer maps it and
// answers Imported -> owner asks ReleaseNow -> importer answers Released.
constexpr char kReclaimImportedToken = 'I';
constexpr char kReclaimReleaseNowToken = 'R';
constexpr char kReclaimReleasedToken = 'D';
// Instead of any of the above, a failing importer answers Failed plus a
// fixed-size NUL-padded reason, so the owner's message names the HIP call that
// broke on the importer side rather than just an exit code.
constexpr char kReclaimFailedToken = 'F';
constexpr size_t kReclaimReasonBytes = 200;

// Fill a default Pinned+Device VMM allocation prop for the current device.
// When `exportable` is set, request a POSIX-fd shareable handle type.
hipMemAllocationProp make_vmm_prop(int dev, bool exportable) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = dev;
  if (exportable)
    prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  return prop;
}

// True if device `dev` advertises memory-pool support (hipMallocAsync / mempool).
bool device_has_mempool(int dev) {
  int v = 0;
  return hipDeviceGetAttribute(&v, hipDeviceAttributeMemoryPoolsSupported, dev) ==
             hipSuccess &&
         v != 0;
}

bool device_has_managed(int dev) {
  int v = 0;
  return hipDeviceGetAttribute(&v, hipDeviceAttributeManagedMemory, dev) ==
             hipSuccess &&
         v != 0;
}

bool device_has_vmm(int dev) {
  int v = 0;
  return hipDeviceGetAttribute(
             &v, hipDeviceAttributeVirtualMemoryManagementSupported, dev) ==
             hipSuccess &&
         v != 0;
}

// Generic device-attribute query (used by the ROCM-2550 RDMA-VMM invariant).
bool query_device_attr(int dev, hipDeviceAttribute_t attr, int* out) {
  int v = 0;
  if (hipDeviceGetAttribute(&v, attr, dev) != hipSuccess) return false;
  *out = v;
  return true;
}

// ROCM-2550 multimap helper: `count` hipMemCreate/map rounds with the given
// gpuDirectRDMACapable flag; kernel-verify the last mapping. Empty -> success.
std::string vmm_rdma_multimap(unsigned char rdma_flag, int count) {
  const size_t bytes = sizeof(int) * 4096;
  hipMemAllocationProp prop = make_vmm_prop(gpu(), /*exportable=*/false);
  prop.allocFlags.gpuDirectRDMACapable = rdma_flag;
  size_t gran = 0;
  hipError_t gran_err = hipMemGetAllocationGranularity(
      &gran, &prop, hipMemAllocationGranularityMinimum);
  if (gran_err != hipSuccess)
    return std::string("hipMemGetAllocationGranularity: ") +
           hipGetErrorString(gran_err);
  const size_t size = round_up(bytes, gran);

  // Each VmmMapping owns its handle and VA, so the whole set unmaps/releases on
  // scope exit -- no manual rollback loop.
  std::vector<VmmMapping> mappings;
  for (int i = 0; i < count; ++i) {
    hipMemGenericAllocationHandle_t handle{};
    hipError_t create_err = hipMemCreate(&handle, size, &prop, 0);
    if (create_err != hipSuccess)
      return "hipMemCreate #" + std::to_string(i) + ": " +
             hipGetErrorString(create_err);
    VmmMapping mapping;
    std::string map_err = mapping.map(handle, size, gpu(), /*own_handle=*/true);
    if (!map_err.empty()) return "map #" + std::to_string(i) + ": " + map_err;
    mappings.push_back(std::move(mapping));
  }

  if (!mappings.empty()) {
    const std::string err =
        fill_indices_and_verify(mappings.back().as<int>(), kIndexCount, nullptr);
    if (!err.empty())
      return "last VMM buffer (flag=" +
             std::to_string(static_cast<unsigned>(rdma_flag)) + "): " + err;
  }
  return {};
}

// hipMemImportFromShareableHandle POSIX-fd osHandle has two runtime ABIs: 
// 1. PointerToFd (runtime reads *osHandle, caller passes &fd)
// 2. FdAsPointer (runtime reads the fd value from osHandle, caller passes (void*)(uintptr_t)fd).
// ROCm >= 7.1 is always FdAsPointer.
// Below 7.1 the version cannot discriminate: a vanilla 7.0.2 (PointerToFd) and 
// a patched 7.0.2.x that back-ported FdAsPointer report the same
// hipRuntimeGetVersion (70051831), so there we probe the ABI directly.
// Unknown means the probe itself could not run (allocate/export failed), so the
// caller must fail rather than guess a convention.
enum class PosixFdImportStyle { FdAsPointer, PointerToFd, Unknown };

static PosixFdImportStyle posix_fd_import_style() {
  static const PosixFdImportStyle style = []() {
    int version = 0;
    if (hipRuntimeGetVersion(&version) == hipSuccess && version >= 70100000)
      return PosixFdImportStyle::FdAsPointer;

    hipMemAllocationProp prop = make_vmm_prop(gpu(), /*exportable=*/true);
    size_t granularity = 0;
    hipMemAllocationGranularity_flags flags = hipMemAllocationGranularityMinimum;

    hipError_t err = hipMemGetAllocationGranularity(&granularity, &prop,
                                                    flags);
    if (err != hipSuccess || granularity == 0)
      return PosixFdImportStyle::Unknown;

    hipMemGenericAllocationHandle_t handle{};
    err = hipMemCreate(&handle, granularity, &prop, 0);
    if (err != hipSuccess)
      return PosixFdImportStyle::Unknown;

    int fd = -1;
    err = hipMemExportToShareableHandle(&fd, handle,
                                        hipMemHandleTypePosixFileDescriptor, 0);
    if (err != hipSuccess) {
      (void)hipMemRelease(handle);
      return PosixFdImportStyle::Unknown;
    }

    // Probe PointerToFd first: it is crash-safe (a real &fd on a FdAsPointer
    // runtime just yields a bad fd), whereas FdAsPointer on a PointerToFd runtime
    // dereferences the tiny fd value and SIGSEGVs. A failed import here therefore
    // means the runtime wants FdAsPointer.
    PosixFdImportStyle detected;
    hipMemGenericAllocationHandle_t imported{};
    err = hipMemImportFromShareableHandle(&imported, static_cast<void*>(&fd),
                                          hipMemHandleTypePosixFileDescriptor);
    if (err == hipSuccess) {
      detected = PosixFdImportStyle::PointerToFd;
      (void)hipMemRelease(imported);
    } else {
      detected = PosixFdImportStyle::FdAsPointer;
      (void)hipGetLastError();
    }

    (void)close(fd);
    (void)hipMemRelease(handle);
    return detected;
  }();
  return style;
}

void* posix_fd_os_handle(int* fd_ptr, PosixFdImportStyle style) {
  return (style == PosixFdImportStyle::PointerToFd)
             ? static_cast<void*>(fd_ptr)
             : reinterpret_cast<void*>(static_cast<intptr_t>(*fd_ptr));
}

// Map an already-imported VMM handle at a fresh VA and confirm the index
// pattern written by the exporter is visible. `imported` is consumed by the
// mapping (own_handle=true). `label` names the sub-test in error messages.
// Returns "" on success.
std::string map_and_verify_imported(hipMemGenericAllocationHandle_t imported,
                                    size_t size, int n, const char* label) {
  VmmMapping dst;
  std::string map_err = dst.map(imported, size, gpu(), /*own_handle=*/true);
  if (!map_err.empty()) return std::string(label) + " " + map_err;
  int bad_index = -1, bad_value = -1;
  if (!hip_verify_indices(dst.as<int>(), n, &bad_index, &bad_value))
    return std::string(label) + " buffer wrong at index " +
           std::to_string(bad_index) + ": got " + std::to_string(bad_value);
  return "";
}

// Export `handle` to a POSIX fd, re-import it through the standard runtime
// entry point, then map+verify. The exported fd is closed right after import
// (before map/setAccess), mirroring RCCL's ncclP2pImportShareableBuffer.
std::string export_import_roundtrip(hipMemGenericAllocationHandle_t handle,
                                    size_t size, int n,
                                    PosixFdImportStyle import_style) {
  int fd = -1;
  hipError_t export_err = hipMemExportToShareableHandle(
      &fd, handle, hipMemHandleTypePosixFileDescriptor, 0);
  if (export_err != hipSuccess)
    return std::string("hipMemExportToShareableHandle: ") +
           hipGetErrorString(export_err);

  hipMemGenericAllocationHandle_t imported{};
  hipError_t import_err = hipMemImportFromShareableHandle(
      &imported, posix_fd_os_handle(&fd, import_style),
      hipMemHandleTypePosixFileDescriptor);
  if (import_err != hipSuccess) {
    (void)close(fd);
    return std::string("hipMemImportFromShareableHandle: ") +
           hipGetErrorString(import_err);
  }
  (void)close(fd);
  return map_and_verify_imported(imported, size, n, "imported");
}

// Exercise the optional compat import shim some runtimes expose for older
// callers (hipMemImportFromShareableHandleCompat). It is not in the public
// headers, so resolve it dynamically: absent -> "" (nothing to test, not a
// failure); present -> a fresh export/import must round-trip the same way.
std::string compat_roundtrip(hipMemGenericAllocationHandle_t handle,
                             size_t size, int n,
                             PosixFdImportStyle import_style) {
  using CompatFn = hipError_t (*)(hipMemGenericAllocationHandle_t*, void*,
                                  hipMemAllocationHandleType);
  auto compat = reinterpret_cast<CompatFn>(
      dlsym(RTLD_DEFAULT, "hipMemImportFromShareableHandleCompat"));
  if (compat == nullptr) return "";

  int fd = -1;
  hipError_t reexport_err = hipMemExportToShareableHandle(
      &fd, handle, hipMemHandleTypePosixFileDescriptor, 0);
  if (reexport_err != hipSuccess)
    return std::string("compat hipMemExportToShareableHandle: ") +
           hipGetErrorString(reexport_err);

  hipMemGenericAllocationHandle_t imported{};
  hipError_t import_err =
      compat(&imported, posix_fd_os_handle(&fd, import_style),
             hipMemHandleTypePosixFileDescriptor);
  if (import_err != hipSuccess) {
    (void)close(fd);
    return std::string("hipMemImportFromShareableHandleCompat: ") +
           hipGetErrorString(import_err);
  }
  (void)close(fd);
  return map_and_verify_imported(imported, size, n, "compat imported");
}

// hipMemMap on the null stream is gated by in-flight work there, so the map is
// raced against this watchdog; exceeding it is the ROCM-25387 deadlock.
constexpr auto kVmmMapWatchdog = std::chrono::seconds(20);

// Outcome of racing hipMemMap against the watchdog while a kernel is in flight.
struct WatchdogMapResult {
  bool timed_out = false;         // worker did not return before the deadline
  hipError_t map_status = hipSuccess;  // hipMemMap-path status when it did return
  void* va = nullptr;             // reserved VA (maybe mapped); caller tears down
};

// Launch the host-flag spin kernel on the NULL stream (stream 0) — that is
// where CLR queues hipMemMap's VirtualMapCommand Marker, so this is the stream
// whose in-flight work gates the map. Returns "" on success.
std::string launch_spinner(int* done_flag) {
  SpinArgs spin_args{done_flag};
  std::string launch_err;
  hipError_t launch_status =
      hip_kernel_launch("k_spin_on_host_flag", dim3(1), dim3(1), 0, 0,
                        &spin_args, sizeof(spin_args), &launch_err);
  if (launch_status != hipSuccess)
    return std::string("spin kernel launch: ") + launch_err;
  return "";
}

// Reserve a VA and run hipMemMap+SetAccess for `handle` on a worker thread while
// the main thread enforces kVmmMapWatchdog. Once the deadline passes (or the
// worker finishes) `done_flag` is released so the spinner — and any blocked
// hipMemMap — can complete, then the worker is always joined (no leak).
WatchdogMapResult map_on_worker_with_watchdog(
    hipMemGenericAllocationHandle_t handle, size_t size, int* done_flag) {
  std::atomic<bool> map_done{false};
  std::atomic<hipError_t> worker_status{hipSuccess};
  void* va = nullptr;
  std::thread worker([&]() {
    hipError_t status = hipMemAddressReserve(&va, size, 0, nullptr, 0);
    if (status == hipSuccess)
      status = hipMemMap(va, size, 0, handle, 0);  // gated call
    if (status == hipSuccess) {
      hipMemAccessDesc desc{};
      desc.location.type = hipMemLocationTypeDevice;
      desc.location.id = gpu();
      desc.flags = hipMemAccessFlagsProtReadWrite;
      status = hipMemSetAccess(va, size, &desc, 1);
    }
    worker_status.store(status, std::memory_order_release);
    map_done.store(true, std::memory_order_release);
  });

  const auto deadline = std::chrono::steady_clock::now() + kVmmMapWatchdog;
  while (!map_done.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  WatchdogMapResult result;
  result.timed_out = !map_done.load(std::memory_order_acquire);
  __atomic_store_n(done_flag, 1, __ATOMIC_RELEASE);
  worker.join();
  (void)hipDeviceSynchronize();
  result.map_status = worker_status.load(std::memory_order_acquire);
  result.va = va;
  return result;
}

// Tear down whatever the worker mapped: unmap + free the VA (if any) and
// release the backing handle.
void teardown_vmm(void* va, size_t size,
                  hipMemGenericAllocationHandle_t handle) {
  if (va != nullptr) {
    (void)hipMemUnmap(va, size);
    (void)hipMemAddressFree(va, size);
  }
  (void)hipMemRelease(handle);
}

// Descriptor for a device-pinned memory pool on `dev`, shared by the MemPool
// tests (a non-exportable pinned pool located on the device).
hipMemPoolProps make_device_pinned_pool_props(int dev) {
  hipMemPoolProps props{};
  props.allocType = hipMemAllocationTypePinned;
  props.handleTypes = hipMemHandleTypeNone;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id = dev;
  return props;
}

// Shared body for the shareable-handle D2D tests: map `imported`, D2D-copy an index
// pattern into it and verify (a shared buffer is a real copy destination, not just mappable).
TestResult import_d2d_copy_verify(hipMemGenericAllocationHandle_t imported,
                                  size_t size, int n) {
  VmmMapping dst;
  const std::string map_err = dst.map(imported, size, gpu(), /*own_handle=*/true);
  if (!map_err.empty()) return fail("imported dst " + map_err);

  const size_t bytes = sizeof(int) * static_cast<size_t>(n);
  std::vector<int> host_src(static_cast<size_t>(n));
  std::iota(host_src.begin(), host_src.end(), 0);  // 0, 1, 2, ...
  DeviceBuffer src;
  HIP_CHECK(src.allocate(bytes));
  HIP_CHECK(hipMemcpy(src.get(), host_src.data(), bytes, hipMemcpyHostToDevice));
  // The point of the test: D2D copy into the imported shareable-handle mapping.
  HIP_CHECK(hipMemcpy(dst.get(), src.get(), bytes, hipMemcpyDeviceToDevice));
  HIP_CHECK(hipDeviceSynchronize());

  int bad_index = -1, bad_value = -1;
  if (!hip_verify_indices(dst.as<int>(), n, &bad_index, &bad_value))
    return fail("D2D into imported buffer wrong at index " +
                std::to_string(bad_index) + ": got " +
                std::to_string(bad_value));
  return pass();
}

// ---------------------------------------------------------------------------
// Reclaim cycles
// ---------------------------------------------------------------------------

// hipMemGetInfo is not per-process, so a reading only means something on a node
// with no other GPU job.
std::string device_settled_used_bytes(size_t* used) {
  std::this_thread::sleep_for(std::chrono::milliseconds(kReclaimSettleMs));
  std::vector<size_t> samples;
  samples.reserve(kReclaimSamples);
  for (int sample = 0; sample < kReclaimSamples; ++sample) {
    if (sample > 0)
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kReclaimSampleGapMs));
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const hipError_t status = hipMemGetInfo(&free_bytes, &total_bytes);
    if (status != hipSuccess)
      return std::string("hipMemGetInfo: ") + hipGetErrorString(status);
    samples.push_back(total_bytes - free_bytes);
  }
  const auto middle = samples.begin() + kReclaimSamples / 2;
  std::nth_element(samples.begin(), middle, samples.end());
  *used = *middle;
  return "";
}

size_t reclaim_tolerance(size_t granularity) {
  return std::max(kReclaimToleranceBytes, 2 * granularity);
}

// Map `handle` and fault in every page. The VA is handed back instead of
// scope-owned because these probes must keep the reservation past the release.
std::string vmm_map_and_touch(hipMemGenericAllocationHandle_t handle,
                              size_t size, int dev, void** va) {
  *va = nullptr;
  VmmMapping mapping;
  const std::string map_err = mapping.map(handle, size, dev,
                                          /*own_handle=*/false);
  if (!map_err.empty()) return map_err;

  hipError_t status = hipMemset(mapping.get(), kPoisonByte, size);
  if (status != hipSuccess)
    return std::string("hipMemset: ") + hipGetErrorString(status);
  status = hipDeviceSynchronize();
  if (status != hipSuccess)
    return std::string("hipDeviceSynchronize: ") + hipGetErrorString(status);

  *va = mapping.release();
  return "";
}

// Unmap and release, leaving the VA reserved: the defect only shows itself while
// the reservation is still held.
std::string vmm_unmap_release_keep_va(void* va, size_t size,
                                      hipMemGenericAllocationHandle_t handle) {
  hipError_t status = hipMemUnmap(va, size);
  if (status != hipSuccess)
    return std::string("hipMemUnmap: ") + hipGetErrorString(status);
  status = hipMemRelease(handle);
  if (status != hipSuccess)
    return std::string("hipMemRelease: ") + hipGetErrorString(status);
  status = hipDeviceSynchronize();
  if (status != hipSuccess)
    return std::string("hipDeviceSynchronize: ") + hipGetErrorString(status);
  return "";
}

// The first cycle is the reference rather than a pre-loop baseline because the
// first touch pulls in one-time HIP allocations; the peak rather than the last
// cycle because a peer exiting at the end frees what its own release did not.
std::string reclaim_verdict(const std::vector<size_t>& used_per_cycle,
                            size_t tolerance) {
  const size_t reference = used_per_cycle.front();
  size_t peak_growth = 0;
  for (size_t cycle = 1; cycle < used_per_cycle.size(); ++cycle) {
    const size_t used = used_per_cycle[cycle];
    const size_t growth = used > reference ? used - reference : 0;
    peak_growth = std::max(peak_growth, growth);
  }
  if (peak_growth <= tolerance) return "";

  std::string curve;
  for (size_t used : used_per_cycle) {
    if (!curve.empty()) curve += " ";
    curve += std::to_string(used / kMiB);
  }
  return "peak growth " + std::to_string(peak_growth / kMiB) + " MiB over " +
         std::to_string(used_per_cycle.size()) + " cycles exceeds the " +
         std::to_string(tolerance / kMiB) +
         " MiB tolerance (used after release per cycle, MiB: " + curve + ")";
}

// Both directions: a wedged peer must not turn a blocking send into a hang the
// runner can only report as a timeout.
void reclaim_set_socket_timeouts(int sock) {
  timeval timeout{};
  timeout.tv_sec = kReclaimPeerTimeoutSec;
  (void)::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  (void)::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

// A VMM POSIX fd only crosses a process boundary as an SCM_RIGHTS message.
bool send_fd(int sock, int fd) {
  char payload = 0;
  iovec io{&payload, sizeof(payload)};
  char control[CMSG_SPACE(sizeof(int))] = {};
  msghdr msg{};
  msg.msg_iov = &io;
  msg.msg_iovlen = 1;
  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);
  cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));
  ssize_t sent = -1;
  do {
    sent = ::sendmsg(sock, &msg, MSG_NOSIGNAL);
  } while (sent < 0 && errno == EINTR);
  return sent > 0;
}

int recv_fd(int sock) {
  char payload = 0;
  iovec io{&payload, sizeof(payload)};
  char control[CMSG_SPACE(sizeof(int))] = {};
  msghdr msg{};
  msg.msg_iov = &io;
  msg.msg_iovlen = 1;
  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);
  ssize_t got = -1;
  do {
    got = ::recvmsg(sock, &msg, MSG_CMSG_CLOEXEC);
  } while (got < 0 && errno == EINTR);
  if (got <= 0 || (msg.msg_flags & MSG_CTRUNC) != 0) return -1;

  // An unchecked short or foreign cmsg yields a bogus fd the caller would close.
  const cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  if (cmsg == nullptr || cmsg->cmsg_level != SOL_SOCKET ||
      cmsg->cmsg_type != SCM_RIGHTS || cmsg->cmsg_len != CMSG_LEN(sizeof(int)))
    return -1;
  int fd = -1;
  std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
  return fd;
}

bool send_token(int sock, char token) {
  return sockcompat::send_all(sock, &token, 1);
}

bool expect_token(int sock, char token) {
  char got = 0;
  return sockcompat::recv_all(sock, &got, 1) && got == token;
}

// Importer side: report `reason` to the owner and exit. Best-effort by design —
// the exit still fails the test if the socket is already gone.
[[noreturn]] void fail_as_importer(int sock, const std::string& reason) {
  char padded[kReclaimReasonBytes] = {};
  reason.copy(padded, sizeof(padded) - 1);
  if (send_token(sock, kReclaimFailedToken))
    (void)sockcompat::send_all(sock, padded, sizeof(padded));
  _exit(1);
}

// Owner side: read the importer's answer to one handshake step. Empty when
// `expected` arrived, the importer's own reason when it reported a failure, and
// `lost` when no answer came at all.
std::string await_importer_token(int sock, char expected, const char* lost) {
  char got = 0;
  if (!sockcompat::recv_all(sock, &got, 1)) return lost;
  if (got == expected) return "";
  // One byte over kReclaimReasonBytes stays zero, so the reason is terminated
  // whatever the peer sent.
  char reason[kReclaimReasonBytes + 1] = {};
  if (got == kReclaimFailedToken &&
      sockcompat::recv_all(sock, reason, kReclaimReasonBytes) &&
      reason[0] != '\0')
    return std::string("importer: ") + reason;
  return lost;
}

// Importer: per cycle import the owner's fd, map and touch it, then release on
// request while keeping the VA reserved, like the owner.
[[noreturn]] void run_reclaim_importer_child(int sock) {
  reclaim_set_socket_timeouts(sock);
  const hipError_t set_device_err = hipSetDevice(gpu());
  if (set_device_err != hipSuccess)
    fail_as_importer(sock, std::string("hipSetDevice: ") +
                               hipGetErrorString(set_device_err));
  // Force context creation so a broken child is attributable to HIP, not to the
  // handshake below.
  (void)hipFree(nullptr);

  const PosixFdImportStyle import_style = posix_fd_import_style();
  if (import_style == PosixFdImportStyle::Unknown)
    fail_as_importer(sock, "could not probe the POSIX-fd import ABI");

  size_t size = 0;
  if (!sockcompat::recv_all(sock, &size, sizeof(size))) _exit(1);

  for (int cycle = 0; cycle < kReclaimCycles; ++cycle) {
    int fd = recv_fd(sock);
    if (fd < 0) _exit(1);

    hipMemGenericAllocationHandle_t imported{};
    const hipError_t import_err = hipMemImportFromShareableHandle(
        &imported, posix_fd_os_handle(&fd, import_style),
        hipMemHandleTypePosixFileDescriptor);
    // The fd is not needed past the import.
    (void)close(fd);
    if (import_err != hipSuccess)
      fail_as_importer(sock,
                       std::string("hipMemImportFromShareableHandle: ") +
                           hipGetErrorString(import_err));

    void* va = nullptr;
    const std::string map_error = vmm_map_and_touch(imported, size, gpu(), &va);
    if (!map_error.empty()) {
      (void)hipMemRelease(imported);
      fail_as_importer(sock, map_error);
    }
    if (!send_token(sock, kReclaimImportedToken)) _exit(1);

    if (!expect_token(sock, kReclaimReleaseNowToken)) _exit(1);
    const std::string release_error =
        vmm_unmap_release_keep_va(va, size, imported);
    if (!release_error.empty()) fail_as_importer(sock, release_error);
    if (!send_token(sock, kReclaimReleasedToken)) _exit(1);
  }
  _exit(0);
}

std::string describe_wait_status(int status) {
  if (status == posix_util::kWaitStatusUnavailable)
    return "wait status unavailable";
  if (WIFEXITED(status))
    return "exit " + std::to_string(WEXITSTATUS(status));
  if (WIFSIGNALED(status))
    return "signal " + std::to_string(WTERMSIG(status));
  return "wait status " + std::to_string(status);
}

// Hand the owner's handle to the importer, wait for it to map, then have it
// release before the owner does.
std::string reclaim_peer_roundtrip(int sock,
                                   hipMemGenericAllocationHandle_t handle) {
  int fd = -1;
  const hipError_t export_err = hipMemExportToShareableHandle(
      &fd, handle, hipMemHandleTypePosixFileDescriptor, 0);
  if (export_err != hipSuccess)
    return std::string("hipMemExportToShareableHandle: ") +
           hipGetErrorString(export_err);

  std::string error;
  if (!send_fd(sock, fd))
    error = "could not hand the handle to the importer";
  else
    error = await_importer_token(sock, kReclaimImportedToken,
                                 "importer did not confirm the import");
  if (error.empty() && !send_token(sock, kReclaimReleaseNowToken))
    error = "could not ask the importer to release";
  if (error.empty())
    error = await_importer_token(sock, kReclaimReleasedToken,
                                 "importer did not confirm its release");
  (void)close(fd);
  return error;
}

// Create/touch/release cycles, collecting the settled device-wide reading after
// each. With `sock >= 0` every buffer also round-trips through the forked
// importer first. The per-cycle VA reservations are never freed: both callers
// are force-isolated, so they die with the test process.
std::string run_reclaim_cycles(int sock, size_t size,
                               const hipMemAllocationProp& prop,
                               std::vector<size_t>* used_per_cycle) {
  for (int cycle = 0; cycle < kReclaimCycles; ++cycle) {
    hipMemGenericAllocationHandle_t handle{};
    const hipError_t create_err = hipMemCreate(&handle, size, &prop, 0);
    if (create_err != hipSuccess)
      return std::string("hipMemCreate: ") + hipGetErrorString(create_err);

    void* va = nullptr;
    std::string error = vmm_map_and_touch(handle, size, gpu(), &va);
    if (!error.empty()) {
      (void)hipMemRelease(handle);
      return error;
    }

    if (sock >= 0) error = reclaim_peer_roundtrip(sock, handle);

    // Released on every path so the reading below is comparable across cycles.
    const std::string release_error =
        vmm_unmap_release_keep_va(va, size, handle);
    if (error.empty()) error = release_error;
    if (!error.empty()) return error;

    size_t used = 0;
    error = device_settled_used_bytes(&used);
    if (!error.empty()) return error;
    used_per_cycle->push_back(used);
  }
  return "";
}

}  // namespace

// ===========================================================================
// Baseline
// ===========================================================================

// hipMalloc, kernel fill, host verify, hipFree — basic device memory integrity.
ACCEPT_TEST(HipMalloc_Free_Integrity, "hip/memmgr", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  const int n = kIndexCount;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);

  DeviceBuffer buf;
  HIP_CHECK(buf.allocate(bytes));
  HIP_CHECK(hipMemset(buf.get(), kPoisonByte, bytes));

  const std::string err = fill_indices_and_verify(buf.as<int>(), n, nullptr);
  if (!err.empty()) return fail(err);
  return pass();
}

// hipHostMalloc pinned buffer: host write/read and kernel touch through the same pointer.
ACCEPT_TEST(HostAlloc_Pinned, "hip/memmgr", Topology::Single, Kind::Functional,
            30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  const int n = kIndexCount;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);

  HostBuffer host;
  HIP_CHECK(host.allocate(bytes, hipHostMallocMapped));
  int* host_ptr = host.as<int>();

  // Poison from the host so a no-op device write is detectable.
  for (int i = 0; i < n; ++i) host_ptr[i] = -1;

  // Obtain the device pointer for the pinned/mapped host buffer and have a kernel
  // write the index pattern through it — this is what proves the memory is
  // device-addressable (not just host-accessible).
  int* device_ptr = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&device_ptr),
                                    host_ptr, 0));

  FillArgs args{device_ptr, n};
  std::string launch_err;
  hipError_t write_err =
      hip_kernel_launch("k_fill_indices", grid_for(n), dim3(kBlockSize), 0, 0,
                        &args, sizeof(args), &launch_err);
  if (write_err == hipSuccess) write_err = hipDeviceSynchronize();
  if (write_err != hipSuccess)
    return fail(launch_err.empty()
                    ? std::string("hipDeviceSynchronize: ") +
                          hipGetErrorString(write_err)
                    : launch_err);

  // Read back on the host through the same pinned pointer.
  const std::string err = verify_host_indices(host_ptr, n);
  if (!err.empty()) return fail("pinned host buffer " + err);

  // Pinned-host H2D/D2H bandwidth at a transfer-sized buffer: one copy each way
  // (perf off) or warmup+N (perf on, emitting GB/s) -- one path, no perf branch.
  TestResult result = pass();
  const size_t bandwidth_bytes = 32u * 1024u * 1024u;

  HostBuffer host_bandwidth_buf;
  DeviceBuffer device_bandwidth_buf;
  if (host_bandwidth_buf.allocate(bandwidth_bytes, hipHostMallocDefault) !=
      hipSuccess) {
    return fail("pinned bandwidth: host alloc failed");
  }
  if (device_bandwidth_buf.allocate(bandwidth_bytes) != hipSuccess) {
    return fail("pinned bandwidth: device alloc failed");
  }

  hipError_t h2d_status = perf_memcpy(
      result.perf, "hostpinned.h2d", device_bandwidth_buf.get(),
      host_bandwidth_buf.get(), bandwidth_bytes, hipMemcpyHostToDevice, nullptr);
  if (h2d_status != hipSuccess) {
    return fail("hostpinned.h2d: " + std::string(hipGetErrorString(h2d_status)));
  }

  hipError_t d2h_status = perf_memcpy(
      result.perf, "hostpinned.d2h", host_bandwidth_buf.get(),
      device_bandwidth_buf.get(), bandwidth_bytes, hipMemcpyDeviceToHost, nullptr);
  if (d2h_status != hipSuccess) {
    return fail("hostpinned.d2h: " + std::string(hipGetErrorString(d2h_status)));
  }

  return result;
}

// ===========================================================================
// Virtual Memory Management (VMM) — ROCm >= 6.4
// ===========================================================================

// Query VMM allocation granularity and confirm size rounding is consistent.
ACCEPT_TEST(Vmm_Granularity, "hip/memmgr", Topology::Single, Kind::Functional,
            30) {
  (void)ctx;
  if (!device_has_vmm(gpu()))
    return skip("device does not advertise VMM support");  // TODO: policy gate

  hipMemAllocationProp prop = make_vmm_prop(gpu(), /*exportable=*/true);
  size_t gran = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop,
                                           hipMemAllocationGranularityMinimum));
  if (gran == 0) return fail("minimum granularity reported as 0");

  const size_t req = gran + 1;            // deliberately unaligned
  const size_t r1 = round_up(req, gran);  // -> 2*gran
  const size_t r2 = round_up(r1, gran);   // already aligned -> unchanged
  if (r1 != 2 * gran || r2 != r1)
    return fail("granularity rounding inconsistent: gran=" +
                std::to_string(gran) + " r1=" + std::to_string(r1) +
                " r2=" + std::to_string(r2));
  return pass();
}

// Full VMM path: create, reserve, map, kernel verify, unmap, release, free.
ACCEPT_TEST(Vmm_Create_Map_Roundtrip, "hip/memmgr", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  if (!device_has_vmm(gpu()))
    return skip("device does not advertise VMM support");  // TODO: policy gate

  const int n = kIndexCount;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);
  hipMemGenericAllocationHandle_t handle{};
  size_t size = 0;
  std::string create_err = vmm_create_pinned(bytes, gpu(), &handle, &size);
  if (!create_err.empty()) return fail(create_err);

  // Map the retained handle at a new VA and unmap on scope exit.
  // The functional map is timed; verification runs only on the first pass.
  PerfLoop loop;
  std::string failure;
  for (auto pass : loop) {
    VmmMapping mapping;
    failure = mapping.map(handle, size, gpu(), /*own_handle=*/false);
    if (failure.empty() && pass.first()) {
      const std::string verify_err =
          fill_indices_and_verify(mapping.as<int>(), n, nullptr);
      if (!verify_err.empty()) failure = "VMM " + verify_err;
    }
    if (!failure.empty()) break;
  }
  (void)hipMemRelease(handle);

  if (!failure.empty()) return fail(failure);
  return pass_with_latency(loop, "vmm.map_unmap");
}

// After hipMemSetAccess(ReadWrite), hipMemGetAccess must report read-write access.
ACCEPT_TEST(Vmm_SetGetAccess, "hip/memmgr", Topology::Single, Kind::Functional,
            30) {
  (void)ctx;
  if (!device_has_vmm(gpu()))
    return skip("device does not advertise VMM support");  // TODO: policy gate

  const size_t bytes = sizeof(int) * 256;
  hipMemGenericAllocationHandle_t handle{};
  size_t size = 0;
  std::string create_err = vmm_create_pinned(bytes, gpu(), &handle, &size);
  if (!create_err.empty()) return fail(create_err);

  VmmMapping mapping;
  std::string map_err = mapping.map(handle, size, gpu(), /*own_handle=*/true);
  if (!map_err.empty()) return fail(map_err);

  hipMemLocation loc{};
  loc.type = hipMemLocationTypeDevice;
  loc.id = gpu();
  unsigned long long flags = 0;
  // Region: the access-query latency, once (perf off) or warmup+N (perf on).
  PerfLoop loop;
  hipError_t get_err = hipSuccess;
  for (auto pass : loop) {
    get_err = hipMemGetAccess(&flags, &loc, mapping.get());
    if (get_err != hipSuccess) break;
  }
  if (get_err != hipSuccess)
    return fail(std::string("hipMemGetAccess: ") + hipGetErrorString(get_err));
  if (flags != static_cast<unsigned long long>(hipMemAccessFlagsProtReadWrite))
    return fail("access flags not read-write after SetAccess: got " +
                std::to_string(flags) + ", expected " +
                std::to_string(
                    static_cast<int>(hipMemAccessFlagsProtReadWrite)));
  return pass_with_latency(loop, "vmm.get_access");
}

// ROCM-2550: GPUDirectRDMA+VMM attribute must equal (VMM supported && DMA-Buf
// supported). This is a deliberately strict regression invariant — if a future
// runtime gates the attribute on extra conditions (topology/BAR) it will need
// revisiting (likely reclassify toward REGRESSION).
ACCEPT_TEST(Vmm_GPUDirectRdmaVmm_Attribute, "hip/memmgr", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!device_has_vmm(gpu()))
    return skip("device does not advertise VMM support");  // TODO: policy gate

  // The attribute enum is always declared by the headers; whether the RUNTIME
  // can answer the query is detected here. A runtime that does not expose the
  // GPUDirectRDMA-with-VMM attribute (older HIP) is a clean skip, not a failure.
  // Region: the attribute query itself, once (perf off) or warmup+N (perf on).
  PerfLoop loop;
  int rdma_vmm = 0;
  bool queryable = true;
  for (auto pass : loop) {
    queryable = query_device_attr(
        gpu(), hipDeviceAttributeGPUDirectRDMAWithHipVMMSupported, &rdma_vmm);
    if (!queryable) break;
  }
  if (!queryable) {
    (void)hipGetLastError();
    return skip(
        "hipDeviceAttributeGPUDirectRDMAWithHipVMMSupported not queryable on "
        "this runtime");
  }

  int vmm = 0, dmabuf = 0;
  if (!query_device_attr(gpu(),
                         hipDeviceAttributeVirtualMemoryManagementSupported, &vmm))
    return fail("hipDeviceAttributeVirtualMemoryManagementSupported query failed");

  if (!query_device_attr(gpu(), hipDeviceAttributeDmaBufSupported, &dmabuf))
    return fail("hipDeviceAttributeDmaBufSupported query failed");

  const int expected = (vmm != 0 && dmabuf != 0) ? 1 : 0;
  if (rdma_vmm != expected)
    return fail("ROCM-2550: GPUDirectRDMAWithHipVMMSupported=" +
                std::to_string(rdma_vmm) + " but expected " +
                std::to_string(expected) + " (VMM=" + std::to_string(vmm) +
                " DmaBuf=" + std::to_string(dmabuf) + ")");
  return pass_with_latency(loop, "vmm.rdma_attr_query");
}

// ROCM-2550: map 16 VMM allocations with gpuDirectRDMACapable=1 and kernel-verify the last.
ACCEPT_TEST(Vmm_GpuDirectRDMACapable_MultiMap, "hip/memmgr", Topology::Single,
            Kind::Functional, 60) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  if (!device_has_vmm(gpu()))
    return skip("device does not advertise VMM support");  // TODO: policy gate
  // Region: 16-way create+map cycle, once (perf off) or warmup+N (perf on).
  PerfLoop loop;
  std::string error;
  for (auto pass : loop) {
    error = vmm_rdma_multimap(/*rdma_flag=*/1, /*count=*/16);
    if (!error.empty()) break;
  }
  if (!error.empty()) return fail(error);
  return pass_with_latency(loop, "vmm.rdma_multimap");
}

// ROCM-2550: map 16 VMM allocations without the RDMA flag and kernel-verify the last.
ACCEPT_TEST(Vmm_RDMA_MultiMap_NoFlag, "hip/memmgr", Topology::Single,
            Kind::Functional, 60) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  if (!device_has_vmm(gpu()))
    return skip("device does not advertise VMM support");  // TODO: policy gate
  // Region: 16-way create+map cycle, once (perf off) or warmup+N (perf on).
  PerfLoop loop;
  std::string error;
  for (auto pass : loop) {
    error = vmm_rdma_multimap(/*rdma_flag=*/0, /*count=*/16);
    if (!error.empty()) break;
  }
  if (!error.empty()) return fail(error);
  return pass_with_latency(loop, "vmm.multimap_noflag");
}

// Zero-size hipMemCreate must fail cleanly; an odd-size reserve must return a
// deterministic result (success or error) without crashing.
ACCEPT_TEST(Vmm_Negative_ZeroSize_Unaligned, "hip/memmgr", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!device_has_vmm(gpu()))
    return skip("device does not advertise VMM support");  // TODO: policy gate

  hipMemAllocationProp prop = make_vmm_prop(gpu(), /*exportable=*/false);
  size_t gran = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop,
                                           hipMemAllocationGranularityMinimum));

  // (1) zero-size create must fail cleanly.
  hipMemGenericAllocationHandle_t handle{};
  hipError_t create_err = hipMemCreate(&handle, 0, &prop, 0);
  if (create_err == hipSuccess) {
    (void)hipMemRelease(handle);
    return fail("zero-size hipMemCreate unexpectedly succeeded");
  }

  // (2) odd-size reserve. alignment=1 is a valid power-of-two, so a runtime may
  // legitimately succeed (rounding the size up) or reject it — we only require a
  // deterministic, non-crashing outcome. If it succeeds we free the reservation.
  void* ptr = nullptr;
  hipError_t reserve_err =
      hipMemAddressReserve(&ptr, gran + 1, /*alignment=*/1, nullptr, 0);
  if (reserve_err == hipSuccess) (void)hipMemAddressFree(ptr, gran + 1);
  // Clear any sticky error so later tests are not affected.
  (void)hipGetLastError();
  return pass();
}

// Export a VMM allocation to a POSIX fd in-process and re-import it with kernel verify.
ACCEPT_TEST(Vmm_PosixFd_Export_Import, "hip/memmgr", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  if (!device_has_vmm(gpu()))
    return skip("device does not advertise VMM support");  // TODO: policy gate

  const int n = kIndexCount;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);
  hipMemGenericAllocationHandle_t handle{};
  size_t size = 0;
  std::string create_err =
      vmm_create_pinned(bytes, gpu(), &handle, &size, /*exportable=*/true);
  if (!create_err.empty()) return fail(create_err);

  // The source keeps `handle` alive (own_handle=false) so the compat sub-test can
  // re-export it; the handle is released explicitly at the end of the test.
  VmmMapping source;
  std::string src_map_err = source.map(handle, size, gpu(), /*own_handle=*/false);
  if (!src_map_err.empty()) {
    (void)hipMemRelease(handle);
    return fail("source " + src_map_err);
  }

  // Write indices through the source mapping.
  const std::string write_err =
      fill_indices_and_verify(source.as<int>(), n, nullptr);
  if (!write_err.empty()) {
    (void)hipMemRelease(handle);
    return fail("source write: " + write_err);
  }

  // POSIX-fd osHandle convention is runtime-version dependent; see
  // posix_fd_import_style(). Round-trip through the standard entry point, then
  // through the optional compat shim (no-op when the runtime lacks it).
  const PosixFdImportStyle import_style = posix_fd_import_style();
  if (import_style == PosixFdImportStyle::Unknown) {
    (void)hipMemRelease(handle);
    return fail("could not probe POSIX-fd import ABI");
  }

  // Region: export -> import -> map+verify, once (perf off) or warmup+N (perf on).
  PerfLoop loop;
  std::string roundtrip_err;
  for (auto pass : loop) {
    roundtrip_err = export_import_roundtrip(handle, size, n, import_style);
    if (!roundtrip_err.empty()) break;
  }
  if (!roundtrip_err.empty()) {
    (void)hipMemRelease(handle);
    return fail(roundtrip_err);
  }
  std::string compat_err = compat_roundtrip(handle, size, n, import_style);
  if (!compat_err.empty()) {
    (void)hipMemRelease(handle);
    return fail(compat_err);
  }
  (void)hipMemRelease(handle);
  return pass_with_latency(loop, "vmm.posixfd.export_import");
}

// ROCM-25387: CLR's hipMemMap enqueues a Marker on the null stream and waits on
// it, so a kernel already in flight there blocks the map. We reproduce single-
// process: spin a kernel on the null stream, then hipMemMap a separate VMM alloc
// on a watchdog thread; a timeout is the deadlock. The kernel is always released
// and joined afterward so the spinner never leaks.
ACCEPT_TEST(Vmm_Map_NoDeadlock_KernelInFlight, "hip/memmgr", Topology::Single,
            Kind::Functional, 60) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  if (!device_has_vmm(gpu()))
    return skip("device does not advertise VMM support");  // TODO: policy gate

  HIP_CHECK(hipSetDevice(gpu()));

  // Fine-grained host flag the kernel polls and the host releases. Fine-grain so
  // the device sees the host store without an explicit copy.
  HostBuffer done_flag;
  HIP_CHECK(done_flag.allocate(sizeof(int),
                               hipHostMallocMapped | hipHostMallocCoherent));
  int* done = done_flag.as<int>();
  *done = 0;

  // A separate VMM allocation to map WHILE the kernel is in flight.
  hipMemAllocationProp prop = make_vmm_prop(gpu(), /*exportable=*/false);
  size_t gran = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop,
                                           hipMemAllocationGranularityMinimum));
  const size_t size = round_up(sizeof(int) * 4096, gran);
  hipMemGenericAllocationHandle_t handle{};
  hipError_t create_err = hipMemCreate(&handle, size, &prop, 0);
  if (create_err != hipSuccess)
    return fail(std::string("hipMemCreate: ") + hipGetErrorString(create_err));

  std::string launch_err = launch_spinner(done);
  if (!launch_err.empty()) {
    *done = 1;  // nothing launched, but be safe
    (void)hipMemRelease(handle);
    return fail(launch_err);
  }

  const WatchdogMapResult map = map_on_worker_with_watchdog(handle, size, done);
  teardown_vmm(map.va, size, handle);

  if (map.timed_out)
    return fail("hipMemMap did not return within 20s with a kernel in flight "
                "on the null stream (ROCM-25387 deadlock signature)");
  if (map.map_status != hipSuccess)
    return fail(std::string("hipMemMap path: ") +
                hipGetErrorString(map.map_status));
  return pass();
}

// Export/import a VMM allocation through a fabric handle and kernel-verify the mapping.
ACCEPT_TEST(Vmm_Fabric_Export_Import, "hip/memmgr", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  if (!device_has_vmm(gpu()))
    return skip("device does not advertise VMM support");

  // The fabric handle type/enum is always declared by the headers; fabric
  // support is driver/topology dependent and detected at run time below, so an
  // unsupported runtime is a clean skip rather than a compile-time exclusion.
  hipMemAllocationProp prop = make_vmm_prop(gpu(), /*exportable=*/false);
  prop.requestedHandleType = hipMemHandleTypeFabric;
  size_t gran = 0;
  if (hipMemGetAllocationGranularity(&gran, &prop,
                                     hipMemAllocationGranularityMinimum) !=
      hipSuccess)
    return skip("fabric handle type not supported by driver/topology");
  const size_t size = round_up(sizeof(int) * 1024, gran);

  hipMemGenericAllocationHandle_t handle{};
  if (hipMemCreate(&handle, size, &prop, 0) != hipSuccess)
    return skip("fabric allocation not supported by driver/topology");

  // Region: export -> import -> map+verify, once (perf off) or warmup+N (perf on).
  PerfLoop loop;
  std::string verify_err;
  for (auto pass : loop) {
    hipMemFabricHandle_t fabric{};
    hipError_t export_err = hipMemExportToShareableHandle(
        &fabric, handle, hipMemHandleTypeFabric, 0);
    if (export_err != hipSuccess) {
      verify_err = std::string("hipMemExportToShareableHandle(fabric): ") +
                   hipGetErrorString(export_err);
      break;
    }
    hipMemGenericAllocationHandle_t imported{};
    hipError_t import_err = hipMemImportFromShareableHandle(
        &imported, &fabric, hipMemHandleTypeFabric);
    if (import_err != hipSuccess) {
      verify_err = std::string("hipMemImportFromShareableHandle(fabric): ") +
                   hipGetErrorString(import_err);
      break;
    }
    VmmMapping dst;
    std::string map_err = dst.map(imported, size, gpu(), /*own_handle=*/true);
    verify_err = map_err.empty()
                     ? fill_indices_and_verify(dst.as<int>(), kIndexCount,
                                               nullptr)
                     : map_err;
    if (!verify_err.empty()) {
      verify_err = "fabric " + verify_err;
      break;
    }
  }
  (void)hipMemRelease(handle);
  if (!verify_err.empty()) return fail(verify_err);
  return pass_with_latency(loop, "vmm.fabric.export_import");
}

// Export a VMM allocation to a POSIX fd, re-import, D2D-copy a pattern in and
// verify. Perf mode auto-captures export/import/copy latency via HIP_CHECK.
ACCEPT_TEST(Vmm_PosixFd_Import_D2D_Copy, "hip/memmgr", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!device_has_vmm(gpu()))
    return skip("device does not advertise VMM support");  // TODO: policy gate
  HIP_CHECK(hipSetDevice(gpu()));

  const int n = kIndexCount;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);
  hipMemGenericAllocationHandle_t handle{};
  size_t size = 0;
  const std::string create_err =
      vmm_create_pinned(bytes, gpu(), &handle, &size, /*exportable=*/true);
  if (!create_err.empty()) return fail(create_err);

  const PosixFdImportStyle import_style = posix_fd_import_style();
  if (import_style == PosixFdImportStyle::Unknown) {
    (void)hipMemRelease(handle);
    return fail("could not probe POSIX-fd import ABI");
  }

  int fd = -1;
  HIP_CHECK(hipMemExportToShareableHandle(
      &fd, handle, hipMemHandleTypePosixFileDescriptor, 0));

  hipMemGenericAllocationHandle_t imported{};
  HIP_CHECK(hipMemImportFromShareableHandle(
      &imported, posix_fd_os_handle(&fd, import_style),
      hipMemHandleTypePosixFileDescriptor));

  (void)close(fd);  // fd not needed past import (mirrors RCCL)

  const TestResult r = import_d2d_copy_verify(imported, size, n);
  (void)hipMemRelease(handle);
  return r;
}

// Fabric-handle variant of the D2D-copy test; skips when fabric is unsupported.
ACCEPT_TEST(Vmm_Fabric_Import_D2D_Copy, "hip/memmgr", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!device_has_vmm(gpu()))
    return skip("device does not advertise VMM support");  // TODO: policy gate
  HIP_CHECK(hipSetDevice(gpu()));

  const int n = kIndexCount;
  hipMemAllocationProp prop = make_vmm_prop(gpu(), /*exportable=*/false);
  prop.requestedHandleType = hipMemHandleTypeFabric;
  size_t gran = 0;
  if (hipMemGetAllocationGranularity(&gran, &prop,
                                     hipMemAllocationGranularityMinimum) !=
      hipSuccess)
    return skip("fabric handle type not supported by driver/topology");
  const size_t size = round_up(sizeof(int) * static_cast<size_t>(n), gran);

  hipMemGenericAllocationHandle_t handle{};
  if (hipMemCreate(&handle, size, &prop, 0) != hipSuccess)
    return skip("fabric allocation not supported by driver/topology");

  hipMemFabricHandle_t fabric{};
  HIP_CHECK(hipMemExportToShareableHandle(&fabric, handle,
                                          hipMemHandleTypeFabric, 0));
  hipMemGenericAllocationHandle_t imported{};
  HIP_CHECK(hipMemImportFromShareableHandle(&imported, &fabric,
                                            hipMemHandleTypeFabric));

  const TestResult r = import_d2d_copy_verify(imported, size, n);
  (void)hipMemRelease(handle);
  return r;
}

// Create/map an uncached VMM allocation, kernel-write, and verify host readback.
ACCEPT_TEST(Vmm_Uncached_Type, "hip/memmgr", Topology::Single, Kind::Functional,
            30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  if (!device_has_vmm(gpu()))
    return skip("device does not advertise VMM support");

  // hipMemAllocationTypeUncached is always a declared enum value; whether the
  // RUNTIME honours it is detected here by attempting the allocation. A runtime
  // that rejects the uncached type (hipErrorNotSupported) is a clean skip.
  const int n = kIndexCount;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);
  hipMemAllocationProp prop = make_vmm_prop(gpu(), /*exportable=*/false);
  prop.type = hipMemAllocationTypeUncached;
  size_t gran = 0;
  hipError_t gran_err = hipMemGetAllocationGranularity(
      &gran, &prop, hipMemAllocationGranularityMinimum);
  if (gran_err == hipErrorNotSupported) {
    (void)hipGetLastError();
    return skip("uncached allocation type not supported by this runtime");
  }
  if (gran_err != hipSuccess)
    return fail(std::string("hipMemGetAllocationGranularity(uncached): ") +
                hipGetErrorString(gran_err));
  const size_t size = round_up(bytes, gran);

  // Create and map an uncached allocation per pass. Verify only the first pass;
  // teardown stays outside the timed region.
  PerfLoop loop;
  std::vector<VmmMapping> mappings;
  mappings.reserve(static_cast<size_t>(loop.run_count()));
  std::string failure;
  bool unsupported = false;
  for (auto pass : loop) {
    hipMemGenericAllocationHandle_t handle{};
    hipError_t create_err = hipMemCreate(&handle, size, &prop, 0);
    if (create_err != hipSuccess) {
      if (create_err == hipErrorNotSupported) {
        (void)hipGetLastError();
        unsupported = true;
      } else {
        failure = std::string("hipMemCreate(uncached): ") +
                  hipGetErrorString(create_err);
      }
      break;
    }

    mappings.emplace_back();
    failure = mappings.back().map(handle, size, gpu(), /*own_handle=*/true);
    if (!failure.empty()) {
      failure = "uncached " + failure;
    } else if (pass.first()) {
      const std::string verify_err =
          fill_indices_and_verify(mappings.back().as<int>(), n, nullptr);
      if (!verify_err.empty()) failure = "uncached VMM " + verify_err;
    }
    if (!failure.empty()) break;
  }
  mappings.clear();  // teardown, outside the timed region

  if (unsupported)
    return skip("uncached allocation type not supported by this runtime");
  if (!failure.empty()) return fail(failure);
  return pass_with_latency(loop, "vmm.uncached.create_map");
}

// ROCM-20991, single process: the device-wide reading after unmap+release, with
// the VA still reserved, must come back to its first-cycle value. On ROCm
// 7.0.2.2/gfx950 the backing returns only at hipMemAddressFree, so no export is
// needed to see the defect. Force-isolated: it keeps its VA to the end.
ACCEPT_TEST(Vmm_Reclaim_Cycles_InProcess, "hip/memmgr", Topology::Single,
            Kind::Functional, 60) {
  (void)ctx;
  if (!device_has_vmm(gpu()))
    return skip("device does not advertise VMM support");  // TODO: policy gate
  HIP_CHECK(hipSetDevice(gpu()));

  hipMemAllocationProp prop = make_vmm_prop(gpu(), /*exportable=*/false);
  size_t gran = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop,
                                           hipMemAllocationGranularityMinimum));
  const size_t size = round_up(kReclaimBytes, gran);

  std::vector<size_t> used_per_cycle;
  const std::string error =
      run_reclaim_cycles(/*sock=*/-1, size, prop, &used_per_cycle);
  if (!error.empty()) return fail(error);

  const std::string verdict =
      reclaim_verdict(used_per_cycle, reclaim_tolerance(gran));
  if (!verdict.empty()) return fail(verdict);
  return pass();
}

// ROCM-20991 over the exported + imported path: the backing must come back once
// both sides release. Teardown order mirrors ncclCommMemSuspend (importer
// releases first, then the owner closes its fd and releases) with both VAs still
// reserved. HIP_CHECK is unusable past the fork: it returns without reaping.
// A leaking runtime keeps both sides' backing, ~3 GiB by the last cycle, so a
// small or busy GPU reports hipMemCreate failing instead of the verdict.
ACCEPT_TEST(Vmm_Reclaim_Cycles_CrossProcess, "hip/memmgr", Topology::Single,
            Kind::Functional, 60) {
  (void)ctx;
  int sockets[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0)
    return fail(std::string("socketpair: ") + std::strerror(errno));

  // fork before any HIP call: a child inheriting an initialized context cannot
  // perform memory operations, so even the capability gate waits.
  const pid_t pid = ::fork();
  if (pid < 0) {
    (void)::close(sockets[0]);
    (void)::close(sockets[1]);
    return fail(std::string("fork: ") + std::strerror(errno));
  }
  if (pid == 0) {
    (void)::close(sockets[0]);
    run_reclaim_importer_child(sockets[1]);
  }
  (void)::close(sockets[1]);
  reclaim_set_socket_timeouts(sockets[0]);

  // Closing the owner's end releases an importer blocked on us, so it exits
  // promptly instead of sitting out its receive timeout.
  auto finish = [&](TestResult result) {
    (void)::close(sockets[0]);
    const int status = posix_util::reap_child_bounded(pid, kReclaimReapGraceMs);
    if (result.status == Status::Fail)
      result.reason += " (importer " + describe_wait_status(status) + ")";
    return result;
  };

  if (!device_has_vmm(gpu()))
    return finish(skip("device does not advertise VMM support"));

  // hipMemGetInfo reports the CURRENT device, so the owner's device is explicit.
  const hipError_t set_device_err = hipSetDevice(gpu());
  if (set_device_err != hipSuccess)
    return finish(fail(std::string("hipSetDevice: ") +
                       hipGetErrorString(set_device_err)));

  hipMemAllocationProp prop = make_vmm_prop(gpu(), /*exportable=*/true);
  size_t gran = 0;
  const hipError_t gran_err = hipMemGetAllocationGranularity(
      &gran, &prop, hipMemAllocationGranularityMinimum);
  if (gran_err != hipSuccess)
    return finish(fail(std::string("hipMemGetAllocationGranularity: ") +
                       hipGetErrorString(gran_err)));
  const size_t size = round_up(kReclaimBytes, gran);

  if (!sockcompat::send_all(sockets[0], &size, sizeof(size)))
    return finish(fail("failed to send the allocation size to the importer"));

  std::vector<size_t> used_per_cycle;
  const std::string error =
      run_reclaim_cycles(sockets[0], size, prop, &used_per_cycle);
  if (!error.empty()) return finish(fail(error));

  const std::string verdict =
      reclaim_verdict(used_per_cycle, reclaim_tolerance(gran));
  if (!verdict.empty()) return finish(fail(verdict));
  return finish(pass());
}

// ===========================================================================
// Stream-ordered / pooled / managed / fine-grained allocators
// ===========================================================================

// Stream-ordered hipMallocAsync/hipFreeAsync with a kernel write and integrity check.
ACCEPT_TEST(MallocAsync_FreeAsync, "hip/memmgr", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  if (!device_has_mempool(gpu()))
    return skip("device does not advertise memory-pool support");  // TODO: policy

  const int n = kIndexCount;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);
  Stream stream;
  HIP_CHECK(stream.create_with_flags(hipStreamNonBlocking));

  void* ptr = nullptr;
  hipError_t alloc_err = hipMallocAsync(&ptr, bytes, stream.get());
  if (alloc_err != hipSuccess)
    return fail(std::string("hipMallocAsync: ") + hipGetErrorString(alloc_err));

  const std::string err =
      fill_indices_and_verify(static_cast<int*>(ptr), n, stream.get());

  hipError_t free_err = hipFreeAsync(ptr, stream.get());
  (void)hipStreamSynchronize(stream.get());

  if (!err.empty()) return fail("async-alloc " + err);
  if (free_err != hipSuccess)
    return fail(std::string("hipFreeAsync: ") + hipGetErrorString(free_err));
  return pass();
}

// Create a memory pool, alloc/free from it on a stream, then destroy the pool.
ACCEPT_TEST(MemPool_Alloc, "hip/memmgr", Topology::Single, Kind::Functional,
            30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  if (!device_has_mempool(gpu()))
    return skip("device does not advertise memory-pool support");  // TODO: policy

  const int n = kIndexCount;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);

  hipMemPoolProps props = make_device_pinned_pool_props(gpu());

  // Create the stream before the pool so a stream-creation failure cannot leak
  // an already-created pool via HIP_CHECK's early return.
  Stream stream;
  HIP_CHECK(stream.create_with_flags(hipStreamNonBlocking));

  hipMemPool_t pool = nullptr;
  hipError_t pool_err = hipMemPoolCreate(&pool, &props);
  if (pool_err != hipSuccess)
    return fail(std::string("hipMemPoolCreate: ") + hipGetErrorString(pool_err));

  void* ptr = nullptr;
  hipError_t alloc_err = hipMallocFromPoolAsync(&ptr, bytes, pool, stream.get());
  std::string err;
  if (alloc_err == hipSuccess) {
    err = fill_indices_and_verify(static_cast<int*>(ptr), n, stream.get());
    (void)hipFreeAsync(ptr, stream.get());
    (void)hipStreamSynchronize(stream.get());
  }

  hipError_t destroy_err = hipMemPoolDestroy(pool);

  if (alloc_err != hipSuccess)
    return fail(std::string("hipMallocFromPoolAsync: ") +
                hipGetErrorString(alloc_err));
  if (!err.empty()) return fail("pool-alloc " + err);
  if (destroy_err != hipSuccess)
    return fail(std::string("hipMemPoolDestroy: ") +
                hipGetErrorString(destroy_err));
  return pass();
}

// Set pool reuse/threshold attributes, alloc from pool, use, read attribute back, destroy.
ACCEPT_TEST(MemPool_SetAttribute, "hip/memmgr", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  if (!device_has_mempool(gpu()))
    return skip("device does not advertise memory-pool support");  // TODO: policy

  const int n = kIndexCount;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);

  hipMemPoolProps props = make_device_pinned_pool_props(gpu());

  hipMemPool_t pool = nullptr;
  HIP_CHECK(hipMemPoolCreate(&pool, &props));

  // Centralized cleanup so each step can check-and-return immediately.
  hipStream_t stream = nullptr;
  void* ptr = nullptr;
  auto fail_clean = [&](const std::string& msg) {
    if (ptr) (void)hipFreeAsync(ptr, stream);
    if (stream) {
      (void)hipStreamSynchronize(stream);
      (void)hipStreamDestroy(stream);
    }
    if (pool) (void)hipMemPoolDestroy(pool);
    return fail(msg);
  };

  int reuse = 1;
  if (hipMemPoolSetAttribute(pool, hipMemPoolReuseFollowEventDependencies,
                             &reuse) != hipSuccess)
    return fail_clean("hipMemPoolSetAttribute(ReuseFollowEventDependencies)");

  uint64_t threshold = 64ull << 20;  // 64 MiB
  if (hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold) !=
      hipSuccess)
    return fail_clean("hipMemPoolSetAttribute(ReleaseThreshold)");

  // Read the threshold back to confirm it was accepted.
  uint64_t got = 0;
  if (hipMemPoolGetAttribute(pool, hipMemPoolAttrReleaseThreshold, &got) !=
      hipSuccess)
    return fail_clean("hipMemPoolGetAttribute(ReleaseThreshold)");
  if (got != threshold)
    return fail_clean("release threshold not honored: got " +
                      std::to_string(got) + ", set " +
                      std::to_string(threshold));

  if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess)
    return fail_clean("hipStreamCreateWithFlags failed");

  if (hipMallocFromPoolAsync(&ptr, bytes, pool, stream) != hipSuccess)
    return fail_clean("hipMallocFromPoolAsync failed");

  {
    const std::string err =
        fill_indices_and_verify(static_cast<int*>(ptr), n, stream);
    if (!err.empty()) return fail_clean("pool-attr " + err);
  }

  // Success: free the buffer, trim retained pool memory, then re-alloc to prove
  // the pool is still usable after hipMemPoolTrimTo.
  (void)hipFreeAsync(ptr, stream);
  ptr = nullptr;
  if (hipStreamSynchronize(stream) != hipSuccess)
    return fail_clean("hipStreamSynchronize before trim failed");
  if (hipMemPoolTrimTo(pool, 0) != hipSuccess)
    return fail_clean("hipMemPoolTrimTo failed");

  if (hipMallocFromPoolAsync(&ptr, bytes, pool, stream) != hipSuccess)
    return fail_clean("hipMallocFromPoolAsync after trim failed");
  {
    const std::string err =
        fill_indices_and_verify(static_cast<int*>(ptr), n, stream);
    if (!err.empty()) return fail_clean("pool buffer after trim " + err);
  }

  (void)hipFreeAsync(ptr, stream);
  ptr = nullptr;
  (void)hipStreamSynchronize(stream);
  (void)hipStreamDestroy(stream);
  stream = nullptr;
  hipError_t destroy_err = hipMemPoolDestroy(pool);
  pool = nullptr;
  if (destroy_err != hipSuccess)
    return fail(std::string("hipMemPoolDestroy: ") +
                hipGetErrorString(destroy_err));
  return pass();
}

// Managed memory: host write, kernel update, sync, host verify, free.
ACCEPT_TEST(MallocManaged_Access, "hip/memmgr", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  if (!device_has_managed(gpu()))
    return skip("device does not advertise managed-memory support");  // TODO: policy

  const int n = kIndexCount;
  const int addend = 7;  // constant the kernel adds to every element
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);

  int* ptr = nullptr;
  HIP_CHECK(hipMallocManaged(reinterpret_cast<void**>(&ptr), bytes,
                             hipMemAttachGlobal));
  for (int i = 0; i < n; ++i) ptr[i] = i;  // host write

  AddArgs args{ptr, n, addend};  // device read-modify-write
  std::string launch_err;
  hipError_t launch_status =
      hip_kernel_launch("k_add_constant", grid_for(n), dim3(kBlockSize), 0, 0,
                        &args, sizeof(args), &launch_err);
  if (launch_status != hipSuccess) {
    (void)hipFree(ptr);
    return fail(launch_err);
  }
  HIP_CHECK(hipDeviceSynchronize());

  int bad_index = -1, bad_value = -1;
  for (int i = 0; i < n; ++i) {
    if (ptr[i] != i + addend) {
      bad_index = i;
      bad_value = ptr[i];
      break;
    }
  }
  (void)hipFree(ptr);
  if (bad_index >= 0)
    return fail("managed buffer wrong at index " + std::to_string(bad_index) +
                ": got " + std::to_string(bad_value) + ", expected " +
                std::to_string(bad_index + addend));
  return pass();
}

// Fine-grained device allocation: kernel write with host-visible coherent read, no explicit copy.
ACCEPT_TEST(ExtMalloc_FineGrain, "hip/memmgr", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  const int n = 256;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);

  int* ptr = nullptr;
  hipError_t alloc_err = hipExtMallocWithFlags(reinterpret_cast<void**>(&ptr),
                                               bytes, hipDeviceMallocFinegrained);
  if (alloc_err != hipSuccess)
    return skip(std::string("fine-grain alloc unsupported: ") +
                hipGetErrorString(alloc_err));  // TODO: policy capability gate

  FillArgs args{ptr, n};
  std::string launch_err;
  hipError_t launch_status =
      hip_kernel_launch("k_fill_indices", grid_for(n), dim3(kBlockSize), 0, 0,
                        &args, sizeof(args), &launch_err);
  if (launch_status != hipSuccess) {
    (void)hipFree(ptr);
    return fail(launch_err);
  }
  HIP_CHECK(hipDeviceSynchronize());

  // Coherent: read directly on the host, no hipMemcpy.
  const std::string err = verify_host_indices(ptr, n);
  (void)hipFree(ptr);
  if (!err.empty()) return fail("fine-grain buffer " + err);
  return pass();
}

}  // namespace acceptance
#endif  // ACCEPT_HAVE_HIP
