/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
#pragma once
// HipTest — shared scaffolding for the HIP tests: RAII owners for HIP resources,
// the packed kernel-arg structs, the common constants, and the launch+verify
// routine, so cleanup is automatic and early returns stay one-liners. Resolves
// through the dynamic wrapper (HipApi.hpp). Gated on ACCEPT_HAVE_HIP.

#ifdef ACCEPT_HAVE_HIP

#include "common/Topology.hpp"
#include "hip/utils/HipApi.hpp"
#include "hip/utils/HipHelpers.hpp"
#include "hip/utils/HipModule.hpp"

#include <cstddef>
#include <string>
#include <utility>

namespace acceptance {

// Canonical element count / launch geometry / poison byte reused across tests.
inline constexpr int kIndexCount = 1024;
inline constexpr unsigned kBlockSize = 256;
inline constexpr unsigned char kPoisonByte = 0xFF;

// 1-D grid that covers `n` elements at `block` threads/block.
inline dim3 grid_for(int n, unsigned block = kBlockSize) {
  return dim3((static_cast<unsigned>(n) + block - 1) / block);
}

// Round `v` up to a multiple of `a` (returns `v` unchanged when `a` is 0).
inline size_t round_up(size_t v, size_t a) {
  return a == 0 ? v : ((v + a - 1) / a) * a;
}

// Packed kernel-arg structs (device ABI: ptr 8B, int 4B) for the hiprtc bundle.
struct FillArgs {  // k_fill_indices(int* buf, int n)
  int* buf;
  int n;
};
struct AddArgs {  // k_add_constant(int* buf, int n, int k)
  int* buf;
  int n;
  int k;
};
struct SpinArgs {  // k_spin_on_host_flag(volatile int* done)
  int* done;
};
struct MismArgs {  // k_count_mismatch_indices(const int* buf, int n, int* mism)
  const int* buf;
  int n;
  int* mism;
};

// RAII owners: each owns one handle and releases it on scope exit. Non-copyable.

// Device allocation (hipMalloc/hipFree). Records its device so the free is
// device-correct for peer buffers (current device saved/restored around it).
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), dev_(o.dev_) {
    o.ptr_ = nullptr;
  }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      reset();
      ptr_ = o.ptr_;
      dev_ = o.dev_;
      o.ptr_ = nullptr;
    }
    return *this;
  }

  hipError_t allocate(size_t bytes) {
    reset();
    int dev = 0;
    (void)hipGetDevice(&dev);
    void* p = nullptr;
    const hipError_t status = hipMalloc(&p, bytes);
    if (status == hipSuccess) {
      ptr_ = p;
      dev_ = dev;
    }
    return status;
  }
  void reset() {
    if (ptr_ == nullptr) return;
    int cur = 0;
    (void)hipGetDevice(&cur);
    if (cur != dev_) (void)hipSetDevice(dev_);
    (void)hipFree(ptr_);
    if (cur != dev_) (void)hipSetDevice(cur);
    ptr_ = nullptr;
  }
  void* get() const { return ptr_; }
  template <class T>
  T* as() const {
    return static_cast<T*>(ptr_);
  }
  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  void* ptr_ = nullptr;
  int dev_ = 0;
};

// Pinned host allocation (hipHostMalloc/hipHostFree).
class HostBuffer {
 public:
  HostBuffer() = default;
  ~HostBuffer() { reset(); }
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;

  hipError_t allocate(size_t bytes, unsigned flags) {
    reset();
    return hipHostMalloc(&ptr_, bytes, flags);
  }
  void reset() {
    if (ptr_ != nullptr) {
      (void)hipHostFree(ptr_);
      ptr_ = nullptr;
    }
  }
  void* get() const { return ptr_; }
  template <class T>
  T* as() const {
    return static_cast<T*>(ptr_);
  }
  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  void* ptr_ = nullptr;
};

// Stream (hipStreamCreate[WithFlags]/hipStreamDestroy).
class Stream {
 public:
  Stream() = default;
  ~Stream() { reset(); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  hipError_t create() { reset(); return hipStreamCreate(&s_); }
  hipError_t create_with_flags(unsigned flags) {
    reset();
    return hipStreamCreateWithFlags(&s_, flags);
  }
  void reset() {
    if (s_ != nullptr) {
      (void)hipStreamDestroy(s_);
      s_ = nullptr;
    }
  }
  hipStream_t get() const { return s_; }
  explicit operator bool() const { return s_ != nullptr; }

 private:
  hipStream_t s_ = nullptr;
};

// Event (hipEventCreate[WithFlags]/hipEventDestroy).
class Event {
 public:
  Event() = default;
  ~Event() { reset(); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  hipError_t create() { reset(); return hipEventCreate(&e_); }
  hipError_t create_with_flags(unsigned flags) {
    reset();
    return hipEventCreateWithFlags(&e_, flags);
  }
  void reset() {
    if (e_ != nullptr) {
      (void)hipEventDestroy(e_);
      e_ = nullptr;
    }
  }
  hipEvent_t get() const { return e_; }
  explicit operator bool() const { return e_ != nullptr; }

 private:
  hipEvent_t e_ = nullptr;
};

// Instantiated graph exec (hipGraphExecDestroy). The handle is filled by
// hipGraphInstantiate(exec.receive(), ...).
class GraphExec {
 public:
  GraphExec() = default;
  ~GraphExec() { reset(); }
  GraphExec(const GraphExec&) = delete;
  GraphExec& operator=(const GraphExec&) = delete;

  hipGraphExec_t* receive() {
    reset();
    return &exec_;
  }
  void reset() {
    if (exec_ != nullptr) {
      (void)hipGraphExecDestroy(exec_);
      exec_ = nullptr;
    }
  }
  hipGraphExec_t get() const { return exec_; }

 private:
  hipGraphExec_t exec_ = nullptr;
};

// Captured/authored graph (hipGraphDestroy). The handle is filled by
// hipStreamEndCapture(s, graph.receive()) or hipStreamGetCaptureInfo.
class Graph {
 public:
  Graph() = default;
  ~Graph() { reset(); }
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  hipGraph_t* receive() {
    reset();
    return &graph_;
  }
  // Take ownership of an already-obtained graph handle (e.g. the one returned by
  // hipStreamEndCapture into a raw local) so it is destroyed on scope exit.
  void adopt(hipGraph_t g) {
    reset();
    graph_ = g;
  }
  void reset() {
    if (graph_ != nullptr) {
      (void)hipGraphDestroy(graph_);
      graph_ = nullptr;
    }
  }
  hipGraph_t get() const { return graph_; }

 private:
  hipGraph_t graph_ = nullptr;
};

// Full VMM mapping lifecycle: reserve VA + map a handle + grant RW access, with
// the reverse teardown (unmap/free/release) on scope exit.
class VmmMapping {
 public:
  VmmMapping() = default;
  ~VmmMapping() { reset(); }
  VmmMapping(const VmmMapping&) = delete;
  VmmMapping& operator=(const VmmMapping&) = delete;
  VmmMapping(VmmMapping&& o) noexcept
      : ptr_(o.ptr_),
        size_(o.size_),
        handle_(o.handle_),
        own_handle_(o.own_handle_) {
    o.ptr_ = nullptr;
    o.handle_ = 0;
    o.own_handle_ = false;
  }
  VmmMapping& operator=(VmmMapping&& o) noexcept {
    if (this != &o) {
      reset();
      ptr_ = o.ptr_;
      size_ = o.size_;
      handle_ = o.handle_;
      own_handle_ = o.own_handle_;
      o.ptr_ = nullptr;
      o.handle_ = 0;
      o.own_handle_ = false;
    }
    return *this;
  }

  // Reserve+map+set-access for `handle`. own_handle=true releases the handle on
  // reset (false: caller keeps it alive, e.g. to re-export). Rolls back on
  // partial failure. Returns "" on success or a reason naming the failing HIP
  // call; on failure the handle is still owned per own_handle (released on reset).
  std::string map(hipMemGenericAllocationHandle_t handle, size_t size, int dev,
                  bool own_handle) {
    reset();
    handle_ = handle;
    size_ = size;
    own_handle_ = own_handle;
    hipError_t status = hipMemAddressReserve(&ptr_, size, 0, nullptr, 0);
    if (status != hipSuccess) {
      ptr_ = nullptr;
      return std::string("hipMemAddressReserve: ") + hipGetErrorString(status);
    }
    status = hipMemMap(ptr_, size, 0, handle, 0);
    if (status != hipSuccess) {
      (void)hipMemAddressFree(ptr_, size);
      ptr_ = nullptr;
      return std::string("hipMemMap: ") + hipGetErrorString(status);
    }
    hipMemAccessDesc desc{};
    desc.location.type = hipMemLocationTypeDevice;
    desc.location.id = dev;
    desc.flags = hipMemAccessFlagsProtReadWrite;
    status = hipMemSetAccess(ptr_, size, &desc, 1);
    if (status != hipSuccess) {
      (void)hipMemUnmap(ptr_, size);
      (void)hipMemAddressFree(ptr_, size);
      ptr_ = nullptr;
      return std::string("hipMemSetAccess: ") + hipGetErrorString(status);
    }
    return "";
  }
  // Hand the mapped VA to the caller, who takes over the teardown. For probes
  // that must outlive the mapping, e.g. keeping the VA reserved past release.
  // The caller then owes the reservation a hipMemAddressFree: the current
  // callers are force-isolated and let process exit reclaim it instead, so a
  // non-isolated caller would leak VA for the life of the worker.
  void* release() {
    void* ptr = ptr_;
    ptr_ = nullptr;
    size_ = 0;
    handle_ = 0;
    own_handle_ = false;
    return ptr;
  }
  void reset() {
    if (ptr_ != nullptr) {
      (void)hipMemUnmap(ptr_, size_);
      (void)hipMemAddressFree(ptr_, size_);
      ptr_ = nullptr;
    }
    if (own_handle_ && handle_ != 0) {
      (void)hipMemRelease(handle_);
      handle_ = 0;
    }
  }
  void* get() const { return ptr_; }
  size_t size() const { return size_; }
  template <class T>
  T* as() const {
    return static_cast<T*>(ptr_);
  }
  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  void* ptr_ = nullptr;
  size_t size_ = 0;
  hipMemGenericAllocationHandle_t handle_ = 0;
  bool own_handle_ = false;
};

// Create a pinned device VMM handle sized to `bytes` rounded up to the minimum
// allocation granularity. Returns "" (and sets *handle/*size) on success, or a
// reason naming the failing HIP call. exportable=true requests a POSIX-fd
// shareable handle.
inline std::string vmm_create_pinned(size_t bytes, int dev,
                                     hipMemGenericAllocationHandle_t* handle,
                                     size_t* size, bool exportable = false) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = dev;
  if (exportable)
    prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
  size_t gran = 0;
  hipError_t status = hipMemGetAllocationGranularity(
      &gran, &prop, hipMemAllocationGranularityMinimum);
  if (status != hipSuccess)
    return std::string("hipMemGetAllocationGranularity: ") +
           hipGetErrorString(status);
  if (gran == 0) gran = 4096;
  *size = round_up(bytes, gran);
  status = hipMemCreate(handle, *size, &prop, 0);
  if (status != hipSuccess)
    return std::string("hipMemCreate: ") + hipGetErrorString(status);
  return "";
}

// Enable src -> dst GPU peer access, treating an already-enabled pair as
// success. Sets the current device to src (callers re-select their device
// afterward). Returns "" on success or a reason naming the failing call and the
// src->dst pair. Clears the sticky last-error the runtime may leave set.
inline std::string enable_peer_access_idempotent(int src, int dst) {
  const hipError_t set_err = hipSetDevice(src);
  if (set_err != hipSuccess)
    return "hipSetDevice(" + std::to_string(src) + "): " +
           hipGetErrorString(set_err);
  const hipError_t enable_err = hipDeviceEnablePeerAccess(dst, 0);
  if (enable_err != hipSuccess &&
      enable_err != hipErrorPeerAccessAlreadyEnabled)
    return "hipDeviceEnablePeerAccess(" + std::to_string(src) + "->" +
           std::to_string(dst) + "): " + hipGetErrorString(enable_err);
  (void)hipGetLastError();
  return "";
}

// Verify a host-side copy holds buf[i] == i for i in [0, n). Returns "" or a
// reason. Mirrors the device-side hip_verify_indices for host readbacks.
inline std::string verify_host_indices(const int* host, int n) {
  for (int i = 0; i < n; ++i)
    if (host[i] != i)
      return "wrong at index " + std::to_string(i) + ": got " +
             std::to_string(host[i]);
  return "";
}

// Launch k_fill_indices over `n` elements on `stream`, sync, verify buf[i] == i.
// Returns "" on success or a reason. Collapses the repeated launch/sync/verify.
inline std::string fill_indices_and_verify(int* d_buf, int n,
                                           hipStream_t stream) {
  FillArgs args{d_buf, n};
  std::string launch_err;
  if (hip_kernel_launch("k_fill_indices", grid_for(n), dim3(kBlockSize), 0,
                        stream, &args, sizeof(args),
                        &launch_err) != hipSuccess)
    return launch_err.empty() ? "k_fill_indices launch failed" : launch_err;
  const hipError_t sync =
      stream ? hipStreamSynchronize(stream) : hipDeviceSynchronize();
  if (sync != hipSuccess)
    return std::string(stream ? "hipStreamSynchronize: " : "hipDeviceSynchronize: ") +
           hipGetErrorString(sync);
  int bad_index = -1, bad_value = -1;
  if (!hip_verify_indices(d_buf, n, &bad_index, &bad_value))
    return "buffer wrong at index " + std::to_string(bad_index) + ": got " +
           std::to_string(bad_value);
  return "";
}

}  // namespace acceptance

#endif  // ACCEPT_HAVE_HIP
