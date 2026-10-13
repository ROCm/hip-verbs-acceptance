/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
#include "hip/utils/HipModule.hpp"

#ifdef ACCEPT_HAVE_HIP

#include <cstdio>
#include <map>
#include <mutex>
#include <vector>

namespace acceptance {

namespace {

// The shared test-kernel bundle, compiled at run time by hiprtc for the current
// device. extern "C" names are resolved with hipModuleGetFunction. These mirror
// the device kernels formerly launched via <<<>>> from the HIP test sources;
// hiprtc supplies the device built-ins (blockIdx/threadIdx/atomicAdd), so no
// includes are required here.
const char* kKernelSource = R"HIPSRC(
extern "C" __global__ void k_write_flag(int* flag, int val) { *flag = val; }

extern "C" __global__ void k_read_flag(const int* flag, int* out) { *out = *flag; }

extern "C" __global__ void k_noop() {}

extern "C" __global__ void k_fill_indices(int* buf, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] = i;
}

extern "C" __global__ void k_write_mapped_pattern(int* buf, int n, int value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] = value;
}

extern "C" __global__ void k_add_constant(int* buf, int n, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] += k;
}

extern "C" __global__ void k_count_mismatch_indices(const int* buf, int n, int* mism) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && buf[i] != i) atomicAdd(mism, 1);
}

// Writes 64*i into out[i] through a per-thread stack array so the kernel
// actually exercises the device stack-size limit raised before launch.
extern "C" __global__ void k_stack_heavy(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  volatile int local[64];
  for (int k = 0; k < 64; ++k) local[k] = i + k;
  int s = 0;
  for (int k = 0; k < 64; ++k) s += local[k];
  out[i] = s - (64 * 63) / 2;  // sum(i+k) - sum(k) = 64*i
}

// Trivial kernel whose hipFunction_t feeds the module-occupancy calculator.
extern "C" __global__ void k_occupancy_probe(int* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = i;
}

// A module __device__ global + a kernel that reads it, for the runtime
// module-loading tests (hipModuleGetGlobal / hipModuleGetFunction from a .co).
extern "C" __device__ int g_test_global = 0;

extern "C" __global__ void k_touch_global(int* out) { *out = g_test_global; }

// Spins on the NULL stream polling a fine-grained host flag until the host
// stores done=1 (ROCM-25387 deadlock repro). s_sleep backs off the fabric.
extern "C" __global__ void k_spin_on_host_flag(volatile int* done) {
  while (__atomic_load_n(done, __ATOMIC_ACQUIRE) == 0) {
    __builtin_amdgcn_s_sleep(1);
  }
}

// Reads a 64-bit signal through the GPU shader path (path A: GPU MC/L2) into a
// sink, warming the L2 cache line with the current value. Used by the GPU-NIC
// PCIe coherence test (put_first / cross_nic variants) to seed a stale line
// before the NIC writes the counter via PCIe (path B). The sink store prevents
// dead-code elimination of the load.
extern "C" __global__ void k_read_u64(const unsigned long long* signal,
                                      unsigned long long* sink) {
  *sink = __atomic_load_n(signal, __ATOMIC_RELAXED);
}
)HIPSRC";

std::mutex g_mu;
bool g_compiled = false;       // compile attempted (success or failure)
hipModule_t g_module = nullptr;
std::string g_compile_err;     // reason if g_module stayed null
std::map<std::string, hipFunction_t> g_fns;
std::vector<char> g_code;      // hiprtc code object bytes (for the file-load test)

// Compile kKernelSource with hiprtc for the current device and load it. Must be
// called with g_mu held. Sets g_module on success, else g_compile_err.
void compile_locked() {
  g_compiled = true;

  if (!hip_kernels_available()) {
    g_compile_err = "requires hiprtc (libhiprtc.so not loaded)";
    return;
  }

  // Target the arch of the current device so the produced code object matches
  // the GPU we will launch on -- discovered at run time, never baked in.
  int dev = 0;
  (void)hipGetDevice(&dev);
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
    g_compile_err = "hipGetDeviceProperties failed for current device";
    return;
  }
  const std::string arch_opt =
      std::string("--gpu-architecture=") + prop.gcnArchName;

  hiprtcProgram prog = nullptr;
  hiprtcResult rc = hiprtcCreateProgram(&prog, kKernelSource,
                                        "acceptance_kernels.hip", 0, nullptr,
                                        nullptr);
  if (rc != HIPRTC_SUCCESS) {
    g_compile_err =
        std::string("hiprtcCreateProgram: ") + hiprtcGetErrorString(rc);
    return;
  }

  const char* opts[] = {arch_opt.c_str()};
  rc = hiprtcCompileProgram(prog, 1, opts);
  if (rc != HIPRTC_SUCCESS) {
    std::string log;
    size_t log_size = 0;
    if (hiprtcGetProgramLogSize(prog, &log_size) == HIPRTC_SUCCESS &&
        log_size > 1) {
      log.resize(log_size);
      (void)hiprtcGetProgramLog(prog, &log[0]);
    }
    g_compile_err = std::string("hiprtcCompileProgram (") + arch_opt +
                    "): " + hiprtcGetErrorString(rc) +
                    (log.empty() ? "" : (" -- " + log));
    (void)hiprtcDestroyProgram(&prog);
    return;
  }

  size_t code_size = 0;
  if (hiprtcGetCodeSize(prog, &code_size) != HIPRTC_SUCCESS || code_size == 0) {
    g_compile_err = "hiprtcGetCodeSize returned empty code object";
    (void)hiprtcDestroyProgram(&prog);
    return;
  }
  std::vector<char> code(code_size);
  if (hiprtcGetCode(prog, code.data()) != HIPRTC_SUCCESS) {
    g_compile_err = "hiprtcGetCode failed";
    (void)hiprtcDestroyProgram(&prog);
    return;
  }
  (void)hiprtcDestroyProgram(&prog);

  hipError_t load_err = hipModuleLoadData(&g_module, code.data());
  if (load_err != hipSuccess) {
    g_module = nullptr;
    g_compile_err =
        std::string("hipModuleLoadData: ") + hipGetErrorString(load_err);
    return;
  }
  g_code = std::move(code);  // keep for the hipModuleLoad(file) test
}

// Resolve `name` from the loaded module, compiling on first use. Returns the
// function or nullptr (with *err set) on any failure.
hipFunction_t resolve(const char* name, std::string* err) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!g_compiled) compile_locked();
  if (!g_module) {
    if (err) *err = g_compile_err;
    return nullptr;
  }
  auto it = g_fns.find(name);
  if (it != g_fns.end()) return it->second;

  hipFunction_t fn = nullptr;
  hipError_t func_err = hipModuleGetFunction(&fn, g_module, name);
  if (func_err != hipSuccess || fn == nullptr) {
    if (err)
      *err = std::string("hipModuleGetFunction(") + name +
             "): " + hipGetErrorString(func_err);
    return nullptr;
  }
  g_fns[name] = fn;
  return fn;
}

}  // namespace

bool hip_kernels_available() {
  return ::acceptance::hipw::init_result().hiprtc_lib_loaded;
}

hipFunction_t hip_kernel_function(const char* name, std::string* err) {
  return resolve(name, err);
}

hipError_t hip_kernel_launch(const char* name, dim3 grid, dim3 block,
                             unsigned shared_bytes, hipStream_t stream,
                             void* args, size_t args_size, std::string* err) {
  hipFunction_t fn = resolve(name, err);
  if (!fn) return hipErrorInvalidImage;

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                    HIP_LAUNCH_PARAM_END};
  // No-argument kernels pass an empty config buffer.
  void** cfg = (args == nullptr || args_size == 0) ? nullptr : config;

  hipError_t launch_err = hipModuleLaunchKernel(
      fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_bytes,
      stream, nullptr, cfg);
  if (launch_err != hipSuccess && err)
    *err = std::string("hipModuleLaunchKernel(") + name +
           "): " + hipGetErrorString(launch_err);
  return launch_err;
}

bool hip_kernels_write_code_object(const std::string& path, std::string* err) {
  std::lock_guard<std::mutex> lk(g_mu);
  if (!g_compiled) compile_locked();
  if (!g_module || g_code.empty()) {
    if (err) *err = g_compile_err.empty() ? "no code object" : g_compile_err;
    return false;
  }
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    if (err) *err = "cannot open " + path + " for write";
    return false;
  }
  const size_t w = std::fwrite(g_code.data(), 1, g_code.size(), f);
  std::fclose(f);
  if (w != g_code.size()) {
    if (err) *err = "short write to " + path;
    return false;
  }
  return true;
}

}  // namespace acceptance

#endif  // ACCEPT_HAVE_HIP
