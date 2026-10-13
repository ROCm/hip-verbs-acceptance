/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#if defined(ACCEPT_HAVE_HIP) && defined(ACCEPT_HAVE_IBVERBS)

#include "verbs/GdrMem.hpp"

#include <cerrno>
#include <cstring>
#include <vector>

namespace acceptance {
namespace verbsx {

namespace {

std::string hip_err(const char* what, hipError_t status) {
  const char* s = hipGetErrorString(status);
  return std::string(what) + ": " + (s ? s : "hip error");
}

}  // namespace

const char* alloc_name(Alloc a) {
  switch (a) {
    case Alloc::DeviceMalloc:    return "hipMalloc";
    case Alloc::HostDefault:     return "hipHostMalloc(Default)";
    case Alloc::HostWC:          return "hipHostMalloc(WriteCombined)";
    case Alloc::HostCoherent:    return "hipHostMalloc(Coherent)";
    case Alloc::HostNonCoherent: return "hipHostMalloc(NonCoherent)";
    case Alloc::FineGrain:       return "hipExtMalloc(FineGrain)";
    case Alloc::Managed:         return "hipMallocManaged";
    default:                     return "unknown";
  }
}

const char* reg_name(Reg r) {
  switch (r) {
    case Reg::Std:          return "ibv_reg_mr";
    case Reg::DmabufV1:     return "dmabuf_v1";
    case Reg::DmabufV2Pcie: return "dmabuf_v2_pcie";
    default:                return "unknown";
  }
}

void* alloc(Alloc mode, size_t size, std::string* err) {
  void* ptr = nullptr;
  hipError_t status = hipSuccess;
  switch (mode) {
    case Alloc::DeviceMalloc:
      status = hipMalloc(&ptr, size); break;
    case Alloc::HostDefault:
      status = hipHostMalloc(&ptr, size, hipHostMallocDefault); break;
    case Alloc::HostWC:
      status = hipHostMalloc(&ptr, size, hipHostMallocWriteCombined); break;
    case Alloc::HostCoherent:
      status = hipHostMalloc(&ptr, size, hipHostMallocCoherent); break;
    case Alloc::HostNonCoherent:
      status = hipHostMalloc(&ptr, size, hipHostMallocNonCoherent); break;
    case Alloc::FineGrain:
      status = hipExtMallocWithFlags(&ptr, size, hipDeviceMallocFinegrained); break;
    case Alloc::Managed:
      status = hipMallocManaged(&ptr, size, hipMemAttachGlobal); break;
    default:
      if (err) *err = "invalid allocation mode";
      return nullptr;
  }
  if (status != hipSuccess) {
    if (err) *err = hip_err(alloc_name(mode), status);
    return nullptr;
  }
  return ptr;
}

void free_(Alloc mode, void* ptr) {
  if (ptr == nullptr) return;
  switch (mode) {
    case Alloc::HostDefault:
    case Alloc::HostWC:
    case Alloc::HostCoherent:
    case Alloc::HostNonCoherent:
      (void)hipHostFree(ptr); break;
    default:
      (void)hipFree(ptr); break;  // device / finegrain / managed
  }
}

Registration reg(struct ibv_pd* pd, void* ptr, size_t size, unsigned int access,
                 Reg mode, std::string* err) {
  Registration registration;
  if (mode == Reg::Std) {
    registration.mr = ibv_reg_mr(pd, ptr, size, static_cast<int>(access));
    if (registration.mr == nullptr && err)
      *err = ibvw::errno_reason("ibv_reg_mr failed");
    return registration;
  }

  // DMA-BUF paths: export then register the exported fd.
  int fd = -1;
  uint64_t offset = 0;
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;
  if (mode == Reg::DmabufV1) {
    hsa_status = hsa_amd_portable_export_dmabuf(ptr, size, &fd, &offset);
  } else {  // DmabufV2Pcie
    hsa_status = hsa_amd_portable_export_dmabuf_v2(
        ptr, size, &fd, &offset, HSA_AMD_DMABUF_MAPPING_TYPE_PCIE);
  }
  if (hsa_status != HSA_STATUS_SUCCESS) {
    if (err)
      *err = std::string(reg_name(mode)) + ": export_dmabuf failed (hsa_status=" +
             std::to_string(static_cast<int>(hsa_status)) + ")";
    return registration;
  }

  // iova = the buffer VA so a remote WRITE's remote_addr maps 1:1.
  registration.mr = ibv_reg_dmabuf_mr(pd, offset, size,
                                      reinterpret_cast<uint64_t>(ptr), fd,
                                      static_cast<int>(access));
  if (registration.mr == nullptr) {
    if (err) *err = ibvw::errno_reason("ibv_reg_dmabuf_mr failed");
    hsa_amd_portable_close_dmabuf(fd);
    return registration;
  }
  registration.dma_fd = fd;
  return registration;
}

void dereg(Registration& registration) {
  if (registration.mr) ibvw::safe_dereg_mr(registration.mr);
  if (registration.dma_fd >= 0) hsa_amd_portable_close_dmabuf(registration.dma_fd);
  registration = Registration{};
}

bool fill_pattern(void* ptr, size_t size, uint8_t seed, std::string* err) {
  std::vector<uint8_t> tmp(size);
  for (size_t i = 0; i < size; ++i)
    tmp[i] = static_cast<uint8_t>(seed + i);  // uint8_t cast wraps mod 256
  // hipMemcpyDefault: `ptr` may be device OR host-pinned; the runtime infers
  // the direction. The plain (non-async) copy is synchronous, so no explicit
  // hipDeviceSynchronize is needed.
  hipError_t status = hipMemcpy(ptr, tmp.data(), size, hipMemcpyDefault);
  if (status != hipSuccess) { if (err) *err = hip_err("hipMemcpy", status); return false; }
  return true;
}

bool verify_pattern(const void* ptr, size_t size, uint8_t seed, std::string* err) {
  std::vector<uint8_t> tmp(size);
  hipError_t status = hipMemcpy(tmp.data(), ptr, size, hipMemcpyDefault);
  if (status != hipSuccess) { if (err) *err = hip_err("hipMemcpy", status); return false; }
  for (size_t i = 0; i < size; ++i) {
    const uint8_t expected = static_cast<uint8_t>(seed + i);
    if (tmp[i] != expected) {
      if (err)
        *err = "data mismatch at byte " + std::to_string(i) + ": expected " +
               std::to_string(expected) + " got " + std::to_string(tmp[i]);
      return false;
    }
  }
  return true;
}

}  // namespace verbsx
}  // namespace acceptance

#endif  // ACCEPT_HAVE_HIP && ACCEPT_HAVE_IBVERBS
