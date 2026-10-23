/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/
#pragma once

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#ifndef __HIP_DISABLE_CPP_FUNCTIONS__
#define __HIP_DISABLE_CPP_FUNCTIONS__
#endif

#include <hip/hip_runtime.h>

#include <string>
#include <vector>

#include "wrapper/hip/api_wrapper.hpp"

namespace acceptance {
namespace hipw {

typedef struct hip_lib_ops_t {
  decltype(::hipCreateTextureObject)* hipCreateTextureObject;
  decltype(::hipDestroyTextureObject)* hipDestroyTextureObject;
  decltype(::hipDeviceCanAccessPeer)* hipDeviceCanAccessPeer;
  decltype(::hipDeviceEnablePeerAccess)* hipDeviceEnablePeerAccess;
  decltype(::hipDeviceGetAttribute)* hipDeviceGetAttribute;
  decltype(::hipDeviceGetByPCIBusId)* hipDeviceGetByPCIBusId;
  decltype(::hipDeviceGetDefaultMemPool)* hipDeviceGetDefaultMemPool;
  decltype(::hipDeviceGetPCIBusId)* hipDeviceGetPCIBusId;
  decltype(::hipDeviceGetStreamPriorityRange)* hipDeviceGetStreamPriorityRange;
  decltype(::hipDeviceReset)* hipDeviceReset;
  decltype(::hipDeviceSetLimit)* hipDeviceSetLimit;
  decltype(::hipDeviceSynchronize)* hipDeviceSynchronize;
  decltype(::hipDeviceTotalMem)* hipDeviceTotalMem;
  decltype(::hipDriverGetVersion)* hipDriverGetVersion;
  decltype(::hipEventCreate)* hipEventCreate;
  decltype(::hipEventCreateWithFlags)* hipEventCreateWithFlags;
  decltype(::hipEventDestroy)* hipEventDestroy;
  decltype(::hipEventElapsedTime)* hipEventElapsedTime;
  decltype(::hipEventQuery)* hipEventQuery;
  decltype(::hipEventRecord)* hipEventRecord;
  decltype(::hipEventSynchronize)* hipEventSynchronize;
  decltype(::hipExtMallocWithFlags)* hipExtMallocWithFlags;
  decltype(::hipFree)* hipFree;
  decltype(::hipFreeAsync)* hipFreeAsync;
  decltype(::hipFuncGetAttributes)* hipFuncGetAttributes;
  decltype(::hipFuncSetAttribute)* hipFuncSetAttribute;
  decltype(::hipGetDevice)* hipGetDevice;
  decltype(::hipGetDeviceCount)* hipGetDeviceCount;
  decltype(::hipGetDevicePropertiesR0600)* hipGetDevicePropertiesR0600;
  decltype(::hipGetErrorString)* hipGetErrorString;
  decltype(::hipGetFuncBySymbol)* hipGetFuncBySymbol;
  decltype(::hipGetLastError)* hipGetLastError;
  decltype(::hipGetProcAddress)* hipGetProcAddress;
  decltype(::hipGraphAddDependencies)* hipGraphAddDependencies;
  decltype(::hipGraphAddEventRecordNode)* hipGraphAddEventRecordNode;
  decltype(::hipGraphAddHostNode)* hipGraphAddHostNode;
  decltype(::hipGraphDestroy)* hipGraphDestroy;
  decltype(::hipGraphExecDestroy)* hipGraphExecDestroy;
  decltype(::hipGraphGetEdges)* hipGraphGetEdges;
  decltype(::hipGraphInstantiate)* hipGraphInstantiate;
  decltype(::hipGraphLaunch)* hipGraphLaunch;
  decltype(::hipHostFree)* hipHostFree;
  decltype(::hipHostGetDevicePointer)* hipHostGetDevicePointer;
  decltype(::hipHostMalloc)* hipHostMalloc;
  decltype(::hipHostRegister)* hipHostRegister;
  decltype(::hipHostUnregister)* hipHostUnregister;
  decltype(::hipIpcCloseMemHandle)* hipIpcCloseMemHandle;
  decltype(::hipIpcGetMemHandle)* hipIpcGetMemHandle;
  decltype(::hipIpcOpenMemHandle)* hipIpcOpenMemHandle;
  // hipLaunchCooperativeKernel is overloaded in the HIP headers (C decl + C++
  // template), so decltype cannot name it -- spell the C signature explicitly.
  hipError_t (*hipLaunchCooperativeKernel)(const void*, dim3, dim3, void**, unsigned int,
                                           hipStream_t);
  decltype(::hipLaunchHostFunc)* hipLaunchHostFunc;
  decltype(::hipLaunchKernel)* hipLaunchKernel;
  decltype(::hipLaunchKernelExC)* hipLaunchKernelExC;
  decltype(::hipMalloc)* hipMalloc;
  decltype(::hipMallocManaged)* hipMallocManaged;
  decltype(::hipMemAddressFree)* hipMemAddressFree;
  decltype(::hipMemAddressReserve)* hipMemAddressReserve;
  decltype(::hipMemAdvise)* hipMemAdvise;
  decltype(::hipMemCreate)* hipMemCreate;
  decltype(::hipMemExportToShareableHandle)* hipMemExportToShareableHandle;
  decltype(::hipMemGetAccess)* hipMemGetAccess;
  decltype(::hipMemGetAddressRange)* hipMemGetAddressRange;
  decltype(::hipMemGetAllocationGranularity)* hipMemGetAllocationGranularity;
  decltype(::hipMemGetHandleForAddressRange)* hipMemGetHandleForAddressRange;
  decltype(::hipMemGetInfo)* hipMemGetInfo;
  decltype(::hipMemImportFromShareableHandle)* hipMemImportFromShareableHandle;
  decltype(::hipMemMap)* hipMemMap;
  decltype(::hipMemPoolCreate)* hipMemPoolCreate;
  decltype(::hipMemPoolDestroy)* hipMemPoolDestroy;
  decltype(::hipMemPoolGetAttribute)* hipMemPoolGetAttribute;
  decltype(::hipMemPoolSetAttribute)* hipMemPoolSetAttribute;
  decltype(::hipMemPoolTrimTo)* hipMemPoolTrimTo;
  decltype(::hipMemPtrGetInfo)* hipMemPtrGetInfo;
  decltype(::hipMemRelease)* hipMemRelease;
  decltype(::hipMemSetAccess)* hipMemSetAccess;
  decltype(::hipMemUnmap)* hipMemUnmap;
  decltype(::hipMemcpy)* hipMemcpy;
  decltype(::hipMemcpyAsync)* hipMemcpyAsync;
  decltype(::hipMemcpyBatchAsync)* hipMemcpyBatchAsync;
  // hipMemcpyTo/FromSymbol are overloaded in the HIP headers (C decl + C++
  // template), so decltype cannot name them -- spell the C signature explicitly.
  hipError_t (*hipMemcpyFromSymbol)(void*, const void*, size_t, size_t, hipMemcpyKind);
  decltype(::hipMemcpyPeer)* hipMemcpyPeer;
  decltype(::hipMemcpyPeerAsync)* hipMemcpyPeerAsync;
  hipError_t (*hipMemcpyToSymbol)(const void*, const void*, size_t, size_t, hipMemcpyKind);
  decltype(::hipMemset)* hipMemset;
  decltype(::hipMemsetAsync)* hipMemsetAsync;
  decltype(::hipMemsetD32Async)* hipMemsetD32Async;
  decltype(::hipModuleGetFunction)* hipModuleGetFunction;
  decltype(::hipModuleGetGlobal)* hipModuleGetGlobal;
  decltype(::hipModuleLaunchCooperativeKernel)* hipModuleLaunchCooperativeKernel;
  decltype(::hipModuleLaunchKernel)* hipModuleLaunchKernel;
  decltype(::hipModuleLoad)* hipModuleLoad;
  decltype(::hipModuleLoadData)* hipModuleLoadData;
  decltype(::hipModuleOccupancyMaxActiveBlocksPerMultiprocessor)* hipModuleOccupancyMaxActiveBlocksPerMultiprocessor;
  decltype(::hipModuleOccupancyMaxPotentialBlockSize)* hipModuleOccupancyMaxPotentialBlockSize;
  decltype(::hipModuleUnload)* hipModuleUnload;
  decltype(::hipPointerGetAttributes)* hipPointerGetAttributes;
  decltype(::hipRuntimeGetVersion)* hipRuntimeGetVersion;
  decltype(::hipSetDevice)* hipSetDevice;
  decltype(::hipStreamBatchMemOp)* hipStreamBatchMemOp;
  decltype(::hipStreamBeginCapture)* hipStreamBeginCapture;
  decltype(::hipStreamCreate)* hipStreamCreate;
  decltype(::hipStreamCreateWithFlags)* hipStreamCreateWithFlags;
  decltype(::hipStreamDestroy)* hipStreamDestroy;
  decltype(::hipStreamEndCapture)* hipStreamEndCapture;
  decltype(::hipStreamGetCaptureInfo_v2)* hipStreamGetCaptureInfo_v2;
  decltype(::hipStreamGetId)* hipStreamGetId;
  decltype(::hipStreamQuery)* hipStreamQuery;
  decltype(::hipStreamSynchronize)* hipStreamSynchronize;
  decltype(::hipStreamUpdateCaptureDependencies)* hipStreamUpdateCaptureDependencies;
  decltype(::hipStreamWaitEvent)* hipStreamWaitEvent;
  decltype(::hipStreamWaitValue32)* hipStreamWaitValue32;
  decltype(::hipStreamWaitValue64)* hipStreamWaitValue64;
  decltype(::hipStreamWriteValue32)* hipStreamWriteValue32;
  decltype(::hipThreadExchangeStreamCaptureMode)* hipThreadExchangeStreamCaptureMode;
  hipError_t (*hipMallocAsync)(void**, size_t, hipStream_t);
  hipError_t (*hipMallocFromPoolAsync)(void**, size_t, hipMemPool_t, hipStream_t);
} hip_lib_ops_t;

static std::vector<std::string> hip_fn_names = {
    "hipCreateTextureObject",
    "hipDestroyTextureObject",
    "hipDeviceCanAccessPeer",
    "hipDeviceEnablePeerAccess",
    "hipDeviceGetAttribute",
    "hipDeviceGetByPCIBusId",
    "hipDeviceGetDefaultMemPool",
    "hipDeviceGetPCIBusId",
    "hipDeviceGetStreamPriorityRange",
    "hipDeviceReset",
    "hipDeviceSetLimit",
    "hipDeviceSynchronize",
    "hipDeviceTotalMem",
    "hipDriverGetVersion",
    "hipEventCreate",
    "hipEventCreateWithFlags",
    "hipEventDestroy",
    "hipEventElapsedTime",
    "hipEventQuery",
    "hipEventRecord",
    "hipEventSynchronize",
    "hipExtMallocWithFlags",
    "hipFree",
    "hipFreeAsync",
    "hipFuncGetAttributes",
    "hipFuncSetAttribute",
    "hipGetDevice",
    "hipGetDeviceCount",
    "hipGetDevicePropertiesR0600",
    "hipGetErrorString",
    "hipGetFuncBySymbol",
    "hipGetLastError",
    "hipGetProcAddress",
    "hipGraphAddDependencies",
    "hipGraphAddEventRecordNode",
    "hipGraphAddHostNode",
    "hipGraphDestroy",
    "hipGraphExecDestroy",
    "hipGraphGetEdges",
    "hipGraphInstantiate",
    "hipGraphLaunch",
    "hipHostFree",
    "hipHostGetDevicePointer",
    "hipHostMalloc",
    "hipHostRegister",
    "hipHostUnregister",
    "hipIpcCloseMemHandle",
    "hipIpcGetMemHandle",
    "hipIpcOpenMemHandle",
    "hipLaunchCooperativeKernel",
    "hipLaunchHostFunc",
    "hipLaunchKernel",
    "hipLaunchKernelExC",
    "hipMalloc",
    "hipMallocManaged",
    "hipMemAddressFree",
    "hipMemAddressReserve",
    "hipMemAdvise",
    "hipMemCreate",
    "hipMemExportToShareableHandle",
    "hipMemGetAccess",
    "hipMemGetAddressRange",
    "hipMemGetAllocationGranularity",
    "hipMemGetHandleForAddressRange",
    "hipMemGetInfo",
    "hipMemImportFromShareableHandle",
    "hipMemMap",
    "hipMemPoolCreate",
    "hipMemPoolDestroy",
    "hipMemPoolGetAttribute",
    "hipMemPoolSetAttribute",
    "hipMemPoolTrimTo",
    "hipMemPtrGetInfo",
    "hipMemRelease",
    "hipMemSetAccess",
    "hipMemUnmap",
    "hipMemcpy",
    "hipMemcpyAsync",
    "hipMemcpyBatchAsync",
    "hipMemcpyFromSymbol",
    "hipMemcpyPeer",
    "hipMemcpyPeerAsync",
    "hipMemcpyToSymbol",
    "hipMemset",
    "hipMemsetAsync",
    "hipMemsetD32Async",
    "hipModuleGetFunction",
    "hipModuleGetGlobal",
    "hipModuleLaunchCooperativeKernel",
    "hipModuleLaunchKernel",
    "hipModuleLoad",
    "hipModuleLoadData",
    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
    "hipModuleOccupancyMaxPotentialBlockSize",
    "hipModuleUnload",
    "hipPointerGetAttributes",
    "hipRuntimeGetVersion",
    "hipSetDevice",
    "hipStreamBatchMemOp",
    "hipStreamBeginCapture",
    "hipStreamCreate",
    "hipStreamCreateWithFlags",
    "hipStreamDestroy",
    "hipStreamEndCapture",
    "hipStreamGetCaptureInfo_v2",
    "hipStreamGetId",
    "hipStreamQuery",
    "hipStreamSynchronize",
    "hipStreamUpdateCaptureDependencies",
    "hipStreamWaitEvent",
    "hipStreamWaitValue32",
    "hipStreamWaitValue64",
    "hipStreamWriteValue32",
    "hipThreadExchangeStreamCaptureMode",
    "hipMallocAsync",
    "hipMallocFromPoolAsync",
};

// Note: no "optional symbols" list. Symbol availability is decided per call at
// run time -- the #defines below route every ops-table pointer through
// checked_op(), so any symbol the running ROCm did not export makes the
// dependent test skip (MissingSymbol -> Skip at the invocation boundary) rather
// than failing the load or dereferencing null. This keeps the suite agnostic to
// which exact APIs a given ROCm version provides.

extern hip_lib_ops_t hip_lib_ops;

// Per-symbol call-site guard. HIPW_HIP_OP(name) yields the function pointer when
// present and throws MissingSymbol{"name"} when the runtime lacks it.
#define HIPW_HIP_OP(name) \
  (::acceptance::hipw::checked_op(::acceptance::hipw::hip_lib_ops.name, #name))

// Typed-pointer wrappers (HIP C++ overloads disabled). Must precede the hipMalloc*
// #defines below. They go through checked_op too so a missing allocator skips.
template <class T, class... Rest>
inline hipError_t malloc_typed(T** p, Rest... rest) {
  return checked_op(hip_lib_ops.hipMalloc, "hipMalloc")(
      reinterpret_cast<void**>(p), rest...);
}
template <class T, class... Rest>
inline hipError_t malloc_async_typed(T** p, Rest... rest) {
  return checked_op(hip_lib_ops.hipMallocAsync, "hipMallocAsync")(
      reinterpret_cast<void**>(p), rest...);
}
template <class T, class... Rest>
inline hipError_t malloc_from_pool_typed(T** p, Rest... rest) {
  return checked_op(hip_lib_ops.hipMallocFromPoolAsync, "hipMallocFromPoolAsync")(
      reinterpret_cast<void**>(p), rest...);
}

#define hipCreateTextureObject HIPW_HIP_OP(hipCreateTextureObject)
#define hipDestroyTextureObject HIPW_HIP_OP(hipDestroyTextureObject)
#define hipDeviceCanAccessPeer HIPW_HIP_OP(hipDeviceCanAccessPeer)
#define hipDeviceEnablePeerAccess HIPW_HIP_OP(hipDeviceEnablePeerAccess)
#define hipDeviceGetAttribute HIPW_HIP_OP(hipDeviceGetAttribute)
#define hipDeviceGetByPCIBusId HIPW_HIP_OP(hipDeviceGetByPCIBusId)
#define hipDeviceGetDefaultMemPool HIPW_HIP_OP(hipDeviceGetDefaultMemPool)
#define hipDeviceGetPCIBusId HIPW_HIP_OP(hipDeviceGetPCIBusId)
#define hipDeviceGetStreamPriorityRange HIPW_HIP_OP(hipDeviceGetStreamPriorityRange)
#define hipDeviceReset HIPW_HIP_OP(hipDeviceReset)
#define hipDeviceSetLimit HIPW_HIP_OP(hipDeviceSetLimit)
#define hipDeviceSynchronize HIPW_HIP_OP(hipDeviceSynchronize)
#define hipDeviceTotalMem HIPW_HIP_OP(hipDeviceTotalMem)
#define hipDriverGetVersion HIPW_HIP_OP(hipDriverGetVersion)
#define hipEventCreate HIPW_HIP_OP(hipEventCreate)
#define hipEventCreateWithFlags HIPW_HIP_OP(hipEventCreateWithFlags)
#define hipEventDestroy HIPW_HIP_OP(hipEventDestroy)
#define hipEventElapsedTime HIPW_HIP_OP(hipEventElapsedTime)
#define hipEventQuery HIPW_HIP_OP(hipEventQuery)
#define hipEventRecord HIPW_HIP_OP(hipEventRecord)
#define hipEventSynchronize HIPW_HIP_OP(hipEventSynchronize)
#define hipExtMallocWithFlags HIPW_HIP_OP(hipExtMallocWithFlags)
#define hipFree HIPW_HIP_OP(hipFree)
#define hipFreeAsync HIPW_HIP_OP(hipFreeAsync)
#define hipFuncGetAttributes HIPW_HIP_OP(hipFuncGetAttributes)
#define hipFuncSetAttribute HIPW_HIP_OP(hipFuncSetAttribute)
#define hipGetDevice HIPW_HIP_OP(hipGetDevice)
#define hipGetDeviceCount HIPW_HIP_OP(hipGetDeviceCount)
#undef hipGetDeviceProperties
// hipGetDeviceProperties resolves to the R0600 ABI symbol; guard under that name.
#define hipGetDeviceProperties \
  (::acceptance::hipw::checked_op(                                            \
      ::acceptance::hipw::hip_lib_ops.hipGetDevicePropertiesR0600,            \
      "hipGetDevicePropertiesR0600"))
#define hipGetErrorString HIPW_HIP_OP(hipGetErrorString)
#define hipGetFuncBySymbol HIPW_HIP_OP(hipGetFuncBySymbol)
#define hipGetLastError HIPW_HIP_OP(hipGetLastError)
#define hipGetProcAddress HIPW_HIP_OP(hipGetProcAddress)
#define hipGraphAddDependencies HIPW_HIP_OP(hipGraphAddDependencies)
#define hipGraphAddEventRecordNode HIPW_HIP_OP(hipGraphAddEventRecordNode)
#define hipGraphAddHostNode HIPW_HIP_OP(hipGraphAddHostNode)
#define hipGraphDestroy HIPW_HIP_OP(hipGraphDestroy)
#define hipGraphExecDestroy HIPW_HIP_OP(hipGraphExecDestroy)
#define hipGraphGetEdges HIPW_HIP_OP(hipGraphGetEdges)
#define hipGraphInstantiate HIPW_HIP_OP(hipGraphInstantiate)
#define hipGraphLaunch HIPW_HIP_OP(hipGraphLaunch)
#define hipHostFree HIPW_HIP_OP(hipHostFree)
#define hipHostGetDevicePointer HIPW_HIP_OP(hipHostGetDevicePointer)
#define hipHostMalloc HIPW_HIP_OP(hipHostMalloc)
#define hipHostRegister HIPW_HIP_OP(hipHostRegister)
#define hipHostUnregister HIPW_HIP_OP(hipHostUnregister)
#define hipIpcCloseMemHandle HIPW_HIP_OP(hipIpcCloseMemHandle)
#define hipIpcGetMemHandle HIPW_HIP_OP(hipIpcGetMemHandle)
#define hipIpcOpenMemHandle HIPW_HIP_OP(hipIpcOpenMemHandle)
#define hipLaunchCooperativeKernel HIPW_HIP_OP(hipLaunchCooperativeKernel)
#define hipLaunchHostFunc HIPW_HIP_OP(hipLaunchHostFunc)
#define hipLaunchKernel HIPW_HIP_OP(hipLaunchKernel)
#define hipLaunchKernelExC HIPW_HIP_OP(hipLaunchKernelExC)
#define hipMallocManaged HIPW_HIP_OP(hipMallocManaged)
#define hipMemAddressFree HIPW_HIP_OP(hipMemAddressFree)
#define hipMemAddressReserve HIPW_HIP_OP(hipMemAddressReserve)
#define hipMemAdvise HIPW_HIP_OP(hipMemAdvise)
#define hipMemCreate HIPW_HIP_OP(hipMemCreate)
#define hipMemExportToShareableHandle HIPW_HIP_OP(hipMemExportToShareableHandle)
#define hipMemGetAccess HIPW_HIP_OP(hipMemGetAccess)
#define hipMemGetAddressRange HIPW_HIP_OP(hipMemGetAddressRange)
#define hipMemGetAllocationGranularity HIPW_HIP_OP(hipMemGetAllocationGranularity)
#define hipMemGetHandleForAddressRange HIPW_HIP_OP(hipMemGetHandleForAddressRange)
#define hipMemGetInfo HIPW_HIP_OP(hipMemGetInfo)
#define hipMemImportFromShareableHandle HIPW_HIP_OP(hipMemImportFromShareableHandle)
#define hipMemMap HIPW_HIP_OP(hipMemMap)
#define hipMemPoolCreate HIPW_HIP_OP(hipMemPoolCreate)
#define hipMemPoolDestroy HIPW_HIP_OP(hipMemPoolDestroy)
#define hipMemPoolGetAttribute HIPW_HIP_OP(hipMemPoolGetAttribute)
#define hipMemPoolSetAttribute HIPW_HIP_OP(hipMemPoolSetAttribute)
#define hipMemPoolTrimTo HIPW_HIP_OP(hipMemPoolTrimTo)
#define hipMemPtrGetInfo HIPW_HIP_OP(hipMemPtrGetInfo)
#define hipMemRelease HIPW_HIP_OP(hipMemRelease)
#define hipMemSetAccess HIPW_HIP_OP(hipMemSetAccess)
#define hipMemUnmap HIPW_HIP_OP(hipMemUnmap)
#define hipMemcpy HIPW_HIP_OP(hipMemcpy)
#define hipMemcpyAsync HIPW_HIP_OP(hipMemcpyAsync)
#define hipMemcpyBatchAsync HIPW_HIP_OP(hipMemcpyBatchAsync)
#define hipMemcpyFromSymbol HIPW_HIP_OP(hipMemcpyFromSymbol)
#define hipMemcpyPeer HIPW_HIP_OP(hipMemcpyPeer)
#define hipMemcpyPeerAsync HIPW_HIP_OP(hipMemcpyPeerAsync)
#define hipMemcpyToSymbol HIPW_HIP_OP(hipMemcpyToSymbol)
#define hipMemset HIPW_HIP_OP(hipMemset)
#define hipMemsetAsync HIPW_HIP_OP(hipMemsetAsync)
#define hipMemsetD32Async HIPW_HIP_OP(hipMemsetD32Async)
#define hipModuleGetFunction HIPW_HIP_OP(hipModuleGetFunction)
#define hipModuleGetGlobal HIPW_HIP_OP(hipModuleGetGlobal)
#define hipModuleLaunchCooperativeKernel HIPW_HIP_OP(hipModuleLaunchCooperativeKernel)
#define hipModuleLaunchKernel HIPW_HIP_OP(hipModuleLaunchKernel)
#define hipModuleLoad HIPW_HIP_OP(hipModuleLoad)
#define hipModuleLoadData HIPW_HIP_OP(hipModuleLoadData)
#define hipModuleOccupancyMaxActiveBlocksPerMultiprocessor HIPW_HIP_OP(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor)
#define hipModuleOccupancyMaxPotentialBlockSize HIPW_HIP_OP(hipModuleOccupancyMaxPotentialBlockSize)
#define hipModuleUnload HIPW_HIP_OP(hipModuleUnload)
#define hipPointerGetAttributes HIPW_HIP_OP(hipPointerGetAttributes)
#define hipRuntimeGetVersion HIPW_HIP_OP(hipRuntimeGetVersion)
#define hipSetDevice HIPW_HIP_OP(hipSetDevice)
#define hipStreamBatchMemOp HIPW_HIP_OP(hipStreamBatchMemOp)
#define hipStreamBeginCapture HIPW_HIP_OP(hipStreamBeginCapture)
#define hipStreamCreate HIPW_HIP_OP(hipStreamCreate)
#define hipStreamCreateWithFlags HIPW_HIP_OP(hipStreamCreateWithFlags)
#define hipStreamDestroy HIPW_HIP_OP(hipStreamDestroy)
#define hipStreamEndCapture HIPW_HIP_OP(hipStreamEndCapture)
#define hipStreamGetCaptureInfo_v2 HIPW_HIP_OP(hipStreamGetCaptureInfo_v2)
#define hipStreamGetId HIPW_HIP_OP(hipStreamGetId)
#define hipStreamQuery HIPW_HIP_OP(hipStreamQuery)
#define hipStreamSynchronize HIPW_HIP_OP(hipStreamSynchronize)
#define hipStreamUpdateCaptureDependencies HIPW_HIP_OP(hipStreamUpdateCaptureDependencies)
#define hipStreamWaitEvent HIPW_HIP_OP(hipStreamWaitEvent)
#define hipStreamWaitValue32 HIPW_HIP_OP(hipStreamWaitValue32)
#define hipStreamWaitValue64 HIPW_HIP_OP(hipStreamWaitValue64)
#define hipStreamWriteValue32 HIPW_HIP_OP(hipStreamWriteValue32)
#define hipThreadExchangeStreamCaptureMode HIPW_HIP_OP(hipThreadExchangeStreamCaptureMode)
#undef hipMalloc
#define hipMalloc ::acceptance::hipw::malloc_typed
#undef hipMallocAsync
#define hipMallocAsync ::acceptance::hipw::malloc_async_typed
#undef hipMallocFromPoolAsync
#define hipMallocFromPoolAsync ::acceptance::hipw::malloc_from_pool_typed

bool hip_api_init();

void hip_api_fini();

// The dlopen handle for libamdhip64.so (valid only after a successful
// hip_api_init(); nullptr otherwise). Exposed as a function, not a direct
// extern on hip_lib_info, because every hip_lib_ops_t member name (including
// a hypothetical `handle` look-alike) is shadowed by the HIPW_HIP_OP(...)
// object-like macros above once this header is included -- callers outside
// this TU need dlinfo()/dladdr() on the loaded library (e.g. to locate its
// ROCm install root), not the ops table itself.
void* hip_dlopen_handle();

}  // namespace hipw
}  // namespace acceptance
