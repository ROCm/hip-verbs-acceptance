/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// Capability — best-effort HW/FW discovery + a flat, serializable environment
// descriptor. The probe (detect()) reads sysfs and a couple of well-known text
// files via libc/POSIX (opendir/readdir/ifstream/uname), and additionally runs a
// small set of dlopen/dlsym capability probes for optional feature-gating
// libraries (libdl is a hard harness dep). It does NOT popen or link
// libibverbs/HIP in the default config — the optional ibverbs/HIP enrichment is
// compiled only behind ACCEPT_HAVE_IBVERBS / ACCEPT_HAVE_HIP, which are off by
// default. Every read/probe is best-effort: a missing file/dir/symbol leaves the
// corresponding field empty/default/false and never throws.
//
// The descriptor is serialized into the per-rank ResultIO stream (an `@env`
// block, see common/ResultIO.hpp) so every worker's environment rides its own
// stdout back to the launcher, and emitted into report.json's per-node
// `environment` object (see common/Reporter.cpp).
//
// to_kv()/from_kv() reuse the codebase's flat key=value convention (no
// JSON anywhere but the external report). Scalars serialize as `key=value`; the
// rdma_devices list is flattened with indexed keys `rdma.<i>.<field>=value`.
// from_kv() is tolerant of missing/unknown keys for forward-compat.
//
// TEST SEAM: when the environment variable ACCEPT_FAKE_ENV is set (any value),
// detect() returns a deterministic fixed descriptor instead of probing hardware,
// so CI/selftest can exercise the whole serialize -> parse -> report flow with no
// RDMA/GPU hardware present.

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "common/Topology.hpp"

namespace acceptance {

// One GPU device discovered from the KFD topology.
struct GpuDevice {
  int         ordinal = 0;   // 0-based index (order of KFD topology node)
  // HIP ordinal (hipSetDevice / --device numbering, relative to
  // HIP_VISIBLE_DEVICES); -1 when HIP doesn't see this GPU. Equals `ordinal`
  // when HIP isn't loaded.
  int         hip_ordinal = -1;
  std::string bdf;           // PCIe bus-device-function, e.g. "0000:1a:00.0"
  int         numa_node = -1; // /sys/bus/pci/devices/<bdf>/numa_node (-1 = unknown)
  std::string arch;          // e.g. "gfx942" (from KFD props or HIP enrichment)
};

// Nearest-NIC affinity for one GPU (nearest by PCIe BDF prefix).
struct GpuNicAffinity {
  int         gpu_ordinal = 0;      // GpuDevice::ordinal (KFD)
  std::string nearest_nic;          // RdmaDevice::device name, e.g. "bnxt_re0"
  bool        ambiguous = false;    // another NIC scored the same (no real nearest)
};

// One RDMA device discovered under /sys/class/infiniband/<device>.
struct RdmaDevice {
  std::string device;            // e.g. "bnxt_re0"
  std::string netdev;           // associated netdev (.../device/net/<one>)
  std::string fw_ver;           // /sys/class/infiniband/<dev>/fw_ver
  std::string board_id;         // .../board_id
  std::string node_guid;        // .../node_guid
  int         gid_index = -1;   // selected RoCEv2 gid index (-1 = none found)
  std::string gid_type;         // .../ports/<p>/gid_attrs/types/<i> (e.g. "RoCE v2")
  std::string port_state;       // first port's state, e.g. "ACTIVE", "DOWN" ("" = unknown)
  // PCI IDs of the NIC function from .../device/{vendor,device,subsystem_device},
  // normalized to "0x%04x" ("" = unreadable). The vendor ID alone picks
  // `vendor`; device/subsystem_device identify the model (e.g. Broadcom
  // 0x1760 = Thor 2) and are recorded for the report only.
  std::string pci_vendor;
  std::string pci_device;
  std::string pci_subsystem_device;
  std::string vendor;           // connectx | thor | ionic | unknown (from pci_vendor)
  std::string vendor_source;    // "pci" (pci_vendor read) | "none" (unreadable)
  // Optional ibverbs enrichment (only populated under ACCEPT_HAVE_IBVERBS).
  // -1 means "not queried / unknown". ibv_query_device reports these as 32-bit.
  int         vendor_id = -1;       // ibv_device_attr.vendor_id
  int         vendor_part_id = -1;  // ibv_device_attr.vendor_part_id
};

// The per-node environment descriptor.
struct EnvironmentInfo {
  // Vendor of this rank's HCA (rdma_devices[0] after detect()); "unknown" with
  // no devices. Other devices may differ, see mixed_vendor_warning().
  std::string vendor;       // connectx | thor | ionic | unknown
  // Name prefix of the first device: mlx5_ | bnxt_re | ionic, "" otherwise.
  // Informational only; the vendor comes from the PCI vendor ID.
  std::string rdma_prefix;
  // ROCm version actually in use, resolved via a priority chain so it matches
  // the ROCm that hva's HIP dlopen wrapper loaded rather than whatever happens
  // to sit at /opt/rocm (see resolve_rocm_version() in Capability.cpp):
  //   1. "runtime_info"  — directory of the LOADED libamdhip64.so (dlinfo on
  //                        the wrapper's handle): .info/version in its lib/
  //                        dir or that dir's parent.
  //   2. "rocm_path"     — $ROCM_PATH/.info/version (absolute $ROCM_PATH only).
  //   3. "hip_runtime"   — hipRuntimeGetVersion() decoded (no path known).
  //   4. "opt_rocm"      — /opt/rocm/.info/version (today's behaviour).
  //   (none found -> rocm_version empty, rocm_version_source == "none")
  // Sources 1 and 3 require ACCEPT_HAVE_HIP and a loaded HIP runtime.
  std::string rocm_version;
  // Root directory the winning source points to (empty for "hip_runtime"/
  // "none", where no filesystem path is known).
  std::string rocm_path;
  // Which source above won: "runtime_info" | "rocm_path" | "hip_runtime" |
  // "opt_rocm" | "none".
  std::string rocm_version_source;
  int         gpu_count = 0;
  std::string kernel;       // uname(2) .release
  std::string os;           // PRETTY_NAME from /etc/os-release
  std::vector<RdmaDevice>    rdma_devices;
  std::vector<GpuDevice>     gpu_devices;     // per-GPU BDF/NUMA/arch (dep-free)
  std::vector<GpuNicAffinity> gpu_nic_affinity; // nearest NIC per GPU

  // Optional-library / symbol presence probes (dlopen/dlsym). Each entry is
  // name -> present. Populated by detect() from an extensible probe table; empty
  // keys are never inserted. Serialized as `sym.<name>=true|false`.
  std::map<std::string, bool> symbols;
  // amdgpu KMD version from /sys/module/amdgpu/version (the `driver` dimension
  // in `when` predicates). Detected in the sysfs-only build; NOT the HIP
  // runtime/driver version.
  std::string driver_version;
  // HIP enrichment (only populated under ACCEPT_HAVE_HIP). The default
  // sysfs-only build leaves this empty.
  std::string gpu_model;       // props.gcnArchName (and/or props.name)
  // The GPU/HCA pair this rank resolved (see common/Topology.hpp). Set by
  // detect(); its HCA is also moved to the front of rdma_devices.
  topology::Pair rank_pair;

  // Flat key=value serialization (see file header). Deterministic ordering:
  // scalars first, then each rdma device in index order.
  std::string to_kv() const;

  // Reconstruct from key=value lines. Tolerant of missing/unknown keys; the
  // device list is reassembled from the highest rdma.<i>.* index seen.
  static EnvironmentInfo from_kv(const std::vector<std::string>& lines);
};

// Decode a HIP_VERSION-style integer (HIP_VERSION = major*10000000 +
// minor*100000 + patch, e.g. from hipRuntimeGetVersion()) into
// "major.minor.patch". Returns an empty string for a non-positive input. Pure
// arithmetic, no HIP dependency -- unit-testable without HIP.
std::string decode_hip_version(int version);

// Walk from `start_dir` up through at most `max_levels` parent directories
// (start_dir itself is level 0) looking for "<dir>/.info/version". Returns the
// first line of the first one found (trailing CR/LF stripped), or
// std::nullopt if none exists within the search depth. Only a regular file
// whose first line is a version string ([A-Za-z0-9._+~-]) counts as found.
// Pure filesystem read -- unit-testable without HIP or dlopen.
std::optional<std::string> find_info_version(const std::string& start_dir,
                                              int max_levels);

// NIC vendor key for a PCI vendor ID string ("0x15b3" -> connectx, "0x14e4" ->
// thor, "0x1dd8" -> ionic); "unknown" for anything else, including "".
std::string nic_vendor_from_pci(const std::string& pci_vendor);

// Enumerate the RDMA devices under `ib_root` (normally /sys/class/infiniband),
// sorted by name. Best-effort: unreadable attributes stay empty/default.
std::vector<RdmaDevice> probe_rdma_devices(const std::string& ib_root);

// Make `hca` this rank's device: move it to the front of env.rdma_devices (no
// move when absent or ""), then set env.vendor from the front device's stored
// vendor ("unknown" with no devices) and env.rdma_prefix from its name.
void select_rank_hca(EnvironmentInfo& env, const std::string& hca);

// The device named `name` in env.rdma_devices, or nullptr.
const RdmaDevice* find_rdma_device(const EnvironmentInfo& env,
                                   const std::string& name);

// "" when all RDMA devices share one vendor; otherwise one line (no "WARN: "
// prefix) listing the devices per vendor and env.vendor, the one in use.
std::string mixed_vendor_warning(const EnvironmentInfo& env);

// Probe the local node. Best-effort, never throws. Honors ACCEPT_FAKE_ENV (see
// file header) as a deterministic test seam.
EnvironmentInfo detect();

// Cached detect(): runs the probe once per process and returns the shared
// result. Prefer this over detect() for read-only lookups so a worker does not
// repeat the sysfs walk / device opens on every call site.
const EnvironmentInfo& environment();

}  // namespace acceptance
