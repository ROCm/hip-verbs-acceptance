/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// Topology — where GPUs and RDMA devices (HCAs) sit on the PCIe tree, and the
// GPU/HCA pair one rank uses.
//
// detect() computes each GPU's nearest NIC (compute_affinity) and resolves
// this rank's pair from its own environment (get_rank_pair).
//
// --hca and --device take one value, or one per local rank (-ppn). The
// launcher picks each rank's entry and passes it on with --local-rank; the
// rank publishes them as ACCEPT_HCA / ACCEPT_DEVICE / ACCEPT_LOCAL_RANK, and
// select_pair() resolves the pair:
//
//   --hca + --device  both as given; `local` says whether the HCA is the GPU's
//                     nearest NIC
//   --hca only        GPU = the lowest HIP ordinal whose nearest NIC is the HCA
//   --device only     HCA = that GPU's nearest NIC if ACTIVE, else the first
//                     ACTIVE NIC
//   neither           GPU = the local rank (0 if not visible), then the
//                     --device-only rule
//
// When affinity is unknown or can't tell NICs apart, the fallback applies and
// `note` says why; that is never an error. A --device this host does not show
// or a --hca it does not have is: `error` says which, and the rank exits 2
// instead of running on something else. The rank then writes the resolved
// pair back into ACCEPT_HCA / ACCEPT_DEVICE, so isolated test children
// (hva --exec) use it without probing again: get_rank_hca() and get_rank_gpu()
// read those.
//
// GPU numbers are HIP ordinals (what hipSetDevice takes, relative to
// HIP_VISIBLE_DEVICES), not KFD topology ordinals; GpuDevice::hip_ordinal maps
// one to the other.

#include <string>
#include <vector>

namespace acceptance {

struct EnvironmentInfo;
struct GpuDevice;
struct GpuNicAffinity;
struct RdmaDevice;

namespace topology {

// One rank's GPU and HCA.
struct Pair {
  std::string hca;          // RDMA device name; "" when the host has none
  int         gpu = 0;      // HIP ordinal
  std::string hca_source;   // "user" | "affinity" | "default" ("" with no HCA)
  std::string gpu_source;   // "user" | "affinity" | "default"
  bool        local = false; // affinity confirms hca is gpu's nearest NIC
  std::string note;         // why a fallback was taken; "" when none
  std::string error;        // an explicit --hca / --device this host can't
                            // satisfy; "" when none. The rank must not run.
};

// For each GPU, the RDMA device whose sysfs PCIe path shares the deepest
// ancestor with the GPU's (same switch beats same root port beats same root
// complex). BDF numbering says nothing about topology, so the paths are what
// get compared.
std::vector<GpuNicAffinity> compute_affinity(const std::vector<GpuDevice>& gpus,
                                             const std::vector<RdmaDevice>& nics);

// A KFD topology node's PCIe BDF ("dddd:bb:dd.f") from its `location_id` and
// `domain` properties. KFD prints both in decimal; location_id is the PCI
// device id, (bus << 8) | devfn. "" when location_id is not a number.
std::string parse_kfd_bdf(const std::string& location_id, const std::string& domain);

// How many leading components two sysfs device paths share
// ("/sys/devices/pci0000:00/0000:00:01.1/..."): the deeper the shared PCIe
// ancestor, the closer the devices. 0 when either path is empty.
int get_pci_shared_depth(const std::string& a, const std::string& b);

// The affinity entry for HIP ordinal `gpu`, or nullptr when unknown.
const GpuNicAffinity* find_gpu_affinity(const EnvironmentInfo& env, int gpu);

// Resolve one rank's pair. `hca` is "" and `device` is -1 when not given.
Pair select_pair(const EnvironmentInfo& env, int local_rank, const std::string& hca,
                 int device);

// select_pair() with the inputs read from ACCEPT_HCA, ACCEPT_DEVICE and
// ACCEPT_LOCAL_RANK (unset or malformed => not given / 0).
Pair get_rank_pair(const EnvironmentInfo& env);

// One line for the pre-run header, e.g. "GPU 0 (default), HCA bnxt_re0
// (affinity)", or "..., HCA none" on a host without RDMA devices.
std::string format_pair(const Pair& p);

// Check --hca / --device lists for a launch with `ppn` ranks per host. Returns
// "" when valid, else the error (without "error: "). Each list must have 1 or
// `ppn` entries, and --device entries must be non-negative integers.
std::string validate_lists(const std::string& hca, const std::string& device, int ppn);

// The entry of a 1-or-P list for `local_rank`; "" for an empty list.
std::string get_rank_entry(const std::string& list, int local_rank);

// The resolved pair for this process (ACCEPT_HCA, "" if unset; ACCEPT_DEVICE,
// 0 if unset).
std::string get_rank_hca();
int get_rank_gpu();

}  // namespace topology
}  // namespace acceptance
