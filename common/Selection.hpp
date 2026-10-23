/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

// Selection — the GPU and RDMA device (HCA) one rank uses.
//
// --hca and --device take one value, or one per local rank (-ppn). The
// launcher picks each rank's entry and passes it on with --local-rank; the
// rank publishes them as ACCEPT_HCA / ACCEPT_DEVICE / ACCEPT_LOCAL_RANK, and
// detect() resolves the pair from its own environment (resolve_selection):
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
// `note` says why; that is never an error. The rank then writes the resolved
// pair back into ACCEPT_HCA / ACCEPT_DEVICE, so isolated test children
// (hva --exec) use it without probing again: hca() and gpu() read those.
//
// GPU numbers are HIP ordinals (what hipSetDevice takes, relative to
// HIP_VISIBLE_DEVICES), not KFD topology ordinals; GpuDevice::hip_ordinal maps
// one to the other.

#include <string>

namespace acceptance {

struct EnvironmentInfo;
struct GpuNicAffinity;

struct Selection {
  std::string hca;          // RDMA device name; "" when the host has none
  int         gpu = 0;      // HIP ordinal
  std::string hca_source;   // "user" | "affinity" | "default" ("" with no HCA)
  std::string gpu_source;   // "user" | "affinity" | "default"
  bool        local = false; // affinity confirms hca is gpu's nearest NIC
  std::string note;         // why a fallback was taken; "" when none
};

// Resolve one rank's pair. `hca` is "" and `device` is -1 when not given.
Selection resolve_selection(const EnvironmentInfo& env, int local_rank,
                            const std::string& hca, int device);

// resolve_selection() with the inputs read from ACCEPT_HCA, ACCEPT_DEVICE and
// ACCEPT_LOCAL_RANK (unset or malformed => not given / 0).
Selection selection_from_env(const EnvironmentInfo& env);

// One line for the pre-run header, e.g. "GPU 0 (default), HCA bnxt_re0
// (affinity)", or "..., HCA none" on a host without RDMA devices.
std::string format_selection(const Selection& s);

// The affinity entry for HIP ordinal `gpu`, or nullptr when unknown.
const GpuNicAffinity* affinity_for_gpu(const EnvironmentInfo& env, int gpu);

// Check --hca / --device lists for a launch with `ppn` ranks per host. Returns
// "" when valid, else the error (without "error: "). Each list must have 1 or
// `ppn` entries, and --device entries must be non-negative integers.
std::string validate_selection_lists(const std::string& hca,
                                     const std::string& device, int ppn);

// The entry of a 1-or-P list for `local_rank`; "" for an empty list.
std::string pick_for_local_rank(const std::string& list, int local_rank);

// The resolved pair for this process (ACCEPT_HCA, "" if unset; ACCEPT_DEVICE,
// 0 if unset).
std::string hca();
int gpu();

}  // namespace acceptance
