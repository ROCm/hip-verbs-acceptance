/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#include "common/Selection.hpp"

#include <cstdlib>
#include <string>
#include <vector>

#include "common/Capability.hpp"
#include "common/SshLauncher.hpp"

namespace acceptance {

namespace {

// Strict non-negative integer parse; -1 for anything else.
int parse_ordinal(const std::string& s) {
  if (s.empty() || s.size() > 9) return -1;
  for (const char c : s)
    if (c < '0' || c > '9') return -1;
  return std::stoi(s);
}

const GpuDevice* gpu_by_hip(const EnvironmentInfo& env, int hip) {
  for (const GpuDevice& g : env.gpu_devices)
    if (g.hip_ordinal == hip) return &g;
  return nullptr;
}

bool gpu_visible(const EnvironmentInfo& env, int hip) {
  // Without a KFD GPU list, the count is all there is to go on.
  if (env.gpu_devices.empty()) return hip >= 0 && hip < env.gpu_count;
  return gpu_by_hip(env, hip) != nullptr;
}

const RdmaDevice* nic_by_name(const EnvironmentInfo& env, const std::string& n) {
  for (const RdmaDevice& d : env.rdma_devices)
    if (d.device == n) return &d;
  return nullptr;
}

bool nic_active(const EnvironmentInfo& env, const std::string& n) {
  const RdmaDevice* d = nic_by_name(env, n);
  return d != nullptr && d->port_state == "ACTIVE";
}

// The first NIC with an ACTIVE port, else the first NIC, else "".
std::string first_active_nic(const EnvironmentInfo& env) {
  for (const RdmaDevice& d : env.rdma_devices)
    if (d.port_state == "ACTIVE") return d.device;
  return env.rdma_devices.empty() ? "" : env.rdma_devices.front().device;
}

// The lowest visible HIP ordinal whose (unambiguous) nearest NIC is `hca`, or
// -1 when there is none.
int gpu_nearest_to(const EnvironmentInfo& env, const std::string& hca) {
  int best = -1;
  for (const GpuNicAffinity& a : env.gpu_nic_affinity) {
    if (a.nearest_nic != hca || a.ambiguous) continue;
    for (const GpuDevice& g : env.gpu_devices) {
      if (g.ordinal != a.gpu_ordinal || g.hip_ordinal < 0) continue;
      if (best < 0 || g.hip_ordinal < best) best = g.hip_ordinal;
    }
  }
  return best;
}

}  // namespace

const GpuNicAffinity* affinity_for_gpu(const EnvironmentInfo& env, int gpu) {
  const GpuDevice* g = gpu_by_hip(env, gpu);
  if (g == nullptr) return nullptr;
  for (const GpuNicAffinity& a : env.gpu_nic_affinity)
    if (a.gpu_ordinal == g->ordinal) return &a;
  return nullptr;
}

Selection resolve_selection(const EnvironmentInfo& env, int local_rank,
                            const std::string& hca, int device) {
  Selection s;
  std::vector<std::string> notes;
  auto gpu_name = [](int g) { return "GPU " + std::to_string(g); };

  // GPU.
  auto default_gpu = [&]() {
    s.gpu_source = "default";
    if (local_rank <= 0 || gpu_visible(env, local_rank)) {
      s.gpu = local_rank < 0 ? 0 : local_rank;
      return;
    }
    s.gpu = 0;
    notes.push_back(gpu_name(local_rank) + " (local rank) is not visible; using GPU 0");
  };
  if (device >= 0) {
    s.gpu = device;
    s.gpu_source = "user";
    if (!gpu_visible(env, device) && (env.gpu_count > 0 || !env.gpu_devices.empty()))
      notes.push_back(gpu_name(device) + " is not visible on this host");
  } else if (!hca.empty()) {
    const int g = gpu_nearest_to(env, hca);
    if (g >= 0) {
      s.gpu = g;
      s.gpu_source = "affinity";
    } else {
      default_gpu();
      if (!env.gpu_devices.empty())
        notes.push_back("no GPU is known to be nearest to " + hca + "; using " +
                        gpu_name(s.gpu));
    }
  } else {
    default_gpu();
  }

  // HCA.
  const GpuNicAffinity* aff = affinity_for_gpu(env, s.gpu);
  const bool aff_usable = aff != nullptr && !aff->nearest_nic.empty() && !aff->ambiguous;
  if (!hca.empty()) {
    s.hca = hca;
    s.hca_source = "user";
    if (nic_by_name(env, hca) == nullptr)
      notes.push_back("HCA " + hca + " is not present on this host");
    else if (device >= 0 && aff_usable && aff->nearest_nic != hca)
      notes.push_back("HCA " + hca + " is not the nearest NIC to " + gpu_name(s.gpu) +
                      " (nearest: " + aff->nearest_nic + ")");
  } else if (aff_usable && nic_active(env, aff->nearest_nic)) {
    s.hca = aff->nearest_nic;
    s.hca_source = "affinity";
  } else {
    s.hca = first_active_nic(env);
    if (!s.hca.empty()) {
      s.hca_source = "default";
      // Only worth a note when there is a GPU whose NIC we tried to find.
      if (!env.gpu_devices.empty()) {
        const std::string why =
            aff == nullptr || aff->nearest_nic.empty()
                ? "no GPU-NIC affinity for " + gpu_name(s.gpu)
            : aff->ambiguous
                ? "affinity can't tell NICs apart for " + gpu_name(s.gpu)
                : "nearest NIC " + aff->nearest_nic + " to " + gpu_name(s.gpu) +
                      " is not ACTIVE";
        notes.push_back(why + "; using " + s.hca);
      }
    }
  }
  s.local = aff_usable && !s.hca.empty() && aff->nearest_nic == s.hca;

  for (const std::string& n : notes) s.note += (s.note.empty() ? "" : "; ") + n;
  return s;
}

Selection selection_from_env(const EnvironmentInfo& env) {
  auto get = [](const char* name) {
    const char* v = std::getenv(name);
    return std::string(v == nullptr ? "" : v);
  };
  const int local_rank = parse_ordinal(get("ACCEPT_LOCAL_RANK"));
  return resolve_selection(env, local_rank < 0 ? 0 : local_rank,
                           get("ACCEPT_HCA"), parse_ordinal(get("ACCEPT_DEVICE")));
}

std::string format_selection(const Selection& s) {
  std::string line = "GPU " + std::to_string(s.gpu) + " (" + s.gpu_source + "), HCA ";
  if (s.hca.empty()) return line + "none";
  return line + s.hca + " (" + s.hca_source + ")";
}

std::string validate_selection_lists(const std::string& hca,
                                     const std::string& device, int ppn) {
  const std::vector<std::string> h = split_nodes(hca);
  const std::vector<std::string> d = split_nodes(device);
  for (const std::string& v : d) {
    if (parse_ordinal(v) < 0)
      return "--device selects a GPU ordinal; use --hca for the RDMA device "
             "(got '" + v + "')";
  }
  auto check = [ppn](const char* flag, std::size_t n) -> std::string {
    if (n <= 1 || static_cast<int>(n) == ppn) return "";
    return std::string(flag) + " lists " + std::to_string(n) +
           " entries; give 1, or one per local rank (" + std::to_string(ppn) + ")";
  };
  if (std::string e = check("--hca", h.size()); !e.empty()) return e;
  return check("--device", d.size());
}

std::string pick_for_local_rank(const std::string& list, int local_rank) {
  const std::vector<std::string> v = split_nodes(list);
  if (v.empty()) return "";
  if (v.size() == 1) return v.front();
  const std::size_t i = local_rank < 0 ? 0 : static_cast<std::size_t>(local_rank);
  return i < v.size() ? v[i] : v.back();
}

std::string hca() {
  const char* v = std::getenv("ACCEPT_HCA");
  return v == nullptr ? "" : v;
}

int gpu() {
  const char* v = std::getenv("ACCEPT_DEVICE");
  const int g = parse_ordinal(v == nullptr ? "" : v);
  return g < 0 ? 0 : g;
}

}  // namespace acceptance
