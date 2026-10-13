/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#include "common/Capability.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <dirent.h>
#include <dlfcn.h>
#include <sys/utsname.h>
#include <unistd.h>   // readlink

#include "common/TestResult.hpp"  // int_to_chars (locale-independent)

// Optional compile-time enrichment backends. Off by default: the headers
// are pulled in ONLY when the corresponding ACCEPT_HAVE_* macro is defined by the
// build (see CMakeLists.txt ACCEPT_WITH_IBVERBS / ACCEPT_WITH_HIP). The default
// CXX build never sees these includes, so it has zero ibverbs/HIP build deps.
#ifdef ACCEPT_HAVE_IBVERBS
// Verbs entry points resolve at run time through the dlopen wrapper (no
// libibverbs linked); IbvApi.hpp pulls in the vendored declarations and the
// ops-table redirects.
#include "wrapper/verbs/IbvApi.hpp"
#endif
#ifdef ACCEPT_HAVE_HIP
// Login-node build model: pull the HIP declarations + dlopen wrapper (not the
// HIP toolchain). HipApi.hpp defines __HIP_PLATFORM_AMD__ and redirects the HIP
// entry points to the run-time-loaded ops table, so no ROCm library is linked.
#include "hip/utils/HipApi.hpp"
#endif

namespace acceptance {

namespace {

// Read the whole content of a file (best-effort). Returns empty on any error.
std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return std::string();
  std::ostringstream os;
  os << f.rdbuf();
  return os.str();
}

// Read the first line of a file with trailing CR/LF stripped (best-effort).
std::string read_first_line(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return std::string();
  std::string line;
  std::getline(f, line);
  if (!line.empty() && line.back() == '\r') line.pop_back();
  return line;
}

// Read a single-value sysfs attribute (one line, trailing whitespace trimmed).
std::string read_attr(const std::string& path) {
  std::string v = read_first_line(path);
  while (!v.empty() &&
         (v.back() == '\n' || v.back() == '\r' || v.back() == ' ' ||
          v.back() == '\t')) {
    v.pop_back();
  }
  return v;
}

// List entries of a directory (excluding "." and ".."), sorted by name.
// Returns empty when the directory cannot be opened.
std::vector<std::string> list_dir(const std::string& path) {
  std::vector<std::string> names;
  DIR* d = ::opendir(path.c_str());
  if (d == nullptr) return names;
  for (struct dirent* e = ::readdir(d); e != nullptr; e = ::readdir(d)) {
    const std::string name = e->d_name;
    if (name == "." || name == "..") continue;
    names.push_back(name);
  }
  ::closedir(d);
  std::sort(names.begin(), names.end());
  return names;
}

// A gid value line is "non-zero" if it contains any hex digit other than 0
// (ignoring the ':' group separators). An all-zero gid (0000:...:0000) means the
// slot is unused.
bool gid_nonzero(const std::string& gid) {
  for (char c : gid) {
    if (c == ':' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
    if (c != '0') return true;
  }
  return false;
}

// True if a sysfs GID value is IPv6 link-local — present on RoCE ports but NOT
// routable between nodes, so it must not be preferred for a cross-node connect.
// Link-local is fe80::/10, but the kernel always auto-generates RoCE link-local
// GIDs as exactly fe80::/64 (RFC 4291 / addrconf), so matching the "fe80:"
// prefix covers every link-local GID that actually appears in the table, and no
// routable RoCE GID (IPv4-mapped "::ffff:", or GUA 2xxx/3xxx) starts with it.
bool gid_is_link_local(const std::string& gid) {
  return gid.rfind("fe80:", 0) == 0;
}

// True if a gid_type string denotes RoCEv2. The kernel spells it "RoCE v2"
// (space), but accept the compact "RoCEv2" too for robustness. Case-sensitive
// match is sufficient — sysfs uses this exact casing.
bool gid_type_is_v2(const std::string& type) {
  return type.find("RoCE v2") != std::string::npos ||
         type.find("RoCEv2") != std::string::npos;
}

// A kv value must stay on one line: collapse any embedded CR/LF to a space so a
// stray newline in a sysfs field can't inject a phantom key=value line (parity
// with TestResult::one_line). Our fields are single-line in practice; defensive.
std::string kv_safe(std::string s) {
  for (char& c : s) {
    if (c == '\n' || c == '\r') c = ' ';
  }
  return s;
}

// Map an infiniband device NAME to (vendor, rdma_prefix) by its canonical prefix.
void vendor_from_device(const std::string& dev, std::string& vendor,
                        std::string& prefix) {
  auto starts_with = [&](const char* p) {
    return dev.compare(0, std::strlen(p), p) == 0;
  };
  if (starts_with("mlx5_")) {
    vendor = "connectx";
    prefix = "mlx5_";
  } else if (starts_with("bnxt_re")) {
    vendor = "thor";
    prefix = "bnxt_re";
  } else if (starts_with("ionic") || starts_with("rocep")) {
    vendor = "ionic";
    prefix = "ionic";
  } else {
    vendor = "unknown";
    prefix.clear();
  }
}

// Parse PRETTY_NAME="..." out of /etc/os-release content.
std::string parse_pretty_name(const std::string& content) {
  std::istringstream is(content);
  std::string line;
  while (std::getline(is, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const std::string key = "PRETTY_NAME=";
    if (line.compare(0, key.size(), key) != 0) continue;
    std::string v = line.substr(key.size());
    // Strip surrounding double quotes if present.
    if (v.size() >= 2 && v.front() == '"' && v.back() == '"') {
      v = v.substr(1, v.size() - 2);
    }
    return v;
  }
  return std::string();
}

// Parse a single KFD property value (the word after the key in "key value\n").
std::string kfd_prop(const std::string& props, const std::string& key) {
  std::istringstream is(props);
  std::string line;
  while (std::getline(is, line)) {
    if (line.compare(0, key.size(), key) != 0) continue;
    if (line.size() <= key.size() || line[key.size()] != ' ') continue;
    return line.substr(key.size() + 1);
  }
  return std::string();
}

// Resolve a KFD topology node's PCIe BDF from its `location_id` property.
// location_id is a hex-encoded BDF: upper 16 bits = domain, bits 8-15 = bus,
// bits 3-7 = device, bits 0-2 = function. Format: DDDD:BB:dd.f
// Fallback: read the "device" symlink target's last component (older kernels).
std::string kfd_node_bdf(const std::string& node_path,
                         const std::string& props) {
  const std::string loc_str = kfd_prop(props, "location_id");
  if (!loc_str.empty()) {
    try {
      const unsigned long loc = std::stoul(loc_str, nullptr, 16);
      const unsigned domain = (loc >> 16) & 0xFFFF;
      const unsigned bus    = (loc >>  8) & 0xFF;
      const unsigned dev    = (loc >>  3) & 0x1F;
      const unsigned func   = (loc >>  0) & 0x07;
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%04x:%02x:%02x.%x",
                    domain, bus, dev, func);
      return buf;
    } catch (...) {}
  }
  // Fallback: readlink .../device -> .../pci/devices/<bdf>
  char buf[512];
  const std::string link = node_path + "/device";
  const ssize_t n = ::readlink(link.c_str(), buf, sizeof(buf) - 1);
  // n == sizeof(buf)-1 may indicate truncation (readlink doesn't report it);
  // treat a maxed-out read as a failure rather than parse a truncated path.
  if (n <= 0 || static_cast<std::size_t>(n) >= sizeof(buf) - 1) return std::string();
  buf[n] = '\0';
  const std::string target(buf);
  const std::size_t slash = target.rfind('/');
  return (slash == std::string::npos) ? target : target.substr(slash + 1);
}

// Read the NUMA node for a PCIe BDF from sysfs.
int bdf_numa_node(const std::string& bdf) {
  if (bdf.empty()) return -1;
  const std::string path = "/sys/bus/pci/devices/" + bdf + "/numa_node";
  const std::string v = read_first_line(path);
  if (v.empty()) return -1;
  try { return std::stoi(v); } catch (...) { return -1; }
}

// Enumerate GPU devices from the KFD topology, filling gpu_devices[].
// Returns the GPU count (simd_count > 0 heuristic).
int detect_gpu_devices(std::vector<GpuDevice>& out) {
  const std::string base = "/sys/class/kfd/kfd/topology/nodes";
  const std::vector<std::string> nodes = list_dir(base);
  int gpus = 0, parsed = 0, ordinal = 0;
  for (const std::string& n : nodes) {
    const std::string node_path = base + "/" + n;
    const std::string props = read_file(node_path + "/properties");
    if (props.empty()) continue;
    const std::string simd_str = kfd_prop(props, "simd_count");
    if (simd_str.empty()) continue;
    ++parsed;
    long long simd = 0;
    try { simd = std::stoll(simd_str); } catch (...) {}
    if (simd <= 0) continue;  // CPU node
    ++gpus;
    GpuDevice g;
    g.ordinal = ordinal++;
    g.bdf = kfd_node_bdf(node_path, props);
    g.numa_node = bdf_numa_node(g.bdf);
    // Best-effort arch from gfx_target_version (decimal, format:
    // MAJOR*10000 + MINOR*100 + STEPPING, e.g. 90402 -> "gfx942").
    // NOTE: this is a numeric approximation — it cannot reproduce lettered
    // canonical names (e.g. gfx90a). The HIP gcnArchName path (ACCEPT_WITH_HIP)
    // is authoritative when available; this is the dep-free fallback.
    const std::string gfxver = kfd_prop(props, "gfx_target_version");
    if (!gfxver.empty()) {
      try {
        const long v = std::stol(gfxver);
        const long major    = v / 10000;
        const long minor    = (v % 10000) / 100;
        const long stepping = v % 100;
        // FORMAT ASSUMPTION: in canonical arch names minor and stepping are each
        // a single hex nibble (gfx942 = 9,4,2; gfx1100 = 11,0,0), so we print
        // them with no padding. This is an approximation: a minor/stepping >= 10
        // is a hex digit (e.g. gfx90a, stepping=10='a') that the decimal version
        // cannot round-trip — such parts misformat here. No shipping arch in
        // scope hits that case; treat the result as a best-effort label.
        char buf[32];
        std::snprintf(buf, sizeof(buf), "gfx%ld%ld%ld", major, minor, stepping);
        g.arch = buf;
      } catch (...) {
        g.arch = "gfx" + gfxver;  // raw fallback
      }
    } else {
      const std::string nm = kfd_prop(props, "name");
      if (!nm.empty()) g.arch = nm;
    }
    out.push_back(std::move(g));
  }
  if (parsed > 0) return gpus;
  return static_cast<int>(nodes.size());
}

// Compute GPU↔NIC affinity: for each GPU, find the RDMA device whose PCIe BDF
// shares the longest prefix with the GPU's BDF (same PCIe root → lowest hop
// count). BDFs are "domain:bus:dev.func"; we compare the "domain:bus" prefix so
// devices on the same bus score highest. Falls back to bnxt_re0/mlx5_0/etc.
std::vector<GpuNicAffinity> compute_affinity(
    const std::vector<GpuDevice>& gpus,
    const std::vector<RdmaDevice>& nics) {
  if (gpus.empty() || nics.empty()) return {};

  // Build NIC BDF table: read /sys/class/infiniband/<dev>/device -> BDF.
  auto nic_bdf = [](const std::string& dev) -> std::string {
    char buf[512];
    const std::string link =
        "/sys/class/infiniband/" + dev + "/device";
    const ssize_t n = ::readlink(link.c_str(), buf, sizeof(buf) - 1);
    if (n <= 0) return std::string();
    buf[n] = '\0';
    const std::string t(buf);
    const std::size_t s = t.rfind('/');
    return (s == std::string::npos) ? t : t.substr(s + 1);
  };

  struct NicEntry { std::string dev; std::string bdf; };
  std::vector<NicEntry> nic_entries;
  for (const auto& nic : nics) {
    nic_entries.push_back({nic.device, nic_bdf(nic.device)});
  }

  // Common-prefix length of two BDF strings (domain:bus share = high affinity).
  auto common_prefix = [](const std::string& a, const std::string& b) -> std::size_t {
    std::size_t n = std::min(a.size(), b.size()), i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
  };

  std::vector<GpuNicAffinity> result;
  for (const auto& gpu : gpus) {
    GpuNicAffinity aff;
    aff.gpu_ordinal = gpu.ordinal;
    std::size_t best = 0;
    for (const auto& ne : nic_entries) {
      const std::size_t score = common_prefix(gpu.bdf, ne.bdf);
      if (score > best || aff.nearest_nic.empty()) {
        best = score;
        aff.nearest_nic = ne.dev;
      }
    }
    result.push_back(std::move(aff));
  }
  return result;
}

// Manual GID-index override: ACCEPT_IB_GID_INDEX (set from --gid). >=0 forces
// that index and skips auto-selection (mirrors RCCL's NCCL_IB_GID_INDEX). -1 or
// unset means auto-select. Invalid/negative values are ignored (auto-select).
int gid_index_override() {
  const char* v = std::getenv("ACCEPT_IB_GID_INDEX");
  if (v == nullptr || *v == '\0') return -1;
  try {
    std::size_t pos = 0;
    const std::string s(v);
    const int idx = std::stoi(s, &pos);
    // Reject trailing junk ("3junk") — pos must consume the whole string.
    if (pos != s.size() || idx < 0) return -1;
    return idx;
  } catch (...) {
    return -1;
  }
}

// Select the RoCEv2 gid for a device's first port; fills dev.gid_index/gid_type.
void select_gid(const std::string& dev_path, RdmaDevice& dev) {
  const std::string ports_path = dev_path + "/ports";
  const std::vector<std::string> ports = list_dir(ports_path);
  if (ports.empty()) return;
  const std::string port = ports.front();  // first port (typically "1")
  const std::string gids_dir = ports_path + "/" + port + "/gids";
  const std::string types_dir =
      ports_path + "/" + port + "/gid_attrs/types";

  auto record = [&](int idx, const std::string& type) {
    dev.gid_index = idx;
    dev.gid_type = type;
  };

  // Manual override: honor ACCEPT_IB_GID_INDEX (--gid) when it points at a
  // valid, non-zero slot. If the forced slot is missing or all-zero (bad index
  // or container GID isolation), fall through to auto-selection so a typo in
  // --gid doesn't silently produce an unusable gid_index.
  if (const int forced = gid_index_override(); forced >= 0) {
    const std::string g = read_attr(gids_dir + "/" + std::to_string(forced));
    if (!g.empty() && gid_nonzero(g)) {
      const std::string type =
          read_attr(types_dir + "/" + std::to_string(forced));
      record(forced, type);
      return;
    }
  }

  // Enumerate the gid slots numerically (their names are integer indices) in
  // ascending order. Selection must pick a ROUTABLE RoCEv2 gid for cross-node
  // connects: a link-local (fe80::) gid is RoCEv2-typed but does NOT route
  // between nodes (RTR times out), so routability is preferred over slot index.
  // Priority, highest first:
  //   1. Routable RoCEv2 (non-link-local): prefer the conventional index-3 slot
  //      when it is routable+v2 (bnxt_re convention), else the lowest routable v2.
  //   2. Any RoCEv2 gid, even link-local (single-node fallback so loopback still
  //      works on a host with only a link-local v2 gid).
  //   3. Lowest non-zero gid of any type (RoCEv1 last resort).
  // Type/value are always checked before committing — never select a slot blindly.
  std::vector<int> indices;
  for (const std::string& name : list_dir(gids_dir)) {
    try {
      std::size_t pos = 0;
      const int idx = std::stoi(name, &pos);
      if (pos == name.size()) indices.push_back(idx);
    } catch (...) {
      // non-numeric entry: ignore
    }
  }
  std::sort(indices.begin(), indices.end());

  int routable_v2 = -1;     // lowest routable (non-link-local) RoCEv2 gid
  std::string routable_v2_type;
  bool idx3_routable_v2 = false;
  std::string idx3_type;
  int first_v2 = -1;        // lowest RoCEv2 gid of any form (incl link-local)
  std::string first_v2_type;
  int first_nonzero = -1;   // lowest non-zero gid of any type (v1 fallback)
  std::string first_nonzero_type;
  for (int idx : indices) {
    const std::string g = read_attr(gids_dir + "/" + std::to_string(idx));
    if (g.empty() || !gid_nonzero(g)) continue;
    const std::string type = read_attr(types_dir + "/" + std::to_string(idx));
    if (first_nonzero < 0) { first_nonzero = idx; first_nonzero_type = type; }
    if (gid_type_is_v2(type)) {
      if (first_v2 < 0) { first_v2 = idx; first_v2_type = type; }
      if (!gid_is_link_local(g)) {
        if (routable_v2 < 0) { routable_v2 = idx; routable_v2_type = type; }
        if (idx == 3) { idx3_routable_v2 = true; idx3_type = type; }
      }
    }
  }
  if (idx3_routable_v2)         record(3, idx3_type);
  else if (routable_v2 >= 0)    record(routable_v2, routable_v2_type);
  else if (first_v2 >= 0)       record(first_v2, first_v2_type);
  else if (first_nonzero >= 0)  record(first_nonzero, first_nonzero_type);
}

// --- capability probes (dlopen/dlsym) --------------------------------------
// Side-effect-light by design: dlopen(RTLD_LAZY|RTLD_LOCAL) then dlclose
// immediately, never calling any library entry point. Opening a shared object
// runs only its ELF constructors — it does not init a GPU (libamdhip64 /
// libhsa-runtime64 init lazily on first API call, which we never make). All
// best-effort: a missing lib/symbol records false and never throws.

void* try_dlopen(std::initializer_list<const char*> candidates) {
  for (const char* name : candidates) {
    // RTLD_NODELETE: don't unmap on dlclose. A provider plugin (e.g. libionic)
    // registers itself into libibverbs on load; unmapping it leaves a dangling
    // driver entry -> SIGSEGV in the next ibv_get_device_list().
    void* h = ::dlopen(name, RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE);
    if (h != nullptr) return h;
  }
  return nullptr;
}

// True iff one of the candidate libraries can be dlopen'd. dlclose immediately.
bool probe_lib(std::initializer_list<const char*> candidates) {
  void* h = try_dlopen(candidates);
  if (h == nullptr) return false;
  ::dlclose(h);
  return true;
}

// True iff one of the candidate libraries loads AND exports `symbol`. The library
// is dlclose'd before returning regardless of the outcome.
bool probe_symbol(std::initializer_list<const char*> candidates,
                  const char* symbol) {
  void* h = try_dlopen(candidates);
  if (h == nullptr) return false;
  const bool present = ::dlsym(h, symbol) != nullptr;
  ::dlclose(h);
  return present;
}

// Run the extensible probe table and fill env.symbols. Add a row here to gate a
// new feature on an optional library/symbol — nothing else needs to change
// (serialization/report iterate the map generically).
//
// CONTRIBUTOR WARNING: dlopen runs the library's ELF constructors. The "no GPU
// init" guarantee above holds ONLY for libraries whose constructors are inert
// (no device open, no thread spawn, no env-mutating side effects). Before adding
// a probe, confirm the target lib's constructor is safe to run in this context;
// if it is not, probe for the file on disk instead of dlopen'ing it.
void probe_symbols(EnvironmentInfo& env) {
  // libionic — ionic direct-verbs userspace lib.
  env.symbols["libionic"] = probe_lib({"libionic.so.1", "libionic.so"});
  // librccl-anp — ionic RCCL net plugin.
  env.symbols["librccl-anp"] = probe_lib({"librccl-anp.so"});
  // hsa_amd_portable_export_dmabuf — HSA runtime dmabuf export (GDRDMA gating).
  env.symbols["hsa_amd_portable_export_dmabuf"] = probe_symbol(
      {"libhsa-runtime64.so.1", "libhsa-runtime64.so"},
      "hsa_amd_portable_export_dmabuf");
  // hipMemGetHandleForAddressRange — HIP dmabuf-style export entry point.
  env.symbols["hipMemGetHandleForAddressRange"] =
      probe_symbol({"libamdhip64.so"}, "hipMemGetHandleForAddressRange");
  // ibv_reg_dmabuf_mr — verbs dmabuf MR registration (GDR mixed tests gating).
  env.symbols["ibv_reg_dmabuf_mr"] = probe_symbol(
      {"libibverbs.so.1", "libibverbs.so"}, "ibv_reg_dmabuf_mr");
  // hipStreamWaitValue64 — GPU stream wait-on-value (coherence test gating).
  env.symbols["hipStreamWaitValue64"] =
      probe_symbol({"libamdhip64.so"}, "hipStreamWaitValue64");
}

// --- optional compile-time enrichment (guarded) -----------------------------

// ibverbs: open the FIRST device and record vendor_id/vendor_part_id on the
// matching RdmaDevice (matched by ibv name). Best-effort; compiled only under
// ACCEPT_HAVE_IBVERBS. A no-op in the default build.
void enrich_ibverbs(EnvironmentInfo& env) {
#ifdef ACCEPT_HAVE_IBVERBS
  // libibverbs may be absent (e.g. a login node): the ops table is empty, so
  // skip cleanly rather than throwing MissingSymbol through the first call.
  if (!::acceptance::ibvw::ibv_lib_loaded()) return;
  // Runs outside the test-invocation boundary, so swallow a MissingSymbol here
  // (a runtime lacking one of these queries) rather than letting it escape.
  try {
    int num = 0;
    struct ibv_device** list = ibv_get_device_list(&num);
    if (list == nullptr) return;
    for (int i = 0; i < num; ++i) {
      struct ibv_device* dev = list[i];
      if (dev == nullptr) continue;
      const char* name = ibv_get_device_name(dev);
      struct ibv_context* ctx = ibv_open_device(dev);
      if (ctx == nullptr) continue;
      struct ibv_device_attr attr;
      if (ibv_query_device(ctx, &attr) == 0 && name != nullptr) {
        for (RdmaDevice& d : env.rdma_devices) {
          if (d.device == name) {
            d.vendor_id = static_cast<int>(attr.vendor_id);
            d.vendor_part_id = static_cast<int>(attr.vendor_part_id);
            break;
          }
        }
      }
      ibv_close_device(ctx);
    }
    ibv_free_device_list(list);
  } catch (const ::acceptance::wrap::MissingSymbol&) {
    // Best-effort enrichment only; leave vendor_id/part_id unset.
  }
#else
  (void)env;
#endif
}

// HIP: record gpu_model only (ACCEPT_HAVE_HIP; no-op otherwise). Queries device 0
// properties — no kernel launch/alloc. driver_version is NOT set here: the HIP
// runtime version duplicates rocm_version and is not the KMD driver; detect()
// reads the amdgpu KMD version from sysfs (dep-free) instead.
void enrich_hip(EnvironmentInfo& env) {
#ifdef ACCEPT_HAVE_HIP
  // HIP entry points resolve at run time through the dlopen wrapper; if the HIP
  // runtime was not present (e.g. a login node) the ops table is empty, so skip
  // cleanly rather than dereferencing a null op.
  if (!::acceptance::hipw::init_result().hip_lib_loaded) {
    (void)env;
    return;
  }
  // This runs outside the test-invocation boundary, so swallow a MissingSymbol
  // (a runtime lacking this query) here rather than letting it escape detect().
  try {
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, 0) == hipSuccess) {
      std::string model = props.gcnArchName;
      if (model.empty()) model = props.name;
      env.gpu_model = model;
    }
  } catch (const ::acceptance::hipw::MissingSymbol&) {
    // Leave gpu_model unset; best-effort enrichment only.
  }
#else
  (void)env;
#endif
}

// The deterministic fixed descriptor returned when ACCEPT_FAKE_ENV is set.
EnvironmentInfo fake_env() {
  EnvironmentInfo env;
  env.vendor = "thor";
  env.rdma_prefix = "bnxt_re";
  env.rocm_version = "0.0-fake";
  env.gpu_count = 8;
  env.kernel = "0.0.0-fake";
  env.os = "FakeOS 1.0";
  env.gpu_model = "gfx-fake";
  env.driver_version = "0.0-fake";
  env.symbols["libionic"] = false;
  env.symbols["hsa_amd_portable_export_dmabuf"] = true;
  // Seed the verbs/HIP GDR symbols the mixed + coherence policies gate on, so
  // the fake environment classifies those tests the same way real hardware
  // with a full stack would (dmabuf MR registration + GPU stream wait-on-value).
  env.symbols["ibv_reg_dmabuf_mr"] = true;
  env.symbols["hipStreamWaitValue64"] = true;
  // gpu_devices/gpu_nic_affinity left empty: the NIC device already exercises the
  // serialize→parse→report flow this seam exists for.
  RdmaDevice d;
  d.device = "bnxt_re0";
  d.netdev = "eth0";
  d.fw_ver = "0.0.0";
  d.board_id = "FAKE0000";
  d.node_guid = "0000:0000:0000:0000";
  d.gid_index = 3;
  d.gid_type = "RoCE v2";
  env.rdma_devices.push_back(d);
  return env;
}

}  // namespace

EnvironmentInfo detect() {
  // TEST SEAM (checked FIRST): deterministic descriptor, no hardware reads.
  if (std::getenv("ACCEPT_FAKE_ENV") != nullptr) return fake_env();

  EnvironmentInfo env;

  // --- RDMA device enumeration -------------------------------------------
  const std::string ib_base = "/sys/class/infiniband";
  const std::vector<std::string> devs = list_dir(ib_base);
  for (const std::string& name : devs) {
    const std::string dev_path = ib_base + "/" + name;
    RdmaDevice d;
    d.device = name;
    d.fw_ver = read_attr(dev_path + "/fw_ver");
    d.board_id = read_attr(dev_path + "/board_id");
    d.node_guid = read_attr(dev_path + "/node_guid");
    // Associated netdev: the single entry under .../device/net/.
    const std::vector<std::string> nets = list_dir(dev_path + "/device/net");
    if (!nets.empty()) d.netdev = nets.front();
    select_gid(dev_path, d);
    env.rdma_devices.push_back(std::move(d));
  }

  // Manual device override: ACCEPT_IB_DEVICE (set from --device) moves the named
  // device to the front so vendor detection, GID selection, and "first device"
  // pickers prefer it (mirrors RCCL's NCCL_IB_HCA). No-op if unset or not found.
  if (const char* want = std::getenv("ACCEPT_IB_DEVICE");
      want != nullptr && *want != '\0') {
    const std::string target = want;
    for (std::size_t i = 0; i < env.rdma_devices.size(); ++i) {
      if (env.rdma_devices[i].device == target) {
        if (i != 0) std::rotate(env.rdma_devices.begin(),
                                env.rdma_devices.begin() + i,
                                env.rdma_devices.begin() + i + 1);
        break;
      }
    }
  }

  // Vendor/prefix come from the FIRST device's name; no devices -> unknown.
  if (!env.rdma_devices.empty()) {
    vendor_from_device(env.rdma_devices.front().device, env.vendor,
                       env.rdma_prefix);
  } else {
    env.vendor = "unknown";
  }

  // --- ROCm version (best-effort) ----------------------------------------
  env.rocm_version = read_first_line("/opt/rocm/.info/version");

  // --- GPU devices + count (KFD topology, dep-free sysfs) -------------------
  env.gpu_count = detect_gpu_devices(env.gpu_devices);

  // --- GPU↔NIC affinity (nearest PCIe BDF prefix) -------------------------
  env.gpu_nic_affinity = compute_affinity(env.gpu_devices, env.rdma_devices);

  // --- kernel (uname) ----------------------------------------------------
  struct utsname uts;
  if (::uname(&uts) == 0) env.kernel = uts.release;

  // --- os (PRETTY_NAME) --------------------------------------------------
  env.os = parse_pretty_name(read_file("/etc/os-release"));

  // --- amdgpu KMD driver version (best-effort sysfs; the fleet-comparable
  // "driver" for M9 predicates — distinct from the HIP runtime / rocm version) --
  env.driver_version = read_first_line("/sys/module/amdgpu/version");

  // --- capability symbol probes (dlopen/dlsym) ---------------------------
  probe_symbols(env);

  // --- optional compile-time enrichment (no-ops unless built in) ----------
  enrich_ibverbs(env);  // vendor_id/vendor_part_id (ACCEPT_HAVE_IBVERBS)
  enrich_hip(env);      // gpu_model only (ACCEPT_HAVE_HIP)

  return env;
}

const EnvironmentInfo& environment() {
  static const EnvironmentInfo env = detect();
  return env;
}

std::string EnvironmentInfo::to_kv() const {
  std::ostringstream out;
  out << "vendor=" << kv_safe(vendor) << "\n";
  out << "rdma_prefix=" << kv_safe(rdma_prefix) << "\n";
  out << "rocm_version=" << kv_safe(rocm_version) << "\n";
  out << "gpu_count=" << TestResult::int_to_chars(gpu_count) << "\n";
  out << "kernel=" << kv_safe(kernel) << "\n";
  out << "os=" << kv_safe(os) << "\n";
  out << "gpu_model=" << kv_safe(gpu_model) << "\n";
  out << "driver_version=" << kv_safe(driver_version) << "\n";
  // Symbol probes: one `sym.<name>=true|false` line each, in map (sorted) order.
  for (const auto& [name, present] : symbols) {
    out << "sym." << kv_safe(name) << "=" << (present ? "true" : "false") << "\n";
  }
  for (std::size_t i = 0; i < rdma_devices.size(); ++i) {
    const RdmaDevice& d = rdma_devices[i];
    const std::string p = "rdma." + std::to_string(i) + ".";
    out << p << "device=" << kv_safe(d.device) << "\n";
    out << p << "netdev=" << kv_safe(d.netdev) << "\n";
    out << p << "fw_ver=" << kv_safe(d.fw_ver) << "\n";
    out << p << "board_id=" << kv_safe(d.board_id) << "\n";
    out << p << "node_guid=" << kv_safe(d.node_guid) << "\n";
    out << p << "gid_index=" << TestResult::int_to_chars(d.gid_index) << "\n";
    out << p << "gid_type=" << kv_safe(d.gid_type) << "\n";
    out << p << "vendor_id=" << TestResult::int_to_chars(d.vendor_id) << "\n";
    out << p << "vendor_part_id=" << TestResult::int_to_chars(d.vendor_part_id)
        << "\n";
  }
  for (std::size_t i = 0; i < gpu_devices.size(); ++i) {
    const GpuDevice& g = gpu_devices[i];
    const std::string p = "gpu." + std::to_string(i) + ".";
    out << p << "ordinal=" << TestResult::int_to_chars(g.ordinal) << "\n";
    out << p << "bdf=" << kv_safe(g.bdf) << "\n";
    out << p << "numa_node=" << TestResult::int_to_chars(g.numa_node) << "\n";
    out << p << "arch=" << kv_safe(g.arch) << "\n";
  }
  for (std::size_t i = 0; i < gpu_nic_affinity.size(); ++i) {
    const GpuNicAffinity& a = gpu_nic_affinity[i];
    const std::string p = "affinity." + std::to_string(i) + ".";
    out << p << "gpu_ordinal=" << TestResult::int_to_chars(a.gpu_ordinal) << "\n";
    out << p << "nearest_nic=" << kv_safe(a.nearest_nic) << "\n";
  }
  return out.str();
}

EnvironmentInfo EnvironmentInfo::from_kv(const std::vector<std::string>& lines) {
  EnvironmentInfo env;
  // Reassemble indexed arrays into sparse maps keyed by index, then flatten in
  // order. Using a map (not a resized vector) bounds memory by the number of
  // lines actually present — a hostile/corrupt `@env` stream (which crosses the
  // ssh trust boundary) cannot force a huge allocation via a large `<i>` index.
  std::map<std::size_t, RdmaDevice>     devmap;
  std::map<std::size_t, GpuDevice>      gpumap;
  std::map<std::size_t, GpuNicAffinity> affmap;

  auto to_int = [](const std::string& s, int fallback) {
    try {
      std::size_t pos = 0;
      const int v = std::stoi(s, &pos);
      if (pos == s.size()) return v;
    } catch (...) {
    }
    return fallback;
  };

  for (const std::string& raw : lines) {
    std::string line = raw;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const std::size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = line.substr(0, eq);
    const std::string val = line.substr(eq + 1);  // verbatim remainder

    if (key == "vendor")            env.vendor = val;
    else if (key == "rdma_prefix")  env.rdma_prefix = val;
    else if (key == "rocm_version") env.rocm_version = val;
    else if (key == "gpu_count")    env.gpu_count = to_int(val, 0);
    else if (key == "kernel")       env.kernel = val;
    else if (key == "os")           env.os = val;
    else if (key == "gpu_model")    env.gpu_model = val;
    else if (key == "driver_version") env.driver_version = val;
    else if (key.compare(0, 4, "sym.") == 0) {
      // sym.<name>=true|false
      const std::string name = key.substr(4);
      if (!name.empty()) env.symbols[name] = (val == "true");
    }
    else if (key.compare(0, 4, "gpu.") == 0) {
      const std::size_t dot = key.find('.', 4);
      if (dot == std::string::npos) continue;
      const std::string idx_str = key.substr(4, dot - 4);
      std::size_t idx = 0;
      try {
        std::size_t pos = 0;
        idx = static_cast<std::size_t>(std::stoul(idx_str, &pos));
        if (pos != idx_str.size()) continue;
      } catch (...) { continue; }
      const std::string field = key.substr(dot + 1);
      GpuDevice& g = gpumap[idx];
      if (field == "ordinal")        g.ordinal = to_int(val, 0);
      else if (field == "bdf")       g.bdf = val;
      else if (field == "numa_node") g.numa_node = to_int(val, -1);
      else if (field == "arch")      g.arch = val;
    }
    else if (key.compare(0, 9, "affinity.") == 0) {
      const std::size_t dot = key.find('.', 9);
      if (dot == std::string::npos) continue;
      const std::string idx_str = key.substr(9, dot - 9);
      std::size_t idx = 0;
      try {
        std::size_t pos = 0;
        idx = static_cast<std::size_t>(std::stoul(idx_str, &pos));
        if (pos != idx_str.size()) continue;
      } catch (...) { continue; }
      const std::string field = key.substr(dot + 1);
      GpuNicAffinity& a = affmap[idx];
      if (field == "gpu_ordinal")      a.gpu_ordinal = to_int(val, 0);
      else if (field == "nearest_nic") a.nearest_nic = val;
    }
    else if (key.compare(0, 5, "rdma.") == 0) {
      // rdma.<i>.<field>
      const std::size_t dot = key.find('.', 5);
      if (dot == std::string::npos) continue;
      const std::string idx_str = key.substr(5, dot - 5);
      std::size_t idx = 0;
      try {
        std::size_t pos = 0;
        idx = static_cast<std::size_t>(std::stoul(idx_str, &pos));
        if (pos != idx_str.size()) continue;
      } catch (...) {
        continue;
      }
      const std::string field = key.substr(dot + 1);
      RdmaDevice& d = devmap[idx];
      if (field == "device")         d.device = val;
      else if (field == "netdev")    d.netdev = val;
      else if (field == "fw_ver")    d.fw_ver = val;
      else if (field == "board_id")  d.board_id = val;
      else if (field == "node_guid") d.node_guid = val;
      else if (field == "gid_index") d.gid_index = to_int(val, -1);
      else if (field == "gid_type")  d.gid_type = val;
      else if (field == "vendor_id") d.vendor_id = to_int(val, -1);
      else if (field == "vendor_part_id") d.vendor_part_id = to_int(val, -1);
      // unknown rdma fields ignored
    }
    // unknown top-level keys ignored (forward-compat)
  }

  env.rdma_devices.reserve(devmap.size());
  for (auto& [idx, d] : devmap) {
    (void)idx;
    env.rdma_devices.push_back(std::move(d));
  }
  env.gpu_devices.reserve(gpumap.size());
  for (auto& [idx, g] : gpumap) {
    (void)idx;
    env.gpu_devices.push_back(std::move(g));
  }
  env.gpu_nic_affinity.reserve(affmap.size());
  for (auto& [idx, a] : affmap) {
    (void)idx;
    env.gpu_nic_affinity.push_back(std::move(a));
  }
  return env;
}

}  // namespace acceptance
