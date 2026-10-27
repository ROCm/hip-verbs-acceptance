/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#include "common/Launch.hpp"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <system_error>

#include "common/Topology.hpp"

namespace acceptance {

namespace {

std::string short_name(const std::string& h) {
  return h.substr(0, h.find('.'));
}

bool same_address(const sockaddr* a, const sockaddr* b) {
  if (a->sa_family != b->sa_family) return false;
  if (a->sa_family == AF_INET) {
    sockaddr_in x{}, y{};
    std::memcpy(&x, a, sizeof(x));
    std::memcpy(&y, b, sizeof(y));
    return x.sin_addr.s_addr == y.sin_addr.s_addr;
  }
  if (a->sa_family == AF_INET6) {
    sockaddr_in6 x{}, y{};
    std::memcpy(&x, a, sizeof(x));
    std::memcpy(&y, b, sizeof(y));
    return std::memcmp(&x.sin6_addr, &y.sin6_addr, sizeof(x.sin6_addr)) == 0;
  }
  return false;
}

}  // namespace

bool host_is_local(const std::string& host, const std::string& self_hostname) {
  if (host == "localhost" || host == "127.0.0.1") return true;
  if (self_hostname.empty()) return false;
  if (host == self_hostname) return true;
  // "n0" vs "n0.cluster.example" in either direction. Comparing short names
  // only when one side is short keeps two different FQDNs that share a first
  // label ("n0.a" vs "n0.b") apart.
  const bool host_short = host.find('.') == std::string::npos;
  const bool self_short = self_hostname.find('.') == std::string::npos;
  if (host_short || self_short) return short_name(host) == short_name(self_hostname);
  return false;
}

bool host_has_local_address(const std::string& host) {
  if (host.empty()) return false;
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  if (::getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0) return false;
  ifaddrs* ifs = nullptr;
  if (::getifaddrs(&ifs) != 0) {
    ::freeaddrinfo(res);
    return false;
  }
  bool found = false;
  for (const addrinfo* a = res; a != nullptr && !found; a = a->ai_next)
    for (const ifaddrs* i = ifs; i != nullptr && !found; i = i->ifa_next)
      found = i->ifa_addr != nullptr && same_address(a->ai_addr, i->ifa_addr);
  ::freeifaddrs(ifs);
  ::freeaddrinfo(res);
  return found;
}

int effective_ppn(int np, int ppn, std::size_t num_hosts) {
  if (ppn > 0) return ppn;
  const int h = num_hosts == 0 ? 1 : static_cast<int>(num_hosts);
  return (np + h - 1) / h;
}

std::string validate_launch(int np, bool np_explicit, int ppn,
                            const std::vector<std::string>& hosts) {
  // Any rank count runs the single-rank tests on every rank; the two-rank
  // tests run only when there are exactly 2 ranks.
  if (np < 1) return "-n must be at least 1 (got " + std::to_string(np) + ")";
  std::set<std::string> seen;
  for (const std::string& h : hosts) {
    if (!seen.insert(h).second)
      return "-hosts lists '" + h + "' more than once; use -ppn to put "
             "several ranks on one host";
  }
  // -n defaults to 1, so "-hosts a,b" alone would quietly run one rank on a.
  if (hosts.size() > 1 && !np_explicit)
    return "-hosts lists " + std::to_string(hosts.size()) +
           " hosts but -n was not given (it defaults to 1); pass -n";
  const int h = hosts.empty() ? 1 : static_cast<int>(hosts.size());
  const int p = effective_ppn(np, ppn, hosts.size());
  if (static_cast<long>(p) * h < np)
    return "-n " + std::to_string(np) + " does not fit in -ppn " +
           std::to_string(p) + " x " + std::to_string(h) + " host(s)";
  return "";
}

std::vector<RankPlacement> plan_ranks(int np, int ppn,
                                      const std::vector<std::string>& hosts,
                                      const std::string& self_hostname) {
  const std::vector<std::string> hs =
      hosts.empty() ? std::vector<std::string>{"localhost"} : hosts;
  const int p = effective_ppn(np, ppn, hs.size());
  std::vector<RankPlacement> out;
  out.reserve(static_cast<std::size_t>(np));
  for (int r = 0; r < np; ++r) {
    RankPlacement pl;
    pl.rank = r;
    pl.local_rank = r % p;
    pl.host = hs[static_cast<std::size_t>(r / p)];
    pl.is_local = host_is_local(pl.host, self_hostname);
    out.push_back(std::move(pl));
  }
  return out;
}

void mark_local_addresses(std::vector<RankPlacement>& plan) {
  std::map<std::string, bool> local;
  for (RankPlacement& p : plan) {
    if (p.is_local) continue;
    auto it = local.find(p.host);
    if (it == local.end())
      it = local.emplace(p.host, host_has_local_address(p.host)).first;
    p.is_local = it->second;
  }
}

std::vector<std::string> build_worker_args(const WorkerParams& p,
                                           const RankPlacement& placement,
                                           int size,
                                           const std::string& rendezvous,
                                           const std::string& vendor) {
  std::vector<std::string> a = {
      "--worker",
      "--rank", std::to_string(placement.rank),
      "--local-rank", std::to_string(placement.local_rank),
      "--size", std::to_string(size),
      "--rendezvous", rendezvous,
      "--results-out", "-",
      "--vendor", vendor,
  };
  auto add = [&a](const char* flag, const std::string& v) {
    if (v.empty()) return;
    a.push_back(flag);
    a.push_back(v);
  };
  if (!p.policy.empty()) {
    std::error_code ec;
    const std::string abs = std::filesystem::absolute(p.policy, ec).string();
    add("--policy", ec ? p.policy : abs);
  }
  add("--filter", p.filter);
  add("--filter-class", p.filter_class);
  add("--filter-category", p.filter_category);
  if (p.hide_na) a.push_back("--hide-na");
  add("--kind", p.kind);
  if (p.isolate_explicit) add("--isolate", p.isolate);
  add("--gid", p.gid);
  add("--hca", topology::get_rank_entry(p.hca, placement.local_rank));
  add("--device", topology::get_rank_entry(p.device, placement.local_rank));
  if (p.deadline_sec > 0) add("--deadline-sec", std::to_string(p.deadline_sec));
  return a;
}

}  // namespace acceptance
