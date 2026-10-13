/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#include "common/ResultIO.hpp"

#include <sstream>

namespace acceptance {

namespace result_io {

std::string serialize(int rank, const std::vector<TestResult>& results) {
  std::ostringstream os;
  os << "rank=" << TestResult::int_to_chars(rank) << "\n";
  for (const TestResult& r : results) {
    os << "@test\n";
    r.to_kv(os);
  }
  return os.str();
}

std::string serialize(int rank, const std::vector<TestResult>& results,
                      const EnvironmentInfo& env) {
  std::ostringstream os;
  os << "rank=" << TestResult::int_to_chars(rank) << "\n";
  // The @env block rides right after the rank header, before any @test block.
  os << "@env\n";
  os << env.to_kv();
  for (const TestResult& r : results) {
    os << "@test\n";
    r.to_kv(os);
  }
  return os.str();
}

bool parse(const std::string& text, int& out_rank,
           std::vector<TestResult>& out_results, EnvironmentInfo& out_env) {
  out_results.clear();
  out_rank = 0;
  out_env = EnvironmentInfo{};

  // Split into lines (CRLF tolerant — TestResult::from_kv strips a trailing
  // '\r' per line; we keep raw lines here and let the record parser handle it).
  std::vector<std::string> lines;
  {
    std::istringstream is(text);
    std::string line;
    while (std::getline(is, line)) lines.push_back(line);
  }

  bool have_rank = false;
  std::vector<std::string> block;     // current @test block
  std::vector<std::string> env_lines; // accumulated @env block
  // Section state machine: Header (before any @env/@test) -> Env -> Test.
  enum class Section { Header, Env, Test };
  Section section = Section::Header;

  auto flush_block = [&]() {
    if (section == Section::Test) {
      out_results.push_back(TestResult::from_kv(block));
      block.clear();
    }
  };

  for (std::string line : lines) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line == "@env") {
      // @env is only meaningful in the header region; ignore a stray one inside
      // the test section (defensive — it would not appear in well-formed output).
      if (section == Section::Header) {
        section = Section::Env;
        continue;
      }
      // Otherwise fall through to treat it like any non-separator line below.
    }
    if (line == "@test") {
      flush_block();
      section = Section::Test;
      continue;
    }
    if (section == Section::Header) {
      // Header region (before the first @env/@test): look for rank=.
      const std::size_t eq = line.find('=');
      if (eq != std::string::npos && line.substr(0, eq) == "rank") {
        const std::string v = line.substr(eq + 1);
        // A malformed rank value (non-integer or trailing junk) means the output
        // is structurally invalid — fail the parse rather than silently coercing
        // to rank 0, which would misattribute results to the wrong rank.
        try {
          std::size_t pos = 0;
          out_rank = std::stoi(v, &pos);
          if (pos != v.size()) return false;  // trailing junk after the integer
        } catch (...) {
          return false;  // empty or non-integer rank
        }
        have_rank = true;
      }
      continue;
    }
    if (section == Section::Env) {
      env_lines.push_back(line);
      continue;
    }
    // Inside a @test block: accumulate the record's key=value lines.
    block.push_back(line);
  }
  flush_block();

  if (!env_lines.empty()) out_env = EnvironmentInfo::from_kv(env_lines);

  return have_rank;
}

bool parse(const std::string& text, int& out_rank,
           std::vector<TestResult>& out_results) {
  EnvironmentInfo ignored;
  return parse(text, out_rank, out_results, ignored);
}

}  // namespace result_io

}  // namespace acceptance
