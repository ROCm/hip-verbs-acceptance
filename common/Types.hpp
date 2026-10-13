/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

#pragma once

#include <string>

namespace acceptance {

// Rank-level outcome a test/EXEC child can produce.
// None = non-participating / passive (identity in the reduction lattice).
enum class Status { Pass, Fail, Skip, None };

// Final per-test outcome after reduction + classification (Reporter only).
// KnownFailure and Unclassified removed; Skip is now a final status.
enum class FinalStatus { Pass, Fail, Skip, Na };

enum class Topology { Single, Loopback, TwoNode };
enum class Kind     { Functional, Performance };

// Policy class taxonomy: Unclassified removed (unlisted tests are not
// run); Skip added (software-gated — would pass with a stack upgrade).
enum class Class    { MustPass, KnownFailure, Skip, Na };

// ---------------------------------------------------------------------------
// enum -> string (lowercase, used as JSON tokens)
// ---------------------------------------------------------------------------

inline const char* to_string(Status s) {
  switch (s) {
    case Status::Pass: return "pass";
    case Status::Fail: return "fail";
    case Status::Skip: return "skip";
    case Status::None: return "none";
  }
  return "none";
}

inline const char* to_string(FinalStatus s) {
  switch (s) {
    case FinalStatus::Pass: return "pass";
    case FinalStatus::Fail: return "fail";
    case FinalStatus::Skip: return "skip";
    case FinalStatus::Na:   return "na";
  }
  return "fail";
}

inline const char* to_string(Topology t) {
  switch (t) {
    case Topology::Single:   return "single";
    case Topology::Loopback: return "loopback";
    case Topology::TwoNode:  return "two_node";
  }
  return "single";
}

inline const char* to_string(Kind k) {
  switch (k) {
    case Kind::Functional:  return "functional";
    case Kind::Performance: return "performance";
  }
  return "functional";
}

inline const char* to_string(Class c) {
  switch (c) {
    case Class::MustPass:     return "must_pass";
    case Class::KnownFailure: return "known_failure";
    case Class::Skip:         return "skip";
    case Class::Na:           return "na";
  }
  return "na";
}

// ---------------------------------------------------------------------------
// string -> enum (JSON parsing). Unknown tokens map to a safe default.
// ---------------------------------------------------------------------------

inline Status status_from(const std::string& s) {
  if (s == "pass") return Status::Pass;
  if (s == "fail") return Status::Fail;
  if (s == "skip") return Status::Skip;
  return Status::None;  // "none" and anything unrecognized
}

// Lenient parse used for policy CSV rows. Unknown tokens fall through to Na so
// a stale token doesn't crash the load — but callers (parse_row) should WARN.
// Do NOT use for CLI validation: use class_is_valid() + explicit rejection there.
inline Class class_from(const std::string& s) {
  if (s == "must_pass")     return Class::MustPass;
  // REGRESSION was folded into MUST_PASS.
  if (s == "regression")    return Class::MustPass;
  // Both spellings accepted; only "known_failure" is emitted.
  if (s == "known_failure") return Class::KnownFailure;
  if (s == "known_broken")  return Class::KnownFailure;
  if (s == "skip")          return Class::Skip;
  if (s == "na")            return Class::Na;
  return Class::Na;  // unknown token; callers should have warned already
}

// True when `s` is a recognised class token (for CLI and policy validation).
inline bool class_is_valid(const std::string& s) {
  return s == "must_pass" || s == "known_failure" || s == "skip" || s == "na";
}

}  // namespace acceptance
