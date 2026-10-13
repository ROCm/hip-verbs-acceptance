/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// Failure-injection synthetic tests. These deliberately exercise every failure
// mode the Runner's watchdog + synthesis logic must contain:
//   crash (abort / SIGSEGV), hang (timeout), non-zero exit, corrupt result file.
// All are isolated by the Runner because their policy class is REGRESSION (see
// policy/<vendor>.csv), so they only ever run in a forked child — they must
// NEVER run in-process.
//
// Each maps to a synthesized Fail row (synth = crash | timeout | corrupt).

#include <cstdio>    // FILE, fprintf, fclose
#include <cstdlib>   // abort, std::_Exit, getenv
#include <string>
#include <unistd.h>  // pause, fork, getpid
#include <sys/types.h>

#include "common/TestRegistry.hpp"

namespace acceptance {

// abort() raises SIGABRT -> WIFSIGNALED in the parent -> synth="crash".
ACCEPT_TEST(syn_crash_abort, "synthetic", Topology::Single, Kind::Functional, 30) {
  (void)ctx;
  std::abort();  // never returns
  return fail("unreachable");
}

// Dereference a volatile null pointer to force SIGSEGV. `volatile` prevents the
// compiler from optimizing the load away or treating it as pure UB to delete.
ACCEPT_TEST(syn_crash_segv, "synthetic", Topology::Single, Kind::Functional, 30) {
  (void)ctx;
  volatile int* p = nullptr;
  return fail(std::to_string(*p));  // load from null -> SIGSEGV before fail()
}

// Block forever; must outlast its (short) timeout so the watchdog fires fast.
// timeout_sec = 2 keeps the watchdog-test runtime to ~2-4s.
ACCEPT_TEST(syn_hang, "synthetic", Topology::Single, Kind::Functional, 2) {
  (void)ctx;
  for (;;) ::pause();  // sleep until a signal; SIGTERM/SIGKILL from watchdog
  return fail("unreachable");
}

// Exit the process with code 7 WITHOUT writing a result file. Non-zero exit is
// authoritative -> synth="crash", exit_code=7.
ACCEPT_TEST(syn_exit_nonzero, "synthetic", Topology::Single, Kind::Functional, 30) {
  (void)ctx;
  std::_Exit(7);  // bypasses atexit/stdio flush; no result file written
  return fail("unreachable");
}

// Returns a normal pass, BUT the Runner launches this id with ACCEPT_SYN_CORRUPT
// set in the child env. do_exec() in main.cc honors that hook by writing garbage
// to the result file and exiting 0 -> exit-0-but-corrupt-file -> synth="corrupt".
// The test body itself cannot reach the result-file path, so the corruption is
// injected at the harness boundary; this is the cleanest deterministic way to
// drive the exit-0 corrupt-file path.
ACCEPT_TEST(syn_corrupt_result, "synthetic", Topology::Single, Kind::Functional, 30) {
  (void)ctx;
  return pass();
}

// syn_spawn_grandchild_hang — proves the watchdog's process-GROUP kill reaches a
// grandchild, not just the direct EXEC child. The test (running inside the
// isolated EXEC child) fork()s a grandchild that writes its OWN pid to the file
// named by $ACCEPT_GC_PIDFILE and then blocks forever; the test body then also
// blocks forever. Both hang past the (short) 2s timeout, so the Runner's
// watchdog escalates SIGTERM->SIGKILL to the whole group (kill(-pid,...)). The
// selftest then asserts the recorded grandchild pid is no longer alive — i.e.
// the group signal killed the grandchild too.
//
// Inert when $ACCEPT_GC_PIDFILE is unset: it then skips the grandchild fork and
// just hangs (still a clean timeout), so this test never depends on the env var
// being present to behave as a REGRESSION timeout.
ACCEPT_TEST(syn_spawn_grandchild_hang, "synthetic", Topology::Single,
            Kind::Functional, 2) {
  (void)ctx;
  const char* pidfile = ::getenv("ACCEPT_GC_PIDFILE");
  if (pidfile != nullptr && pidfile[0] != '\0') {
    const pid_t gc = ::fork();
    if (gc == 0) {
      // ---- grandchild: record own pid, then hang forever ----
      std::FILE* f = std::fopen(pidfile, "w");
      if (f != nullptr) {
        std::fprintf(f, "%ld\n", static_cast<long>(::getpid()));
        std::fclose(f);
      }
      for (;;) ::pause();  // only a signal stops us; group SIGKILL must reach us
      ::_exit(0);          // unreachable
    }
    // gc < 0 (fork failed) falls through: the test still hangs and times out.
  }
  for (;;) ::pause();  // parent (the test) hangs -> watchdog timeout
  return fail("unreachable");
}

}  // namespace acceptance
