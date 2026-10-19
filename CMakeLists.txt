# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20)
project(hip-verbs-acceptance CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Optional -DACCEPT_WERROR=ON for CI.
option(ACCEPT_WERROR "Treat compiler warnings as errors" OFF)
set(ACCEPT_WARN_FLAGS -Wall -Wextra)
if(ACCEPT_WERROR)
  list(APPEND ACCEPT_WARN_FLAGS -Werror)
endif()

# Optional backends. Off by default for harness work (Step 0..4); later steps
# flip these ON to pull in ibverbs / HIP. When ON we resolve the dependency and
# define a feature macro the sources gate on.
option(ACCEPT_WITH_IBVERBS "Build with libibverbs support" OFF)
option(ACCEPT_WITH_HIP     "Build with HIP support"        OFF)

# Synthetic test sources are listed as DIRECT target sources (not a static
# library) so their static-init registrations are never dropped as unreferenced.
add_executable(hva
  main.cc
  wrapper/DynLib.cpp
  wrapper/hip/api_wrapper.cpp
  wrapper/hip/hip_api_wrapper.cpp
  wrapper/hip/hsa_api_wrapper.cpp
  wrapper/hip/hiprtc_api_wrapper.cpp
  common/TestRegistry.cpp
  common/Runner.cpp
  common/Policy.cpp
  common/Reporter.cpp
  common/PerfSummary.cpp
  common/ResultIO.cpp
  common/Capability.cpp
  common/SockCompat.cpp
  common/SocketBootstrap.cpp
  common/SshLauncher.cpp
  synthetic/basic.cc
  synthetic/failures.cc
  synthetic/classified.cc
  synthetic/two_rank.cc
)

# No third-party libraries (nlohmann removed). All formats are
# hand-emitted / hand-parsed. project root -> #include "common/Types.hpp".
# vendor/rocm-headers provides the HIP/HSA *declarations* for HipApi.hpp; no ROCm
# library is linked (symbols load via dlopen/dlsym at init()).
target_include_directories(hva PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Build-time version info: common/Version.hpp holds the hand-maintained
# semantic version; generated/common/version_info.hpp holds the git SHA
# (kGitSha), re-resolved on EVERY build (not just at configure time) by
# cmake/GenerateVersion.cmake -- see that file for why this is a custom
# target and not a configure_file()/execute_process() at configure time.
set(ACCEPT_GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)
set(ACCEPT_VERSION_HEADER ${ACCEPT_GENERATED_DIR}/common/version_info.hpp)
add_custom_target(generate_version ALL
  COMMAND ${CMAKE_COMMAND}
          -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}
          -DOUT_FILE=${ACCEPT_VERSION_HEADER}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateVersion.cmake
  BYPRODUCTS ${ACCEPT_VERSION_HEADER}
  COMMENT "Resolving git SHA for build-time version header"
  VERBATIM
)
target_include_directories(hva PRIVATE ${ACCEPT_GENERATED_DIR})
add_dependencies(hva generate_version)

# Vendored third-party declarations are marked SYSTEM so their own warnings (e.g.
# -Wdeprecated-enum-enum-conversion in rdma-core verbs.h) don't pollute our
# -Wall -Wextra build. Only vendored dirs are SYSTEM; project root stays normal.
target_include_directories(hva SYSTEM PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vendor/rocm-headers
)

target_compile_options(hva PRIVATE ${ACCEPT_WARN_FLAGS})

# Hard deps for the harness: dynamic loader + threads.
find_package(Threads REQUIRED)
target_link_libraries(hva PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)

if(ACCEPT_WITH_IBVERBS)
  # Login-node build model (mirrors ACCEPT_WITH_HIP): the verbs tests compile as
  # PLAIN C++ against the vendored rdma-core *declarations*
  # (vendor/rdma-core-headers) and resolve every ibv_* symbol at run time through
  # the dlopen/dlsym wrapper (wrapper/verbs/*). No libibverbs is linked -- only
  # -ldl (already linked above) -- so the binary BUILDS on a node with no
  # libibverbs-dev and runs where the runtime libibverbs.so.1 is present.
  target_include_directories(hva SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/rdma-core-headers)
  target_compile_definitions(hva PRIVATE ACCEPT_HAVE_IBVERBS)
  target_sources(hva PRIVATE
    wrapper/verbs/ibv_api_wrapper.cpp
    wrapper/verbs/IonicDvApi.cpp
    verbs/foundation.cc
    verbs/RcConn.cpp
    verbs/UdConn.cpp
    verbs/data_path.cc
    verbs/connection.cc
    verbs/recovery.cc
  )
endif()

if(ACCEPT_WITH_HIP)
  # Login-node build model: HIP acceptance tests compile as PLAIN C++ against the
  # vendored ROCm declarations (vendor/rocm-headers, already on the include path
  # above) and resolve the HIP/HSA runtime symbols at run time through the
  # dlopen/dlsym wrapper (HipApi.hpp + hip/wrapper/*). Consequences:
  #   * no enable_language(HIP), no HIP toolchain (hipcc/clang) required,
  #   * no find_package(hip), no hip::host / hsa-runtime64 linked,
  #   * only -ldl is linked (added unconditionally above).
  # This lets the whole suite BUILD on a node with no ROCm install; the binary is
  # then carried to a GPU node where the runtime libraries are present and the
  # wrapper's init() loads the symbols. Sources gate their bodies on ACCEPT_HAVE_HIP.
  target_compile_definitions(hva PRIVATE ACCEPT_HAVE_HIP)

  # Feature availability is no longer decided at build time: the vendored headers
  # always declare the newer batch APIs/types, so tests compile their REAL bodies
  # unconditionally and null-check the optional ops-table pointer at run time
  # (see hip_optional_fn_names). No ACCEPT_HAVE_HIP_<FEATURE> probe macros.

  # Migrated HIP test sources, compiled as CXX (default language). Device kernels
  # are compiled at run time by hiprtc (HipModule.cpp); <<<>>> launches have been
  # ported to the runtime module-launch path. The host-symbol launch APIs
  # (hipLaunchKernel(host)/hipGetFuncBySymbol/...) need statically registered
  # device code, provided by the vendored multi-arch object kernels_static.o plus
  # the __hipRegister* forwarding shim (hip_register_shim.cpp). Remaining graph /
  # IPC / HSA-direct units move over in following steps. Listed as DIRECT target
  # sources so their static-init test registrations are never dropped.
  target_sources(hva PRIVATE
    hip/utils/HipModule.cpp
    wrapper/hip/hip_register_shim.cpp
    hip/data_movement.cc
    hip/streams_events.cc
    hip/kernel_launch/kernel_launch.cc
    hip/kernel_launch/module_runtime.cc
    hip/graph_capture.cc
    hip/ipc.cc
    hip/host_memory_version.cc
    hip/memory_manager.cc
    hip/device_management.cc
    hip/hsa_interop.cc
  )

  # Vendored prebuilt multi-arch device kernels (host-symbol launch path). This is
  # a relocatable object built ONCE by hipcc (hip/kernel_launch/build_static_kernels.sh)
  # and committed; it is LINKED, not compiled, so the login-node build needs no HIP
  # toolchain. Its load-time registration reaches the runtime via the shim above.
  set(_accept_static_kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/hip/kernel_launch/prebuilt/kernels_static.o)
  set_source_files_properties(${_accept_static_kernels}
    PROPERTIES EXTERNAL_OBJECT TRUE GENERATED FALSE)
  target_sources(hva PRIVATE ${_accept_static_kernels})
endif()

# HIP + Verbs mixed data-path tests: require BOTH backends. GdrMem (HIP alloc x
# verbs registration) plus the mixed acceptance tests build only when
# ACCEPT_WITH_HIP and ACCEPT_WITH_IBVERBS are both ON; each source additionally
# gates its body on (ACCEPT_HAVE_HIP && ACCEPT_HAVE_IBVERBS). Listed as DIRECT
# target sources so their static-init test registrations are never dropped.
if(ACCEPT_WITH_HIP AND ACCEPT_WITH_IBVERBS)
  target_sources(hva PRIVATE
    verbs/GdrMem.cpp
    verbs/hip_mixed.cc
    verbs/hip_alloc_matrix.cc
    verbs/hip_coherence.cc
  )
endif()

# unit_tests — Reporter reduce/classify + Policy/ResultIO/SshLauncher units.
add_executable(unit_tests
  tests/unit_reduce.cc
  tests/unit_policy.cc
  tests/unit_resultio.cc
  tests/unit_ssh.cc
  tests/unit_perf.cc
  common/Reporter.cpp
  common/PerfSummary.cpp
  common/Policy.cpp
  common/ResultIO.cpp
  common/Capability.cpp
  common/SshLauncher.cpp
  common/TestRegistry.cpp
)
target_include_directories(unit_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${ACCEPT_GENERATED_DIR}
)
target_compile_options(unit_tests PRIVATE ${ACCEPT_WARN_FLAGS})
add_dependencies(unit_tests generate_version)

enable_testing()
add_test(NAME unit_tests COMMAND unit_tests)

# install() for --nodes same-path binary assumption. The policy CSVs are
# installed next to the binary (<prefix>/bin/policy/*.csv) to match the
# auto-policy lookup (<binary_dir>/policy/<gpu>_<nic>.csv, see
# resolve_policy_path() in main.cc), so a packaged install works without the
# caller also copying policy/ by hand.
include(GNUInstallDirs)
install(TARGETS hva RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY policy/ DESTINATION ${CMAKE_INSTALL_BINDIR}/policy
        FILES_MATCHING PATTERN "*.csv")
