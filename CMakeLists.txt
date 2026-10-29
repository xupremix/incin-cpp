cmake_minimum_required(VERSION 3.18)

project(incin-cpp VERSION 0.1.0 LANGUAGES CXX)

option(INCIN_CPP_BUILD_TESTS "Build the incin-cpp test suite" ON)
option(INCIN_CPP_BUILD_EXAMPLES "Build the incin-cpp examples" ON)

# Prefer a libtorch checkout in the repository root when the caller did not
# provide one explicitly.
if(NOT DEFINED CMAKE_PREFIX_PATH AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
  set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
endif()

find_package(Torch REQUIRED)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(incin_cpp INTERFACE)
add_library(incin::incin ALIAS incin_cpp)
set_target_properties(incin_cpp PROPERTIES EXPORT_NAME incin)

target_include_directories(
  incin_cpp
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(incin_cpp INTERFACE cxx_std_17)
target_compile_options(incin_cpp INTERFACE ${TORCH_CXX_FLAGS})

# Link the imported `torch` target rather than the expanded TORCH_LIBRARIES
# list. The latter contains absolute paths to the libtorch/CUDA libraries of
# the machine that built incin-cpp, which makes the installed package
# non-relocatable. The `torch` target is re-created by find_dependency(Torch)
# on the consumer machine and already propagates include directories, compile
# requirements, and the transitive link interface.
target_link_libraries(incin_cpp INTERFACE torch)

if(INCIN_CPP_BUILD_EXAMPLES)
  add_executable(incin_example examples/matmul.cpp)
  target_link_libraries(incin_example PRIVATE incin::incin)
endif()

if(INCIN_CPP_BUILD_TESTS)
  enable_testing()
  add_executable(incin_cpp_tests tests/tensor_test.cpp)
  target_link_libraries(incin_cpp_tests PRIVATE incin::incin)
  add_test(NAME incin_cpp_tests COMMAND incin_cpp_tests)

  # Compile-fail tests. Each fixture must be rejected and the compiler output
  # must contain the diagnostic listed after the `|`, so the test cannot pass
  # for an unrelated reason.
  set(
    INCIN_CPP_COMPILE_FAIL_TESTS
    "matmul_inner_mismatch|the last dimension of the left operand must match"
    "transpose_duplicate|The permutation cannot contain duplicates"
    "reshape_element_count|reshape must preserve the number of elements"
    "arange_length|the shape must match the arange length"
    "narrow_past_end|TensorRange::END is past the end of the dimension"
  )
  foreach(entry IN LISTS INCIN_CPP_COMPILE_FAIL_TESTS)
    string(REPLACE "|" ";" fields "${entry}")
    list(GET fields 0 fixture)
    list(GET fields 1 expected)
    add_executable(incin_compile_fail_${fixture} EXCLUDE_FROM_ALL
      tests/compile_fail/${fixture}.cpp)
    target_link_libraries(incin_compile_fail_${fixture} PRIVATE incin::incin)
    add_test(
      NAME compile_fail_${fixture}
      COMMAND
        ${CMAKE_COMMAND}
        -DBUILD_DIR=${CMAKE_BINARY_DIR}
        -DTARGET=incin_compile_fail_${fixture}
        -DEXPECTED=${expected}
        -DCONFIG=$<CONFIG>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/expect_compile_failure.cmake
    )
    set_tests_properties(compile_fail_${fixture} PROPERTIES
      RUN_SERIAL TRUE
      TIMEOUT 300)
  endforeach()
endif()

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS incin_cpp EXPORT incin-cpp-targets)
install(
  EXPORT incin-cpp-targets
  NAMESPACE incin::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/incin-cpp
)

configure_package_config_file(
  cmake/incin-cpp-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/incin-cpp-config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/incin-cpp
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/incin-cpp-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/incin-cpp-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/incin-cpp-config-version.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/incin-cpp
)
