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
target_link_libraries(incin_cpp INTERFACE "${TORCH_LIBRARIES}")

if(INCIN_CPP_BUILD_EXAMPLES)
  add_executable(incin_example examples/matmul.cpp)
  target_link_libraries(incin_example PRIVATE incin::incin)
endif()

if(INCIN_CPP_BUILD_TESTS)
  enable_testing()
  add_executable(incin_cpp_tests tests/tensor_test.cpp)
  target_link_libraries(incin_cpp_tests PRIVATE incin::incin)
  add_test(NAME incin_cpp_tests COMMAND incin_cpp_tests)
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
