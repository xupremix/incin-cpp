# Compile-fail test driver.
#
# Builds one fixture target and succeeds only when the build fails and the
# compiler output contains the expected diagnostic. Checking the message keeps
# the test from passing for unrelated reasons (missing headers, linker errors,
# or a different static assertion).

foreach(required BUILD_DIR TARGET)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "expect_compile_failure.cmake: ${required} is required")
  endif()
endforeach()

set(build_command "${CMAKE_COMMAND}" --build "${BUILD_DIR}" --target "${TARGET}")
if(DEFINED CONFIG AND NOT CONFIG STREQUAL "")
  list(APPEND build_command --config "${CONFIG}")
endif()

execute_process(
  COMMAND ${build_command}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

set(output "${stdout}\n${stderr}")

if(result EQUAL 0)
  message(FATAL_ERROR
    "Expected target '${TARGET}' to fail compilation, but it succeeded.")
endif()

if(DEFINED EXPECTED AND NOT EXPECTED STREQUAL "")
  string(FIND "${output}" "${EXPECTED}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR
      "Target '${TARGET}' failed, but not with the expected diagnostic.\n"
      "Expected to find: ${EXPECTED}\n\nCompiler output:\n${output}")
  endif()
endif()

message(STATUS "Target '${TARGET}' was rejected as expected.")
