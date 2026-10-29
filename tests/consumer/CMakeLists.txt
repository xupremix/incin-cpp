cmake_minimum_required(VERSION 3.18)

project(incin-cpp-consumer LANGUAGES CXX)

# This project is configured against an installed incin-cpp package, not the
# source tree. It is used by CI to verify that the installed CMake package is
# relocatable and that find_package(incin-cpp) pulls in libtorch correctly.
find_package(incin-cpp REQUIRED)

add_executable(incin_consumer main.cpp)
target_link_libraries(incin_consumer PRIVATE incin::incin)
