cmake_minimum_required(VERSION 3.16)
project(sloader)

enable_testing()
set(CMAKE_CXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

add_library(hello SHARED hello.cc)
SET_TARGET_PROPERTIES(hello PROPERTIES PREFIX "")
target_compile_options(hello PRIVATE -shared -fPIC -I/usr/include/python3.10 -Wall -Wextra -Wno-unused-parameter -O2 -g)
