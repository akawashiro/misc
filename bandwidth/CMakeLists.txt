cmake_minimum_required(VERSION 3.20)
project(bandwidth LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(abseil-cpp)

add_executable(memcpy memcpy.cc)
target_link_libraries(memcpy absl::log absl::base absl::synchronization
                      absl::log_initialize)
add_executable(memcpy_mt memcpy_mt.cc)
target_link_libraries(memcpy_mt absl::log absl::base absl::synchronization
                      absl::log_initialize)
add_executable(udp udp.cc)
target_link_libraries(udp absl::log absl::base absl::synchronization
                      absl::log_initialize)
add_executable(uds uds.cc)
target_link_libraries(uds absl::log absl::base absl::synchronization
                      absl::log_initialize)
add_executable(tcp tcp.cc)
target_link_libraries(tcp absl::log absl::base absl::synchronization
                      absl::log_initialize)
add_executable(pipe pipe.cc)
target_link_libraries(pipe absl::log absl::base absl::synchronization
                      absl::log_initialize)
add_executable(mmap mmap.cc)
target_link_libraries(mmap absl::log absl::base absl::synchronization
                      absl::log_initialize)
add_executable(shm shm.cc)
target_link_libraries(
  shm
  absl::log
  absl::base
  absl::synchronization
  absl::log_initialize
  rt
  pthread)
