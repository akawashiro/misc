cmake_minimum_required(VERSION 3.20)
project(bandwidth LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(abseil-cpp)

set(BANDWIDTH_LIBS absl::log absl::base absl::synchronization
                     absl::log_initialize)

add_library(common common.cc)
target_link_libraries(common ${BANDWIDTH_LIBS})

set(BANDWIDTH_LIBS common ${BANDWIDTH_LIBS})

add_executable(memcpy memcpy.cc)
target_link_libraries(memcpy ${BANDWIDTH_LIBS})
add_executable(memcpy_mt memcpy_mt.cc)
target_link_libraries(memcpy_mt ${BANDWIDTH_LIBS})
add_executable(udp udp.cc)
target_link_libraries(udp ${BANDWIDTH_LIBS})
add_executable(uds uds.cc)
target_link_libraries(uds ${BANDWIDTH_LIBS})
add_executable(tcp tcp.cc)
target_link_libraries(tcp ${BANDWIDTH_LIBS})
add_executable(pipe pipe.cc)
target_link_libraries(pipe ${BANDWIDTH_LIBS})
add_executable(mmap mmap.cc)
target_link_libraries(mmap ${BANDWIDTH_LIBS})
add_executable(shm shm.cc)
target_link_libraries(shm ${BANDWIDTH_LIBS} rt pthread)
