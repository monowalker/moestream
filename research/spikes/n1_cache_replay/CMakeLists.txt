cmake_minimum_required(VERSION 3.20)
project(moestream_n1_replay CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# The product code lives in llamapatch/. Reference it here; never copy it.
add_library(ms_core ../../llamapatch/expert_cache.cpp)
target_include_directories(ms_core PUBLIC ../../llamapatch)
add_executable(replay replay.cpp)
target_link_libraries(replay PRIVATE ms_core)
