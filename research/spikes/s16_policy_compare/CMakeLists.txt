cmake_minimum_required(VERSION 3.20)
project(s16_policy_compare CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Link the product's S3-FIFO directly; do not reimplement it here.
add_executable(s16_policy_compare main.cpp ../../llamapatch/expert_cache.cpp)
target_include_directories(s16_policy_compare PRIVATE ../../llamapatch)
