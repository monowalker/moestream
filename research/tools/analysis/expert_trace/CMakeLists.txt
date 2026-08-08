cmake_minimum_required(VERSION 3.20)
project(moestream_expert_trace CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_ROOT "$ENV{LLAMA_ROOT}" CACHE PATH "llama.cpp source root")
if(NOT LLAMA_ROOT)
    message(FATAL_ERROR "LLAMA_ROOT is not set")
endif()

add_executable(expert_trace main.cpp)

target_include_directories(expert_trace PRIVATE
    ${LLAMA_ROOT}/include
    ${LLAMA_ROOT}/ggml/include
)

set(LIBDIRS
    ${LLAMA_ROOT}/build/bin
    ${LLAMA_ROOT}/build/src
    ${LLAMA_ROOT}/build/ggml/src
    ${LLAMA_ROOT}/build/ggml/src/ggml-vulkan
)

find_library(LLAMA_LIB     llama       PATHS ${LIBDIRS} NO_DEFAULT_PATH)
find_library(GGML_LIB      ggml        PATHS ${LIBDIRS} NO_DEFAULT_PATH)
find_library(GGML_BASE_LIB ggml-base   PATHS ${LIBDIRS} NO_DEFAULT_PATH)
find_library(GGML_CPU_LIB  ggml-cpu    PATHS ${LIBDIRS} NO_DEFAULT_PATH)
find_library(GGML_VK_LIB   ggml-vulkan PATHS ${LIBDIRS} NO_DEFAULT_PATH)

message(STATUS "llama     : ${LLAMA_LIB}")
message(STATUS "ggml      : ${GGML_LIB}")
message(STATUS "ggml-vk   : ${GGML_VK_LIB}")

target_link_libraries(expert_trace PRIVATE ${LLAMA_LIB} ${GGML_LIB} ${GGML_CPU_LIB} ${GGML_BASE_LIB})
if(GGML_VK_LIB)
    target_link_libraries(expert_trace PRIVATE ${GGML_VK_LIB})
endif()

set_target_properties(expert_trace PROPERTIES BUILD_RPATH "${LIBDIRS}")
