cmake_minimum_required(VERSION 3.20)
project(moestream_spike_s1 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ggml is prebuilt inside the image (docker/dev/Dockerfile.dev)
set(GGML_ROOT "$ENV{GGML_ROOT}" CACHE PATH "ggml source root")
if(NOT GGML_ROOT)
    message(FATAL_ERROR "GGML_ROOT is not set")
endif()

add_executable(s1_real_expert_stream main.c)

target_include_directories(s1_real_expert_stream PRIVATE
    ${GGML_ROOT}/include
)

find_library(GGML_LIB      ggml      PATHS ${GGML_ROOT}/build/src ${GGML_ROOT}/build/bin NO_DEFAULT_PATH)
find_library(GGML_BASE_LIB ggml-base PATHS ${GGML_ROOT}/build/src ${GGML_ROOT}/build/bin NO_DEFAULT_PATH)
find_library(GGML_CPU_LIB  ggml-cpu  PATHS ${GGML_ROOT}/build/src ${GGML_ROOT}/build/bin ${GGML_ROOT}/build/src/ggml-cpu NO_DEFAULT_PATH)

message(STATUS "ggml      : ${GGML_LIB}")
message(STATUS "ggml-base : ${GGML_BASE_LIB}")
message(STATUS "ggml-cpu  : ${GGML_CPU_LIB}")

target_link_libraries(s1_real_expert_stream PRIVATE
    ${GGML_LIB} ${GGML_CPU_LIB} ${GGML_BASE_LIB} m pthread
)

set_target_properties(s1_real_expert_stream PROPERTIES
    BUILD_RPATH "${GGML_ROOT}/build/src;${GGML_ROOT}/build/bin"
)
