cmake_minimum_required(VERSION 3.20)
project(hypic_compose_ascendc LANGUAGES CXX)

# Standalone, opt-in build: does not add an unvalidated kernel to the default extension.
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(RUN_MODE npu CACHE STRING "Ascend execution mode")
set(SOC_VERSION Ascend910B2 CACHE STRING "Exact target SoC")
set(ASCEND_CANN_PACKAGE_PATH "" CACHE PATH "CANN toolkit root")
if(NOT ASCEND_CANN_PACKAGE_PATH)
    message(FATAL_ERROR "Pass -DASCEND_CANN_PACKAGE_PATH=<toolkit root>")
endif()
string(TOLOWER "${SOC_VERSION}" HYPIC_SOC)
if(NOT HYPIC_SOC MATCHES "^ascend910b")
    message(FATAL_ERROR "Initial cached compose backend targets Ascend910B only")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND ${Python3_EXECUTABLE} -c
    "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND ${Python3_EXECUTABLE} -c
    "import torch_npu; print(torch_npu.__path__[0])"
    OUTPUT_VARIABLE TORCH_NPU_PATH OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_file(ASCENDC_CMAKE ascendc.cmake PATHS
    "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/ascendc_kernel_cmake"
    "${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake"
    NO_DEFAULT_PATH REQUIRED)
include(${ASCENDC_CMAKE})
ascendc_library(hypic_cached_kernel_v2 SHARED kernel.cpp)
ascendc_library(hypic_tree_kernel SHARED tree_kernel.cpp)
# The legacy generator rewrites __global__ to inline and loses __schedmode__.
# Build only the persistent kernel with CANN's native ASC language so batch
# scheduling survives compilation. Preserve the validated baseline toolchain.
include(ExternalProject)
set(HYPIC_SINGLE_BUILD "${CMAKE_CURRENT_BINARY_DIR}/native_single")
ExternalProject_Add(hypic_tree_single_build
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/native_single"
    BINARY_DIR "${HYPIC_SINGLE_BUILD}"
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
    BUILD_ALWAYS TRUE
    BUILD_BYPRODUCTS "${HYPIC_SINGLE_BUILD}/libhypic_tree_single_kernel.so"
    INSTALL_COMMAND "")
add_library(hypic_tree_single_kernel SHARED IMPORTED)
set_target_properties(hypic_tree_single_kernel PROPERTIES
    IMPORTED_LOCATION "${HYPIC_SINGLE_BUILD}/libhypic_tree_single_kernel.so")
add_dependencies(hypic_tree_single_kernel hypic_tree_single_build)
add_library(hypic_compose_ascendc SHARED binding.cpp)
set(HYPIC_OP_NAMESPACE hypic_ascendc CACHE STRING "Operator namespace for side-by-side benchmark builds")
if(NOT HYPIC_OP_NAMESPACE MATCHES "^[a-z][a-z0-9_]*$")
    message(FATAL_ERROR "Invalid HYPIC_OP_NAMESPACE")
endif()
target_compile_definitions(hypic_compose_ascendc PRIVATE HYPIC_OP_NAMESPACE=${HYPIC_OP_NAMESPACE})
target_include_directories(hypic_compose_ascendc PRIVATE
    "${TORCH_NPU_PATH}/include" "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_directories(hypic_compose_ascendc PRIVATE
    "${TORCH_NPU_PATH}/lib" "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(hypic_compose_ascendc PRIVATE hypic_cached_kernel_v2 hypic_tree_kernel
    hypic_tree_single_kernel
    ${TORCH_LIBRARIES} torch_npu ascendcl tiling_api register platform)
# CANN's legacy helper disables CMake RPATH generation globally.
target_link_options(hypic_compose_ascendc PRIVATE "-Wl,-rpath,$ORIGIN:$ORIGIN/lib:$ORIGIN/native_single")
