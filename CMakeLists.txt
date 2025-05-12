cmake_minimum_required(VERSION 3.20)
project(eternia)

# -----------------------------------------------------------------------------
# Define Options
# -----------------------------------------------------------------------------
option(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" ON)
option(ETERNIA_ENABLE_CMAKE_DOTENV "Load environment variables from .env.cmake" OFF)

option(HERMES_MPICH "Specify that this a MPICH build" OFF)
option(HERMES_OPENMPI "Specify that this a OpenMPI build" OFF)

option(ETERNIA_ENABLE_CUDA "Enable CUDA support" OFF)
option(ETERNIA_ENABLE_ROCM "Enable ROCm support" OFF)

# Ensure CUDA and GDS on at same time
if(HERMES_ENABLE_NVIDIA_GDS_ADAPTER)
    set(ETERNIA_ENABLE_CUDA ON)
endif()

# A hack for spack to get dependencies
option(ETERNIA_NO_COMPILE "Don't compile the code" OFF)

if(ETERNIA_NO_COMPILE)
    install(
        FILES
        ${CMAKE_SOURCE_DIR}/.gitignore
        DESTINATION
        ${CMAKE_INSTALL_PREFIX}/.gitignore
    )
    return()
endif()

# ------------------------------------------------------------------------------
# DOTENV
# ------------------------------------------------------------------------------
if(ETERNIA_ENABLE_CMAKE_DOTENV)
    if(EXISTS ${CMAKE_SOURCE_DIR}/.env.cmake)
        include(.env.cmake)
    endif()
endif()

# -----------------------------------------------------------------------------
# Compiler Optimization
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("IN DEBUG MODE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")
    add_compile_definitions(HERMES_LOG_VERBOSITY=10)
else()
    message("IN RELEASE MODE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O3")
    add_compile_definitions(HERMES_LOG_VERBOSITY=1)
endif()

add_compile_options(-march=native -fomit-frame-pointer)

# -----------------------------------------------------------------------------
# Targets built within this project are exported at Install time for use
# by other projects.
# -----------------------------------------------------------------------------
if(NOT CHIMAERA_EXPORTED_TARGETS)
    set(CHIMAERA_EXPORTED_TARGETS "Eternia")
endif()

# ------------------------------------------------------------------------------
# Setup install and output Directories
# ------------------------------------------------------------------------------
if(NOT ETERNIA_INSTALL_BIN_DIR)
    set(ETERNIA_INSTALL_BIN_DIR ${CMAKE_INSTALL_PREFIX}/bin)
endif()

if(NOT ETERNIA_INSTALL_LIB_DIR)
    set(ETERNIA_INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)
endif()

if(NOT ETERNIA_INSTALL_INCLUDE_DIR)
    set(ETERNIA_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
endif()

if(NOT ETERNIA_INSTALL_DATA_DIR)
    set(ETERNIA_INSTALL_DATA_DIR ${CMAKE_INSTALL_PREFIX}/share)
endif()

# -----------------------------------------------------------------------------
# Find Packages
# -----------------------------------------------------------------------------

# This is for compatability with CLANG + Spack
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_INSTALL_RPATH "${ETERNIA_INSTALL_LIB_DIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Include most packages needed by both client and host
find_package(Hermes REQUIRED)

# ENABLE GPU SUPPORT
if(CHIMAERA_ENABLE_CUDA)
    hshm_enable_cuda(17)
endif()

if(CHIMAERA_ENABLE_ROCM)
    hshm_enable_rocm("HIP" 17)
endif()

# ------------------------------------------------------------------------------
# Setup CMake Environment
# ------------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Executables.")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Libraries")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all static libraries.")

# -----------------------------------------------------------------------------
# Build Hermes Main Packages
# -----------------------------------------------------------------------------
# Main includes
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

# add_subdirectory(src)
add_subdirectory(tasks)

# add_subdirectory(benchmark)

# -----------------------------------------------------------------------------
# Install Hermes Headers
# -----------------------------------------------------------------------------
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_PREFIX})

jarvis_repo_add("${CMAKE_SOURCE_DIR}/test/jarvis_eternia" "${CMAKE_SOURCE_DIR}/test/jarvis_eternia/pipelines")