cmake_minimum_required(VERSION 3.18)
project(FisaModuleGPUWrapper LANGUAGES CXX)

# Build core module from shared Source tree, but place .pyd output in GPU/Source.
set(FISA_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Output directory for fisa_module" FORCE)
add_subdirectory("../../Source" "${CMAKE_BINARY_DIR}/fisa_core")
