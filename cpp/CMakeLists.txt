cmake_minimum_required(VERSION 3.14)
project(model_lab_sam3_bridge LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)

if(NOT DEFINED SAM3_CPP_ROOT)
    message(FATAL_ERROR "Set -DSAM3_CPP_ROOT=/path/to/sam3.cpp")
endif()

set(SAM3_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SAM3_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${SAM3_CPP_ROOT}" sam3cpp EXCLUDE_FROM_ALL)

add_executable(sam3_bridge sam3_bridge.cpp)
target_link_libraries(sam3_bridge PRIVATE sam3)
target_compile_features(sam3_bridge PRIVATE cxx_std_14)

