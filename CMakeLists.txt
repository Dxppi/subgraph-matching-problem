cmake_minimum_required(VERSION 3.18)
project(KCliqueCounter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ---- Libraries ----
find_library(GRAPHBLAS_LIB graphblas REQUIRED)
find_library(LAGRAPH_LIB lagraph REQUIRED)
find_library(LAGRAPHX_LIB lagraphx REQUIRED)

# ---- Include dirs ----
include_directories(
    /usr/local/include
    /usr/local/include/suitesparse
)

# ---- OpenMP ----
find_package(OpenMP REQUIRED)

# ---- Executable ----
add_executable(KCliqueCounter
    main.cpp
    Algorithms.cpp
)

# ---- Link ----
target_link_libraries(KCliqueCounter
    PUBLIC
        ${GRAPHBLAS_LIB}
        ${LAGRAPH_LIB}
        ${LAGRAPHX_LIB}
        OpenMP::OpenMP_CXX
)
