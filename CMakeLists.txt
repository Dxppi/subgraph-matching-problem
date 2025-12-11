cmake_minimum_required(VERSION 3.18)
project(KCliqueCounter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(LAGRAPH_LIB lagraph REQUIRED)

file(GLOB SRC_FILES src/*.cpp)

add_executable(KCliqueCounter
    main.cpp
    ${SRC_FILES}
)

target_include_directories(KCliqueCounter PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    /usr/local/include
    /usr/local/include/suitesparse
)

# ---- OpenMP ----
find_package(OpenMP REQUIRED)

# ---- Link ----
target_link_libraries(KCliqueCounter
    PUBLIC
        ${GRAPHBLAS_LIB}
        ${LAGRAPH_LIB}
        ${LAGRAPHX_LIB}
        OpenMP::OpenMP_CXX
)
