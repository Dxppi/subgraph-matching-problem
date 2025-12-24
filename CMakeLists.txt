cmake_minimum_required(VERSION 3.10)

project(LA_KClique LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    add_compile_options(/O2 /W4)
else()
    add_compile_options(-O3 -Wall -Wextra)
endif()

find_package(OpenMP)

if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
endif()

add_executable(la_file
    main.cpp
    sparse_matrix.h
    la_algorithms.h
)
target_link_libraries(la_file PRIVATE OpenMP::OpenMP_CXX)

add_executable(la_examples
    examples.cpp
    sparse_matrix.h
    la_algorithms.h
)
target_link_libraries(la_examples PRIVATE OpenMP::OpenMP_CXX)


