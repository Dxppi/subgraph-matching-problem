cmake_minimum_required(VERSION 3.10)

project(LA_KClique LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
  gbtl
  GIT_REPOSITORY https://github.com/cmu-sei/gbtl.git
  GIT_TAG        3.0.0
)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_MakeAvailable(gbtl)
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
target_link_libraries(la_file PRIVATE
 OpenMP::OpenMP_CXX
 graphblas
 )

add_executable(la_examples
    examples.cpp
    sparse_matrix.h
    la_algorithms.h
)
target_link_libraries(la_examples PRIVATE
 OpenMP::OpenMP_CXX
 graphblas
 )

enable_testing()

add_executable(
  testing_graphs
  testing_graphs.cc
)
target_link_libraries(
  testing_graphs
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(testing_graphs)

