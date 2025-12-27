cmake_minimum_required(VERSION 3.10)

project(LA_KClique LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# ---------- Dependencies: GBTL + GoogleTest ----------

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
FetchContent_MakeAvailable(googletest gbtl)

add_library(graphblas INTERFACE)
target_include_directories(graphblas INTERFACE
    "${gbtl_SOURCE_DIR}/src"
    "${gbtl_SOURCE_DIR}/src/graphblas/platforms/optimized_sequential"
)
target_compile_definitions(graphblas INTERFACE
    GRAPHBLAS_BACKEND_optimized_sequential
)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    target_compile_options(graphblas INTERFACE
        -Wno-unused-parameter
        -Wno-return-stack-address
        -Wno-deprecated-copy
    )
endif()

# ---------- Compiler flags ----------

if (MSVC)
    add_compile_options(/O2 /W4)
else()
    add_compile_options(-O3 -Wall -Wextra)
endif()

# ---------- OpenMP (optional) ----------

find_package(OpenMP)

# ---------- Main executables ----------

add_executable(la_file
    main.cpp
)
target_include_directories(la_file PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sparseMatrix"
)
target_link_libraries(la_file PRIVATE
    graphblas
)
if (OpenMP_CXX_FOUND)
    target_link_libraries(la_file PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(la_examples
    sparseMatrix/examples.cpp
)
target_include_directories(la_examples PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sparseMatrix"
)
target_link_libraries(la_examples PRIVATE
    graphblas
)
if (OpenMP_CXX_FOUND)
    target_link_libraries(la_examples PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---------- Tests ----------

enable_testing()
add_subdirectory(tests)
