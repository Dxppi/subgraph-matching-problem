# Основные тесты
add_executable(testing_graphs
    testGraphBlasMatrix.cc
)

target_include_directories(testing_graphs PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../graphBlas
    ${CMAKE_CURRENT_SOURCE_DIR}/../data
)

target_link_libraries(testing_graphs PRIVATE
    GTest::gtest_main
    graphblas
)

target_compile_definitions(testing_graphs
    PRIVATE TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests"
)

include(GoogleTest)
gtest_discover_tests(testing_graphs
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)