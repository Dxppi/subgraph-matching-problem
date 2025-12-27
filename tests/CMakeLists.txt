# tests/CMakeLists.txt

add_executable(testing_graphs
    testgraph_gtest.cc
)

target_link_libraries(testing_graphs PRIVATE
    GTest::gtest_main
    graphblas
)

include(GoogleTest)

gtest_discover_tests(testing_graphs
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
