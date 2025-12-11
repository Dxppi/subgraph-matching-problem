#pragma once
#include "Matrix.hpp"
#include <cstddef>


using count_t = unsigned long long;

Matrix algorithm_5(const Matrix &G, size_t k);
Matrix algorithm_6(const Matrix &G, size_t k);

count_t algorithm_3_count_triangles(const Matrix &G);
count_t algorithm_8_count_4cliques(const Matrix &G);

count_t algorithm_7_recursive(Matrix G, size_t k);
count_t solve_k_clique(Matrix G, size_t k);
