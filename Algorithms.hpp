#pragma once

#include "Matrix.hpp"
#include <cstddef>
#include <vector>

using count_t = unsigned long long;

struct SolverConfig {
    enum Preprocessing { NONE, ALG_5, ALG_6 };
    enum BaseCase { DOWN_TO_3, DOWN_TO_4 };

    Preprocessing prep = ALG_5;
    BaseCase base = DOWN_TO_4;

    bool parallel_enabled = true;
};

Matrix algorithm_5(const Matrix& G, size_t k);
Matrix algorithm_6(const Matrix& G, size_t k);

count_t algorithm_3_count_triangles(const Matrix& G);
count_t algorithm_8_count_4cliques(const Matrix& G);

count_t algorithm_7_recursive(Matrix G, size_t k, const SolverConfig& config);

count_t solve_k_clique(Matrix G, size_t k, SolverConfig config = SolverConfig());