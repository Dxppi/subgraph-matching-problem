#pragma once

#include <GraphBLAS.h>
#include <cstddef>
#include <iostream>
#include <vector>

#define CHECK_GRB(статус)                                                      \
  if ((статус) != GrB_SUCCESS) {                                               \
    std::cerr << "Ошибка GraphBLAS: код " << (статус) << " на строке "           \
              << __LINE__ << std::endl;                                        \
    exit(1);                                                                   \
  }

GrB_Index grb_n(GrB_Matrix A);
uint64_t grb_sum_all(GrB_Matrix A);
GrB_Matrix grb_get_upper(GrB_Matrix G);
GrB_Matrix grb_extract_subgraph(GrB_Matrix G,
                                const std::vector<GrB_Index> &indices);

uint64_t algorithm_3_count_triangles_grb(GrB_Matrix G);
GrB_Matrix algorithm_5_grb(GrB_Matrix G, size_t k);
GrB_Matrix algorithm_6_grb(GrB_Matrix G, size_t k);
uint64_t algorithm_8_count_4cliques_grb(GrB_Matrix G);
uint64_t algorithm_7_recursive_grb(GrB_Matrix G, size_t k);
uint64_t solve_k_clique_grb(GrB_Matrix G, size_t k);