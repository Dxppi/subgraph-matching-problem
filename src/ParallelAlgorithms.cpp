#include "ParallelAlgorithms.hpp"
#include <GraphBLAS.h>

GrB_Index grb_n(GrB_Matrix A) {
  GrB_Index n;
  GrB_Matrix_nrows(&n, A);
  return n;
}

uint64_t grb_sum_all(GrB_Matrix A) {
  uint64_t result = 0;
  CHECK_GRB(
      GrB_Matrix_reduce_UINT64(&result, NULL, GrB_PLUS_MONOID_UINT64, A, NULL));
  return result;
}

GrB_Matrix grb_get_upper(GrB_Matrix G) {
  GrB_Matrix U = NULL;
  GrB_Index n = grb_n(G);
  CHECK_GRB(GrB_Matrix_new(&U, GrB_UINT64, n, n));

  GrB_Scalar k = NULL;
  GrB_Scalar_new(&k, GrB_INT64);
  GrB_Scalar_setElement_INT64(k, 1);

  CHECK_GRB(GxB_Matrix_select(U, NULL, NULL, GxB_TRIU, G, k, NULL));

  GrB_Scalar_free(&k);
  return U;
}

GrB_Matrix grb_extract_subgraph(GrB_Matrix G,
                                const std::vector<GrB_Index> &indices) {
  GrB_Index k = indices.size();
  if (k == 0)
    return NULL;

  GrB_Matrix Sub = NULL;
  CHECK_GRB(GrB_Matrix_new(&Sub, GrB_UINT64, k, k));

  CHECK_GRB(GrB_Matrix_extract(Sub, NULL, NULL, G, indices.data(), k,
                               indices.data(), k, NULL));
  return Sub;
}

uint64_t algorithm_3_count_triangles_grb(GrB_Matrix G) {
  GrB_Matrix U = grb_get_upper(G);
  GrB_Index n = grb_n(G);

  GrB_Matrix C = NULL;
  CHECK_GRB(GrB_Matrix_new(&C, GrB_UINT64, n, n));
  CHECK_GRB(GrB_mxm(C, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_UINT64, U, U, NULL));

  GrB_Matrix Tri = NULL;
  CHECK_GRB(GrB_Matrix_new(&Tri, GrB_UINT64, n, n));
  CHECK_GRB(GrB_Matrix_eWiseMult_BinaryOp(Tri, NULL, NULL, GrB_TIMES_UINT64, C, U, NULL));

  uint64_t count = grb_sum_all(Tri);

  GrB_Matrix_free(&U);
  GrB_Matrix_free(&C);
  GrB_Matrix_free(&Tri);
  return count;
}

GrB_Matrix algorithm_5_grb(GrB_Matrix G, size_t k) {
  if (k <= 1) {
    GrB_Matrix G_dup = NULL;
    GrB_Matrix_dup(&G_dup, G);
    return G_dup;
  }

  GrB_Index n = grb_n(G);

  GrB_Vector degrees = NULL;
  CHECK_GRB(GrB_Vector_new(&degrees, GrB_UINT64, n));
  CHECK_GRB(GrB_Matrix_reduce_Monoid(degrees, NULL, NULL,
                                     GrB_PLUS_MONOID_UINT64, G, NULL));

  std::vector<GrB_Index> valid_indices;

  GxB_Iterator iter;
  GxB_Iterator_new(&iter);
  GxB_Vector_Iterator_attach(iter, degrees, NULL);
  GxB_Vector_Iterator_seek(iter, 0);

  while (GxB_Vector_Iterator_next(iter) == GrB_SUCCESS) {
    GrB_Index idx = GxB_Vector_Iterator_getIndex(iter);
    uint64_t deg = GxB_Iterator_get_UINT64(iter);
    if (deg >= k - 1) {
      valid_indices.push_back(idx);
    }
  }
  GxB_Iterator_free(&iter);
  GrB_Vector_free(&degrees);

  if (valid_indices.size() == n) {
    GrB_Matrix G_dup = NULL;
    GrB_Matrix_dup(&G_dup, G);
    return G_dup;
  }

  return grb_extract_subgraph(G, valid_indices);
}

GrB_Matrix algorithm_6_grb(GrB_Matrix G, size_t k) {
  if (k <= 2) {
    GrB_Matrix G_dup = NULL;
    GrB_Matrix_dup(&G_dup, G);
    return G_dup;
  }

  GrB_Index n = grb_n(G);

  GrB_Matrix A2 = NULL;
  CHECK_GRB(GrB_Matrix_new(&A2, GrB_UINT64, n, n));
  CHECK_GRB(
      GrB_mxm(A2, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_UINT64, G, G, NULL));

  GrB_Matrix A_prime = NULL;
  CHECK_GRB(GrB_Matrix_new(&A_prime, GrB_UINT64, n, n));
  CHECK_GRB(GrB_Matrix_eWiseMult_BinaryOp(A_prime, NULL, NULL, GrB_TIMES_UINT64, A2, G, NULL));

  GrB_Matrix Result = NULL;
  CHECK_GRB(GrB_Matrix_new(&Result, GrB_UINT64, n, n));

  GrB_Scalar threshold = NULL;
  GrB_Scalar_new(&threshold, GrB_UINT64);
  GrB_Scalar_setElement_UINT64(threshold, k - 2);

  CHECK_GRB(GxB_Matrix_select(Result, NULL, NULL, GxB_GE_THUNK, A_prime,
                       threshold, NULL));

  GrB_Matrix temp = NULL;
  CHECK_GRB(GrB_Matrix_new(&temp, GrB_UINT64, n, n));
  CHECK_GRB(GrB_Matrix_apply(temp, NULL, NULL, GxB_ONE_UINT64, Result, NULL));
  GrB_Matrix_free(&Result);
  Result = temp;

  GrB_Matrix_free(&A2);
  GrB_Matrix_free(&A_prime);
  GrB_Scalar_free(&threshold);

  return Result;
}

uint64_t algorithm_8_count_4cliques_grb(GrB_Matrix G) {
  uint64_t total_count = 0;
  GrB_Matrix G_vec = grb_get_upper(G);
  GrB_Index n = grb_n(G_vec);

  GrB_Vector row_u = NULL, row_v = NULL, common = NULL;
  CHECK_GRB(GrB_Vector_new(&row_u, GrB_UINT64, n));
  CHECK_GRB(GrB_Vector_new(&row_v, GrB_UINT64, n));
  CHECK_GRB(GrB_Vector_new(&common, GrB_UINT64, n));

  std::vector<GrB_Index> all_indices(n);
  for (GrB_Index i = 0; i < n; ++i) {
    all_indices[i] = i;
  }

  for (GrB_Index v = 0; v < n; ++v) {
    GrB_Index nvals_v;
    CHECK_GRB(GrB_Col_extract(row_v, NULL, NULL, G_vec, all_indices.data(), n, v, GrB_DESC_T0));
    GrB_Vector_nvals(&nvals_v, row_v);
    
    if (nvals_v == 0) continue;

    for (GrB_Index u = v + 1; u < n; ++u) {
      uint64_t val = 0;
      GrB_Matrix_extractElement_UINT64(&val, G_vec, v, u);
      if (val == 0) continue;

      CHECK_GRB(
          GrB_Col_extract(row_u, NULL, NULL, G_vec, all_indices.data(), n, u, GrB_DESC_T0));

      CHECK_GRB(GrB_Vector_eWiseMult_BinaryOp(common, NULL, NULL, GrB_TIMES_UINT64, row_v, row_u,
                              NULL));

      GrB_Index n_common;
      GrB_Vector_nvals(&n_common, common);

      if (n_common > 0) {
        std::vector<GrB_Index> all_common(n_common);
        GrB_Vector_extractTuples_UINT64(all_common.data(), NULL, &n_common, common);

        std::vector<GrB_Index> indices;
        for (GrB_Index w : all_common) {
          if (w > u) {
            indices.push_back(w);
          }
        }

        if (!indices.empty()) {
          GrB_Matrix sub = grb_extract_subgraph(G_vec, indices);
          total_count += grb_sum_all(sub);
          GrB_Matrix_free(&sub);
        }
      }
    }
  }

  GrB_Vector_free(&row_u);
  GrB_Vector_free(&row_v);
  GrB_Vector_free(&common);
  GrB_Matrix_free(&G_vec);

  return total_count;
}

uint64_t algorithm_7_recursive_grb(GrB_Matrix G, size_t k) {
  if (k == 3)
    return algorithm_3_count_triangles_grb(G);
  if (k == 4)
    return algorithm_8_count_4cliques_grb(G);

  GrB_Matrix A1 = grb_get_upper(G);
  uint64_t total_count = 0;
  GrB_Index n = grb_n(A1);

  GrB_Vector neighbors_v = NULL;
  CHECK_GRB(GrB_Vector_new(&neighbors_v, GrB_UINT64, n));

  std::vector<GrB_Index> all_indices(n);
  for (GrB_Index i = 0; i < n; ++i) {
    all_indices[i] = i;
  }

  for (GrB_Index v = 0; v < n; ++v) {
    CHECK_GRB(
        GrB_Col_extract(neighbors_v, NULL, NULL, A1, all_indices.data(), n, v, GrB_DESC_T0));

    GrB_Index degree;
    GrB_Vector_nvals(&degree, neighbors_v);

    if (degree < k - 1)
      continue;

    std::vector<GrB_Index> Nr(degree);
    GrB_Vector_extractTuples_UINT64(Nr.data(), NULL, &degree, neighbors_v);

    GrB_Matrix sub = grb_extract_subgraph(A1, Nr);

    total_count += algorithm_7_recursive_grb(sub, k - 1);

    GrB_Matrix_free(&sub);
  }

  GrB_Vector_free(&neighbors_v);
  GrB_Matrix_free(&A1);
  return total_count;
}

uint64_t solve_k_clique_grb(GrB_Matrix G, size_t k) {
  if (k == 1)
    return grb_n(G);
  if (k == 2)
    return grb_sum_all(G) / 2;

  GrB_Matrix WorkG = NULL;
  GrB_Matrix_dup(&WorkG, G);

  if (k == 3) {
    uint64_t res = algorithm_3_count_triangles_grb(WorkG);
    GrB_Matrix_free(&WorkG);
    return res;
  }
  if (k == 4) {
    uint64_t res = algorithm_8_count_4cliques_grb(WorkG);
    GrB_Matrix_free(&WorkG);
    return res;
  }

  GrB_Matrix PrunedG = algorithm_5_grb(WorkG, k);
  GrB_Matrix_free(&WorkG);

  uint64_t res = algorithm_7_recursive_grb(PrunedG, k);

  GrB_Matrix_free(&PrunedG);
  return res;
}