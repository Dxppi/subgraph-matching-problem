#include "Algorithms.hpp"
#include <vector>

Matrix algorithm_5(const Matrix &G, size_t k) {
  if (k <= 1)
    return G;
  std::vector<size_t> valid_indices;
  for (size_t i = 0; i < G.n; ++i)
    if (G.get_neighbors(i).size() >= k - 1)
      valid_indices.push_back(i);
  if (valid_indices.size() == G.n)
    return G;
  return G.extract_subgraph(valid_indices);
}

Matrix algorithm_6(const Matrix &G, size_t k) {
  if (k <= 2)
    return G;
  Matrix A2 = G.multiply(G);
  Matrix A_prime = A2.hadamard(G);
  Matrix Result(G.n);
  for (size_t i = 0; i < G.n; ++i)
    for (size_t j = 0; j < G.n; ++j)
      if (A_prime.data[i][j] >= k - 2)
        Result.data[i][j] = 1;
  return Result;
}

count_t algorithm_3_count_triangles(const Matrix &G) {
  Matrix G_vec = G.get_upper_triangular();
  Matrix G_vec_sq = G_vec.multiply(G_vec);
  Matrix A_star = G_vec_sq.hadamard(G_vec);
  return A_star.sum_all();
}

count_t algorithm_8_count_4cliques(const Matrix &G) {
  count_t total_count = 0;
  Matrix G_vec = G.get_upper_triangular();
  for (size_t v = 0; v < G_vec.n; ++v)
    for (size_t u = v + 1; u < G_vec.n; ++u) {
      if (!G_vec.data[v][u])
        continue;
      std::vector<size_t> common_neighbors;
      for (size_t w = u + 1; w < G_vec.n; ++w)
        if (G_vec.data[v][w] && G_vec.data[u][w])
          common_neighbors.push_back(w);
      if (!common_neighbors.empty()) {
        Matrix sub = G_vec.extract_subgraph(common_neighbors);
        total_count += sub.get_upper_triangular().sum_all();
      }
    }
  return total_count;
}

count_t algorithm_7_recursive(Matrix G, size_t k) {
  if (k == 3)
    return algorithm_3_count_triangles(G);
  if (k == 4)
    return algorithm_8_count_4cliques(G);

  Matrix A1 = G.get_upper_triangular();
  count_t total_count = 0;

  for (size_t v = 0; v < A1.n; ++v) {
    std::vector<size_t> Nr = A1.get_neighbors(v);
    if (Nr.size() < k - 1)
      continue;
    Matrix sub = A1.extract_subgraph(Nr);
    total_count += algorithm_7_recursive(sub, k - 1);
  }

  return total_count;
}

count_t solve_k_clique(Matrix G, size_t k) {
  if (k == 1)
    return G.n;
  if (k == 2)
    return G.sum_all() / 2;
  if (k == 3)
    return algorithm_3_count_triangles(G);
  if (k == 4)
    return algorithm_8_count_4cliques(G);

  G = algorithm_5(G, k);
  return algorithm_7_recursive(G, k);
}
