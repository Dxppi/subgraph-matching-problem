#include <LAGraph.h>
#include <LAGraphX.h>

#include <chrono>
#include <iomanip>
#include <iostream>

#include "Algorithms.hpp"
#include "Matrix.hpp"

// ----------------------------
// Создание полного графа
// ----------------------------
Matrix create_complete_graph(size_t n) {
  Matrix adj(n);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
      if (i != j)
        adj.set(i, j, 1);
  return adj;
}

int main() {
  // ----------------------------
  // Инициализация GraphBLAS и LAGraph
  // ----------------------------
  if (GrB_init(GrB_NONBLOCKING) != GrB_SUCCESS) {
    std::cerr << "GraphBLAS init failed\n";
    return -1;
  }

  if (LAGraph_Init(NULL) != 0) {
    std::cerr << "LAGraph init failed\n";
    GrB_finalize();
    return -1;
  }

  // ----------------------------
  // Тестовый граф: K5
  // ----------------------------
  Matrix adj = create_complete_graph(5);

  std::cout << "--- Graph: K5 (Complete graph with 5 vertices) ---\n";
  std::cout << "3-Cliques: " << solve_k_clique(adj, 3u) << " (Expected: 10)\n";
  std::cout << "4-Cliques: " << solve_k_clique(adj, 4u) << " (Expected: 5)\n";
  std::cout << "5-Cliques: " << solve_k_clique(adj, 5u) << " (Expected: 1)\n";

  // ----------------------------
  // Сравнение производительности
  // ----------------------------
  Matrix K_test = create_complete_graph(6);
  size_t test_k = 4u;

  auto start_seq = std::chrono::high_resolution_clock::now();
  count_t count_seq = solve_k_clique(K_test, test_k);
  auto end_seq = std::chrono::high_resolution_clock::now();
  auto duration_seq =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_seq - start_seq)
          .count();

  std::cout << "\nSequential Time (K6, k=" << test_k << "): " << duration_seq
            << " ms\n";
  std::cout << "Clique count: " << count_seq << " (Expected: 15)\n";

  // ----------------------------
  // LAGraph TriangleCount
  // ----------------------------
  GrB_Matrix A = convert_to_grb(adj);
  LAGraph_Graph G = NULL;

  if (LAGraph_New(&G, &A, LAGraph_ADJACENCY_UNDIRECTED, NULL) != 0) {
    std::cerr << "LAGraph_New failed\n";
  } else {
    uint64_t ntri = 0;
    if (LAGraph_TriangleCount(&ntri, G, NULL) != 0) {
      std::cerr << "LAGraph TriangleCount failed\n";
    } else {
      std::cout << "LAGraph triangles: " << ntri << std::endl;
    }
    LAGraph_Delete(&G, NULL);
  }

  GrB_Matrix_free(&A);

  // ----------------------------
  // Финализация
  // ----------------------------
  LAGraph_Finalize(NULL);
  GrB_finalize();

  return 0;
}
