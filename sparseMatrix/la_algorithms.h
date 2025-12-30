#ifndef LA_ALGORITHMS_H
#define LA_ALGORITHMS_H

#include "sparse_matrix.h"
#include <chrono>
#include <iostream>
#include <vector>

namespace LA {

// Algorithm 5: удалить вершины со степенью < k-1
pair<SparseMatrix, vector<int>> filterByDegreeAlg5(const SparseMatrix &G,
                                                   int k) {
  auto deg = G.degrees();
  vector<int> keepNodes;
  for (int i = 0; i < G.size(); ++i) {
    if (deg[i] >= k - 1)
      keepNodes.push_back(i);
  }
  if (keepNodes.empty())
    return {SparseMatrix(0), {}};
  return {G.inducedSubgraph(keepNodes), keepNodes};
}

// Подсчёт k-клик на основе Algorithm 7 из Emelin et al. (2023)
// Спуск к 4-кликам через Algorithm 5
inline long long countKCliquesWithStages(const SparseMatrix &G, int k,
                                         std::ostream &log = std::cerr) {
  using namespace std::chrono;
  auto now = []() { return high_resolution_clock::now(); };
  auto dur = [](auto t1, auto t2) {
    return duration_cast<milliseconds>(t2 - t1).count();
  };

  int n = G.size();
  log << "\n[k=" << k << "] Start n=" << n << " nnz=" << G.nnz() << "\n";

  if (k == 3)
    return G.countTriangles();
  if (k == 4)
    return G.count4Cliques();

  auto t0 = now();

  // Algorithm 5: удалить вершины со степенью < k-1
  auto [G1, _] = filterByDegreeAlg5(G, k);
  auto t1 = now();
  log << "Alg.5 (удалить вершины < k-1): " << dur(t0, t1)
      << "ms, n=" << G1.size() << "\n";
  if (G1.size() < k)
    return 0;

  // Algorithm 6: итеративная фильтрация рёбер
  SparseMatrix G2 = G1.iterativeEdgeFiltering(k - 3);
  auto t2 = now();
  log << "Alg.6 (фильтрация рёбер): " << dur(t1, t2) << "ms, n=" << G2.size()
      << "\n";
  if (G2.size() < k)
    return 0;

  // Ориентация графа в DAG: i -> j если i < j
  SparseMatrix Or = G2.orientById();
  auto t3 = now();
  log << "Ориентация (по ID): " << dur(t2, t3) << "ms\n";

  // Генерируем рёбра для параллелизма
  auto tasks = Or.getAllEdges();
  auto t4 = now();
  log << "Генерация задач: " << dur(t3, t4) << "ms, задач=" << tasks.size()
      << "\n";

  long long total = 0;

  // Параллелизм по рёбрам (u,v)
#pragma omp parallel for reduction(+ : total) schedule(dynamic, 10)
  for (size_t i = 0; i < tasks.size(); ++i) {
    int u = tasks[i].first;
    int v = tasks[i].second;

    // Находим общих соседей u и v
    const auto &Nu_map = Or.a[u];
    const auto &Nv_map = Or.a[v];

    std::vector<int> common;
    if (Nu_map.size() < Nv_map.size()) {
      for (auto &pair : Nu_map) {
        if (Nv_map.count(pair.first))
          common.push_back(pair.first);
      }
    } else {
      for (auto &pair : Nv_map) {
        if (Nu_map.count(pair.first))
          common.push_back(pair.first);
      }
    }

    // Подсчитываем (k-2)-клики в подграфе на общих соседях
    if ((int)common.size() >= k - 2) {
      if (k == 5) {
        // k=5: считаем треугольники в H
        SparseMatrix H = G2.inducedSubgraph(common);
        total += H.countTriangles();
      } else if (k == 6) {
        // k=6: считаем 4-клики в H
        SparseMatrix H = G2.inducedSubgraph(common);
        total += H.count4Cliques();
      } else {
        // k >= 7: спускаемся рекурсивно к 4-кликам
        SparseMatrix H = G2.inducedSubgraph(common);
        total += countKCliquesWithStages(H, k - 2, log);
      }
    }
  }

  auto t5 = now();

  log << "Основной этап (параллелизм по рёбрам): " << dur(t4, t5) << "ms\n";
  log << "ИТОГО: " << total << " (Время: " << dur(t0, t5) << "ms)\n";

  return total;
}

} // namespace LA

#endif // LA_ALGORITHMS_H