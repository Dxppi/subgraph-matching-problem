#ifndef LA_ALGORITHMS_H
#define LA_ALGORITHMS_H

#include "sparse_matrix.h"
#include <chrono>
#include <iostream>
#include <set>
#include <vector>

namespace LA {

// Построение неориентированного индуцированного подграфа на вершинах subset
inline SparseMatrix inducedSubgraph(const SparseMatrix &A,
                                    const std::vector<int> &subset) {
  int k = (int)subset.size();
  SparseMatrix B(k);
  // old -> new
  std::vector<int> pos(A.size(), -1);
  for (int i = 0; i < k; ++i)
    pos[subset[i]] = i;

  for (int i = 0; i < k; ++i) {
    int u = subset[i];
    for (auto &kv : A.a[u]) {
      int v = kv.first;
      if (pos[v] == -1)
        continue;
      int j = pos[v];
      if (i == j)
        continue;
      // неориентированный граф
      B.set(i, j, 1);
    }
  }
  return B;
}

// ---------- Algorithm 3: счёт треугольников ----------
// Count = SUM( (A^2 ⊙ A) ) / 6
inline long long countTriangles(const SparseMatrix &A) {
  SparseMatrix A2 = A.multiply(A);
  SparseMatrix T = A2.hadamard(A);
  return T.sumAll() / 6;
}

// ---------- Algorithm 5: удаление вершин по степени ----------
// Удаляем вершины с degree < k-2. Возвращаем новую матрицу и mapping[newIdx] =
// oldIdx.
inline std::pair<SparseMatrix, std::vector<int>>
deleteVerticesByDegree(const SparseMatrix &A, int k) {
  int n = A.size();
  auto deg = A.degrees();
  int minDeg = k - 2;

  std::vector<int> mapNewToOld;
  mapNewToOld.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (deg[i] >= minDeg) {
      mapNewToOld.push_back(i);
    }
  }

  SparseMatrix B((int)mapNewToOld.size());
  if (mapNewToOld.empty()) {
    return {B, mapNewToOld};
  }

  // old -> new: O(1) доступ
  std::vector<int> oldToNew(n, -1);
  for (int newI = 0; newI < (int)mapNewToOld.size(); ++newI) {
    oldToNew[mapNewToOld[newI]] = newI;
  }

  for (int newI = 0; newI < (int)mapNewToOld.size(); ++newI) {
    int oldI = mapNewToOld[newI];
    for (auto &kv : A.a[oldI]) {
      int oldJ = kv.first;
      int val = kv.second;
      int newJ = oldToNew[oldJ];
      if (newJ != -1) {
        B.set(newI, newJ, val);
      }
    }
  }

  return {B, mapNewToOld};
}

// Один проход фильтрации рёбер по |N(u) ∩ N(v)| < k-3.
// Возвращает новый граф и флаг changed: были ли удалены рёбра.
inline std::pair<SparseMatrix, bool> onePassEdgeFiltering(const SparseMatrix &G,
                                                          int k) {
  int n = G.size();
  SparseMatrix R(n);
  bool changed = false;

  int minCommon = k - 3;
  if (minCommon <= 0) {
    // Ничего не фильтруем
    return {G, false};
  }

  // Для каждого ребра (u,v) проверяем размер пересечения N(u) и N(v)
  for (int u = 0; u < n; ++u) {
    auto Nu = G.rowIndices(u);
    for (int v : Nu) {
      if (u >= v)
        continue; // каждое ребро один раз
      auto Nv = G.rowIndices(v);

      // пересечение N(u) и N(v)
      std::vector<int> inter;
      inter.reserve(std::min(Nu.size(), Nv.size()));
      std::set_intersection(Nu.begin(), Nu.end(), Nv.begin(), Nv.end(),
                            std::back_inserter(inter));

      if ((int)inter.size() < minCommon) {
        // ребро (u,v) удаляем
        changed = true;
      } else {
        // оставляем в результирующем графе
        R.set(u, v, 1);
        R.set(v, u, 1);
      }
    }
  }

  return {R, changed};
}

// Algorithm 6 (упрощённый): удаление рёбер с малым |N(u) ∩ N(v)| и последующее
// удаление изолированных вершин (компактирование).
inline SparseMatrix deleteEdgesByNeighborhood(const SparseMatrix &A, int k) {
  SparseMatrix G = A;
  if (G.size() == 0)
    return G;

  // Итерации до стабилизации: пока хотя бы одно ребро удаляется
  while (true) {
    auto [Gnext, changed] = onePassEdgeFiltering(G, k);
    G = std::move(Gnext);
    if (!changed)
      break;
  }

  // Компактирование вершин: выбрасываем вершины со степенью 0
  int n = G.size();
  auto deg = G.degrees();

  std::vector<int> mapNewToOld;
  mapNewToOld.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (deg[i] > 0) {
      mapNewToOld.push_back(i);
    }
  }

  SparseMatrix R((int)mapNewToOld.size());
  if (mapNewToOld.empty()) {
    return R;
  }

  std::vector<int> oldToNew(n, -1);
  for (int newI = 0; newI < (int)mapNewToOld.size(); ++newI) {
    oldToNew[mapNewToOld[newI]] = newI;
  }

  for (int newI = 0; newI < (int)mapNewToOld.size(); ++newI) {
    int oldI = mapNewToOld[newI];
    for (auto &kv : G.a[oldI]) {
      int oldJ = kv.first;
      int newJ = oldToNew[oldJ];
      if (newJ != -1) {
        R.set(newI, newJ, 1);
      }
    }
  }

  return R;
}

// ---------- простая ациклическая ориентация: по номеру вершины ----------
// ориентируем ребро всегда i->j при i<j
inline SparseMatrix orientById(const SparseMatrix &A) {
  int n = A.size();
  SparseMatrix Or(n);
  for (int i = 0; i < n; ++i) {
    for (auto &kv : A.a[i]) {
      int j = kv.first;
      if (i < j)
        Or.set(i, j, 1); // дуга i→j
    }
  }
  return Or;
}

// правое соседство N_r(v): все j, такие что v→j
inline std::vector<int> rightNeighbors(const SparseMatrix &Or, int v) {
  return Or.rowIndices(v);
}

// ---------- Algorithm 8: подсчёт 4-клик через треугольники в N(v) ----------
// Для каждой вершины v считаем треугольники в подграфе, индуцированном на N(v).
// Каждая 4-клика считается 2 раза -> делим результат на 2.
inline long long count4Cliques(const SparseMatrix &A) {
  int n = A.size();
  long long total = 0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : total) schedule(dynamic)
#endif
  for (int v = 0; v < n; ++v) {
    auto Nv = A.rowIndices(v); // соседи v
    if ((int)Nv.size() < 3)
      continue;

    SparseMatrix Ind = inducedSubgraph(A, Nv);
    long long triInNv = countTriangles(Ind); // Alg.3
    total += triInNv;
  }

  // каждая 4-клика считается по числу вершин (4 раза)
  return total / 4;
}

// Рекурсивный подсчёт k-клик в стиле Algorithm 7 (упрощённый вариант).
// ВНИМАНИЕ: работает на маленьких/средних графах, для очень больших — без
// оптимизаций статьи может быть медленным.
inline long long countKCliquesRecursive(const SparseMatrix &G, int k,
                                        int depth = 0) {
  int n = G.size();
  if (k < 3 || n < k)
    return 0;

  // Базы
  if (k == 3) {
    return countTriangles(G);
  }
  if (k == 4) {
    return count4Cliques(G);
  }

  // Alg.5
  auto prV = deleteVerticesByDegree(G, k);
  const SparseMatrix &G1 = prV.first;
  if (G1.size() < k)
    return 0;

  // Alg.6
  SparseMatrix G2 = deleteEdgesByNeighborhood(G1, k);
  if (G2.size() < k)
    return 0;

  // Ориентация
  SparseMatrix Or = orientById(G2);

  long long total = 0;
  int n2 = Or.size();

  // Параллелим только верхние уровни, например depth <= 1
  bool doParallel = (depth <= 1);

  if (doParallel) {
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : total) schedule(dynamic)
#endif
    for (int v = 0; v < n2; ++v) {
      auto Nr = rightNeighbors(Or, v);
      if ((int)Nr.size() < k - 1)
        continue;

      SparseMatrix H = inducedSubgraph(G2, Nr);
      long long sub = countKCliquesRecursive(H, k - 1, depth + 1);
      total += sub;
    }
  } else {
    // Глубже считаем последовательно, чтобы не плодить слишком много задач
    for (int v = 0; v < n2; ++v) {
      auto Nr = rightNeighbors(Or, v);
      if ((int)Nr.size() < k - 1)
        continue;

      SparseMatrix H = inducedSubgraph(G2, Nr);
      long long sub = countKCliquesRecursive(H, k - 1, depth + 1);
      total += sub;
    }
  }

  return total;
}

// Обёртка для внешнего использования
inline long long countKCliques(const SparseMatrix &G, int k) {
  return countKCliquesRecursive(G, k, 0);
}

inline long long countKCliquesWithStages(const SparseMatrix &G, int k,
                                         std::ostream &log = std::cerr) {
  using clk = std::chrono::high_resolution_clock;

  int n = G.size();
  if (k < 3 || n < k)
    return 0;

  if (k == 3) {
    auto t0 = clk::now();
    long long res = countTriangles(G);
    auto t1 = clk::now();
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    log << "[k=" << k << "] base k=3, time=" << ms << " ms\n";
    return res;
  }

  if (k == 4) {
    auto t0 = clk::now();
    long long res = count4Cliques(G);
    auto t1 = clk::now();
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    log << "[k=" << k << "] base k=4 (Alg.8), time=" << ms << " ms\n";
    return res;
  }

  log << "[k=" << k << "] n=" << n << " nnz≈" << G.nnz() / 2 << "\n";

  // Alg.5
  auto t0 = clk::now();
  auto prV = deleteVerticesByDegree(G, k);
  const SparseMatrix &G1 = prV.first;
  auto t1 = clk::now();
  auto ms5 =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  log << "[k=" << k << "] Alg.5: n'=" << G1.size() << " nnz'≈" << G1.nnz() / 2
      << " time=" << ms5 << " ms\n";

  if (G1.size() < k) {
    log << "[k=" << k << "] Alg.5 removed too much, result=0\n";
    return 0;
  }

  // Alg.6
  t0 = clk::now();
  SparseMatrix G2 = deleteEdgesByNeighborhood(G1, k);
  t1 = clk::now();
  auto ms6 =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  log << "[k=" << k << "] Alg.6: n''=" << G2.size() << " nnz''≈" << G2.nnz() / 2
      << " time=" << ms6 << " ms\n";

  if (G2.size() < k) {
    log << "[k=" << k << "] Alg.6 removed too much, result=0\n";
    return 0;
  }

  // Ориентация
  t0 = clk::now();
  SparseMatrix Or = orientById(G2);
  t1 = clk::now();
  auto msOr =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  log << "[k=" << k << "] orient: arcs=" << Or.nnz() << " time=" << msOr
      << " ms\n";

  // Основная фаза (цикл по v + рекурсия (k-1))
  t0 = clk::now();
  long long total = 0;
  int n2 = Or.size();

#pragma omp parallel for reduction(+ : total) schedule(dynamic)
  for (int v = 0; v < n2; ++v) {
    auto Nr = rightNeighbors(Or, v);
    if ((int)Nr.size() < k - 1)
      continue;

    SparseMatrix H = inducedSubgraph(G2, Nr);
    long long sub = countKCliquesRecursive(H, k - 1, 1);
    total += sub;
  }

  t1 = clk::now();
  auto msCore =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  log << "[k=" << k << "] core phase: time=" << msCore << " ms\n";

  log << "[k=" << k << "] total=" << total
      << " T=" << (ms5 + ms6 + msOr + msCore) << " ms\n";

  return total;
}

} // namespace LA

#endif // LA_ALGORITHMS_H
