#include "graphblas/graphblas.hpp"
#include <algorithm>
#include <algorithms/triangle_count.hpp>
#include <set>
#include <vector>

/**
 * Подсчитывает количество треугольников в графе
 */
template <typename MatrixT> long long count_triangles(MatrixT const &graph) {
  return algorithms::triangle_count(graph);
}

/**
 * Удаляет вершины со степенью < k-2
 * @param A матрица смежности
 * @param k размер клики (для 4-клик k=4, поэтому удаляем вершины с deg < 2)
 * @return пару (новая матрица, отображение newIdx -> oldIdx)
 */
template <typename MatrixT>
std::pair<MatrixT, std::vector<grb::IndexType>>
deleteVerticesByDegree(const MatrixT &A, int k) {
  using T = typename MatrixT::ScalarType;
  grb::IndexType n = A.nrows();
  int minDeg = k - 2;

  // Вычисляем степени вершин через умножение на вектор единиц
  grb::Vector<T> ones(n);
  for (grb::IndexType i = 0; i < n; ++i) {
    ones.setElement(i, 1);
  }

  grb::Vector<T> degrees(n);
  grb::mxv(degrees, grb::NoMask(), grb::NoAccumulate(),
           grb::ArithmeticSemiring<T>(), A, ones, grb::REPLACE);

  // Выбираем вершины с достаточной степенью
  std::vector<grb::IndexType> mapNewToOld;
  mapNewToOld.reserve(n);

  for (grb::IndexType i = 0; i < n; ++i) {
    try {
      T deg_i = degrees.extractElement(i);
      if (deg_i >= minDeg) {
        mapNewToOld.push_back(i);
      }
    } catch (const grb::NoValueException &) {
      continue;
    }
  }

  // Новая матрица
  MatrixT B(mapNewToOld.size(), mapNewToOld.size());
  if (mapNewToOld.empty()) {
    return {B, mapNewToOld};
  }

  // old -> new mapping
  std::vector<int> oldToNew(n, -1);
  for (grb::IndexType newI = 0; newI < (grb::IndexType)mapNewToOld.size();
       ++newI) {
    oldToNew[mapNewToOld[newI]] = newI;
  }

  // Строим новую матрицу
  std::vector<grb::IndexType> rows, cols;
  std::vector<T> vals;

  for (grb::IndexType newI = 0; newI < (grb::IndexType)mapNewToOld.size();
       ++newI) {
    grb::IndexType oldI = mapNewToOld[newI];

    // Извлекаем строку oldI из матрицы A
    grb::Vector<T> row_i(n);

    std::vector<grb::IndexType> all_indices;
    for (grb::IndexType j = 0; j < n; ++j) {
      all_indices.push_back(j);
    }

    grb::extract(row_i, grb::NoMask(), grb::NoAccumulate(), A, all_indices,
                 oldI, grb::REPLACE);

    // Копируем ненулевые элементы в новую матрицу
    grb::IndexArrayType indices(row_i.nvals());
    std::vector<T> row_vals(row_i.nvals());
    row_i.extractTuples(indices.begin(), row_vals.begin());

    for (grb::IndexType idx = 0; idx < row_i.nvals(); ++idx) {
      grb::IndexType oldJ = indices[idx];
      int newJ = oldToNew[oldJ];
      if (newJ != -1) {
        rows.push_back(newI);
        cols.push_back(newJ);
        vals.push_back(row_vals[idx]);
      }
    }
  }

  if (!rows.empty()) {
    B.build(rows.begin(), cols.begin(), vals.begin(), rows.size());
  }

  return {B, mapNewToOld};
}

/**
 * Один проход фильтрации рёбер по |N(u) ∩ N(v)| < k-3
 * @param G матрица смежности
 * @param k размер клики
 * @return пару (новая матрица, флаг изменения)
 */
template <typename MatrixT>
std::pair<MatrixT, bool> onePassEdgeFiltering(const MatrixT &G, int k) {
  using T = typename MatrixT::ScalarType;
  grb::IndexType n = G.nrows();
  MatrixT R(n, n);
  bool changed = false;

  int minCommon = k - 3;
  if (minCommon <= 0) {
    return {G, false};
  }

  std::vector<grb::IndexType> all_indices;
  for (grb::IndexType i = 0; i < n; ++i) {
    all_indices.push_back(i);
  }

  // Для каждого ребра (u,v) проверяем пересечение соседей
  for (grb::IndexType u = 0; u < n; ++u) {
    // Получаем соседей u
    grb::Vector<T> row_u(n);
    grb::extract(row_u, grb::NoMask(), grb::NoAccumulate(), G, all_indices, u,
                 grb::REPLACE);

    grb::IndexArrayType indices_u(row_u.nvals());
    std::vector<T> vals_u(row_u.nvals());
    row_u.extractTuples(indices_u.begin(), vals_u.begin());

    // Сортируем для set_intersection
    std::vector<grb::IndexType> Nu(indices_u.begin(), indices_u.end());
    std::sort(Nu.begin(), Nu.end());

    for (grb::IndexType v_idx = 0; v_idx < (grb::IndexType)Nu.size(); ++v_idx) {
      grb::IndexType v = Nu[v_idx];
      if (u >= v)
        continue;

      // Получаем соседей v
      grb::Vector<T> row_v(n);
      grb::extract(row_v, grb::NoMask(), grb::NoAccumulate(), G, all_indices, v,
                   grb::REPLACE);

      grb::IndexArrayType indices_v(row_v.nvals());
      std::vector<T> vals_v(row_v.nvals());
      row_v.extractTuples(indices_v.begin(), vals_v.begin());

      std::vector<grb::IndexType> Nv(indices_v.begin(), indices_v.end());
      std::sort(Nv.begin(), Nv.end());

      // Пересечение N(u) и N(v)
      std::vector<grb::IndexType> inter;
      std::set_intersection(Nu.begin(), Nu.end(), Nv.begin(), Nv.end(),
                            std::back_inserter(inter));

      if ((int)inter.size() >= minCommon) {
        // Оставляем ребро
        R.setElement(u, v, 1);
        R.setElement(v, u, 1);
      } else {
        changed = true;
      }
    }
  }

  return {R, changed};
}

/**
 * Алгоритм 6: удаление рёбер по соседству и компактирование
 * @param A матрица смежности
 * @param k размер клики
 * @return новая матрица после фильтрации
 */
template <typename MatrixT>
MatrixT deleteEdgesByNeighborhood(const MatrixT &A, int k) {
  using T = typename MatrixT::ScalarType;
  MatrixT G = A;

  if (G.nrows() == 0)
    return G;

  // Итерации до стабилизации
  while (true) {
    auto [Gnext, changed] = onePassEdgeFiltering(G, k);
    G = Gnext;
    if (!changed)
      break;
  }

  // Компактирование: удаляем вершины со степенью 0
  grb::IndexType n = G.nrows();

  // Вычисляем степени
  grb::Vector<T> ones(n);
  for (grb::IndexType i = 0; i < n; ++i) {
    ones.setElement(i, 1);
  }

  grb::Vector<T> degrees(n);
  grb::mxv(degrees, grb::NoMask(), grb::NoAccumulate(),
           grb::ArithmeticSemiring<T>(), G, ones, grb::REPLACE);

  std::vector<grb::IndexType> mapNewToOld;
  mapNewToOld.reserve(n);

  for (grb::IndexType i = 0; i < n; ++i) {
    try {
      T deg_i = degrees.extractElement(i);
      if (deg_i > 0) {
        mapNewToOld.push_back(i);
      }
    } catch (const grb::NoValueException &) {
      continue;
    }
  }

  MatrixT R(mapNewToOld.size(), mapNewToOld.size());
  if (mapNewToOld.empty()) {
    return R;
  }

  std::vector<int> oldToNew(n, -1);
  for (grb::IndexType newI = 0; newI < (grb::IndexType)mapNewToOld.size();
       ++newI) {
    oldToNew[mapNewToOld[newI]] = newI;
  }

  std::vector<grb::IndexType> rows, cols;
  std::vector<T> vals;

  for (grb::IndexType newI = 0; newI < (grb::IndexType)mapNewToOld.size();
       ++newI) {
    grb::IndexType oldI = mapNewToOld[newI];

    // Извлекаем строку oldI
    grb::Vector<T> row_i(n);
    std::vector<grb::IndexType> all_indices;
    for (grb::IndexType j = 0; j < n; ++j) {
      all_indices.push_back(j);
    }

    grb::extract(row_i, grb::NoMask(), grb::NoAccumulate(), G, all_indices,
                 oldI, grb::REPLACE);

    grb::IndexArrayType indices(row_i.nvals());
    std::vector<T> row_vals(row_i.nvals());
    row_i.extractTuples(indices.begin(), row_vals.begin());

    for (grb::IndexType idx = 0; idx < row_i.nvals(); ++idx) {
      grb::IndexType oldJ = indices[idx];
      int newJ = oldToNew[oldJ];
      if (newJ != -1) {
        rows.push_back(newI);
        cols.push_back(newJ);
        vals.push_back(1);
      }
    }
  }

  if (!rows.empty()) {
    R.build(rows.begin(), cols.begin(), vals.begin(), rows.size());
  }

  return R;
}

/**
 * Ориентирует граф по номерам вершин: ребро i-j становится i→j если i<j
 * @param A неориентированная матрица смежности
 * @return ориентированная матрица
 */
template <typename MatrixT> MatrixT orientById(const MatrixT &A) {
  using T = typename MatrixT::ScalarType;
  grb::IndexType n = A.nrows();
  MatrixT Or(n, n);

  std::vector<grb::IndexType> all_indices;
  for (grb::IndexType i = 0; i < n; ++i) {
    all_indices.push_back(i);
  }

  for (grb::IndexType i = 0; i < n; ++i) {
    grb::Vector<T> row_i(n);
    grb::extract(row_i, grb::NoMask(), grb::NoAccumulate(), A, all_indices, i,
                 grb::REPLACE);

    grb::IndexArrayType indices(row_i.nvals());
    std::vector<T> row_vals(row_i.nvals());
    row_i.extractTuples(indices.begin(), row_vals.begin());

    for (grb::IndexType idx = 0; idx < row_i.nvals(); ++idx) {
      grb::IndexType j = indices[idx];
      if (i < j) {
        Or.setElement(i, j, 1);
      }
    }
  }

  return Or;
}

/**
 * Возвращает правых соседей вершины v в ориентированном графе
 * @param Or ориентированная матрица смежности
 * @param v индекс вершины
 * @return вектор индексов соседей
 */
template <typename MatrixT>
std::vector<grb::IndexType> rightNeighbors(const MatrixT &Or,
                                           grb::IndexType v) {
  using T = typename MatrixT::ScalarType;
  grb::IndexType n = Or.ncols();

  grb::Vector<T> row_v(n);
  std::vector<grb::IndexType> all_indices;
  for (grb::IndexType j = 0; j < n; ++j) {
    all_indices.push_back(j);
  }

  grb::extract(row_v, grb::NoMask(), grb::NoAccumulate(), Or, all_indices, v,
               grb::REPLACE);

  grb::IndexArrayType indices(row_v.nvals());
  std::vector<T> vals(row_v.nvals());
  row_v.extractTuples(indices.begin(), vals.begin());

  std::vector<grb::IndexType> result(indices.begin(), indices.end());
  std::sort(result.begin(), result.end());
  return result;
}

/**
 * Строит индуцированный подграф на подмножестве вершин
 * @param A матрица смежности
 * @param subset индексы вершин подграфа
 * @return новая матрица - индуцированный подграф
 */
template <typename MatrixT>
MatrixT inducedSubgraph(const MatrixT &A,
                        const std::vector<grb::IndexType> &subset) {
  using T = typename MatrixT::ScalarType;
  grb::IndexType n = A.nrows();
  grb::IndexType k = subset.size();

  MatrixT B(k, k);

  // old -> new mapping
  std::vector<int> pos(n, -1);
  for (grb::IndexType i = 0; i < k; ++i) {
    pos[subset[i]] = i;
  }

  std::vector<grb::IndexType> all_indices;
  for (grb::IndexType i = 0; i < n; ++i) {
    all_indices.push_back(i);
  }

  std::vector<grb::IndexType> rows, cols;
  std::vector<T> vals;

  for (grb::IndexType newI = 0; newI < k; ++newI) {
    grb::IndexType oldI = subset[newI];

    grb::Vector<T> row_i(n);
    grb::extract(row_i, grb::NoMask(), grb::NoAccumulate(), A, all_indices,
                 oldI, grb::REPLACE);

    grb::IndexArrayType indices(row_i.nvals());
    std::vector<T> row_vals(row_i.nvals());
    row_i.extractTuples(indices.begin(), row_vals.begin());

    for (grb::IndexType idx = 0; idx < row_i.nvals(); ++idx) {
      grb::IndexType oldJ = indices[idx];
      int newJ = pos[oldJ];
      if (newJ != -1 && newI != (grb::IndexType)newJ) {
        rows.push_back(newI);
        cols.push_back(newJ);
        vals.push_back(1);
      }
    }
  }

  if (!rows.empty()) {
    B.build(rows.begin(), cols.begin(), vals.begin(), rows.size());
  }

  return B;
}

/**
 * Подсчитывает количество 4-клик в графе
 */
template <typename MatrixT> long long count4Cliques(const MatrixT &A) {
  using T = typename MatrixT::ScalarType;
  grb::IndexType n = A.nrows();
  long long total = 0;

  std::vector<grb::IndexType> all_indices;
  for (grb::IndexType i = 0; i < n; ++i) {
    all_indices.push_back(i);
  }

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : total) schedule(dynamic)
#endif
  for (grb::IndexType v = 0; v < n; ++v) {
    grb::Vector<T> row_v(n);
    grb::extract(row_v, grb::NoMask(), grb::NoAccumulate(), A, all_indices, v,
                 grb::REPLACE);

    if (row_v.nvals() < 3) {
      continue;
    }

    grb::IndexArrayType Nv(row_v.nvals());
    std::vector<T> vals(row_v.nvals());
    row_v.extractTuples(Nv.begin(), vals.begin());

    MatrixT Ind = inducedSubgraph(
        A, std::vector<grb::IndexType>(Nv.begin(), Nv.begin() + row_v.nvals()));

    long long triInNv = count_triangles(Ind);
    total += triInNv;
  }

  return total / 4;
}

/**
 * Рекурсивный подсчёт k-клик (Algorithm 7)
 * @param G матрица смежности
 * @param k размер клики
 * @param depth глубина рекурсии (для контроля параллелизма)
 * @return количество k-клик
 */
template <typename MatrixT>
long long countKCliquesRecursive(const MatrixT &G, int k, int depth = 0) {
  using T = typename MatrixT::ScalarType;
  grb::IndexType n = G.nrows();

  if (k < 3 || n < (grb::IndexType)k)
    return 0;

  // Базовые случаи
  if (k == 3) {
    return count_triangles(G);
  }
  if (k == 4) {
    return count4Cliques(G);
  }

  // Algorithm 5: удаление вершин по степени
  auto [G1, mapping1] = deleteVerticesByDegree(G, k);
  if (G1.nrows() < (grb::IndexType)k)
    return 0;

  // Algorithm 6: удаление рёбер по соседству
  MatrixT G2 = deleteEdgesByNeighborhood(G1, k);
  if (G2.nrows() < (grb::IndexType)k)
    return 0;

  // Ориентирование графа
  MatrixT Or = orientById(G2);

  long long total = 0;
  grb::IndexType n2 = Or.nrows();

  // Параллелизм только на верхних уровнях
  bool doParallel = (depth <= 1);

  if (doParallel) {
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : total) schedule(dynamic)
#endif
    for (grb::IndexType v = 0; v < n2; ++v) {
      auto Nr = rightNeighbors(Or, v);
      if ((int)Nr.size() < k - 1)
        continue;

      MatrixT H = inducedSubgraph(G2, Nr);
      long long sub = countKCliquesRecursive(H, k - 1, depth + 1);
      total += sub;
    }
  } else {
    // Глубже - последовательно
    for (grb::IndexType v = 0; v < n2; ++v) {
      auto Nr = rightNeighbors(Or, v);
      if ((int)Nr.size() < k - 1)
        continue;

      MatrixT H = inducedSubgraph(G2, Nr);
      long long sub = countKCliquesRecursive(H, k - 1, depth + 1);
      total += sub;
    }
  }

  return total;
}

/**
 * Обёртка для подсчёта k-клик
 * @param G матрица смежности
 * @param k размер клики
 * @return количество k-клик
 */
template <typename MatrixT>
long long countKCliques_blas(const MatrixT &G, int k) {
  return countKCliquesRecursive(G, k, 0);
}
