#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <algorithm>
#include <bits/stdc++.h>
#include <omp.h>
#include <unordered_map>
#include <vector>

using namespace std;

class SparseMatrix {
public:
  using SparseRow = unordered_map<int, int>;

  int n;
  vector<SparseRow> a;

  explicit SparseMatrix(int n_ = 0) : n(n_), a(n_) {}

  int size() const { return n; }

  void set(int i, int j, int val) {
    if (val)
      a[i][j] = val;
    else
      a[i].erase(j);
  }

  int get(int i, int j) const {
    auto it = a[i].find(j);
    return it == a[i].end() ? 0 : it->second;
  }

  vector<int> rowIndices(int i) const {
    vector<int> res;
    res.reserve(a[i].size());
    for (auto &p : a[i])
      res.push_back(p.first);
    sort(res.begin(), res.end());
    return res;
  }

  // Умножение матриц
  SparseMatrix multiply(const SparseMatrix &b) const {
    SparseMatrix c(n);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
      for (auto &kv : a[i]) {
        int k = kv.first;
        int v1 = kv.second;
        for (auto &kv2 : b.a[k]) {
          int j = kv2.first;
          int cur = c.get(i, j);
          c.set(i, j, cur + v1 * kv2.second);
        }
      }
    }
    return c;
  }

  // Поэлементное произведение (Hadamard product)
  SparseMatrix hadamard(const SparseMatrix &b) const {
    SparseMatrix c(n);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
      for (auto &kv : a[i]) {
        int j = kv.first;
        auto it = b.a[i].find(j);
        if (it != b.a[i].end())
          c.set(i, j, kv.second * it->second);
      }
    }
    return c;
  }

  // Сумма всех элементов
  long long sumAll() const {
    long long s = 0;
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (int i = 0; i < n; ++i)
      for (auto &kv : a[i])
        s += kv.second;
    return s;
  }

  // Степени вершин
  vector<int> degrees() const {
    vector<int> d(n);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
      d[i] = (int)a[i].size();
    return d;
  }

  // Количество ненулевых элементов
  long long nnz() const {
    long long c = 0;
#pragma omp parallel for reduction(+ : c) schedule(static)
    for (int i = 0; i < n; ++i)
      c += (int)a[i].size();
    return c;
  }

  // Индуцированный подграф на подмножестве вершин
  SparseMatrix inducedSubgraph(const vector<int> &subset) const {
    int k = (int)subset.size();
    SparseMatrix B(k);
    unordered_map<int, int> map;
    for (int i = 0; i < k; ++i)
      map[subset[i]] = i;
    for (int i = 0; i < k; ++i) {
      int u = subset[i];
      for (auto &kv : a[u]) {
        int v = kv.first;
        if (map.count(v)) {
          int j = map[v];
          if (i != j)
            B.set(i, j, 1);
        }
      }
    }
    return B;
  }

  // Подсчёт треугольников: sum(A² ⊙ A) / 6
  long long countTriangles() const {
    SparseMatrix A2 = multiply(*this);
    SparseMatrix T = A2.hadamard(*this);
    return T.sumAll() / 6;
  }

  // Подсчёт 4-клик через треугольники в окрестности каждой вершины
  long long count4Cliques() const {
    SparseMatrix A_or = orientById(); // ориентируем
    long long total = 0;

#pragma omp parallel for reduction(+ : total) schedule(dynamic)
    for (int v = 0; v < n; ++v) {
      for (auto [u, _] : A_or.a[v]) { // для каждого ребра v→u
        // mask_e = A[v][:] ⊙ (A[u][:])^T
        const auto &Av = A_or.a[v];
        const auto &Au = A_or.a[u];

        vector<int> common;
        if (Av.size() < Au.size()) {
          for (auto &p : Av) {
            if (Au.count(p.first))
              common.push_back(p.first);
          }
        } else {
          for (auto &p : Au) {
            if (Av.count(p.first))
              common.push_back(p.first);
          }
        }

        if (common.size() >= 2) {
          // A_e = A[mask_e][mask_e]
          SparseMatrix H = A_or.inducedSubgraph(common);

          // Count += SUM(A_e)  ← просто сумма элементов!
          total += H.nnz(); // вместо countTriangles()!
        }
      }
    }
    return total;
  }

  // Algorithm 5: удалить вершины со степенью < minDegree
  pair<SparseMatrix, vector<int>> filterByDegree(int minDegree) const {
    auto deg = degrees();
    vector<int> keepNodes;
    for (int i = 0; i < n; ++i) {
      if (deg[i] >= minDegree)
        keepNodes.push_back(i);
    }
    if (keepNodes.empty())
      return {SparseMatrix(0), {}};
    return {inducedSubgraph(keepNodes), keepNodes};
  }

  // Algorithm 6: удалить рёбра где |N(u) ∩ N(v)| < minCommon
  pair<SparseMatrix, bool> filterEdgesByCommonNeighbors(int minCommon) const {
    SparseMatrix R(n);
    bool changed = false;
    if (minCommon <= 0)
      return {*this, false};

#pragma omp parallel for schedule(dynamic) reduction(| : changed)
    for (int u = 0; u < n; ++u) {
      vector<int> Nu = rowIndices(u);
      for (int v : Nu) {
        if (u >= v)
          continue;
        vector<int> Nv = rowIndices(v);
        vector<int> inter;
        set_intersection(Nu.begin(), Nu.end(), Nv.begin(), Nv.end(),
                         back_inserter(inter));
        if ((int)inter.size() >= minCommon) {
#pragma omp critical
          {
            R.set(u, v, 1);
            R.set(v, u, 1);
          }
        } else {
          changed = true;
        }
      }
    }
    return {R, changed};
  }

  // Итеративная фильтрация рёбер с компактизацией
  SparseMatrix iterativeEdgeFiltering(int minCommon) const {
    SparseMatrix G = *this;
    while (true) {
      auto [Gnext, changed] = G.filterEdgesByCommonNeighbors(minCommon);
      if (!changed) {
        G = Gnext;
        break;
      }
      G = Gnext;
    }
    auto deg = G.degrees();
    vector<int> nonZero;
    for (int i = 0; i < G.size(); ++i) {
      if (deg[i] > 0)
        nonZero.push_back(i);
    }
    return G.inducedSubgraph(nonZero);
  }

  // Ориентация графа в DAG: i -> j если i < j
  SparseMatrix orientById() const {
    SparseMatrix Or(n);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
      for (auto &kv : a[i]) {
        if (i < kv.first) {
#pragma omp critical
          Or.set(i, kv.first, 1);
        }
      }
    }
    return Or;
  }

  // Извлечение всех рёбер как вектор пар
  vector<pair<int, int>> getAllEdges() const {
    vector<pair<int, int>> edges;
    edges.reserve(nnz());
    for (int u = 0; u < n; ++u) {
      for (auto &kv : a[u]) {
        edges.push_back({u, kv.first});
      }
    }
    return edges;
  }
};

#endif // SPARSE_MATRIX_H