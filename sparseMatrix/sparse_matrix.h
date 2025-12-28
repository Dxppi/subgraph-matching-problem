#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <algorithm>
#include <bits/stdc++.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace std;

class SparseMatrix {
public:
  using SparseRow = std::unordered_map<int, int>;

  int n;                    // размер n×n
  std::vector<SparseRow> a; // строки

  explicit SparseMatrix(int n_ = 0) : n(n_), a(n_) {}

  int size() const { return n; }

  void set(int i, int j, int val) {
    if (val)
      a[i][j] = val;
    else {
      auto it = a[i].find(j);
      if (it != a[i].end())
        a[i].erase(it);
    }
  }

  int get(int i, int j) const {
    auto it = a[i].find(j);
    return (it == a[i].end() ? 0 : it->second);
  }

  const SparseRow &row(int i) const { return a[i]; }

  std::vector<int> rowIndices(int i) const {
    std::vector<int> res;
    res.reserve(a[i].size());
    for (auto &p : a[i])
      res.push_back(p.first);
    std::sort(res.begin(), res.end());
    return res;
  }

  SparseMatrix transpose() const {
    SparseMatrix t(n);
    for (int i = 0; i < n; ++i)
      for (auto &p : a[i])
        t.set(p.first, i, p.second);
    return t;
  }

  SparseMatrix multiply(const SparseMatrix &b) const {
    SparseMatrix c(n);
    for (int i = 0; i < n; ++i) {
      for (auto &kv : a[i]) {
        int k = kv.first;
        int v1 = kv.second;
        for (auto &kv2 : b.a[k]) {
          int j = kv2.first;
          int v2 = kv2.second;
          int cur = c.get(i, j);
          c.set(i, j, cur + v1 * v2);
        }
      }
    }
    return c;
  }

  SparseMatrix hadamard(const SparseMatrix &b) const {
    SparseMatrix c(n);
    for (int i = 0; i < n; ++i) {
      for (auto &kv : a[i]) {
        int j = kv.first;
        auto it = b.a[i].find(j);
        if (it != b.a[i].end()) {
          c.set(i, j, kv.second * it->second);
        }
      }
    }
    return c;
  }

  long long sumAll() const {
    long long s = 0;
    for (int i = 0; i < n; ++i)
      for (auto &kv : a[i])
        s += kv.second;
    return s;
  }

  std::vector<int> degrees() const {
    std::vector<int> d(n, 0);
    for (int i = 0; i < n; ++i)
      d[i] = (int)a[i].size();
    return d;
  }

  long long nnz() const {
    long long c = 0;
    for (int i = 0; i < n; ++i)
      c += (int)a[i].size();
    return c;
  }
};

// ===== ФУНКЦИЯ ДЛЯ ЧТЕНИЯ ФАЙЛА (не метод класса!) =====
inline SparseMatrix loadEdgeListFile(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open file: " + path);
  }

  std::vector<std::pair<int, int>> edges;
  edges.reserve(1'000'000);

  int u, v;
  int maxV = -1;
  while (in >> u >> v) {
    if (u == v)
      continue;
    edges.emplace_back(u, v);
    maxV = std::max(maxV, std::max(u, v));
  }

  in.close();

  if (maxV < 0) {
    return SparseMatrix(0);
  }

  int n = maxV + 1;
  SparseMatrix A(n);
  for (auto &e : edges) {
    int x = e.first;
    int y = e.second;
    A.set(x, y, 1);
    A.set(y, x, 1);
  }

  std::cout << "Graph loaded from " << path << std::endl;
  std::cout << "  Vertices: " << n << std::endl;
  std::cout << "  Edges: " << edges.size() << std::endl;

  return A;
}

#endif // SPARSE_MATRIX_H
