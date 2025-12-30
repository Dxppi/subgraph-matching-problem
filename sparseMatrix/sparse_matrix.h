#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <algorithm>
#include <numeric>
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

  // ========== MATRIX MULTIPLICATION ==========
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

  // ========== HADAMARD (element-wise) PRODUCT ==========
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

  // ========== SUM OF ALL ELEMENTS ==========
  long long sumAll() const {
    long long s = 0;
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (int i = 0; i < n; ++i)
      for (auto &kv : a[i])
        s += kv.second;
    return s;
  }

  // ========== VERTEX DEGREES ==========
  vector<int> degrees() const {
    vector<int> d(n);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
      d[i] = (int)a[i].size();
    return d;
  }

  // ========== NUMBER OF NON-ZERO ELEMENTS ==========
  long long nnz() const {
    long long c = 0;
#pragma omp parallel for reduction(+ : c) schedule(static)
    for (int i = 0; i < n; ++i)
      c += (int)a[i].size();
    return c;
  }

  // ========== INDUCED SUBGRAPH ==========
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

  // ========== ALGORITHM 3: TRIANGLE COUNTING (Corrected) ==========
  long long countTriangles() const {
    // Step 1: Orient graph acyclically (i -> j if i < j)
    SparseMatrix A_dir = orientById();

    // Step 2: A* = (A_dir)^2 ⊙ A_dir
    SparseMatrix A2 = A_dir.multiply(A_dir);
    SparseMatrix T = A2.hadamard(A_dir);

    // Step 3: Sum all elements = number of triangles
    // NO division by 6 for directed graph!
    return T.sumAll();
  }

  // ========== ALGORITHM 8/9: 4-CLIQUE COUNTING (Corrected) ==========
  long long count4Cliques() const {
    SparseMatrix A_or = orientById(); // Orient graph
    long long total = 0;

#pragma omp parallel for reduction(+ : total) schedule(dynamic)
    for (int v = 0; v < n; ++v) {
      for (auto [u, _] : A_or.a[v]) { // for each outgoing edge v->u
        // Find common right neighbors of v and u
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
          // Create induced subgraph on common neighbors
          SparseMatrix H = A_or.inducedSubgraph(common);
          // CORRECTED: Count edges (triangles) in directed subgraph H
          total += H.sumAll(); // NOT H.nnz()!
        }
      }
    }
    return total;
  }

  // ========== ALGORITHM 5: FILTER VERTICES BY DEGREE ==========
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

  // ========== ALGORITHM 6: FILTER EDGES BY COMMON NEIGHBORS (LA-based)
  // ==========
  pair<SparseMatrix, bool> filterEdgesByCommonNeighbors(int minCommon) const {
    if (minCommon <= 0)
      return {*this, false};

    // Compute A' = A^2 ⊙ A (common neighbor counts)
    SparseMatrix A2 = multiply(*this);
    SparseMatrix A_prime = A2.hadamard(*this);

    // Filter edges with at least minCommon common neighbors
    SparseMatrix R(n);
    bool changed = false;

#pragma omp parallel for schedule(dynamic) reduction(| : changed)
    for (int i = 0; i < n; ++i) {
      for (auto &kv : A_prime.a[i]) {
        int j = kv.first;
        int common_count = kv.second;
        if (common_count >= minCommon) {
#pragma omp critical
          { R.set(i, j, 1); }
        } else {
          changed = true;
        }
      }
    }

    return {R, changed};
  }

  // ========== ITERATIVE EDGE FILTERING ==========
  SparseMatrix iterativeEdgeFiltering(int minCommon) const {
    SparseMatrix G = *this;
    while (true) {
      auto [Gnext, changed] = G.filterEdgesByCommonNeighbors(minCommon);
      if (!changed) {
        G = Gnext;
        break;
      }
      G = Gnext;

      // Compact: remove isolated vertices
      auto deg = G.degrees();
      vector<int> nonZero;
      for (int i = 0; i < G.size(); ++i) {
        if (deg[i] > 0)
          nonZero.push_back(i);
      }
      G = G.inducedSubgraph(nonZero);
    }
    return G;
  }

  // ========== ACYCLIC ORIENTATION BY ID ==========
  SparseMatrix orientById() const {
    SparseMatrix Or(n);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      for (auto &kv : a[i]) {
        int j = kv.first;
        if (i < j) {
          Or.a[i][j] = 1;
        }
      }
    }
    return Or;
  }

  // ========== GET ALL EDGES ==========
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