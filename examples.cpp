#include "la_algorithms.h"
#include "sparse_matrix.h"
#include <bits/stdc++.h>

using namespace std;

// вспомогательная функция: печать неориентированного графа
void printUndirected(const SparseMatrix &A) {
  cout << "Graph (n=" << A.size() << ", m≈" << A.nnz() / 2 << "):\n";
  for (int i = 0; i < A.size(); ++i) {
    cout << i << ":";
    auto row = A.rowIndices(i);
    for (int j : row) {
      cout << " " << j;
    }
    cout << "\n";
  }
}

void testCompleteGraphs() {
  cout << "=== Complete graphs K_n tests ===\n";
  for (int n = 3; n <= 8; ++n) {
    SparseMatrix A(n);
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) {
        A.set(i, j, 1);
        A.set(j, i, 1);
      }
    cout << "K_" << n << ":\n";
    for (int k = 3; k <= n; ++k) {
      long long ck = LA::countKCliques(A, k);
      // теоретически C(n,k)
      long long expected = 1;
      for (int i = 0; i < k; ++i)
        expected = expected * (n - i) / (i + 1);
      cout << "  k=" << k << "  count=" << ck << "  expected=" << expected
           << "\n";
    }
  }
  cout << "\n";
}

int main() {


  
  // Пример 1: треугольник K3
  {
    cout << "=== Example 1: triangle K3 ===\n";
    SparseMatrix A(3);
    A.set(0, 1, 1);
    A.set(1, 0, 1);
    A.set(1, 2, 1);
    A.set(2, 1, 1);
    A.set(0, 2, 1);
    A.set(2, 0, 1);

    printUndirected(A);
    cout << "Triangles (Alg.3) = " << LA::countTriangles(A) << "\n\n";
  }

  // Пример 2: K4 (4 вершины, полный граф)
  {
    cout << "=== Example 2: complete graph K4 ===\n";
    int n = 4;
    SparseMatrix A(n);
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) {
        A.set(i, j, 1);
        A.set(j, i, 1);
      }
    printUndirected(A);
    cout << "Triangles (Alg.3) = " << LA::countTriangles(A)
         << " (ожидается C(4,3)=4)\n\n";
  }

  // Пример 3: случайный граф и preprocessing Alg.5/6
  {
    cout << "=== Example 3: random graph + Alg.5/6 ===\n";
    int n = 8;
    double p = 0.4; // вероятность ребра
    SparseMatrix A(n);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) {
        if (dist(rng) < p) {
          A.set(i, j, 1);
          A.set(j, i, 1);
        }
      }

    printUndirected(A);
    cout << "Triangles (Alg.3) = " << LA::countTriangles(A) << "\n";

    int k = 4; // хотим искать 4-клики, значит фильтрация по k
    cout << "Preprocessing for k = " << k << "\n";

    auto pr = LA::deleteVerticesByDegree(A, k);
    SparseMatrix G1 = pr.first;
    cout << "After Alg.5: n' = " << G1.size() << ", m'≈" << G1.nnz() / 2
         << "\n";

    SparseMatrix G2 = LA::deleteEdgesByNeighborhood(G1, k);
    cout << "After Alg.6: n'' = " << G2.size() << ", m''≈" << G2.nnz() / 2
         << "\n";

    cout << "Oriented graph (by id):\n";
    auto Or = LA::orientById(G2);
    cout << "oriented arcs ≈ " << Or.nnz() << "\n\n";
  }

  // K4: 4 вершины, полный граф -> одна 4-клика
  {
    cout << "=== Check 4-cliques on K4 ===\n";
    int n = 4;
    SparseMatrix A(n);
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) {
        A.set(i, j, 1);
        A.set(j, i, 1);
      }
    cout << "4-cliques = " << LA::count4Cliques(A) << " (ожидается 1)\n\n";
  }

  // Пример: проверка countKCliques на K5
  {
    cout << "=== Example: k-cliques on K5 ===\n";
    int n = 5;
    SparseMatrix A(n);
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) {
        A.set(i, j, 1);
        A.set(j, i, 1);
      }

    // C(5,3) = 10, C(5,4) = 5, C(5,5) = 1
    cout << "3-cliques = " << LA::countKCliques(A, 3) << " (expected 10)\n";
    cout << "4-cliques = " << LA::countKCliques(A, 4) << " (expected 5)\n";
    cout << "5-cliques = " << LA::countKCliques(A, 5) << " (expected 1)\n\n";
  }

  return 0;
}
