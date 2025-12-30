// graph_loader.hpp
#ifndef GRAPH_LOADER_HPP
#define GRAPH_LOADER_HPP

#include "sparse_matrix.h"
#include <fstream>
#include <iostream>
#include <vector>

class GraphLoader {
public:
  // Загрузить граф из edge list файла (u v в каждой строке)
  static SparseMatrix loadEdgeListFile(const std::string &path) {
    std::ifstream in(path);

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

    if (maxV < 0)
      return SparseMatrix(0);

    int n = maxV + 1;
    SparseMatrix A(n);
    for (const auto &[x, y] : edges) {
      A.set(x, y, 1);
      A.set(y, x, 1);
    }

    std::cout << "Loaded: " << n << " vertices, " << edges.size() << " edges\n";
    return A;
  }
};

#endif