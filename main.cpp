#include "graph_loader.hpp"
#include "la_algorithms.h"
#include "sparse_matrix.h"
#include <bits/stdc++.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

#include <random>

void generateErdosRenyiEdgeList(const std::string &outPath, int n, double p,
                                uint32_t seed = 42) {
  std::mt19937 rng(seed);
  std::bernoulli_distribution bern(p);

  std::ofstream out(outPath);
  if (!out) {
    throw std::runtime_error("Cannot open output file: " + outPath);
  }

  for (int u = 0; u < n; ++u) {
    for (int v = u + 1; v < n; ++v) {
      if (bern(rng)) {
        out << u << " " << v << "\n";
      }
    }
  }
}

int main(int argc, char **argv) {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  if (argc < 2) {
    cerr << "Usage:\n"
         << "  " << argv[0] << " ../data/graph.txt [k]\n"
         << "  " << argv[0] << " ../data/graph.txt exp\n"
         << "  " << argv[0] << " gen n p out.txt\n";
    return 1;
  }

  if (std::string(argv[1]) == "gen") {
    if (argc < 5) {
      cerr << "Usage: " << argv[0] << " gen n p out.txt\n";
      return 1;
    }
    int n = std::stoi(argv[2]);
    double p = std::stod(argv[3]);
    std::string outPath = argv[4];

    try {
      generateErdosRenyiEdgeList(outPath, n, p, 42);
      cerr << "Generated G(" << n << "," << p << ") into " << outPath << "\n";
    } catch (const std::exception &e) {
      cerr << "Error in generator: " << e.what() << "\n";
      return 1;
    }
    return 0;
  }

  string path = argv[1];

  try {
    SparseMatrix G = GraphLoader::loadEdgeListFile(path);
    int n = G.size();
    long long m = G.nnz() / 2;

    cout << "Loaded graph from " << path << "\n";
    cout << "n = " << n << ", m ≈ " << m << "\n";

    if (n == 0) {
      cout << "Empty graph\n";
      return 0;
    }

    if (argc == 2 || (argc >= 3 && string(argv[2]) != "exp")) {
      int k = 4;
      if (argc >= 3) {
        k = stoi(argv[2]);
      }

      auto t0 = chrono::high_resolution_clock::now();
      long long tri = G.countTriangles();
      auto t1 = chrono::high_resolution_clock::now();
      auto ms_tri =
          chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

      cout << "Triangles (k=3) = " << tri << "  time = " << ms_tri << " ms\n";
      if (k >= 3) {
        t0 = chrono::high_resolution_clock::now();
        long long ck = LA::countKCliquesWithStages(G, k);
        t1 = chrono::high_resolution_clock::now();
        auto ms_ck =
            chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

        cout << k << "-cliques (Alg.7) = " << ck << "  time = " << ms_ck
             << " ms\n";
      }

      return 0;
    }

    vector<int> ks = {3, 4, 5, 6};
    vector<int> threads = {24};

    string gname = path;
    {
      size_t pos = gname.find_last_of("/\\");
      if (pos != string::npos)
        gname = gname.substr(pos + 1);
    }

    cout << "### EXPERIMENT MODE ###\n";
    cout << "graph=" << gname << "\n";
    cout << "n=" << n << " m≈" << m << "\n\n";

    cout << "graph;k;threads;time_ms;cliques\n";

    for (int k : ks) {
      for (int t : threads) {
#ifdef _OPENMP
        omp_set_num_threads(t);
#endif
        auto t0 = chrono::high_resolution_clock::now();
        long long ck = LA::countKCliquesWithStages(G, k);
        auto t1 = chrono::high_resolution_clock::now();
        auto ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

        cout << gname << ";" << k << ";" << t << ";" << ms << ";" << ck << "\n";
      }
    }

    return 0;

  } catch (const exception &e) {
    cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}