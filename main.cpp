#include "la_algorithms.h"
#include "sparse_matrix.h"
#include <bits/stdc++.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// Чтение неориентированного графа из файла без заголовка:
// каждая строка: u v (0-based)
SparseMatrix loadEdgeListFile(const string &path) {
  ifstream in(path);
  if (!in) {
    throw runtime_error("Cannot open file: " + path);
  }

  vector<pair<int, int>> edges;
  edges.reserve(1'000'000);

  int u, v;
  int maxV = -1;
  while (in >> u >> v) {
    if (u == v)
      continue;
    edges.emplace_back(u, v);
    maxV = max(maxV, max(u, v));
  }

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

  return A;
}

#include <random>

void generateErdosRenyiEdgeList(const std::string &outPath, int n, double p,
                                uint32_t seed = 42) {
  std::mt19937 rng(seed);
  std::bernoulli_distribution bern(p);

  std::ofstream out(outPath);
  if (!out) {
    throw std::runtime_error("Cannot open output file: " + outPath);
  }

  // Перебираем пары (u,v), u < v
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
         << "  " << argv[0] << " data/graph.txt [k]\n"
         << "  " << argv[0] << " data/graph.txt exp\n"
         << "  " << argv[0] << " gen n p out.txt\n";
    return 1;
  }

  // -------- Режим генерации случайного графа --------
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
    SparseMatrix G = loadEdgeListFile(path);
    int n = G.size();
    long long m = G.nnz() / 2;

    cout << "Loaded graph from " << path << "\n";
    cout << "n = " << n << ", m ≈ " << m << "\n";

    if (n == 0) {
      cout << "Empty graph\n";
      return 0;
    }

    // -------- Обычный режим: ./prog graph.txt [k] --------
    if (argc == 2 || (argc >= 3 && string(argv[2]) != "exp")) {
      int k = 4;
      if (argc >= 3) {
        k = stoi(argv[2]);
      }

      // Треугольники
      auto t0 = chrono::high_resolution_clock::now();
      long long tri = LA::countTriangles(G);
      auto t1 = chrono::high_resolution_clock::now();
      auto ms_tri =
          chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

      cout << "Triangles (k=3) = " << tri << "  time = " << ms_tri << " ms\n";

      // 4-клики (Alg.8)
      t0 = chrono::high_resolution_clock::now();
      long long c4 = LA::count4Cliques(G);
      t1 = chrono::high_resolution_clock::now();
      auto ms_c4 = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

      cout << "4-cliques (Alg.8) = " << c4 << "  time = " << ms_c4 << " ms\n";

      // k-клики через Algorithm 7 (with stages)
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

    // -------- Режим экспериментов: ./prog graph.txt exp --------
    // Здесь делаем серию запусков по k и числу потоков
    vector<int> ks = {3, 4, 5, 6};
    vector<int> threads = {1, 2, 4, 8, 16, 24};

    // короткое имя графа для вывода
    string gname = path;
    {
      // отрежем директорию
      size_t pos = gname.find_last_of("/\\");
      if (pos != string::npos)
        gname = gname.substr(pos + 1);
    }

    cout << "### EXPERIMENT MODE ###\n";
    cout << "graph=" << gname << "\n";
    cout << "n=" << n << " m≈" << m << "\n\n";

    // Заголовок в CSV-стиле
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
