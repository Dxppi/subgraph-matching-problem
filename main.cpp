#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "Algorithms.hpp"
#include "Matrix.hpp"
#include "ParallelAlgorithms.hpp"
#include "LAGraphAlgorithms.hpp"

Matrix create_complete_graph(size_t n) {
  Matrix adj(n);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
      if (i != j)
        adj.set(i, j, 1);
  return adj;
}

Matrix create_random_graph(size_t n, double edge_probability) {
  Matrix adj(n);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (dis(gen) < edge_probability) {
        adj.set(i, j, 1);
        adj.set(j, i, 1);
      }
    }
  }
  return adj;
}

GrB_Matrix convert_to_grb_uint64(const Matrix &G) {
  GrB_Matrix A = NULL;
  GrB_Info info = GrB_Matrix_new(&A, GrB_UINT64, G.n, G.n);
  if (info != GrB_SUCCESS) {
    std::cerr << "Ошибка создания GrB_Matrix\n";
    return NULL;
  }

  for (size_t i = 0; i < G.n; ++i) {
    for (size_t j = 0; j < G.n; ++j) {
      if (G.get(i, j) != 0) {
        GrB_Matrix_setElement_UINT64(A, 1, i, j);
      }
    }
  }

  return A;
}

GrB_Matrix convert_to_grb_bool(const Matrix &G) {
  GrB_Matrix A = NULL;
  GrB_Info info = GrB_Matrix_new(&A, GrB_BOOL, G.n, G.n);
  if (info != GrB_SUCCESS) {
    std::cerr << "Ошибка создания GrB_Matrix\n";
    return NULL;
  }

  for (size_t i = 0; i < G.n; ++i) {
    for (size_t j = 0; j < G.n; ++j) {
      if (G.get(i, j) != 0) {
        GrB_Matrix_setElement_BOOL(A, true, i, j);
      }
    }
  }

  return A;
}

uint64_t count_triangles_lagraph(const Matrix &graph) {
  GrB_Matrix lag_graph = convert_to_grb_bool(graph);
  if (lag_graph == NULL) {
    return 0;
  }
  
  uint64_t result = lagraph_count_triangles(lag_graph);
  GrB_Matrix_free(&lag_graph);
  return result;
}

void test_and_compare(const Matrix &graph, const std::string &graph_name,
                      size_t k, uint64_t expected = 0, bool use_parallel = false) {
  std::cout << "\n========================================\n";
  std::cout << "Граф: " << graph_name << ", k=" << k;
  if (use_parallel) {
    #ifdef _OPENMP
    std::cout << " [OpenMP ПАРАЛЛЕЛЬНО - " << omp_get_max_threads() << " потоков]";
    #else
    std::cout << " [OpenMP запрошен, но недоступен]";
    #endif
  }
  std::cout << "\n";
  if (expected > 0) {
    std::cout << "Ожидаемый результат: " << expected << "\n";
  }
  std::cout << "========================================\n";

  GrB_Matrix grb_graph = convert_to_grb_uint64(graph);
  if (grb_graph == NULL) {
    std::cerr << "Ошибка конвертации графа в GrB_Matrix\n";
    return;
  }

  auto start_seq = std::chrono::high_resolution_clock::now();
  count_t result_seq = solve_k_clique(graph, k, use_parallel);
  auto end_seq = std::chrono::high_resolution_clock::now();
  auto duration_seq_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_seq - start_seq)
                          .count();

  auto start_grb = std::chrono::high_resolution_clock::now();
  uint64_t result_grb = solve_k_clique_grb(grb_graph, k);
  auto end_grb = std::chrono::high_resolution_clock::now();
  auto duration_grb_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_grb - start_grb)
                          .count();

  uint64_t result_lag = 0;
  int64_t duration_lag_ms = 0;
  if (k == 3) {
    auto start_lag = std::chrono::high_resolution_clock::now();
    result_lag = count_triangles_lagraph(graph);
    auto end_lag = std::chrono::high_resolution_clock::now();
    duration_lag_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      end_lag - start_lag)
                      .count();
  }

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Последовательный результат: " << result_seq << "\n";
  std::cout << "GraphBLAS результат:       " << result_grb << "\n";
  if (k == 3) {
    std::cout << "LAGraph результат:         " << result_lag << "\n";
  }
  std::cout << "\n";

  if (result_seq != result_grb) {
    std::cerr << "ОШИБКА: Результаты не совпадают!\n";
  }

  if (k == 3 && result_lag > 0 && result_seq != result_lag) {
    std::cerr << "ПРЕДУПРЕЖДЕНИЕ: Результат LAGraph не совпадает!\n";
  }

  if (expected > 0 && (result_seq != expected || result_grb != expected)) {
    std::cerr << "ПРЕДУПРЕЖДЕНИЕ: Результаты не совпадают с ожидаемым значением!\n";
  }

  std::cout << "\nПроизводительность:\n";
  if (duration_seq_ms < 1000) {
    std::cout << "  Последовательно: " << duration_seq_ms << " мс\n";
  } else {
    std::cout << "  Последовательно: " << duration_seq_ms / 1000.0 << " с\n";
  }
  if (duration_grb_ms < 1000) {
    std::cout << "  GraphBLAS:       " << duration_grb_ms << " мс\n";
  } else {
    std::cout << "  GraphBLAS:       " << duration_grb_ms / 1000.0 << " с\n";
  }
  if (k == 3 && duration_lag_ms > 0) {
    if (duration_lag_ms < 1000) {
      std::cout << "  LAGraph:         " << duration_lag_ms << " мс\n";
    } else {
      std::cout << "  LAGraph:         " << duration_lag_ms / 1000.0 << " с\n";
    }
  }

  if (duration_seq_ms > 0 && duration_grb_ms > 0) {
    double speedup = (double)duration_seq_ms / duration_grb_ms;
    std::cout << "  Ускорение (GraphBLAS): " << std::setprecision(2) << speedup << "x";
    if (speedup > 1.0) {
      std::cout << " (GraphBLAS быстрее)";
    } else if (speedup < 1.0) {
      std::cout << " (Последовательно быстрее)";
    } else {
      std::cout << " (Равно)";
    }
    std::cout << "\n";
  }

  if (k == 3 && duration_lag_ms > 0 && duration_seq_ms > 0) {
    double speedup_lag = (double)duration_seq_ms / duration_lag_ms;
    std::cout << "  Ускорение (LAGraph): " << std::setprecision(2) << speedup_lag << "x";
    if (speedup_lag > 1.0) {
      std::cout << " (LAGraph быстрее)";
    } else if (speedup_lag < 1.0) {
      std::cout << " (Последовательно быстрее)";
    } else {
      std::cout << " (Равно)";
    }
    std::cout << "\n";
  }

  GrB_Matrix_free(&grb_graph);
}

int main() {
  #ifdef _OPENMP
  std::cout << "OpenMP включен: " << omp_get_max_threads() << " потоков доступно\n";
  #else
  std::cout << "OpenMP НЕ включен (скомпилировано без -fopenmp)\n";
  #endif
  
  if (GrB_init(GrB_NONBLOCKING) != GrB_SUCCESS) {
    std::cerr << "Ошибка инициализации GraphBLAS\n";
    return -1;
  }

  if (LAGraph_Init(NULL) != 0) {
    std::cerr << "Ошибка инициализации LAGraph\n";
    GrB_finalize();
    return -1;
  }

  std::cout << "========================================\n";
  std::cout << "  Сравнение Последовательных и GraphBLAS\n";
  std::cout << "  Алгоритмов Подсчета K-Клик\n";
  std::cout << "========================================\n";

  Matrix K5 = create_complete_graph(5);
  test_and_compare(K5, "K5 (Полный граф, n=5)", 3, 10);
  test_and_compare(K5, "K5 (Полный граф, n=5)", 4, 5);
  test_and_compare(K5, "K5 (Полный граф, n=5)", 5, 1);

  Matrix K6 = create_complete_graph(6);
  test_and_compare(K6, "K6 (Полный граф, n=6)", 3, 20);
  test_and_compare(K6, "K6 (Полный граф, n=6)", 4, 15);
  test_and_compare(K6, "K6 (Полный граф, n=6)", 5, 6);
  test_and_compare(K6, "K6 (Полный граф, n=6)", 6, 1);

  Matrix K7 = create_complete_graph(7);
  test_and_compare(K7, "K7 (Полный граф, n=7)", 3, 35);
  test_and_compare(K7, "K7 (Полный граф, n=7)", 4, 35);
  test_and_compare(K7, "K7 (Полный граф, n=7)", 5, 21);

  Matrix random_graph = create_random_graph(20, 0.3);
  test_and_compare(random_graph, "Случайный граф (n=20, p=0.3)", 3);
  test_and_compare(random_graph, "Случайный граф (n=20, p=0.3)", 4);

  std::cout << "\n\n========================================\n";
  std::cout << "  Тест Производительности на Большом Графе\n";
  std::cout << "========================================\n";
  Matrix K8 = create_complete_graph(8);
  test_and_compare(K8, "K8 (Полный граф, n=8)", 4, 70);
  test_and_compare(K8, "K8 (Полный граф, n=8)", 5, 56);

  std::cout << "\n\n========================================\n";
  std::cout << "  ТЕСТ БОЛЬШОГО ГРАФА (n=10000)\n";
  std::cout << "  Должен показать преимущества GraphBLAS/LAGraph!\n";
  std::cout << "========================================\n";
  
  std::cout << "\nСоздание большого случайного графа (n=10000, p=0.01)...\n";
  auto start_create = std::chrono::high_resolution_clock::now();
  Matrix large_graph = create_random_graph(10000, 0.01);
  auto end_create = std::chrono::high_resolution_clock::now();
  auto duration_create = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_create - start_create)
                            .count();
  std::cout << "Время создания графа: " << duration_create << " мс\n";
  
  size_t edge_count = 0;
  for (size_t i = 0; i < large_graph.n; ++i) {
    for (size_t j = i + 1; j < large_graph.n; ++j) {
      if (large_graph.get(i, j) != 0) {
        edge_count++;
      }
    }
  }
  std::cout << "Граф имеет " << edge_count << " ребер\n\n";

  test_and_compare(large_graph, "Большой случайный граф (n=10000, p=0.01)", 3);

  std::cout << "\n\n========================================\n";
  std::cout << "  ТЕСТ ОЧЕНЬ БОЛЬШОГО ГРАФА (n=5000, p=0.05)\n";
  std::cout << "========================================\n";
  
  std::cout << "\nСоздание очень большого случайного графа (n=5000, p=0.05)...\n";
  start_create = std::chrono::high_resolution_clock::now();
  Matrix very_large_graph = create_random_graph(5000, 0.05);
  end_create = std::chrono::high_resolution_clock::now();
  duration_create = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_create - start_create)
                            .count();
  std::cout << "Время создания графа: " << duration_create << " мс\n";
  
  edge_count = 0;
  for (size_t i = 0; i < very_large_graph.n; ++i) {
    for (size_t j = i + 1; j < very_large_graph.n; ++j) {
      if (very_large_graph.get(i, j) != 0) {
        edge_count++;
      }
    }
  }
  std::cout << "Граф имеет " << edge_count << " ребер\n\n";

  test_and_compare(very_large_graph, "Очень большой случайный граф (n=5000, p=0.05)", 3);

  std::cout << "\n\n========================================\n";
  std::cout << "  ТЕСТ ПАРАЛЛЕЛЬНОЙ ВЕРСИИ\n";
  std::cout << "========================================\n";
  
  Matrix test_parallel = create_random_graph(2000, 0.05);
  std::cout << "\nТестирование с включенной параллелизацией OpenMP...\n";
  test_and_compare(test_parallel, "Тестовый граф (n=2000, p=0.05) - ПАРАЛЛЕЛЬНО", 3, 0, true);
  std::cout << "\nТестирование того же графа без параллелизации...\n";
  test_and_compare(test_parallel, "Тестовый граф (n=2000, p=0.05) - ПОСЛЕДОВАТЕЛЬНО", 3, 0, false);

  std::cout << "\n\n========================================\n";
  std::cout << "  Тестирование Завершено\n";
  std::cout << "========================================\n";

  LAGraph_Finalize(NULL);
  GrB_finalize();

  return 0;
}
