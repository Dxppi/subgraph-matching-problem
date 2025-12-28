#include "graph_reader.hpp"
#include "la_algorithms_blas.hpp"
#include <gtest/gtest.h>
#include <vector>

// ============================================================================
// Вспомогательные функции для создания тестовых графов
// ============================================================================
/**
 * Создает полный граф K_n
 */
template <typename MatrixT> MatrixT createCompleteGraph(grb::IndexType n) {
  MatrixT graph(n, n);
  std::vector<grb::IndexType> rows, cols;
  std::vector<typename MatrixT::ScalarType> vals;

  for (grb::IndexType i = 0; i < n; ++i) {
    for (grb::IndexType j = 0; j < n; ++j) {
      if (i != j) {
        rows.push_back(i);
        cols.push_back(j);
        vals.push_back(1);
      }
    }
  }

  graph.build(rows.begin(), cols.begin(), vals.begin(), vals.size());
  return graph;
}

/**
 * Создает пустой граф
 */
template <typename MatrixT> MatrixT createEmptyGraph(grb::IndexType n) {
  MatrixT graph(n, n);
  return graph;
}

/**
 * Создает цикл
 */
template <typename MatrixT> MatrixT createCycle(grb::IndexType n) {
  MatrixT graph(n, n);
  std::vector<grb::IndexType> rows, cols;
  std::vector<typename MatrixT::ScalarType> vals;

  for (grb::IndexType i = 0; i < n; ++i) {
    grb::IndexType next = (i + 1) % n;
    rows.push_back(i);
    cols.push_back(next);
    vals.push_back(1);
    rows.push_back(next);
    cols.push_back(i);
    vals.push_back(1);
  }

  graph.build(rows.begin(), cols.begin(), vals.begin(), vals.size());
  return graph;
}

/**
 * Создает граф-звезда
 */
template <typename MatrixT> MatrixT createStarGraph(grb::IndexType n) {
  MatrixT graph(n, n);
  std::vector<grb::IndexType> rows, cols;
  std::vector<typename MatrixT::ScalarType> vals;

  for (grb::IndexType i = 1; i < n; ++i) {
    rows.push_back(0);
    cols.push_back(i);
    vals.push_back(1);
    rows.push_back(i);
    cols.push_back(0);
    vals.push_back(1);
  }

  graph.build(rows.begin(), cols.begin(), vals.begin(), vals.size());
  return graph;
}

// ============================================================================
// Тесты для countKCliques_blas (рекурсивный алгоритм)
// ============================================================================

class KCliqueCountTest : public ::testing::Test {
protected:
  using MatrixType = grb::Matrix<long long>;
};

// k=3 базовый случай (треугольники)
TEST_F(KCliqueCountTest, K3_EmptyGraph) {
  MatrixType graph = createEmptyGraph<MatrixType>(5);
  long long result = countKCliques_blas(graph, 3);
  EXPECT_EQ(result, 0) << "Empty graph should have 0 triangles";
}

TEST_F(KCliqueCountTest, K3_CompleteGraphK4) {
  // K_4 содержит C(4,3) = 4 треугольника
  MatrixType graph = createCompleteGraph<MatrixType>(4);
  long long result = countKCliques_blas(graph, 3);
  EXPECT_EQ(result, 4) << "K_4 should have 4 triangles";
}

TEST_F(KCliqueCountTest, K3_CompleteGraphK5) {
  // K_5 содержит C(5,3) = 10 треугольников
  MatrixType graph = createCompleteGraph<MatrixType>(5);
  long long result = countKCliques_blas(graph, 3);
  EXPECT_EQ(result, 10) << "K_5 should have 10 triangles";
}

TEST_F(KCliqueCountTest, K3_CycleGraph) {
  // Цикл не содержит треугольников
  MatrixType graph = createCycle<MatrixType>(6);
  long long result = countKCliques_blas(graph, 3);
  EXPECT_EQ(result, 0) << "Cycle should have 0 triangles";
}

TEST_F(KCliqueCountTest, K3_StarGraph) {
  // Звёзда не содержит треугольников
  MatrixType graph = createStarGraph<MatrixType>(10);
  long long result = countKCliques_blas(graph, 3);
  EXPECT_EQ(result, 0) << "Star graph should have 0 triangles";
}

// k=4 (4-клики)
TEST_F(KCliqueCountTest, K4_EmptyGraph) {
  MatrixType graph = createEmptyGraph<MatrixType>(10);
  long long result = countKCliques_blas(graph, 4);
  EXPECT_EQ(result, 0) << "Empty graph should have 0 4-cliques";
}

TEST_F(KCliqueCountTest, K4_CompleteGraphK5) {
  // K_5 содержит C(5,4) = 5 четырёхклик
  MatrixType graph = createCompleteGraph<MatrixType>(5);
  long long result = countKCliques_blas(graph, 4);
  EXPECT_EQ(result, 5) << "K_5 should have 5 4-cliques";
}

TEST_F(KCliqueCountTest, K4_CompleteGraphK6) {
  // K_6 содержит C(6,4) = 15 четырёхклик
  MatrixType graph = createCompleteGraph<MatrixType>(6);
  long long result = countKCliques_blas(graph, 4);
  EXPECT_EQ(result, 15) << "K_6 should have 15 4-cliques";
}

TEST_F(KCliqueCountTest, K4_CycleGraph) {
  // Цикл не содержит 4-клик
  MatrixType graph = createCycle<MatrixType>(8);
  long long result = countKCliques_blas(graph, 4);
  EXPECT_EQ(result, 0) << "Cycle should have 0 4-cliques";
}

// k=5 (5-клики) - основной тест рекурсивного алгоритма
TEST_F(KCliqueCountTest, K5_CompleteGraphK5) {
  // K_5 содержит ровно одну 5-клику (само K_5)
  MatrixType graph = createCompleteGraph<MatrixType>(5);
  long long result = countKCliques_blas(graph, 5);
  EXPECT_EQ(result, 1) << "K_5 should have 1 5-clique (itself)";
}

TEST_F(KCliqueCountTest, K5_CompleteGraphK6) {
  // K_6 содержит C(6,5) = 6 пятиклик
  MatrixType graph = createCompleteGraph<MatrixType>(6);
  long long result = countKCliques_blas(graph, 5);
  EXPECT_EQ(result, 6) << "K_6 should have 6 5-cliques (C(6,5)=6)";
}

TEST_F(KCliqueCountTest, K5_CompleteGraphK7) {
  // K_7 содержит C(7,5) = 21 пятиклику
  MatrixType graph = createCompleteGraph<MatrixType>(7);
  long long result = countKCliques_blas(graph, 5);
  EXPECT_EQ(result, 21) << "K_7 should have 21 5-cliques (C(7,5)=21)";
}

TEST_F(KCliqueCountTest, K5_EmptyGraph) {
  MatrixType graph = createEmptyGraph<MatrixType>(10);
  long long result = countKCliques_blas(graph, 5);
  EXPECT_EQ(result, 0) << "Empty graph should have 0 5-cliques";
}

TEST_F(KCliqueCountTest, K5_StarGraph) {
  // Звёзда не содержит 5-клик
  MatrixType graph = createStarGraph<MatrixType>(15);
  long long result = countKCliques_blas(graph, 5);
  EXPECT_EQ(result, 0) << "Star graph should have 0 5-cliques";
}

// k=6 (6-клики) - проверка глубокой рекурсии
TEST_F(KCliqueCountTest, K6_CompleteGraphK6) {
  // K_6 содержит C(6,6) = 1 шестиклику (само K_6)
  MatrixType graph = createCompleteGraph<MatrixType>(6);
  long long result = countKCliques_blas(graph, 6);
  EXPECT_EQ(result, 1) << "K_6 should have 1 6-clique (itself)";
}

TEST_F(KCliqueCountTest, K6_CompleteGraphK7) {
  // K_7 содержит C(7,6) = 7 шестиклик
  MatrixType graph = createCompleteGraph<MatrixType>(7);
  long long result = countKCliques_blas(graph, 6);
  EXPECT_EQ(result, 7) << "K_7 should have 7 6-cliques (C(7,6)=7)";
}

TEST_F(KCliqueCountTest, K6_EmptyGraph) {
  MatrixType graph = createEmptyGraph<MatrixType>(20);
  long long result = countKCliques_blas(graph, 6);
  EXPECT_EQ(result, 0) << "Empty graph should have 0 6-cliques";
}

// k=7 (7-клики)
TEST_F(KCliqueCountTest, K7_CompleteGraphK7) {
  // K_7 содержит C(7,7) = 1 семиклику (само K_7)
  MatrixType graph = createCompleteGraph<MatrixType>(7);
  long long result = countKCliques_blas(graph, 7);
  EXPECT_EQ(result, 1) << "K_7 should have 1 7-clique (itself)";
}

TEST_F(KCliqueCountTest, K7_CompleteGraphK8) {
  // K_8 содержит C(8,7) = 8 семиклик
  MatrixType graph = createCompleteGraph<MatrixType>(8);
  long long result = countKCliques_blas(graph, 7);
  EXPECT_EQ(result, 8) << "K_8 should have 8 7-cliques (C(8,7)=8)";
}

// Граничные случаи
TEST_F(KCliqueCountTest, EdgeCase_KTooLargeForGraph) {
  // Граф K_5, ищем 10-клики (их нет)
  MatrixType graph = createCompleteGraph<MatrixType>(5);
  long long result = countKCliques_blas(graph, 10);
  EXPECT_EQ(result, 0) << "K_5 should have 0 10-cliques";
}

TEST_F(KCliqueCountTest, EdgeCase_K1) {
  // k=1 - одна вершина это клика?
  MatrixType graph = createCompleteGraph<MatrixType>(5);
  long long result = countKCliques_blas(graph, 1);
  EXPECT_EQ(result, 0) << "k=1 should return 0";
}

TEST_F(KCliqueCountTest, EdgeCase_K2) {
  // k=2 - рёбра? (не реализовано полностью)
  MatrixType graph = createCompleteGraph<MatrixType>(5);
  long long result = countKCliques_blas(graph, 2);
  EXPECT_EQ(result, 0) << "k=2 should return 0";
}

// ============================================================================
// Параметризованные тесты для полных графов K_n
// ============================================================================

class ParametrizedKCliqueTest
    : public ::testing::TestWithParam<
          std::tuple<grb::IndexType, int, long long>> {
protected:
  using MatrixType = grb::Matrix<long long>;

  static long long binomial(grb::IndexType n, int k) {
    if (k > (int)n || k < 0)
      return 0;
    if (k == 0 || k == (int)n)
      return 1;

    long long result = 1;
    for (int i = 0; i < k; ++i) {
      result = result * (n - i) / (i + 1);
    }
    return result;
  }
};

INSTANTIATE_TEST_SUITE_P(
    KCliqueCompleteGraphs, ParametrizedKCliqueTest,
    ::testing::Values(std::make_tuple(5LL, 3, 10LL), // K_5: C(5,3) = 10
                      std::make_tuple(5LL, 4, 5LL),  // K_5: C(5,4) = 5
                      std::make_tuple(5LL, 5, 1LL),  // K_5: C(5,5) = 1
                      std::make_tuple(6LL, 3, 20LL), // K_6: C(6,3) = 20
                      std::make_tuple(6LL, 4, 15LL), // K_6: C(6,4) = 15
                      std::make_tuple(6LL, 5, 6LL),  // K_6: C(6,5) = 6
                      std::make_tuple(6LL, 6, 1LL),  // K_6: C(6,6) = 1
                      std::make_tuple(7LL, 3, 35LL), // K_7: C(7,3) = 35
                      std::make_tuple(7LL, 4, 35LL), // K_7: C(7,4) = 35
                      std::make_tuple(7LL, 5, 21LL), // K_7: C(7,5) = 21
                      std::make_tuple(7LL, 6, 7LL),  // K_7: C(7,6) = 7
                      std::make_tuple(7LL, 7, 1LL),  // K_7: C(7,7) = 1
                      std::make_tuple(8LL, 4, 70LL), // K_8: C(8,4) = 70
                      std::make_tuple(8LL, 5, 56LL), // K_8: C(8,5) = 56
                      std::make_tuple(8LL, 6, 28LL), // K_8: C(8,6) = 28
                      std::make_tuple(8LL, 7, 8LL)   // K_8: C(8,7) = 8
                      ));

TEST_P(ParametrizedKCliqueTest, CompleteGraphKCliques) {
  auto [n, k, expected] = GetParam();
  grb::Matrix<long long> graph = createCompleteGraph<grb::Matrix<long long>>(n);
  long long result = countKCliques_blas(graph, k);
  EXPECT_EQ(result, expected)
      << "K_" << n << " should have " << expected << " " << k << "-cliques "
      << "(C(" << n << "," << k << ")=" << expected << ")";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
