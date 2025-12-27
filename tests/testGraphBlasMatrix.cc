#include "blas_matrix.hpp"
#include "la_algorithms.hpp"
#include <chrono>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <set>
#include <vector>

// Простой таймер для замера времени выполнения
template <typename ClockT = std::chrono::steady_clock,
          typename DurationT = std::chrono::microseconds>
class Timer {
private:
  typename ClockT::time_point start_time;
  typename ClockT::time_point stop_time;

public:
  void start() { start_time = ClockT::now(); }

  void stop() { stop_time = ClockT::now(); }

  long long elapsed() const {
    auto duration =
        std::chrono::duration_cast<DurationT>(stop_time - start_time);
    return duration.count();
  }
};

// ============================================================================
// ТЕСТЫ БЛАЗ МАТРИЦЫ
// ============================================================================

// ТЕСТ 1: Базовые операции (set/get)
TEST(BlasMatrixTest, BasicOperations) {
  BlasMatrix A(3);

  // Установка элементов
  A.set(0, 1, 5);
  A.set(1, 2, 3);
  A.set(2, 0, 7);

  // Проверка получения элементов
  EXPECT_EQ(A.get(0, 1), 5);
  EXPECT_EQ(A.get(1, 2), 3);
  EXPECT_EQ(A.get(2, 0), 7);
  EXPECT_EQ(A.get(0, 0), 0); // Элемент не устанавливался
  EXPECT_EQ(A.nnz(), 3);
}

// ТЕСТ 2: Транспонирование
TEST(BlasMatrixTest, Transpose) {
  BlasMatrix A(4);
  A.set(0, 1, 2);
  A.set(0, 3, 4);
  A.set(2, 1, 5);
  A.set(3, 0, 6);

  BlasMatrix AT = A.transpose();

  // Проверка корректности транспонирования
  EXPECT_EQ(AT.get(1, 0), A.get(0, 1));
  EXPECT_EQ(AT.get(3, 0), A.get(0, 3));
  EXPECT_EQ(AT.get(1, 2), A.get(2, 1));
  EXPECT_EQ(AT.get(0, 3), A.get(3, 0));
  EXPECT_EQ(AT.nnz(), A.nnz());
}

// ТЕСТ 3: Умножение матриц
TEST(BlasMatrixTest, MatrixMultiplication) {
  BlasMatrix A(3);
  BlasMatrix B(3);

  // A диагональная: [1 2 0; 0 3 0; 0 0 4]
  A.set(0, 0, 1);
  A.set(0, 1, 2);
  A.set(1, 1, 3);
  A.set(2, 2, 4);

  // B единичная матрица
  B.set(0, 0, 1);
  B.set(1, 1, 1);
  B.set(2, 2, 1);

  BlasMatrix C = A.multiply(B);

  // C должна быть равна A (так как B - единичная)
  EXPECT_EQ(C.get(0, 0), 1);
  EXPECT_EQ(C.get(0, 1), 2);
  EXPECT_EQ(C.get(1, 1), 3);
  EXPECT_EQ(C.get(2, 2), 4);
}

// ТЕСТ 4: Степени матриц (пути в графе)
TEST(BlasMatrixTest, MatrixPowers) {
  BlasMatrix A(4);
  // Граф: 0->1, 1->2, 2->3, 3->0 (цикл)
  A.set(0, 1, 1);
  A.set(1, 2, 1);
  A.set(2, 3, 1);
  A.set(3, 0, 1);

  BlasMatrix A2 = A.multiply(A);
  EXPECT_EQ(A2.nnz(), 4); // Пути длины 2

  BlasMatrix A3 = A2.multiply(A);
  EXPECT_EQ(A3.nnz(), 4); // Пути длины 3

  BlasMatrix A4 = A3.multiply(A);
  EXPECT_EQ(A4.nnz(), 4); // Пути длины 4
}

// ТЕСТ 5: Поэлементное произведение (Hadamard)
TEST(BlasMatrixTest, HadamardProduct) {
  BlasMatrix A(3);
  BlasMatrix B(3);

  A.set(0, 0, 2);
  A.set(0, 1, 3);
  A.set(1, 1, 4);
  A.set(2, 2, 5);

  B.set(0, 0, 1);
  B.set(0, 1, 2);
  B.set(1, 0, 3);
  B.set(1, 1, 4);
  B.set(2, 2, 5);

  BlasMatrix C = A.hadamard(B);

  // Проверка результатов
  EXPECT_EQ(C.get(0, 0), 2);  // 2*1
  EXPECT_EQ(C.get(0, 1), 6);  // 3*2
  EXPECT_EQ(C.get(1, 1), 16); // 4*4
  EXPECT_EQ(C.get(2, 2), 25); // 5*5
  EXPECT_EQ(C.get(1, 0), 0);  // нет пары
}

// ТЕСТ 6: Сумма элементов
TEST(BlasMatrixTest, SumAll) {
  BlasMatrix A(3);
  A.set(0, 0, 1);
  A.set(0, 1, 2);
  A.set(1, 1, 3);
  A.set(2, 2, 4);

  long long sum = A.sumAll();
  EXPECT_EQ(sum, 10); // 1+2+3+4
}

// ТЕСТ 7: Степени вершин
TEST(BlasMatrixTest, VertexDegrees) {
  BlasMatrix A(5);
  A.set(0, 1, 1);
  A.set(0, 2, 1);
  A.set(1, 2, 1);
  A.set(1, 3, 1);
  A.set(2, 3, 1);
  A.set(2, 4, 1);

  std::vector<int> degrees = A.degrees();

  EXPECT_EQ(degrees[0], 2);
  EXPECT_EQ(degrees[1], 2);
  EXPECT_EQ(degrees[2], 2);
  EXPECT_EQ(degrees[3], 0);
  EXPECT_EQ(degrees[4], 0);
}

// ТЕСТ 8: Поэлементное сложение
TEST(BlasMatrixTest, ElementWiseAddition) {
  BlasMatrix A(3);
  BlasMatrix B(3);

  A.set(0, 0, 1);
  A.set(0, 1, 2);
  A.set(1, 1, 3);

  B.set(0, 1, 4);
  B.set(1, 1, 5);
  B.set(2, 2, 6);

  BlasMatrix C = A.eWiseAdding(B);

  EXPECT_EQ(C.get(0, 0), 1);
  EXPECT_EQ(C.get(0, 1), 6); // 2+4
  EXPECT_EQ(C.get(1, 1), 8); // 3+5
  EXPECT_EQ(C.get(2, 2), 6);
}

// ТЕСТ 9: Умножение на скаляр
TEST(BlasMatrixTest, ScalarMultiply) {
  BlasMatrix A(3);
  A.set(0, 1, 2);
  A.set(1, 2, 3);
  A.set(2, 0, 4);

  BlasMatrix B = A.scalarMultiply(5);

  EXPECT_EQ(B.get(0, 1), 10); // 2*5
  EXPECT_EQ(B.get(1, 2), 15); // 3*5
  EXPECT_EQ(B.get(2, 0), 20); // 4*5
}

// ТЕСТ 10: Случайная разреженная матрица (производительность)
TEST(BlasMatrixTest, RandomSparseMatrixPerformance) {
  int n = 100;
  double sparsity = 0.05; // 5% ненулевых элементов

  std::mt19937 gen(42);
  std::uniform_int_distribution<> row_dist(0, n - 1);
  std::uniform_int_distribution<> col_dist(0, n - 1);
  std::uniform_int_distribution<> val_dist(1, 100);

  BlasMatrix A(n);
  BlasMatrix B(n);

  int nnz_target = (int)(n * n * sparsity);

  Timer<> timer;
  timer.start();

  std::set<std::pair<int, int>> inserted;
  for (int idx = 0; idx < nnz_target; ++idx) {
    int i, j;
    do {
      i = row_dist(gen);
      j = col_dist(gen);
    } while (inserted.count({i, j}));

    inserted.insert({i, j});
    A.set(i, j, val_dist(gen));
    B.set(i, j, val_dist(gen));
  }

  timer.stop();
  long long fill_time = timer.elapsed();

  EXPECT_GT(A.nnz(), 0);
  EXPECT_GT(B.nnz(), 0);

  // Тест умножения
  timer.start();
  BlasMatrix C = A.multiply(B);
  timer.stop();
  long long mult_time = timer.elapsed();

  EXPECT_GT(C.nnz(), 0);

  // Тест транспонирования
  timer.start();
  BlasMatrix AT = A.transpose();
  timer.stop();
  long long transpose_time = timer.elapsed();

  EXPECT_EQ(AT.nnz(), A.nnz());

  // Вывод времени для информации
  std::cout << "\n[PERFORMANCE] 100x100 матрица с 5% элементов:\n"
            << "  Заполнение: " << fill_time << " мкс\n"
            << "  Умножение: " << mult_time << " мкс\n"
            << "  Транспонирование: " << transpose_time << " мкс\n";
}

// ТЕСТ 11: Подсчет треугольников
TEST(BlasMatrixTest, TriangleCounting) {
  BlasMatrix A(5);
  // Неориентированный граф с двумя треугольниками

  // Треугольник 1: 0-1-2
  A.set(0, 1, 1);
  A.set(1, 0, 1);
  A.set(0, 2, 1);
  A.set(2, 0, 1);
  A.set(1, 2, 1);
  A.set(2, 1, 1);

  // Треугольник 2: 1-2-3
  A.set(1, 3, 1);
  A.set(3, 1, 1);
  A.set(2, 3, 1);
  A.set(3, 2, 1);

  BlasMatrix A2 = A.multiply(A);
  BlasMatrix A3 = A2.multiply(A);

  // Подсчитываем trace(A³)
  long long trace = 0;
  for (int i = 0; i < 5; ++i) {
    trace += A3.get(i, i);
  }

  long long num_triangles = trace / 6; // Для неориентированного графа

  EXPECT_EQ(num_triangles, 2);
}

// ТЕСТ 12: Чтение из файла (если файл существует)
TEST(BlasMatrixTest, ReadFromFile) {
  std::string filename = std::string(TEST_DATA_DIR) + "/test_graph.mtx";
  std::ifstream infile(filename);

  if (!infile.is_open()) {
    GTEST_SKIP() << "Файл " << filename << " не найден. Пропускаем тест.";
  }

  grb::IndexType max_id = 0;
  std::vector<std::pair<grb::IndexType, grb::IndexType>> edges;
  std::string line;

  int line_count = 0;
  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#')
      continue;

    grb::IndexType src, dst;
    std::istringstream iss(line);
    if (iss >> src >> dst) {
      max_id = std::max(max_id, std::max(src, dst));
      edges.push_back({src, dst});
      line_count++;
    }
  }

  infile.close();

  EXPECT_GT(line_count, 0);

  // Создаём матрицу только если граф небольшой
  if (max_id < 10000 && edges.size() > 0) {
    BlasMatrix G(max_id + 1);

    for (auto &edge : edges) {
      G.set(edge.first, edge.second, 1);
    }

    EXPECT_EQ(G.nnz(), edges.size());
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
