#ifndef BLAS_MATRIX_HPP
#define BLAS_MATRIX_HPP

#include <algorithm>
#include <graphblas/graphblas.hpp>
#include <vector>

using namespace grb;

class BlasMatrix {
public:
  using MatrixType = Matrix<int>;

private:
  MatrixType matrix;
  int n;

public:
  // Конструктор
  explicit BlasMatrix(int n_ = 0) : n(n_), matrix(n_, n_) {}

  // Размер матрицы
  int size() const { return n; }

  // Установить значение элемента
  void set(int i, int j, int val) {
    if (val != 0) {
      matrix.setElement(i, j, val);
    }
  }

  // Получить значение элемента
  int get(int i, int j) const {
    int val;
    bool exists = matrix.extractElement(i, j);
    return exists ? val : 0;
  }

  // Получить матрицу для прямого доступа (для совместимости)
  const MatrixType &getMatrix() const { return matrix; }

  MatrixType &getMatrix() { return matrix; }

  // Транспонирование
  BlasMatrix transpose() const {
    BlasMatrix result(n);

    grb::transpose(result.matrix, grb::NoMask(), grb::NoAccumulate(),
                   this->matrix);

    return result;
  }

  // Умножение матриц (стандартное произведение)
  // C = A * B, где A = this
  BlasMatrix multiply(const BlasMatrix &b) const {
    BlasMatrix result(n);

    // C = A * B с операциями сложения и умножения
    mxm(result.matrix, NoMask(), NoAccumulate(),
        ArithmeticSemiring<int, int, int>(), this->matrix, b.matrix);

    return result;
  }

  // Поэлементное произведение (A ⊙ B) - Hadamard product
  BlasMatrix hadamard(const BlasMatrix &b) const {
    BlasMatrix result(n);
    eWiseMult(result.matrix, NoMask(), NoAccumulate(), Times<int>(),
              this->matrix, b.matrix);

    return result;
  }

  // Сумма всех элементов
  long long sumAll() const {
    long long sum = 0;

    // Редукция всех элементов
    reduce(sum, NoAccumulate(), Plus<long long>(), this->matrix);

    return sum;
  }

  // Количество ненулевых элементов
  long long nnz() const { return matrix.nvals(); }

  // Степени вершин (количество ненулевых элементов в каждой строке)
  std::vector<int> degrees() const {
    std::vector<int> d(n, 0);

    // Для каждой строки подсчитываем количество ненулевых элементов
    Vector<int> rowCounts(n);
    reduce(rowCounts, NoMask(), NoAccumulate(), Plus<int>(), this->matrix);

    // Извлекаем значения в вектор
    for (int i = 0; i < n; ++i) {
      int val;
      if (rowCounts.extractElement(i)) {
        d[i] = val;
      }
    }

    return d;
  }

  // Построить матрицу из координатного формата
  void buildFromCoordinates(const std::vector<int> &rows,
                            const std::vector<int> &cols,
                            const std::vector<int> &vals) {
    matrix.clear();
    for (size_t i = 0; i < rows.size(); ++i) {
      matrix.setElement(rows[i], cols[i], vals[i]);
    }
  }

  BlasMatrix scalarMultiply(int scalar) const {
    BlasMatrix result(n);

    apply(
        result.matrix, NoMask(), NoAccumulate(),
        [scalar](int x) { return x * scalar; }, this->matrix);

    return result;
  }

  // Поэлементное сложение (A ⊕ B)
  BlasMatrix eWiseAdding(const BlasMatrix &b) const {
    BlasMatrix result(n);

    eWiseAdd(result.matrix, NoMask(), NoAccumulate(), Plus<int>(), this->matrix,
             b.matrix);

    return result;
  }

  // Информация о матрице
  void printInfo() const {
    std::cout << "BlasMatrix " << n << "x" << n << " with " << nnz()
              << " non-zero elements\n";
  }
};

#endif // BLAS_MATRIX_HPP