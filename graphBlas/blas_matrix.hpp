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
  explicit BlasMatrix(int n_ = 0) : matrix(n_, n_), n(n_) {}

  // Размер матрицы
  int size() const { return n; }

  // Установить значение элемента
  void set(int i, int j, int val) {
    if (val != 0) {
      matrix.setElement(i, j, val);
    }
  }

  // Получить значение элемента (ИЗМЕНЕНО: возвращает 0 если нет элемента)
  int get(int i, int j) const {
    if (matrix.hasElement(i, j)) {
      return matrix.extractElement(i, j);
    }
    return 0; // ← ИЗМЕНЕНО: возвращаем 0 вместо ошибки
  }

  // Получить матрицу для прямого доступа
  const MatrixType &getMatrix() const { return matrix; }
  MatrixType &getMatrix() { return matrix; }

  // Транспонирование
  BlasMatrix transpose() const {
    BlasMatrix result(n);
    grb::transpose(result.matrix, grb::NoMask(), grb::NoAccumulate(),
                   this->matrix);
    return result;
  }

  // Умножение матриц
  BlasMatrix multiply(const BlasMatrix &b) const {
    BlasMatrix result(n);
    mxm(result.matrix, NoMask(), NoAccumulate(),
        ArithmeticSemiring<int, int, int>(), this->matrix, b.matrix);
    return result;
  }

  // Поэлементное произведение (Hadamard product)
  BlasMatrix hadamard(const BlasMatrix &b) const {
    BlasMatrix result(n);
    eWiseMult(result.matrix, NoMask(), NoAccumulate(), Times<int>(),
              this->matrix, b.matrix);
    return result;
  }

  // Сумма всех элементов
  long long sumAll() const {
    int tmp = 0;
    grb::PlusMonoid<int> monoid;
    grb::reduce(tmp, grb::NoAccumulate(), monoid, this->matrix);
    return static_cast<long long>(tmp);
  }

  // Количество ненулевых элементов
  long long nnz() const { return matrix.nvals(); }

  // === ОПТИМИЗИРОВАННЫЙ метод для подсчёта степеней ===
  // Использует матрично-векторное умножение вместо полного проскана
  std::vector<int> degrees() const {
    std::vector<int> d(n, 0);

    // Инициализируем вектор единиц
    Vector<int> ones(n);
    for (int i = 0; i < n; ++i) {
      ones.setElement(i, 1);
    }

    // Умножаем матрицу на вектор единиц: A * ones = row_sums
    Vector<int> row_sums(n);
    mxv(row_sums, NoMask(), NoAccumulate(), ArithmeticSemiring<int, int, int>(),
        this->matrix, ones);

    // Извлекаем результаты
    for (int i = 0; i < n; ++i) {
      if (row_sums.hasElement(i)) {
        d[i] = row_sums.extractElement(i);
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

  // Скалярное умножение
  BlasMatrix scalarMultiply(int scalar) const {
    BlasMatrix result(n);
    apply(
        result.matrix, NoMask(), NoAccumulate(),
        [scalar](int x) { return x * scalar; }, this->matrix);
    return result;
  }

  // Поэлементное сложение
  BlasMatrix eWiseAdding(const BlasMatrix &b) const {
    BlasMatrix result(n);
    eWiseAdd(result.matrix, NoMask(), NoAccumulate(), Plus<int>(), this->matrix,
             b.matrix);
    return result;
  }
};

#endif // BLAS_MATRIX_HPP
