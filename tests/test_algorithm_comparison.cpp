#include "graph_reader.hpp"
#include "sparse_matrix.h"
#include <chrono>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <la_algorithms.h>
#include <la_algorithms_blas.hpp>
#include <string>
#include <vector>

class KCliqueCountTest : public ::testing::Test {
protected:
  using MatrixType = grb::Matrix<long long>;
};

TEST_F(KCliqueCountTest, Compare_algorithms_fbook) {
  std::string filepath = std::string(TEST_DATA_DIR) + "/fbook.txt";
  MatrixType grbGraph = readGraphFromFile<MatrixType>(filepath);
  SparseMatrix sparseGraph = loadEdgeListFile(filepath);
  long long grb_result = countKCliques_blas(grbGraph, 3);
  long long sp_result = LA::countKCliquesRecursive(sparseGraph, 3, 0);
  ASSERT_EQ(grb_result, sp_result);
}
