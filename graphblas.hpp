#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#include <graphblas/graphblas.hpp>

using namespace grb;

std::default_random_engine generator;
std::uniform_real_distribution<double> distribution;

IndexType read_edge_list(std::string const &pathname,
                         IndexArrayType &Arow_indices,
                         IndexArrayType &Acol_indices,
                         IndexArrayType &Brow_indices,
                         IndexArrayType &Bcol_indices) {
  std::ifstream infile(pathname);
  IndexType max_id = 0;
  uint64_t num_rows = 0;
  uint64_t src, dst;

  while (infile) {
    infile >> src >> dst;
    // std::cout << "Read: " << src << ", " << dst << std::endl;
    max_id = std::max(max_id, src);
    max_id = std::max(max_id, dst);

    // if (src > max_id) max_id = src;
    // if (dst > max_id) max_id = dst;

    if (distribution(generator) < 0.85) {
      Arow_indices.push_back(src);
      Acol_indices.push_back(dst);
    }

    if (distribution(generator) < 0.85) {
      Brow_indices.push_back(src);
      Bcol_indices.push_back(dst);
    }

    ++num_rows;
  }
  std::cout << "Read " << num_rows << " rows." << std::endl;
  std::cout << "#Nodes = " << (max_id + 1) << std::endl;

  return (max_id + 1);
}

void main() {}