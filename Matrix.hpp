#pragma once
#include <vector>
#include <cstddef>

struct Matrix {
    size_t n;
    std::vector<std::vector<size_t>> data;

    Matrix(size_t size) : n(size), data(size, std::vector<size_t>(size, 0)) {}

    void set(size_t i, size_t j, size_t val) {
        if (i < n && j < n) data[i][j] = val;
    }

    size_t get(size_t i, size_t j) const {
        if (i < n && j < n) return data[i][j];
        return 0;
    }

    Matrix multiply(const Matrix& other) const {
        Matrix res(n);
        for (size_t i = 0; i < n; ++i)
            for (size_t k = 0; k < n; ++k) {
                if (data[i][k] == 0) continue;
                for (size_t j = 0; j < n; ++j)
                    res.data[i][j] += data[i][k] * other.data[k][j];
            }
        return res;
    }

    Matrix hadamard(const Matrix& other) const {
        Matrix res(n);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                res.data[i][j] = data[i][j] * other.data[i][j];
        return res;
    }

    Matrix get_upper_triangular() const {
        Matrix res(n);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                res.data[i][j] = data[i][j];
        return res;
    }

    unsigned long long sum_all() const {
        unsigned long long total = 0;
        for (const auto& row : data)
            for (size_t val : row)
                total += val;
        return total;
    }

    Matrix extract_subgraph(const std::vector<size_t>& indices) const {
        size_t new_size = indices.size();
        Matrix res(new_size);
        for (size_t i = 0; i < new_size; ++i)
            for (size_t j = 0; j < new_size; ++j)
                if (data[indices[i]][indices[j]] != 0)
                    res.data[i][j] = 1;
        return res;
    }

    std::vector<size_t> get_neighbors(size_t row_idx) const {
        std::vector<size_t> neighbors;
        for (size_t j = 0; j < n; ++j)
            if (data[row_idx][j] != 0)
                neighbors.push_back(j);
        return neighbors;
    }
};