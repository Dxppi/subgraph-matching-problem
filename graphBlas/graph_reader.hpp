#ifndef GRAPH_READER_H
#define GRAPH_READER_H

#include "graphblas/graphblas.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

/**
 * Читает граф из файла в формате:
 * каждая строка: u v (ребро между вершинами u и v)
 *
 * Пример файла:
 * 0 1
 * 0 2
 * 0 3
 * 1 2
 * 2 3
 *
 * @param filename путь к файлу
 * @return матрица смежности неориентированного графа
 */
template <typename MatrixT>
MatrixT readGraphFromFile(const std::string &filename) {
  using T = typename MatrixT::ScalarType;

  // Шаг 1: Открываем файл
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  // Шаг 2: Инициализируем контейнеры для рёбер
  std::vector<grb::IndexType> rows;
  std::vector<grb::IndexType> cols;
  std::vector<T> vals;
  grb::IndexType max_vertex = 0;

  // Шаг 3: Читаем строки из файла
  std::string line;
  int edge_count = 0;

  while (std::getline(file, line)) {
    // Пропускаем пустые строки и строки только с пробелами
    if (line.empty() ||
        line.find_first_not_of(" \t\r\n") == std::string::npos) {
      continue;
    }

    // Парсим строку
    std::istringstream iss(line);
    int u, v;

    // Читаем два целых числа из строки
    if (iss >> u >> v) {
      // Шаг 4: Добавляем ребро u->v (для неориентированного графа)
      rows.push_back(u);
      cols.push_back(v);
      vals.push_back(1);

      // Шаг 5: Добавляем обратное ребро v->u (неориентированность)
      rows.push_back(v);
      cols.push_back(u);
      vals.push_back(1);

      // Шаг 6: Обновляем максимальный индекс вершины
      max_vertex = std::max(max_vertex, (grb::IndexType)std::max(u, v));
      edge_count++;
    }
  }

  // Шаг 7: Закрываем файл
  file.close();

  // Шаг 8: Проверяем, что рёбра были прочитаны
  if (rows.empty()) {
    throw std::runtime_error("No edges found in file: " + filename);
  }

  // Шаг 9: Создаём матрицу размером (max_vertex+1) x (max_vertex+1)
  // +1 потому что вершины нумеруются с 0
  grb::IndexType n = max_vertex + 1;
  MatrixT graph(n, n);

  // Шаг 10: Строим матрицу смежности из прочитанных рёбер
  graph.build(rows.begin(), cols.begin(), vals.begin(), vals.size());

  // Выводим информацию о загрузке
  std::cout << "Graph loaded from " << filename << std::endl;
  std::cout << "  Vertices: " << n << std::endl;
  std::cout << "  Edges: " << edge_count << std::endl;

  return graph;
}

/**
 * Читает граф с информацией о статистике
 */
struct GraphInfo {
  grb::IndexType num_vertices;
  grb::IndexType num_edges;
  double density;
};

template <typename MatrixT>
std::pair<MatrixT, GraphInfo> readGraphWithInfo(const std::string &filename) {
  using T = typename MatrixT::ScalarType;

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  std::vector<grb::IndexType> rows, cols;
  std::vector<T> vals;
  grb::IndexType max_vertex = 0;

  std::string line;
  int edge_count = 0;

  while (std::getline(file, line)) {
    if (line.empty() ||
        line.find_first_not_of(" \t\r\n") == std::string::npos) {
      continue;
    }

    std::istringstream iss(line);
    int u, v;

    if (iss >> u >> v) {
      rows.push_back(u);
      cols.push_back(v);
      vals.push_back(1);

      rows.push_back(v);
      cols.push_back(u);
      vals.push_back(1);

      max_vertex = std::max(max_vertex, (grb::IndexType)std::max(u, v));
      edge_count++;
    }
  }

  file.close();

  if (rows.empty()) {
    throw std::runtime_error("No edges found in file: " + filename);
  }

  grb::IndexType n = max_vertex + 1;
  MatrixT graph(n, n);
  graph.build(rows.begin(), cols.begin(), vals.begin(), vals.size());

  // Вычисляем информацию о графе
  GraphInfo info;
  info.num_vertices = n;
  info.num_edges = edge_count;
  // Плотность = 2E / (V*(V-1)) для неориентированного графа
  info.density = (2.0 * edge_count) / (n * (n - 1));

  std::cout << "Graph loaded from " << filename << std::endl;
  std::cout << "  Vertices: " << info.num_vertices << std::endl;
  std::cout << "  Edges: " << info.num_edges << std::endl;
  std::cout << "  Density: " << (info.density * 100) << "%" << std::endl;

  return {graph, info};
}

#endif // GRAPH_READER_H