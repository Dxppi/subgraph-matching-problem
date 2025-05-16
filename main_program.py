import networkx as nx
import numpy as np
import random
import sys
import time
from itertools import combinations

from graph_helpers import graph_to_adjacency_matrix, adjacency_matrix_to_graph
from k_clique_algorithms import la_matching_counting_k_cliques

def create_graph_manual():
    """Создает граф на основе ввода пользователя."""
    while True:
        try:
            num_vertices = int(input("Введите количество вершин в графе: "))
            if num_vertices <= 0:
                print("Количество вершин должно быть положительным числом.")
                continue
            break
        except ValueError:
            print("Некорректный ввод. Пожалуйста, введите целое число.")

    G = nx.Graph()
    G.add_nodes_from(range(num_vertices))

    print("Теперь введите ребра графа в формате 'вершина1 вершина2'. Введите 'готово' для завершения.")
    print("Вершины нумеруются от 0 до", num_vertices - 1)

    edges = []
    while True:
        edge_input = input("Введите ребро (или 'готово'): ")
        if edge_input.lower() == 'готово':
            break
        try:
            u, v = map(int, edge_input.split())
            if 0 <= u < num_vertices and 0 <= v < num_vertices:
                 edges.append((u, v))
            else:
                print("Некорректные номера вершин. Вершины должны быть в диапазоне от 0 до", num_vertices - 1)
        except ValueError:
            print("Некорректный формат ввода. Используйте 'вершина1 вершина2'.")

    G.add_edges_from(edges)
    return G

def create_graph_random(num_vertices, probability):
    """Создает случайный граф."""
    G = nx.erdos_renyi_graph(num_vertices, probability)
    G = nx.convert_node_labels_to_integers(G)
    return G

def run_automatic_test(input_file_path, output_file_path):
    """Выполняет автоматическое тестирование с чтением параметров из input.txt и записью результатов в output.txt."""
    try:
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            lines = infile.readlines()
            line_idx = 0

            test_params = []
            current_params = {}
            reading_section = None

            while line_idx < len(lines):
                line = lines[line_idx].strip()
                line_idx += 1

                if not line or line.startswith('#'):
                    continue

                if line == 'experiment':
                    if current_params:
                        test_params.append(current_params)
                    current_params = {}
                    reading_section = 'experiment'
                elif line == 'end_experiment':
                    reading_section = None
                elif reading_section == 'experiment':
                    parts = line.split()
                    if parts[0] == 'n_range':
                        current_params['n_range'] = list(map(int, parts[1:]))
                    elif parts[0] == 'probability':
                        current_params['probability'] = float(parts[1])
                    elif parts[0] == 'k_values':
                        current_params['k_values'] = list(map(int, parts[1:]))

            if current_params:
                test_params.append(current_params)

            if not test_params:
                outfile.write("Входной файл не содержит параметров эксперимента.\n")
                return

            for params in test_params:
                n_range = params.get('n_range', [])
                probability = params.get('probability', 0.0)
                k_values = params.get('k_values', [])

                if not n_range or not k_values:
                    outfile.write(f"Пропущен эксперимент из-за неполных параметров: {params}\n")
                    outfile.write("-" * 20 + "\n")
                    continue

                outfile.write(f"Начало эксперимента: N в диапазоне {n_range}, P={probability}, K={k_values}\n")

                for n in n_range:
                    if n <= 0:
                        outfile.write(f"Пропущен размер N={n}: некорректное значение.\n")
                        continue

                    outfile.write(f"Генерация случайного графа с N={n}, P={probability}...\n")
                    G = create_graph_random(n, probability)
                    adj_matrix, nodes = graph_to_adjacency_matrix(G)
                    outfile.write(f"Граф сгенерирован: {G.number_of_nodes()} вершин, {G.number_of_edges()} ребер.\n")
                    if n <= 20:
                        outfile.write(f"Вершины: {list(G.nodes())}\n")
                        outfile.write(f"Ребра: {list(G.edges())}\n")
                    else:
                        outfile.write("Списки вершин и ребер не показаны для графов > 20 вершин.\n")
                    outfile.write("\n")


                    for k in k_values:
                        if k <= 0:
                            outfile.write(f"Пропущен поиск для K={k}: некорректное значение.\n")
                            continue

                        outfile.write(f"Поиск {k}-клик...\n")
                        start_time = time.time()
                        cliques = la_matching_counting_k_cliques(adj_matrix, k, nodes)
                        end_time = time.time()
                        outfile.write(f"Время выполнения поиска {k}-клик: {end_time - start_time:.6f} сек.\n")
                        if n <= 20:
                            outfile.write(f"Найденные {k}-клики: {cliques}\n")
                        outfile.write(f"Количество {k}-клик: {len(cliques)}\n")
                        outfile.write("\n")

                outfile.write("-" * 20 + "\n")

    except FileNotFoundError:
        print("Ошибка: Входной файл input.txt или выходной файл output.txt не найдены.")
    except Exception as e:
        print(f"Произошла ошибка при выполнении автоматического теста: {e}")


def main():
    """Основная функция программы."""
    print("Выберите режим работы:")
    print("1. Интерактивный (ввод графа вручную или случайно, поиск клик, опциональное добавление ребер и повторный поиск)")
    print("2. Автоматическое тестирование (чтение параметров из input.txt, запись результатов в output.txt)")

    while True:
        mode_choice = input("Ваш выбор (1 или 2): ")
        if mode_choice == '1':
            print("\nВыберите способ создания графа:")
            print("1. Ввести вручную")
            print("2. Сгенерировать случайный граф")

            while True:
                graph_choice = input("Ваш выбор (1 или 2): ")
                if graph_choice == '1':
                    G = create_graph_manual()
                    break
                elif graph_choice == '2':
                    while True:
                        try:
                            num_vertices = int(input("Введите количество вершин в графе: "))
                            if num_vertices <= 0:
                                print("Количество вершин должно быть положительным числом.")
                                continue
                            break
                        except ValueError:
                            print("Некорректный ввод. Пожалуйста, введите целое число.")
                    while True:
                        try:
                            probability = float(input("Введите вероятность создания ребра (от 0 до 1): "))
                            if 0 <= probability <= 1:
                                break
                            else:
                                print("Вероятность должна быть в диапазоне от 0 до 1.")
                        except ValueError:
                            print("Некорректный ввод. Пожалуйста, введите число.")
                    G = create_graph_random(num_vertices, probability)
                    break
                else:
                    print("Некорректный выбор. Пожалуйста, введите 1 или 2.")

            print("\nГраф создан:")
            if G.number_of_nodes() <= 20:
                print("Вершины:", list(G.nodes()))
                print("Ребра:", list(G.edges()))
            else:
                print("Списки вершин и ребер не показаны для графов > 20 вершин.")


            G_current = G.copy()
            G_current_adj, current_nodes = graph_to_adjacency_matrix(G_current)
            n_current = G_current_adj.shape[0]

            while True:
                while True:
                    try:
                        k = int(input("\nВведите размер k для поиска k-клик в текущем графе: "))
                        if k <= 0:
                            print("Размер k должен быть положительным числом.")
                            continue
                        break
                    except ValueError:
                        print("Некорректный ввод. Пожалуйста, введите целое число.")

                print(f"\nПоиск {k}-клик в текущем графе...")
                current_k_cliques = la_matching_counting_k_cliques(G_current_adj, k, current_nodes)

                if n_current <= 20:
                    print(f"Найденные {k}-клики: {current_k_cliques}")
                print(f"Количество {k}-клик: {len(current_k_cliques)}")

                while True:
                    action = input("\nВыберите действие: (сменить k / добавить ребра / выйти): ").lower()
                    if action in ['сменить k', 'добавить ребра', 'выйти']:
                        break
                    else:
                        print("Некорректный ввод. Пожалуйста, введите 'сменить k', 'добавить ребра' или 'выйти'.")

                if action == 'выйти':
                    return

                if action == 'сменить k':
                    continue

                if action == 'добавить ребра':
                    new_edges_list = []
                    print("Введите новые ребра в формате 'вершина1 вершина2'. Введите 'готово' для завершения.")
                    print("Вершины нумеруются от 0 до", len(current_nodes) - 1)

                    while True:
                        edge_input = input("Введите новое ребро (или 'готово'): ")
                        if edge_input.lower() == 'готово':
                            break
                        try:
                            u, v = map(int, edge_input.split())
                            if 0 <= u < len(current_nodes) and 0 <= v < len(current_nodes):
                                 if not G_current.has_edge(u, v):
                                     new_edges_list.append((u, v))
                                 else:
                                     print(f"Ребро ({u}, {v}) уже существует в графе.")
                            else:
                                print("Некорректные номера вершин. Вершины должны быть в диапазоне от 0 до", len(current_nodes) - 1)
                        except ValueError:
                            print("Некорректный формат ввода. Используйте 'вершина1 вершина2'.")

                    if new_edges_list:
                        G_current.add_edges_from(new_edges_list)
                        G_current_adj, current_nodes = graph_to_adjacency_matrix(G_current)
                        n_current = G_current_adj.shape[0]

                        print(f"Добавлены ребра: {new_edges_list}")
                        if n_current <= 20:
                             print("Обновленные ребра графа:", list(G_current.edges()))
                        else:
                             print("Обновленные списки вершин и ребер не показаны для графов > 20 вершин.")

                    else:
                        print("Не было введено новых ребер.")

        elif mode_choice == '2':
            run_automatic_test('input.txt', 'output.txt')
            break
        else:
            print("Некорректный выбор режима. Пожалуйста, введите 1 или 2.")


if __name__ == "__main__":
    main()
