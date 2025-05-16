import numpy as np
import time
import sys
from itertools import combinations

from graph_helpers import arbitrary_acyclic_orientation, get_induced_subgraph_adj_matrix, get_nodes_from_indices

def la_matching_counting_triangles(adj_matrix, nodes):
    """Алгоритм 3: Линейно-алгебраический поиск треугольников."""
    n = adj_matrix.shape[0]
    if n < 3:
        return []

    oriented_adj_matrix = arbitrary_acyclic_orientation(adj_matrix, nodes)

    A_G_sq = np.linalg.matrix_power(oriented_adj_matrix, 2)
    A_star = A_G_sq * oriented_adj_matrix

    triangles = set()

    for i in range(n):
        for j in range(n):
            if A_star[i, j] > 0:
                for k in range(n):
                     if oriented_adj_matrix[i, k] > 0 and oriented_adj_matrix[k, j] > 0:
                         triangle = tuple(sorted((nodes[i], nodes[k], nodes[j])))
                         triangles.add(triangle)

    return [list(t) for t in triangles]

def remove_low_degree_vertices(adj_matrix, nodes, k):
    """Алгоритм 5: удаляет вершины степени < k - 1."""
    degrees = np.sum(adj_matrix, axis=1)
    valid_indices = [i for i, deg in enumerate(degrees) if deg >= k - 1]
    if not valid_indices:
        return np.zeros((0, 0), dtype=int), []
    reduced_adj_matrix = adj_matrix[np.ix_(valid_indices, valid_indices)]
    reduced_nodes = [nodes[i] for i in valid_indices]
    return reduced_adj_matrix, reduced_nodes

def remove_weak_edges(adj_matrix, k):
    """Алгоритм 6: удаляет рёбра, у которых |N(u) ∩ N(v)| < k - 2."""
    n = adj_matrix.shape[0]
    A_square = adj_matrix @ adj_matrix
    mask = A_square >= (k - 2)
    np.fill_diagonal(mask, 0)
    filtered_adj_matrix = adj_matrix * mask
    filtered_adj_matrix[filtered_adj_matrix > 1] = 1
    return filtered_adj_matrix

def reduce_graph_by_degree_and_common_neighbors(adj_matrix, nodes, k):
    """Итеративное применение Алгоритма 5 и 6, пока граф не стабилизируется."""
    changed = True
    while changed:
        prev_shape = adj_matrix.shape
        adj_matrix, nodes = remove_low_degree_vertices(adj_matrix, nodes, k)
        if adj_matrix.shape[0] < k:
            return np.zeros((0, 0), dtype=int), []
        adj_matrix = remove_weak_edges(adj_matrix, k)
        adj_matrix, nodes = remove_low_degree_vertices(adj_matrix, nodes, k)
        changed = adj_matrix.shape != prev_shape
    return adj_matrix, nodes

def la_matching_counting_k_cliques(adj_matrix, k, nodes):
    """Алгоритм 7: Линейно-алгебраический поиск k-клик + Алгоритмы 5+6."""
    start_time = time.time()
    n = adj_matrix.shape[0]

    if k == 1:
        cliques = [[node] for node in nodes]
        end_time = time.time()
        if sys.stdout == sys.__stdout__:
            print(f"Время выполнения поиска {k}-клик: {end_time - start_time:.6f} сек.")
        return cliques

    if k == 2:
        cliques = []
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] == 1 and i <= j:
                    cliques.append([nodes[i], nodes[j]])
        end_time = time.time()
        if sys.stdout == sys.__stdout__:
            print(f"Время выполнения поиска {k}-клик: {end_time - start_time:.6f} сек.")
        return cliques

    if k == 3:
        cliques = la_matching_counting_triangles(adj_matrix, nodes)
        end_time = time.time()
        if sys.stdout == sys.__stdout__:
            print(f"Время выполнения поиска {k}-клик: {end_time - start_time:.6f} сек.")
        return cliques

    # Алгоритмы 5 и 6
    adj_matrix, nodes = reduce_graph_by_degree_and_common_neighbors(adj_matrix, nodes, k)
    n = adj_matrix.shape[0]
    if n < k:
        end_time = time.time()
        if sys.stdout == sys.__stdout__:
            print(f"Время выполнения поиска {k}-клик: {end_time - start_time:.6f} сек.")
        return []

    oriented_adj_matrix = arbitrary_acyclic_orientation(adj_matrix, nodes)
    node_map = {nodes[i]: i for i in range(n)}

    def find_k_cliques_recursive_alg7(current_adj_matrix, current_nodes_list, current_prefix_nodes, level, target_k):
        current_n = current_adj_matrix.shape[0]
        node_map_current = {current_nodes_list[i]: i for i in range(current_n)}

        if level == target_k - 2:
            triangles_in_subgraph = la_matching_counting_triangles(current_adj_matrix, current_nodes_list)
            found_k_cliques = set()
            for triangle_nodes in triangles_in_subgraph:
                potential_k_clique_nodes = sorted(current_prefix_nodes + triangle_nodes)
                if len(potential_k_clique_nodes) == target_k:
                    is_full_clique = True
                    for u_node, v_node in combinations(potential_k_clique_nodes, 2):
                        u_idx_orig, v_idx_orig = node_map[u_node], node_map[v_node]
                        if adj_matrix[u_idx_orig, v_idx_orig] == 0:
                            is_full_clique = False
                            break
                    if is_full_clique:
                        found_k_cliques.add(tuple(potential_k_clique_nodes))
            return found_k_cliques

        found_k_cliques = set()
        oriented_current_adj_matrix = arbitrary_acyclic_orientation(current_adj_matrix, current_nodes_list)

        for v_node in current_nodes_list:
            v_idx_in_subgraph = node_map_current[v_node]
            v_right_neighbors_indices = [i for i in range(current_n) if oriented_current_adj_matrix[v_idx_in_subgraph, i] == 1]

            if len(v_right_neighbors_indices) >= target_k - level - 1:
                next_subgraph_indices = v_right_neighbors_indices
                next_subgraph_adj_matrix = get_induced_subgraph_adj_matrix(current_adj_matrix, next_subgraph_indices)
                next_subgraph_nodes_list = get_nodes_from_indices(current_nodes_list, next_subgraph_indices)

                found_cliques = find_k_cliques_recursive_alg7(
                    next_subgraph_adj_matrix,
                    next_subgraph_nodes_list,
                    current_prefix_nodes + [v_node],
                    level + 1,
                    target_k
                )
                found_k_cliques.update(found_cliques)

        return found_k_cliques

    k_cliques = find_k_cliques_recursive_alg7(adj_matrix, nodes, [], 1, k)

    end_time = time.time()
    if sys.stdout == sys.__stdout__:
        print(f"Время выполнения поиска {k}-клик: {end_time - start_time:.6f} сек.")
    return [list(c) for c in k_cliques]