import networkx as nx
import numpy as np

def graph_to_adjacency_matrix(graph):
    """Преобразует граф networkx в матрицу смежности numpy."""
    n = graph.number_of_nodes()
    nodes = sorted(list(graph.nodes()))
    node_map = {nodes[i]: i for i in range(n)}

    adj_matrix = np.zeros((n, n), dtype=int)

    for u, v in graph.edges():
        u_idx, v_idx = node_map[u], node_map[v]
        adj_matrix[u_idx, v_idx] = 1
        if u != v:
            adj_matrix[v_idx, u_idx] = 1
    return adj_matrix, nodes

def adjacency_matrix_to_graph(adj_matrix, nodes):
    """Преобразует матрицу смежности numpy в граф networkx."""
    n = adj_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == 1:
                u, v = nodes[i], nodes[j]
                if u == v:
                    G.add_edge(u, v)
                elif i < j:
                    G.add_edge(u, v)
    return G

def arbitrary_acyclic_orientation(adj_matrix, nodes):
    """Произвольная ациклическая ориентация графа."""
    n = adj_matrix.shape[0]
    oriented_adj_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == 1:
                if i < j:
                    oriented_adj_matrix[i, j] = 1
                elif j < i:
                    oriented_adj_matrix[j, i] = 1

    return oriented_adj_matrix

def get_neighborhood_matrix(adj_matrix):
    """Вычисляет матрицу окрестностей N[v]."""
    n = adj_matrix.shape[0]
    N_matrix = adj_matrix + np.eye(n, dtype=int)
    N_matrix[N_matrix > 1] = 1
    return N_matrix

def get_induced_subgraph_adj_matrix(original_adj_matrix, indices):
    """Возвращает матрицу смежности подграфа по индексам вершин."""
    return original_adj_matrix[np.ix_(indices, indices)]

def get_nodes_from_indices(original_nodes, indices):
    """Возвращает список вершин по их индексам."""
    return [original_nodes[i] for i in indices]
