"""Создать данные для теста для большого графа, чтобы их использовать нужно вставить в файл input.txt, не забудь про структуру тестов"""

import networkx as nx

num_vertices_large = 150
probability_large = 0.05
G_large = nx.erdos_renyi_graph(num_vertices_large, probability_large)
G_large = nx.convert_node_labels_to_integers(G_large)
large_edges = list(G_large.edges())

for u, v in large_edges:
    print(f"{u} {v}")