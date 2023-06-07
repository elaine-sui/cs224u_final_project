import networkx as nx
import numpy as np
from collections import Counter


def get_num_edges(G):
    return G.number_of_edges()


def get_num_nodes(G):
    return G.number_of_nodes()


def num_connected_components(G):
    return nx.number_connected_components(G)


def get_all_paths(G, source):
    paths = []

    if source in G.nodes:
        for node in G.nodes:
            if node == source:
                continue
            for path in nx.all_simple_paths(G, source, node):
                weight = nx.path_weight(G, path, "weight")
                paths.append((path, weight))
    else:
        for node in G.nodes:
            for node2 in G.nodes:
                if node2 == node:
                    continue
                for path in nx.all_simple_paths(G, node, node2):
                    weight = nx.path_weight(G, path, "weight")
                    paths.append((path, weight))
    return paths


def get_mean_degree(G, weighted=False):
    agg = []
    iter = G.degree(weight="weight") if weighted else G.degree()
    for node, deg in iter.items():
        agg.append(deg)
    return np.mean(agg)


def get_max_degree(G, weighted=False):
    max_node, max_deg = None, -1
    iter = G.degree(weight="weight") if weighted else G.degree()
    for node, deg in iter.items():
        if deg > max_deg:
            max_deg = deg
            max_node = node

    return max_node, max_deg


def get_avg_edge_weight(G):
    agg = []
    for u, v, w in G.edges.data("weight"):
        agg.append(w)
    return np.mean(agg)


def find_bottleneck_nodes(G, k=5):
    """
    Nodes with most paths going through them
    """
    d = G.degree(weight="weight")
    return sorted(d.items(), key=lambda item: item[1], reverse=True)[:k]


def find_isolated_nodes(G, k=5):
    """
    Nodes with fewest paths going through them
    """
    d = G.degree(weight="weight")
    return sorted(d.items(), key=lambda item: -item[1])[:k]


def get_avg_path_length(G, weighted=False):
    paths = get_all_paths(G, None)
    agg = []
    for path, weight in paths:
        if weighted:
            agg.append(weight)
        else:
            agg.append(len(path))
    return np.mean(agg)


def get_cycles(G):
    return list(nx.simple_cycles(G))


def get_unique_paths_from_source(G, source):
    paths = get_all_paths(G, source)
    # TODO: is there a reason this is not already unique?
    # I think it is and we don't need the rest of this function
    return [p for (p, w) in paths]
    # string_paths = []
    # for path, weight in paths:
    #     # converting path to string so we can hash into set
    #     string_paths.append("-".join(path))
    # string_paths = set(string_paths)

    # unique_paths = []
    # for sp in string_paths:
    #     unique_paths.append(sp.split("-"))

    # return unique_paths


def path_weight_dist(G, source, dest):
    path_weights = []
    for path in nx.all_simple_paths(G, source, dest):
        weight = nx.path_weight(G, path, "weight")
        path_weights.append(weight)
    return Counter(path_weights)


def num_paths(G, source, dest):
    paths = list(nx.all_simple_paths(G, source, dest))
    return len(paths)


def shortcut_exists(G, source, dest):
    min_len, max_len = float("inf"), 0
    for path in nx.all_simple_paths(G, source, dest):
        path_len = len(path)
        if path_len < min_len:
            min_len = path_len
        if path_len > max_len:
            max_len = path_len
    if min_len < max_len:
        return True, min_len
    else:
        return False, None
