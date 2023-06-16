"""
The following code is adapted from the implementation of https://arxiv.org/abs/1802.08773
Credit: https://github.com/JiaxuanYou/graph-generation
"""

import networkx as nx
import numpy as np
from tqdm import trange


def generate_dataset(name: str, verbose: bool = True, **kwargs) -> list[nx.Graph]:
    graphs = []
    if name == "community":
        c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], kwargs["num_communities"])
        for _ in trange(3000, desc="Generating community graphs", disable=not verbose):
            graphs.append(n_community(c_sizes, p_inter=0.01))
    else:
        raise ValueError("Invalid dataset name!")

    return graphs


def n_community(c_sizes, p_inter=0.01) -> nx.Graph:
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = [
        nx.subgraph(G, component) for component in nx.connected_components(G)
    ]
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i + 1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])

    return G
