"""
General Visualization utilities.
"""

from copy import deepcopy

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def visualize_graphs(graphs: list[Data | nx.Graph], title: str):
    fig, axes = plt.subplots(1, len(graphs), figsize=(18, 4))
    for ax, graph in zip(axes, graphs):
        if isinstance(graph, Data):
            g = to_networkx(graph, to_undirected=True)
        else:
            g = graph
        nx.draw(g, ax=ax, node_size=100, width=0.5, pos=nx.spring_layout(g))
        ax.patch.set_edgecolor("black")
    fig.suptitle(title)
    plt.show()


def draw_reconstruction_plot(
    original_graphs: list[nx.Graph], recon_graphs: list[nx.Graph]
):
    _, axes = plt.subplots(2, len(original_graphs), figsize=(20, 10))

    for (gold_g, recon_g), ax in zip(zip(original_graphs, recon_graphs), axes.T):
        upper_ax, lower_ax = ax
        nx.draw(
            gold_g, ax=upper_ax, node_size=100, width=0.5, pos=nx.spring_layout(gold_g)
        )
        nx.draw(
            recon_g,
            ax=lower_ax,
            node_size=100,
            width=0.5,
            pos=nx.spring_layout(recon_g),
        )
        upper_ax.set_title("Original")
        lower_ax.set_title("Reconstructed")

    plt.show()


def plot_pca(z: np.array, title: str = "PCA projection of node representations"):
    reduced = PCA(n_components=2).fit_transform(z)
    ax = sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1])
    ax.set_title(title)
    plt.show()
