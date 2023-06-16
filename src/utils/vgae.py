"""
Utility functions for training and analysis of GraphVAE (https://arxiv.org/abs/1611.07308) model.
"""

import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

from src.utils.visualizations import draw_reconstruction_plot, plot_pca


def visualize_vgae_training_log(training_log: dict[str, list[float]]):
    """Visualizations of training logs."""
    _, (loss_ax, recon_ax) = plt.subplots(1, 2, figsize=(18, 5))

    ax = sns.lineplot(
        x=training_log["epoch"], y=training_log["loss"], ax=loss_ax, label="Train"
    )
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax = sns.lineplot(
        x=training_log["epoch"],
        y=training_log["train_recon_auc"],
        ax=recon_ax,
        label="Train",
    )
    ax = sns.lineplot(
        x=training_log["epoch"],
        y=training_log["val_recon_auc"],
        ax=recon_ax,
        label="Val",
    )
    ax.set_title("Reconstruction ROC-AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC-AUC")

    plt.show()


def visualize_vgae_reconstruction(model, dataset, device, num_samples: int):
    """Visulizations for VGAE reconstruction."""
    original_graphs = []
    recon_graphs = []

    with torch.no_grad():
        model.eval()
        samples = dataset[random.sample(range(len(dataset)), k=num_samples)]
        for data in samples:
            data = data.to(device)
            z = model(data.x, data.edge_index)["z"]
            num_nodes = len(z)
            triu_edge_index = torch.combinations(torch.arange(num_nodes)).T.to(device)
            edge_proba = model.decode(z, triu_edge_index).sigmoid()

            edge_index = triu_edge_index[
                :, torch.bernoulli(edge_proba).flatten().bool()
            ]

            recon_data = Data(edge_index=edge_index, num_nodes=num_nodes)
            recon_graphs.append(to_networkx(recon_data, to_undirected=True))

            original_graphs.append(to_networkx(data, to_undirected=True))

    draw_reconstruction_plot(original_graphs, recon_graphs)


def visualize_vgae_embeddings(model, data_laoder: DataLoader, device):
    """Visulizations of VGAE latent space."""
    model.eval()
    with torch.no_grad():
        latents = []
        for batch in data_laoder:
            batch.to(device)
            out = model.encode(batch.x, batch.edge_index)
            latents.append(out["z"].cpu().numpy())
        z = np.concatenate(latents)

    plot_pca(z)
