""""
Utility functions for training and analysis of GraphVAE (https://arxiv.org/abs/1802.03480) model.
"""

import random
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_networkx
from torchmetrics.classification import BinaryAUROC
from tqdm.auto import tqdm, trange

from src.utils.visualizations import draw_reconstruction_plot, plot_pca


def train_graph_vae(
    model, train_dataloader, val_dataloader, hparams: dict[str, Any], device: str
) -> dict[str, list[float]]:
    """Trains GraphVAE model with the given data and hparams."""
    optimizer = Adam(model.parameters(), lr=hparams["lr"])

    # run training and log loss history
    training_log = {
        "loss": [],
        "train_recon_auc": [],
        "val_recon_auc": [],
        "epoch": [],
    }

    # prepare metrics for reconstruction
    train_auroc = BinaryAUROC()
    val_auroc = BinaryAUROC()

    model.to(device)
    model.train()
    with trange(hparams["epochs"], desc="Epoch") as pbar:
        for epoch in pbar:
            batch_losses = []
            for batch in tqdm(train_dataloader, desc="Batch", leave=False):
                optimizer.zero_grad()

                batch.to(device)

                out = model.forward(batch.x, batch.edge_index, batch.batch)
                adj_matrix = to_dense_adj(
                    batch.edge_index, batch.batch, max_num_nodes=model.max_num_nodes
                )
                loss = model.loss(
                    adj_matrix,
                    out["z_mu"],
                    out["z_logvar"],
                    out["edge_recon"],
                )

                loss["total"].backward()
                optimizer.step()

                batch_losses.append(loss["total"].item())

                adj_triu = adj_matrix[:, model.triu_mask]
                train_auroc(out["edge_recon"].sigmoid(), adj_triu)

            # log training metrics
            training_log["epoch"].append(epoch)
            training_log["train_recon_auc"].append(train_auroc.compute().item())
            train_auroc.reset()
            epoch_loss = np.mean(batch_losses)
            training_log["loss"].append(epoch_loss)
            pbar.set_postfix({"train/loss": epoch_loss})

            # validate model
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Val batches", leave=False):
                    batch.to(device)

                    out = model(batch.x, batch.edge_index, batch.batch)

                    # compute validation metrics
                    adj_matrix = to_dense_adj(
                        batch.edge_index, batch.batch, max_num_nodes=model.max_num_nodes
                    )
                    adj_triu = adj_matrix[:, model.triu_mask]
                    val_auroc(out["edge_recon"].sigmoid(), adj_triu)

            # log validation metrics
            training_log["val_recon_auc"].append(val_auroc.compute().item())
            val_auroc.reset()

        return training_log


def visualize_graph_vae_reconstruction(model, dataset, device, num_samples: int):
    """Visulizations for GraphVAE reconstruction."""
    original_graphs = []
    recon_graphs = []

    with torch.no_grad():
        model.eval()
        samples = dataset[random.sample(range(len(dataset)), k=num_samples)]

        for data in samples:
            data = data.to(device)
            batch_idx = torch.zeros(len(data.x), dtype=torch.long, device=device)
            out = model.forward(data.x, data.edge_index, batch_idx)

            edge_recon = out["edge_recon"].sigmoid().cpu()
            edges = torch.bernoulli(edge_recon)
            recon_adj_lower = model._recover_adj_lower(edges)
            recon_adj_matrix = model._recover_full_adj_from_lower(recon_adj_lower)
            recon_data = nx.from_numpy_array(recon_adj_matrix.squeeze(0).cpu().numpy())

            recon_graphs.append(recon_data)
            original_graphs.append(to_networkx(data, to_undirected=True))

    draw_reconstruction_plot(original_graphs, recon_graphs)


def visualize_graph_vae_embeddings(model, data_laoder, device):
    """Visulizations of GraphVAE latent space."""
    model.eval()
    with torch.no_grad():
        latents = []
        for batch in data_laoder:
            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            latents.append(out["z"].cpu().numpy())
        z = np.concatenate(latents)

    plot_pca(z, title="PCA projections of graph-level representations")
