"""
The following is adapted from DGL repository
Credit: https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgmg
"""

import datetime
import os
import random
from copy import deepcopy
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
from matplotlib import animation
from torch_geometric.utils import to_networkx

########################################################################################################################
#                                                    configuration                                                     #
########################################################################################################################


# def mkdir_p(path):
#     import errno

#     try:
#         os.makedirs(path)
#         print("Created directory {}".format(path))
#     except OSError as exc:
#         if exc.errno == errno.EEXIST and os.path.isdir(path):
#             print("Directory {} already exists.".format(path))
#         else:
#             raise


# def date_filename(base_dir="./"):
#     dt = datetime.datetime.now()
#     return os.path.join(
#         base_dir,
#         "{}_{:02d}-{:02d}-{:02d}".format(dt.date(), dt.hour, dt.minute, dt.second),
#     )


# def setup_log_dir(opts):
#     log_dir = "{}".format(date_filename(opts["log_dir"]))
#     mkdir_p(log_dir)
#     return log_dir


# def save_arg_dict(opts, filename="settings.txt"):
#     def _format_value(v):
#         if isinstance(v, float):
#             return "{:.4f}".format(v)
#         elif isinstance(v, int):
#             return "{:d}".format(v)
#         else:
#             return "{}".format(v)

#     save_path = os.path.join(opts["log_dir"], filename)
#     with open(save_path, "w") as f:
#         for key, value in opts.items():
#             f.write("{}\t{}\n".format(key, _format_value(value)))
#     print("Saved settings to {}".format(save_path))


# def setup(args):
#     opts = args.__dict__.copy()

#     cudnn.benchmark = False
#     cudnn.deterministic = True

#     # Seed
#     if opts["seed"] is None:
#         opts["seed"] = random.randint(1, 10000)
#     random.seed(opts["seed"])
#     torch.manual_seed(opts["seed"])

#     # Dataset
#     from .dmgm_config import dataset_based_configure

#     opts = dataset_based_configure(opts)

#     assert opts["path_to_dataset"] is not None, "Expect path to dataset to be set."
#     if not os.path.exists(opts["path_to_dataset"]):
#         if opts["dataset"] == "cycles":
#             from src.data.cycles import generate_dataset

#             generate_dataset(
#                 opts["min_size"],
#                 opts["max_size"],
#                 opts["ds_size"],
#                 opts["path_to_dataset"],
#             )
#         else:
#             raise ValueError("Unsupported dataset: {}".format(opts["dataset"]))

#     # Optimization
#     if opts["clip_grad"]:
#         assert (
#             opts["clip_grad"] is not None
#         ), "Expect the gradient norm constraint to be set."

#     # Log
#     print("Prepare logging directory...")
#     log_dir = setup_log_dir(opts)
#     opts["log_dir"] = log_dir
#     mkdir_p(log_dir + "/samples")

#     plt.switch_backend("Agg")

#     save_arg_dict(opts)
#     pprint(opts)

#     return opts


def animate_graph_evolution(model):
    """Animates consecutive steps of DGMG model generation."""

    model.eval()
    g = model()

    src_list = g.edges()[1]
    dest_list = g.edges()[0]

    evolution = []

    nx_g = nx.Graph()
    evolution.append(deepcopy(nx_g))

    for i in range(0, len(src_list), 2):
        src = src_list[i].item()
        dest = dest_list[i].item()
        if src not in nx_g.nodes():
            nx_g.add_node(src)
            evolution.append(deepcopy(nx_g))
        if dest not in nx_g.nodes():
            nx_g.add_node(dest)
            evolution.append(deepcopy(nx_g))
        nx_g.add_edges_from([(src, dest), (dest, src)])
        evolution.append(deepcopy(nx_g))

    def animate(i):
        ax.cla()
        g_t = evolution[i]
        nx.draw_circular(
            g_t, with_labels=True, ax=ax, node_color=["#FEBD69"] * g_t.number_of_nodes()
        )

    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, animate, frames=len(evolution), interval=600)
    return ani


# General DGMG utils


def tranform_dataset(dataset):
    """Transforms PyG dataset into sequences of actions for DGMG."""
    sequences = []
    for d in dataset:
        g = to_networkx(d, to_undirected=True)
        decision_sequence = []
        added_edges = set()

        for n in g.nodes:
            decision_sequence.append(0)  # add node

            edges_to_add = [
                edge
                for edge in g.edges(n)
                if not (
                    (edge in added_edges) or (edge[::-1] in added_edges) or edge[1] > n
                )
            ]
            added_edges |= set(edges_to_add)

            if not edges_to_add:
                decision_sequence.append(1)  # not adding edges

            else:
                for e in edges_to_add:
                    decision_sequence.append(0)  # add edge
                    decision_sequence.append(e[1])  # add edge destination

                decision_sequence.append(1)  # stop adding edges

        decision_sequence.append(1)  # stop adding nodes
        sequences.append(decision_sequence)

    return sequences


def collate_single(batch):
    """Collate function preventing batch sizes > 1."""
    assert len(batch) == 1, "Currently we do not support batched training"
    return batch[0]
