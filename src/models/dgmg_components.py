"""
The following is adapted from DGL repository
Credit: https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgmg
"""
from functools import partial

import torch
import torch.nn.functional as F
from dgl import DGLGraph
from torch import Tensor, nn
from torch.distributions import Bernoulli, Categorical


class GraphEmbed(nn.Module):
    """Graph Embedding module, computes representation of the whole graph."""

    def __init__(self, node_hidden_size: int):
        super().__init__()

        # Setting from the paper
        self.graph_hidden_size = 2 * node_hidden_size

        # Embed graphs
        self.node_gating = nn.Sequential(nn.Linear(node_hidden_size, 1), nn.Sigmoid())
        self.node_to_graph = nn.Linear(node_hidden_size, self.graph_hidden_size)

    def forward(self, g: DGLGraph):
        if g.num_nodes() == 0:
            return torch.zeros(1, self.graph_hidden_size)
        else:
            # Node features are stored as hv in ndata.
            hvs = g.ndata["hv"]
            return (self.node_gating(hvs) * self.node_to_graph(hvs)).sum(
                0, keepdim=True
            )


class GraphProp(nn.Module):
    """Graph propagation module, computes representation of each node."""

    def __init__(self, num_prop_rounds: int, node_hidden_size: int):
        super().__init__()

        self.num_prop_rounds = num_prop_rounds

        # Setting from the paper
        self.node_activation_hidden_size = 2 * node_hidden_size

        message_funcs = []
        self.reduce_funcs = []
        node_update_funcs = []

        for t in range(num_prop_rounds):
            # input being [hv, hu, xuv]
            message_funcs.append(
                nn.Linear(2 * node_hidden_size + 1, self.node_activation_hidden_size)
            )

            self.reduce_funcs.append(partial(self.dgmg_reduce, round=t))
            node_update_funcs.append(
                nn.GRUCell(self.node_activation_hidden_size, node_hidden_size)
            )

        self.message_funcs = nn.ModuleList(message_funcs)
        self.node_update_funcs = nn.ModuleList(node_update_funcs)

    def forward(self, g: DGLGraph):
        if g.num_edges() == 0:
            return
        else:
            for t in range(self.num_prop_rounds):
                g.update_all(
                    message_func=self.dgmg_msg, reduce_func=self.reduce_funcs[t]
                )
                g.ndata["hv"] = self.node_update_funcs[t](g.ndata["a"], g.ndata["hv"])

    def dgmg_msg(self, edges):
        """For an edge u->v, return concat([h_u, x_uv])"""
        return {"m": torch.cat([edges.src["hv"], edges.data["he"]], dim=1)}

    def dgmg_reduce(self, nodes, round: int):
        hv_old = nodes.data["hv"]
        m = nodes.mailbox["m"]
        message = torch.cat([hv_old.unsqueeze(1).expand(-1, m.size(1), -1), m], dim=2)
        node_activation = (self.message_funcs[round](message)).sum(1)

        return {"a": node_activation}


def bernoulli_action_log_prob(logit: Tensor, action: int):
    """Calculate the log p of an action with respect to a Bernoulli
    distribution. Use logit rather than prob for numerical stability."""
    if action == 0:
        return F.logsigmoid(-logit)
    else:
        return F.logsigmoid(logit)


class AddNode(nn.Module):
    """AddNode module, computes probability of adding new node."""

    def __init__(self, graph_embed_func: nn.Module, node_hidden_size: int):
        super().__init__()

        self.graph_op = {"embed": graph_embed_func}

        self.stop = 1
        self.add_node = nn.Linear(graph_embed_func.graph_hidden_size, 1)

        # If to add a node, initialize its hv
        self.node_type_embed = nn.Embedding(1, node_hidden_size)
        self.initialize_hv = nn.Linear(
            node_hidden_size + graph_embed_func.graph_hidden_size,
            node_hidden_size,
        )

        self.init_node_activation = torch.zeros(1, 2 * node_hidden_size)

    def _initialize_node_repr(self, g: DGLGraph, node_type: int, graph_embed: Tensor):
        num_nodes = g.num_nodes()
        hv_init = self.initialize_hv(
            torch.cat(
                [
                    self.node_type_embed(torch.LongTensor([node_type])),
                    graph_embed,
                ],
                dim=1,
            )
        )
        g.nodes[num_nodes - 1].data["hv"] = hv_init
        g.nodes[num_nodes - 1].data["a"] = self.init_node_activation

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g: DGLGraph, action: int | None = None):
        graph_embed = self.graph_op["embed"](g)

        logit = self.add_node(graph_embed)
        prob = torch.sigmoid(logit)

        if not self.training:
            action = Bernoulli(prob).sample().item()
        stop = bool(action == self.stop)

        if not stop:
            g.add_nodes(1)
            self._initialize_node_repr(g, action, graph_embed)

        if self.training:
            sample_log_prob = bernoulli_action_log_prob(logit, action)
            self.log_prob.append(sample_log_prob)

        return stop


class AddEdge(nn.Module):
    """AddEdge module, computes probability of adding edge to newly added node."""

    def __init__(self, graph_embed_func: nn.Module, node_hidden_size: int):
        super().__init__()

        self.graph_op = {"embed": graph_embed_func}
        self.add_edge = nn.Linear(
            graph_embed_func.graph_hidden_size + node_hidden_size, 1
        )

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g: DGLGraph, action: int | None = None):
        graph_embed = self.graph_op["embed"](g)
        src_embed = g.nodes[g.num_nodes() - 1].data["hv"]

        logit = self.add_edge(torch.cat([graph_embed, src_embed], dim=1))
        prob = torch.sigmoid(logit)

        if not self.training:
            action = Bernoulli(prob).sample().item()
        to_add_edge = bool(action == 0)

        if self.training:
            sample_log_prob = bernoulli_action_log_prob(logit, action)
            self.log_prob.append(sample_log_prob)

        return to_add_edge


class ChooseDestAndUpdate(nn.Module):
    """ChooseDestAndUpdate module,  computes probability of new node's edges to each previous nodes."""

    def __init__(self, graph_prop_func: nn.Module, node_hidden_size: int):
        super().__init__()

        self.graph_op = {"prop": graph_prop_func}
        self.choose_dest = nn.Linear(2 * node_hidden_size, 1)

    def _initialize_edge_repr(self, g: DGLGraph, src_list: list, dest_list: list):
        # For untyped edges, only add 1 to indicate its existence.
        # For multiple edge types, use a one-hot representation
        # or an embedding module.
        edge_repr = torch.ones(len(src_list), 1)
        g.edges[src_list, dest_list].data["he"] = edge_repr

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g: DGLGraph, dest: int | None):
        src = g.number_of_nodes() - 1
        possible_dests = range(src)

        src_embed_expand = g.nodes[src].data["hv"].expand(src, -1)
        possible_dests_embed = g.nodes[possible_dests].data["hv"]

        dests_scores = self.choose_dest(
            torch.cat([possible_dests_embed, src_embed_expand], dim=1)
        ).view(1, -1)
        dests_probs = F.softmax(dests_scores, dim=1)

        if not self.training:
            dest = Categorical(dests_probs).sample().item()

        if not g.has_edges_between(src, dest):
            # For undirected graphs, add edges for both directions
            # so that you can perform graph propagation.
            src_list = [src, dest]
            dest_list = [dest, src]

            g.add_edges(src_list, dest_list)
            self._initialize_edge_repr(g, src_list, dest_list)

            self.graph_op["prop"](g)

        if self.training:
            if dests_probs.nelement() > 1:
                self.log_prob.append(
                    F.log_softmax(dests_scores, dim=1)[:, dest : dest + 1]
                )
