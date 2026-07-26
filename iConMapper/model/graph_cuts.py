import torch
import torch.nn.functional as F
from sklearn.cluster import spectral_clustering


def graph_cuts(fg_embed, edge_index, num_cg, bandwidth=1.0, kernel='rbf',
               device=None, random_state=None):
    """
    Spectral clustering on graph with RBF affinity computed from embeddings.

    Returns:
        pred_cg_idx: numpy array [num_nodes] cluster assignments
        affinity: torch tensor [num_nodes, num_nodes]
    """
    if device is None:
        device = fg_embed.device

    affinity = compute_affinity(fg_embed, edge_index, bandwidth, kernel, device)
    pred_cg_idx = spectral_clustering(
        affinity.cpu().numpy(),
        n_clusters=num_cg,
        assign_labels='discretize',
        random_state=random_state)

    return pred_cg_idx, affinity


def graph_cuts_with_adj(adj, num_cg, random_state=None):
    """Spectral clustering directly from an adjacency matrix."""
    pred_cg_idx = spectral_clustering(
        adj.cpu().numpy(),
        n_clusters=num_cg,
        assign_labels='discretize',
        random_state=random_state)
    return pred_cg_idx


def compute_affinity(fg_embed, edge_index, bandwidth=1.0, kernel='rbf', device=None):
    """Compute affinity matrix from embeddings and graph structure."""
    if device is None:
        device = fg_embed.device

    if kernel == 'rbf':
        num_nodes = fg_embed.shape[0]
        fg_embed = fg_embed.to(device)
        edge_index = edge_index.to(device)

        pairwise_dist = torch.norm(
            fg_embed[edge_index[0]] - fg_embed[edge_index[1]], dim=1)
        pairwise_dist = pairwise_dist ** 2
        affinity_values = torch.exp(-pairwise_dist / (2 * bandwidth ** 2))

        affinity = torch.sparse_coo_tensor(
            indices=edge_index.long(),
            values=affinity_values,
            size=(num_nodes, num_nodes)
        ).to_dense()

    elif kernel == 'linear':
        fg_embed = fg_embed.to(device)
        affinity = F.relu(fg_embed @ fg_embed.t())

    else:
        raise ValueError(f"Unsupported kernel: '{kernel}'. Use 'rbf' or 'linear'.")

    return affinity