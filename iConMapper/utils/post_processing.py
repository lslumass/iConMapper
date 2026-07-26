import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix, coo_matrix, triu
from sklearn.metrics import precision_recall_fscore_support


def enforce_connectivity(node_cluster_idx_pred, edge_index_lst):
    """Relabel disconnected subgraphs within the same predicted cluster."""
    num_vertices = len(node_cluster_idx_pred)
    filtered_edges = []

    for u, v in edge_index_lst.T:
        if node_cluster_idx_pred[u] == node_cluster_idx_pred[v]:
            filtered_edges.append([u, v])

    if len(filtered_edges) == 0:
        return node_cluster_idx_pred

    filtered_edge_index = np.array(filtered_edges).T
    data = np.ones(len(filtered_edges))
    filtered_adj = csr_matrix(
        (data, (filtered_edge_index[0], filtered_edge_index[1])),
        shape=(num_vertices, num_vertices))

    _, labels = connected_components(csgraph=filtered_adj, directed=False, return_labels=True)
    return labels


def edge_cut_prec_recall_fscore(node_cluster_idx_pred, node_cluster_idx_gt, edge_index_lst):
    """Compute precision/recall/F1 for edge cut classification."""
    num_vertices = len(node_cluster_idx_pred)
    data = np.ones(edge_index_lst.shape[1])
    adj = coo_matrix(
        (data, (edge_index_lst[0], edge_index_lst[1])),
        shape=(num_vertices, num_vertices))

    upper_adj = triu(adj, format='coo')
    num_edges = edge_index_lst.shape[1] // 2

    edge_pred = -1 * np.ones(num_edges, dtype=np.int32)
    edge_gt = -1 * np.ones(num_edges, dtype=np.int32)

    for edge_idx, (u, v) in enumerate(zip(upper_adj.row, upper_adj.col)):
        edge_pred[edge_idx] = int(node_cluster_idx_pred[u] != node_cluster_idx_pred[v])
        edge_gt[edge_idx] = int(node_cluster_idx_gt[u] != node_cluster_idx_gt[v])

    prec, rec, fscore, _ = precision_recall_fscore_support(
        y_true=edge_gt, y_pred=edge_pred, average='binary')
    return prec, rec, fscore


def cg_type_prec_recall_fscore(real_result, predict_result):
    """Weighted precision/recall/F1 for CG type classification."""
    prec, rec, fscore, _ = precision_recall_fscore_support(
        real_result, predict_result, average='weighted')
    return prec, rec, fscore