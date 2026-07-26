import warnings
import torch
import torch.nn as nn


class EntropyLoss(nn.Module):
    """Entropy regularization for soft assignment matrices."""

    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, assign, log_assign, num_fg_atoms):
        return -1.0 * torch.sum(assign * log_assign) / num_fg_atoms.sum()


class LinkPredLoss(nn.Module):
    """Link prediction loss for graph reconstruction."""

    def __init__(self):
        super(LinkPredLoss, self).__init__()

    def forward(self, fg_adj, assign, num_fg_atoms, assign_mask):
        reconstruct = assign @ assign.permute(0, 2, 1) * assign_mask
        return torch.sum((reconstruct - fg_adj) ** 2) / (num_fg_atoms ** 2).sum()


class TripletLoss(nn.Module):
    """Triplet margin loss for metric learning on graph embeddings."""

    def __init__(self, margin: float):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, fg_embed, triplet_index):
        anchor_idx, pos_idx, neg_idx = triplet_index
        anchor_feat = fg_embed[anchor_idx]
        pos_feat = fg_embed[pos_idx]
        neg_feat = fg_embed[neg_idx]

        d_pos = ((anchor_feat - pos_feat) ** 2).sum(dim=1)
        d_neg = ((anchor_feat - neg_feat) ** 2).sum(dim=1)
        loss = nn.functional.relu(d_pos - d_neg + self.margin).mean()
        return loss


class PosPairMSE(nn.Module):
    """MSE loss between positive pair embeddings."""

    def forward(self, fg_embed, pos_pair_index):
        pos1_embed = fg_embed[pos_pair_index[0]]
        pos2_embed = fg_embed[pos_pair_index[1]]
        loss = ((pos1_embed - pos2_embed) ** 2).mean()
        return loss


class LiftedStructureLoss(nn.Module):
    """Lifted Structure Loss with log-sum-exp approximation."""

    def __init__(self, alpha=40, beta=2, margin=0.5, hard_mining=None, **kwargs):
        super(LiftedStructureLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, inputs, targets):
        n = inputs.size(0)
        sim_mat = inputs @ inputs.t()
        loss = []
        num_skipped = 0

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets == targets[i])
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets != targets[i])

            if len(pos_pair_) == 0 or len(neg_pair_) == 0:
                num_skipped += 1
                continue

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            if self.hard_mining is not None:
                neg_pair = torch.masked_select(neg_pair_, neg_pair_ + 0.1 > pos_pair_[0])
                pos_pair = torch.masked_select(pos_pair_, pos_pair_ - 0.1 < neg_pair_[-1])

                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    num_skipped += 1
                    continue
            else:
                pos_pair = pos_pair_
                neg_pair = neg_pair_

            pos_loss = 2.0 / self.beta * torch.logsumexp(-self.beta * pos_pair, 0)
            neg_loss = 2.0 / self.alpha * torch.logsumexp(self.alpha * neg_pair, 0)

            if not torch.isinf(pos_loss):
                loss.append(pos_loss)
            if not torch.isinf(neg_loss):
                loss.append(neg_loss)

        if len(loss) == 0:
            warnings.warn("LiftedStructureLoss: All pairs skipped. Returning zero loss.")
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        return sum(loss) / n


def ncut_criterion(node_cluster_prob, affinity_matrix):
    """Normalized cut loss."""
    assert affinity_matrix.ndim == 2
    degree_vec = affinity_matrix.sum(dim=1)
    vol = node_cluster_prob.t() @ degree_vec.reshape(-1, 1)
    normalized_cut = (node_cluster_prob / vol.reshape(1, -1)) \
                     @ (1 - node_cluster_prob).t() * affinity_matrix
    return normalized_cut.mean()


def bcut_criterion(node_cluster_prob):
    """Balanced cut loss."""
    num_nodes, num_cluster = node_cluster_prob.shape
    loss = ((node_cluster_prob.sum(dim=0) - (num_nodes / num_cluster)) ** 2).mean()
    return loss