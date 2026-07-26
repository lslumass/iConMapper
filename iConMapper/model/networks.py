import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from dataset.ham import ATOMS, MASK_ATOM_INDEX, CG_TYPE_DICT

NUM_ATOMS = len(ATOMS)
NUM_CG_TYPES = len(CG_TYPE_DICT)  # Synced with dataset automatically


class DSGPM_TP(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim,
                 use_degree_feat=True, use_cycle_feat=True,
                 use_charge_feat=True, use_aromatic_feat=False,
                 use_mask_embed=False, num_nn_iter=6, dropout=0.0,
                 num_cg_types=None):
        super(DSGPM_TP, self).__init__()
        self.use_degree_feat = use_degree_feat
        self.use_cycle_feat = use_cycle_feat
        self.use_charge_feat = use_charge_feat
        self.use_aromatic_feat = use_aromatic_feat
        self.use_mask_embed = use_mask_embed
        self.num_nn_iter = num_nn_iter
        self.dropout = dropout

        # Use passed value or default from CG_TYPE_DICT
        self.NUM_CG_TYPES = num_cg_types if num_cg_types is not None else NUM_CG_TYPES

        self.input_fc, self.nn_conv, self.gru, self.output_fc = self._build_layers(
            input_dim, hidden_dim, embedding_dim)

        # feature_num = embedding_dim + one_hot_atoms + extended_features
        if self.use_mask_embed:
            self.feature_num = embedding_dim + NUM_ATOMS + 1
        else:
            self.feature_num = embedding_dim + NUM_ATOMS

        if self.use_degree_feat:
            self.feature_num += 1
        if self.use_cycle_feat:
            self.feature_num += 1
        if self.use_charge_feat:
            self.feature_num += 1
        if self.use_aromatic_feat:
            self.feature_num += 1

        self.cg_type_fc = nn.Sequential(
            nn.Linear(self.feature_num, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, self.NUM_CG_TYPES)
        )

    def _build_layers(self, input_dim, hidden_dim, embedding_dim):
        """Build message passing layers."""
        if self.use_mask_embed:
            input_fc = nn.Embedding(input_dim + 1, hidden_dim, padding_idx=MASK_ATOM_INDEX)
        else:
            input_fc = nn.Embedding(input_dim, hidden_dim)

        # Extend hidden_dim for concatenated extended features
        if self.use_degree_feat:
            hidden_dim += 1
        if self.use_cycle_feat:
            hidden_dim += 1
        if self.use_charge_feat:
            hidden_dim += 1
        if self.use_aromatic_feat:
            hidden_dim += 1

        edge_nn = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim * hidden_dim)
        )
        nn_conv = gnn.NNConv(hidden_dim, hidden_dim, edge_nn, aggr='add')
        gru = nn.GRU(hidden_dim, hidden_dim)
        output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim)
        )
        return input_fc, nn_conv, gru, output_fc

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        out = F.relu(self.input_fc(x)).squeeze(1)

        if self.use_degree_feat or self.use_cycle_feat or self.use_charge_feat or self.use_aromatic_feat:
            out = torch.cat([out, data.extended_feat], dim=1)

        h = out.unsqueeze(0)

        for _ in range(self.num_nn_iter):
            m = F.relu(self.nn_conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.output_fc(out)

        if self.use_mask_embed:
            atom_types_tensor = torch.zeros((x.shape[0], NUM_ATOMS + 1), device=x.device)
        else:
            atom_types_tensor = torch.zeros((x.shape[0], NUM_ATOMS), device=x.device)
        atom_types_tensor.scatter_(1, x, 1)

        feat_lst = [out, atom_types_tensor]
        if self.use_degree_feat or self.use_cycle_feat or self.use_charge_feat or self.use_aromatic_feat:
            feat_lst.append(data.extended_feat)
        out = torch.cat(feat_lst, dim=1)

        fg_embed = F.normalize(out, dim=1)
        node_cg_type_pred = self.cg_type_fc(fg_embed)

        return fg_embed, node_cg_type_pred