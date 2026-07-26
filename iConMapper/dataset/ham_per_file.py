import os
import glob
import json
import torch
import networkx as nx
import numpy as np
import torch.nn.functional as F

from networkx.algorithms.cycles import cycle_basis
from utils.automorphism_group import node_equal, edge_equal
from torch_geometric.data import Data
from torch.utils.data import Dataset
from .ham import ATOMS, BOND_TYPE_DICT, CG_TYPE_DICT, BOND_STR_TO_FLOAT
from rdkit import Chem


class HAMPerFile(Dataset):
    """Per-file dataset for inference. One JSON = one sample."""

    def __init__(self, data_root, cycle_feat=False, degree_feat=False,
                 charge_feat=False, aromatic_feat=False, automorphism=False):
        jsons_root = os.path.join(data_root, '*.json')
        self.json_file_path_lst = glob.glob(jsons_root)
        self.automorphism = automorphism
        self.cycle_feat = cycle_feat
        self.degree_feat = degree_feat
        self.charge_feat = charge_feat
        self.aromatic_feat = aromatic_feat

    def __getitem__(self, index):
        json_fpath = self.json_file_path_lst[index]

        with open(json_fpath) as f:
            json_data = json.load(f)

        data = Data()

        if 'smiles' not in json_data:
            smiles = os.path.splitext(os.path.basename(json_fpath))[0]
        else:
            smiles = json_data['smiles']

        graph = nx.Graph(smiles=smiles)
        for node in json_data['nodes']:
            graph.add_node(node['id'], element=node['element'], cg=node['cg_id'])
        for edge in json_data['edges']:
            bond_type = edge['bondtype']
            if isinstance(bond_type, str):
                bond_type = BOND_STR_TO_FLOAT[bond_type]
            graph.add_edge(edge['source'], edge['target'], bond_type=bond_type)

        # ========== atom types ==========
        fg_beads: list = json_data['nodes']
        fg_beads.sort(key=lambda x: x['id'])
        atom_types = torch.LongTensor(
            [list(ATOMS.keys()).index(bead['element']) for bead in fg_beads]).reshape(-1, 1)
        data.x = atom_types

        # ========== charges ==========
        if 'charge' in fg_beads[0]:
            data.atom_charges = torch.tensor(
                [[bead['charge']] for bead in fg_beads], dtype=torch.float)
        else:
            data.atom_charges = torch.zeros(len(fg_beads), 1)

        # ========== aromatic ==========
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumAtoms() == len(fg_beads):
            atom_aromatics = [int(atom.GetIsAromatic()) for atom in mol.GetAtoms()]
            data.atom_aromatics = torch.tensor(atom_aromatics, dtype=torch.float).reshape(-1, 1)
        else:
            data.atom_aromatics = torch.zeros(len(fg_beads), 1)

        # ========== CG types ==========
        if 'cg_type' in fg_beads[0]:
            data.atom_CG_types = torch.LongTensor(
                [list(CG_TYPE_DICT.keys()).index(bead['cg_type']) for bead in fg_beads]).reshape(-1, 1)

        # ========== extended features ==========
        extended_feats = []
        if self.degree_feat:
            degrees = np.array(graph.degree)[:, 1]
            extended_feats.append(torch.tensor(degrees, dtype=torch.float).unsqueeze(-1) / 4)
        if self.cycle_feat:
            cycle_indicator = torch.zeros(len(fg_beads), 1)
            for cycle in cycle_basis(graph):
                cycle_indicator[torch.tensor(cycle)] = 1
            extended_feats.append(cycle_indicator)
        if self.charge_feat:
            extended_feats.append(data.atom_charges)
        if self.aromatic_feat:
            extended_feats.append(data.atom_aromatics)
        if extended_feats:
            data.extended_feat = torch.cat(extended_feats, dim=1)

        # ========== edges ==========
        edges = []
        bond_types = []
        for x in json_data['edges']:
            edges.append([x['source'], x['target']])
            edges.append([x['target'], x['source']])
            bond_types.append(BOND_TYPE_DICT[x['bondtype']])
            bond_types.append(BOND_TYPE_DICT[x['bondtype']])
        data.edge_index = torch.tensor(edges, dtype=torch.long).t()
        data.edge_attr = F.one_hot(
            torch.tensor(bond_types, dtype=torch.long), num_classes=4).float()

        # ========== ground truth ==========
        original_mapping = self.compute_cluster_idx(json_data)
        if self.automorphism:
            gm = nx.isomorphism.GraphMatcher(graph, graph,
                                             node_match=node_equal,
                                             edge_match=edge_equal)
            mapping_lst = []
            for node_mapping in gm.isomorphisms_iter():
                key_value_lst = torch.tensor(list(node_mapping.items())).transpose(1, 0)
                new_mapping = original_mapping.clone()
                new_mapping[key_value_lst[0]] = new_mapping[key_value_lst[1]]
                mapping_lst.append(new_mapping)
            data.y = torch.stack(mapping_lst)
        else:
            data.y = original_mapping

        data.graph = graph
        data.json = json_data
        data.fname = os.path.splitext(os.path.basename(json_fpath))[0]

        return data

    def __len__(self):
        return len(self.json_file_path_lst)

    @staticmethod
    def compute_cluster_idx(json_data):
        node_cluster_index = -1 * torch.ones((len(json_data['nodes']),)).long()
        for node in json_data['nodes']:
            node_cluster_index[node['id']] = node['cg_id']
        assert torch.all(node_cluster_index >= 0)
        return node_cluster_index